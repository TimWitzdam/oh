import type { ModelSpec } from '../catalog';
import { INFERENCE_URL } from '../paths';
import type { Detector, ScoreUpdate } from './types';

/**
 * The Balanced and Deep tiers are torch classifiers, which need torch rather
 * than onnxruntime. They live in a small loopback-only Python service inside the
 * same container and stream scores back as NDJSON so the UI can update while the
 * remaining windows are still running.
 */

/**
 * Windows per request to the inference service. The service caps a request at
 * the same number, so the two must move together. Batching is what makes an
 * arbitrary text length safe to score: without it a document that segments into
 * more windows than the cap (a table of contents or a CV, one short line per
 * row) is rejected outright no matter how much text the route accepted.
 */
const SCORE_BATCH = 64;

export class TorchDetector implements Detector {
  private loading: Promise<void> | null = null;

  constructor(readonly spec: ModelSpec) {}

  ready(): Promise<void> {
    if (!this.loading) {
      this.loading = this.warm().catch((error: unknown) => {
        this.loading = null;
        throw error;
      });
    }
    return this.loading;
  }

  private async warm(): Promise<void> {
    let response: Response;
    try {
      response = await fetch(`${INFERENCE_URL}/load`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ model: this.spec.id }),
        signal: AbortSignal.timeout(300_000),
      });
    } catch {
      throw new Error(
        'The Balanced and Deep tiers need the Python inference service inside the container, and it is not responding. The Lite tier does not need it.',
      );
    }
    if (!response.ok) {
      throw new Error(`The inference service could not load this model: ${await describe(response)}`);
    }
  }

  async *score(
    windows: string[],
    signal: AbortSignal,
    _batchSize: number,
  ): AsyncGenerator<ScoreUpdate> {
    await this.ready();
    let scored = 0;

    for (let offset = 0; offset < windows.length; offset += SCORE_BATCH) {
      const batch = windows.slice(offset, offset + SCORE_BATCH);
      const response = await fetch(`${INFERENCE_URL}/score`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ model: this.spec.id, texts: batch }),
        signal,
      });
      if (!response.ok) {
        throw new Error(`inference service failed: ${await describe(response)}`);
      }
      const reader = response.body?.getReader();
      if (!reader) throw new Error('inference service returned no stream');
      const decoder = new TextDecoder();
      let buffer = '';

      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';
        for (const line of lines) {
          if (!line.trim()) continue;
          const message = JSON.parse(line) as { index?: number; ai?: number; error?: string };
          if (message.error) throw new Error(message.error);
          if (typeof message.index === 'number' && typeof message.ai === 'number') {
            scored += 1;
            // The service numbers a batch from zero, so it has to be rebased
            // onto the full window list.
            yield { index: offset + message.index, ai: message.ai };
          }
        }
      }
    }

    // A stream can end early without an error line: the service stops when it
    // decides the client is gone, or the connection is cut. Returning quietly
    // would leave the uncovered sentences to be scored 0.5 downstream, which
    // reads in the UI as a real "50% machine" verdict rather than a failure.
    if (scored < windows.length) {
      throw new Error(
        `The inference service returned ${scored} of ${windows.length} scores, so the result would be incomplete.`,
      );
    }
  }

  async dispose(): Promise<void> {
    // The deep weights live in the inference service, so they have to be handed
    // back over there; only then is this side free to forget the warm handle.
    try {
      const response = await fetch(`${INFERENCE_URL}/unload`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ model: this.spec.id }),
        signal: AbortSignal.timeout(30_000),
      });
      if (!response.ok) {
        throw new Error(`the inference service kept the weights: ${await describe(response)}`);
      }
    } finally {
      this.loading = null;
    }
  }
}

async function describe(response: Response): Promise<string> {
  const body = await response.text().catch(() => '');
  return `HTTP ${response.status} ${body.slice(0, 200)}`.trim();
}