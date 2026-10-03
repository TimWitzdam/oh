import type { ModelSpec } from '../catalog';
import { INFERENCE_URL } from '../paths';
import type { Detector, ScoreUpdate } from './types';

/**
 * The deep tier is a fine-tuned 0.6B language-model detector, which needs torch
 * rather than onnxruntime. It lives in a small loopback-only Python service
 * inside the same container and streams scores back as NDJSON so the UI can
 * update while the remaining windows are still running.
 */

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
    const response = await fetch(`${INFERENCE_URL}/score`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ model: this.spec.id, texts: windows }),
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
          yield { index: message.index, ai: message.ai };
        }
      }
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