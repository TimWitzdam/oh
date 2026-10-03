import { NextResponse } from 'next/server';

import { findModel } from '@/lib/catalog';
import { activeInFlight, analyze } from '@/lib/analyze';
import { inspectInstall } from '@/lib/models';
import { readSettings } from '@/lib/store';

export const dynamic = 'force-dynamic';
export const maxDuration = 3600;

const MAX_CHARS = 400_000;

export async function POST(request: Request) {
  let text = '';
  try {
    const body = (await request.json()) as { text?: string };
    text = typeof body.text === 'string' ? body.text : '';
  } catch {
    return NextResponse.json({ error: 'invalid JSON body' }, { status: 400 });
  }

  text = text.replace(/\r\n/g, '\n').trim();
  if (text.length < 40) {
    return NextResponse.json(
      { error: 'Paste at least a paragraph so there is something to compare.' },
      { status: 400 },
    );
  }
  if (text.length > MAX_CHARS) {
    return NextResponse.json(
      { error: `Text is too long (${text.length} characters, limit ${MAX_CHARS}).` },
      { status: 413 },
    );
  }

  const settings = await readSettings();
  const spec = findModel(settings.activeModelId);
  if (!spec) {
    return NextResponse.json({ error: 'No model is active yet.' }, { status: 409 });
  }
  const installed = await inspectInstall(spec);
  if (!installed) {
    return NextResponse.json({ error: `${spec.name} is not installed.` }, { status: 409 });
  }
  if (activeInFlight() > 0) {
    return NextResponse.json({ error: 'Another analysis is already running.' }, { status: 429 });
  }

  const encoder = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      const send = (payload: unknown) => {
        controller.enqueue(encoder.encode(`${JSON.stringify(payload)}\n`));
      };
      let finished = false;
      try {
        for await (const event of analyze(text, spec, settings, request.signal)) {
          if (event.type === 'done' || event.type === 'error') finished = true;
          send(event);
        }
        // A stream that ends without a terminal event used to look identical to
        // a clean finish from the browser's side, so a half-finished run came
        // back as "the analysis stream ended before a result came back" with
        // nothing in the logs to explain it.
        if (!finished && !request.signal.aborted) {
          send({ type: 'error', message: 'The analysis stopped early and no result was produced.' });
        }
      } catch (error) {
        // Only a disconnect is silent. An AbortError while the client is still
        // connected came from somewhere else (a timeout on the hop to the
        // inference service, say) and is the actual reason the run failed.
        const clientGone = request.signal.aborted;
        if (!clientGone) {
          console.error('[analyze] failed', error);
          try {
            send({
              type: 'error',
              message: error instanceof Error ? error.message : String(error),
            });
          } catch {
            // client already gone
          }
        }
      } finally {
        try {
          controller.close();
        } catch {
          // client already gone
        }
      }
    },
  });

  return new Response(stream, {
    headers: {
      'content-type': 'application/x-ndjson; charset=utf-8',
      'cache-control': 'no-store',
      'x-accel-buffering': 'no',
    },
  });
}