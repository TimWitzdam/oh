import { NextResponse } from 'next/server';

import type { DownloadJob } from '@/lib/downloads';
import { downloads } from '@/lib/downloads';

export const dynamic = 'force-dynamic';

const TERMINAL = new Set(['done', 'error', 'canceled']);

export async function GET(
  request: Request,
  { params }: { params: Promise<{ jobId: string }> },
) {
  const { jobId } = await params;
  const existing = downloads.get(jobId);
  if (!existing) {
    return NextResponse.json({ error: `unknown download: ${jobId}` }, { status: 404 });
  }

  const encoder = new TextEncoder();
  let cleanup = () => {};

  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      let closed = false;
      const close = () => {
        if (closed) return;
        closed = true;
        cleanup();
        try {
          controller.close();
        } catch {
          // stream already torn down by the client
        }
      };
      const send = (job: DownloadJob) => {
        if (closed) return;
        controller.enqueue(encoder.encode(`data: ${JSON.stringify(job)}\n\n`));
        if (TERMINAL.has(job.status)) close();
      };

      const unsubscribe = downloads.subscribe(jobId, send);
      const heartbeat = setInterval(() => {
        if (!closed) controller.enqueue(encoder.encode(': keep-alive\n\n'));
      }, 15_000);
      request.signal.addEventListener('abort', close);
      cleanup = () => {
        clearInterval(heartbeat);
        unsubscribe?.();
        request.signal.removeEventListener('abort', close);
      };

      send(downloads.get(jobId) ?? existing);
    },
    cancel() {
      cleanup();
    },
  });

  return new Response(stream, {
    headers: {
      'content-type': 'text/event-stream; charset=utf-8',
      'cache-control': 'no-store',
      connection: 'keep-alive',
    },
  });
}

export async function DELETE(
  _request: Request,
  { params }: { params: Promise<{ jobId: string }> },
) {
  const { jobId } = await params;
  const job = downloads.cancel(jobId);
  if (!job) {
    return NextResponse.json({ error: `unknown download: ${jobId}` }, { status: 404 });
  }
  return NextResponse.json({ job });
}