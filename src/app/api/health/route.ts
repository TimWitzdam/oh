import { NextResponse } from 'next/server';

import { INFERENCE_URL } from '@/lib/paths';

export const dynamic = 'force-dynamic';

const TTL_MS = 15_000;
const PROBE_TIMEOUT_MS = 2_000;

let cached: { at: number; value: unknown } | null = null;

/**
 * The app is healthy on its own; the inference service is reported as extra
 * information. Its result is cached briefly so a busy torch process cannot make
 * a health check time out.
 */
export async function GET() {
  const now = Date.now();
  if (!cached || now - cached.at > TTL_MS) {
    cached = {
      at: now,
      value: await fetch(`${INFERENCE_URL}/health`, {
        cache: 'no-store',
        signal: AbortSignal.timeout(PROBE_TIMEOUT_MS),
      })
        .then(async (response) => (response.ok ? await response.json() : null))
        .catch(() => null),
    };
  }
  return NextResponse.json({ ok: true, inference: cached.value });
}
