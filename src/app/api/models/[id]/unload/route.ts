import { NextResponse } from 'next/server';

import { activeInFlight, releaseDetector, residentModelIds } from '@/lib/analyze';
import { assertKnownModel } from '@/lib/downloads';

export const dynamic = 'force-dynamic';

/** Frees the weights of one detector. The files on disk stay untouched. */
export async function POST(
  _request: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const { id } = await params;
  try {
    assertKnownModel(id);
  } catch {
    return NextResponse.json({ error: `unknown model: ${id}` }, { status: 404 });
  }

  // Disposing mid-run would pull the tensors out from under a running batch.
  if (activeInFlight() > 0) {
    return NextResponse.json(
      { error: 'An analysis is running. Unload once it finishes.' },
      { status: 409 },
    );
  }

  let unloaded: boolean;
  try {
    unloaded = await releaseDetector(id);
  } catch (caught) {
    return NextResponse.json(
      { error: caught instanceof Error ? caught.message : String(caught) },
      { status: 502 },
    );
  }

  return NextResponse.json({ unloaded, loaded: residentModelIds() });
}
