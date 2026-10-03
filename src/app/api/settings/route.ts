import { NextResponse } from 'next/server';

import type { Settings } from '@/lib/catalog';
import { writeSettings } from '@/lib/store';

export const dynamic = 'force-dynamic';

export async function POST(request: Request) {
  let patch: Partial<Settings>;
  try {
    patch = (await request.json()) as Partial<Settings>;
  } catch {
    return NextResponse.json({ error: 'invalid JSON body' }, { status: 400 });
  }
  try {
    return NextResponse.json({ settings: await writeSettings(patch) });
  } catch (error) {
    // The panel keeps working off the last good settings, so the reader only
    // needs to know the change did not stick and why.
    console.error('[settings] write failed', error);
    return NextResponse.json(
      { error: `The settings could not be written: ${message(error)}` },
      { status: 500 },
    );
  }
}

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}