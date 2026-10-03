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
  const settings = await writeSettings(patch);
  return NextResponse.json({ settings });
}