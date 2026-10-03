import { NextResponse } from 'next/server';

import { assertKnownModel, downloads } from '@/lib/downloads';
import { inspectInstall } from '@/lib/models';

export const dynamic = 'force-dynamic';

export async function POST(
  _request: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const { id } = await params;
  let spec;
  try {
    spec = assertKnownModel(id);
  } catch {
    return NextResponse.json({ error: `unknown model: ${id}` }, { status: 404 });
  }

  const installed = await inspectInstall(spec);
  if (installed) {
    return NextResponse.json({ error: 'already installed' }, { status: 409 });
  }

  const existing = downloads.activeFor(spec.id);
  if (existing) {
    return NextResponse.json({ job: existing, resumed: true });
  }

  return NextResponse.json({ job: downloads.start(spec), resumed: false }, { status: 202 });
}