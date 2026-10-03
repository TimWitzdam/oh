import { NextResponse } from 'next/server';

import { assertKnownModel, downloads } from '@/lib/downloads';
import { removeModel } from '@/lib/models';

export const dynamic = 'force-dynamic';

export async function DELETE(
  _request: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const { id } = await params;
  try {
    assertKnownModel(id);
  } catch {
    return NextResponse.json({ error: `unknown model: ${id}` }, { status: 404 });
  }
  await downloads.purge(assertKnownModel(id));
  await removeModel(id);
  return NextResponse.json({ removed: id });
}