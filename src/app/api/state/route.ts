import { NextResponse } from 'next/server';

import { loadState } from '@/lib/state';

export const dynamic = 'force-dynamic';

export async function GET() {
  return NextResponse.json(await loadState());
}