import { NextResponse } from 'next/server';

import { downloads } from '@/lib/downloads';

export const dynamic = 'force-dynamic';

export async function GET() {
  return NextResponse.json({ jobs: downloads.list() });
}