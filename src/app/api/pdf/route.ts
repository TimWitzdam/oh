import { NextResponse } from 'next/server';
import { extractText, getDocumentProxy } from 'unpdf';

export const dynamic = 'force-dynamic';
export const maxDuration = 300;

/** Generous enough for a scanned manuscript, small enough to keep RAM in check. */
const MAX_BYTES = 40 * 1024 * 1024;
/** Same ceiling POST /api/analyze enforces, so imported text always fits. */
const MAX_CHARS = 400_000;

/** Pulls the text layer out of a PDF so it can be scored like pasted text. */
export async function POST(request: Request) {
  const declared = Number(request.headers.get('content-length') ?? '0');
  if (Number.isFinite(declared) && declared > MAX_BYTES) {
    return NextResponse.json({ error: tooBig() }, { status: 413 });
  }

  let bytes: Uint8Array;
  try {
    bytes = new Uint8Array(await request.arrayBuffer());
  } catch {
    return NextResponse.json({ error: 'The upload did not arrive intact.' }, { status: 400 });
  }
  if (bytes.byteLength === 0) {
    return NextResponse.json({ error: 'That file was empty.' }, { status: 400 });
  }
  if (bytes.byteLength > MAX_BYTES) {
    return NextResponse.json({ error: tooBig() }, { status: 413 });
  }
  if (new TextDecoder('latin1').decode(bytes.subarray(0, 5)) !== '%PDF-') {
    return NextResponse.json({ error: 'That file is not a PDF.' }, { status: 415 });
  }

  let pages: number;
  let text: string;
  try {
    const document = await getDocumentProxy(bytes);
    const extracted = await extractText(document, { mergePages: true });
    pages = extracted.totalPages;
    text = extracted.text;
  } catch (error) {
    // pdfjs rejects anything it cannot parse, from a bad header to a damaged
    // xref table. Either way the fix is on the file, not on our side.
    return NextResponse.json(
      { error: `This PDF could not be read: ${error instanceof Error ? error.message : String(error)}` },
      { status: 422 },
    );
  }

  const truncated = text.length > MAX_CHARS;
  if (truncated) text = text.slice(0, MAX_CHARS);
  text = text.replace(/\r\n/g, '\n').trim();

  // A PDF of nothing but page images has no text layer to read. OCR first.
  if (text.length < 40) {
    return NextResponse.json(
      { error: 'No text came out of this PDF. It looks like a scan of page images — run it through OCR first.' },
      { status: 422 },
    );
  }

  return NextResponse.json({ text, pages, truncated });
}

function tooBig(): string {
  return `That PDF is larger than the ${MAX_BYTES / 1024 / 1024} MB limit.`;
}
