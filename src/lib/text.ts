/**
 * Text rules shared by the composer, the result panel and the routes.
 *
 * The composer shows a word count and refuses to run a paragraph-less box, so
 * the numbers it counts with have to be the numbers the server counts with.
 * One place, both sides.
 */

/** Below this a window says nothing about the document. The server refuses it too. */
export const MIN_WORDS = 40;

/** Character ceiling for one run. Enforced by /api/analyze, /api/pdf and imports. */
export const MAX_TEXT_CHARS = 400_000;

/** Whitespace-separated tokens, which is what the segmenter calls a word. */
export function countWords(text: string): number {
  const trimmed = text.trim();
  return trimmed ? trimmed.split(/\s+/).length : 0;
}

/** Zero-width and bidi marks, which ride along with text copied out of documents. */
const ZERO_WIDTH = /[\u200b\u200c\u200d\u2060\u206e\ufeff\u200e\u200f]/g;
/** Counted for the paste note, and including the non-breaking space. */
const STRAY = /[\u00a0\u200b\u200c\u200d\u2060\u206e\ufeff\u200e\u200f]/g;

/**
 * Line endings and invisible characters, which arrive by the cartload from
 * Word, Google Docs and PDFs and are noise to a model. A non-breaking space
 * becomes an ordinary one rather than disappearing, which would weld the two
 * words around it together. Markup is deliberately left alone: the document
 * keeps the characters the user gave us so the highlighting lines up with what
 * is in the box (see normalize.ts, which strips formatting per window at
 * scoring time instead).
 *
 * Returns how many characters were repaired, so a paste can say so.
 */
export function tidyText(raw: string): { text: string; fixes: number } {
  const lineEndings = raw.match(/\r\n?/g)?.length ?? 0;
  const strays = raw.match(STRAY)?.length ?? 0;
  const text = raw
    .replace(/\r\n?/g, '\n')
    .replace(/\u00a0/g, ' ')
    .replace(ZERO_WIDTH, '')
    .replace(/[ \t]+$/gm, '')
    .replace(/\n{3,}/g, '\n\n');
  return { text, fixes: lineEndings + strays };
}

/** Wall clock guess from a tier's measured throughput, for before the run starts. */
export function estimateSeconds(words: number, wordsPerSecond: number): number {
  return words <= 0 ? 0 : words / Math.max(1, wordsPerSecond);
}

/** Rough by design: the point is "seconds, not minutes", not a stopwatch. */
export function formatDuration(seconds: number): string {
  if (seconds < 1) return 'under a second';
  if (seconds < 60) return `about ${Math.round(seconds)} s`;
  return `about ${Math.round(seconds / 60)} min`;
}