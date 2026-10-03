export interface Sentence {
  start: number;
  end: number;
  text: string;
  /** A window should start here rather than continue through this sentence. */
  breakBefore: boolean;
}

export interface Window {
  index: number;
  start: number;
  end: number;
  text: string;
  /** Indexes into the sentence list that fall inside this window. */
  sentences: number[];
}

const segmenter = new Intl.Segmenter('en', { granularity: 'sentence' });

type LineKind = 'blank' | 'heading' | 'list' | 'quote' | 'text';

function classifyLine(line: string): LineKind {
  const trimmed = line.trim();
  if (!trimmed) return 'blank';
  if (/^#{1,6}\s/.test(trimmed)) return 'heading';
  if (/^(?:[-*+]|\d{1,3}[.)])\s/.test(trimmed)) return 'list';
  if (/^>\s?/.test(trimmed)) return 'quote';
  return 'text';
}

/**
 * `Intl.Segmenter` treats "1." as a sentence of its own, which would leave the
 * number floating away from its item in the highlighted view. Markers are swapped
 * for same-length replacements that cannot end a sentence; the offsets still
 * address the original text, so nothing here reaches the UI.
 */
function maskListMarkers(text: string): string {
  return text
    .replace(/^([ \t]*)([-*+])([ \t])/gm, '$1•$3')
    .replace(/^([ \t]*)(\d{1,3})[.)]([ \t])/gm, (_match, indent, digits, gap) => `${indent}${digits}a${gap}`);
}

/**
 * Decides whether a window should break at the gap before this sentence.
 *
 * Blank lines always break, and so do headings and the end of a list. A single
 * newline between ordinary lines breaks too, because pasted text often separates
 * paragraphs that way. Consecutive list items do *not* break: a markdown bullet
 * list is one unit of text, and scoring each bullet alone gives the detector
 * three words to work with.
 */
function breaksHere(text: string, previousEnd: number, start: number): boolean {
  const gap = text.slice(previousEnd, start);
  if (!gap.includes('\n')) return false;

  const lineBefore = text.slice(text.lastIndexOf('\n', previousEnd - 1) + 1, previousEnd);
  const before = classifyLine(lineBefore);

  // A heading is context for what follows it, so it never sits in a window of
  // its own - not even across the blank line a markdown heading is followed by.
  if (before === 'heading') return false;

  if (/\n[^\S\n]*\n/.test(gap)) return true;

  const newlineAt = previousEnd + gap.indexOf('\n');
  // The whole line, not just the slice up to the sentence: classifying "- "
  // as prose would break a bullet list into one window per bullet.
  const lineEnd = text.indexOf('\n', newlineAt + 1);
  const lineAfter = text.slice(newlineAt + 1, lineEnd === -1 ? text.length : lineEnd);
  const after = classifyLine(lineAfter);

  if (after === 'heading') return true;
  if (before === 'list' && after === 'list') return false;
  return true;
}

export function splitSentences(text: string): Sentence[] {
  const out: Sentence[] = [];
  let previousEnd = -1;

  for (const part of segmenter.segment(maskListMarkers(text))) {
    const raw = part.segment;
    const trimmed = raw.trim();
    if (!trimmed) continue;
    const start = part.index + (raw.length - raw.trimStart().length);
    const end = start + trimmed.length;
    out.push({
      start,
      end,
      text: text.slice(start, end),
      breakBefore: previousEnd >= 0 && breaksHere(text, previousEnd, start),
    });
    previousEnd = end;
  }
  return out;
}

/**
 * A window below this is not worth handing a detector: it is a handful of words
 * with no context to judge. Paragraphs are still respected above it.
 */
const MIN_WINDOW_CHARS = 160;

/**
 * Groups sentences into scoring windows.
 *
 * Three rules matter for accuracy: a window never crosses a paragraph break,
 * because detectors score a paragraph as a unit; consecutive windows overlap by
 * one sentence so boundary sentences get two opinions instead of one; and long
 * sentences are split on word boundaries rather than truncated.
 *
 * The paragraph break yields to a window that is still too small to score. A
 * document written one short line per row - a table of contents, a CV, a
 * publication list - otherwise becomes one window per line, which asks the model
 * to judge three words at a time and costs a full round of inference per line.
 */
export function buildWindows(
  sentences: Sentence[],
  targetChars: number,
  maxChars: number,
): Window[] {
  const windows: Window[] = [];
  let current: number[] = [];
  let chars = 0;

  const flush = (carryOver: boolean) => {
    if (current.length === 0) return;
    const start = sentences[current[0]].start;
    const end = sentences[current[current.length - 1]].end;
    windows.push({
      index: windows.length,
      start,
      end,
      text: textBetween(sentences, start, end),
      sentences: [...current],
    });

    const carry = carryOver && current.length > 2 ? [current[current.length - 1]] : [];
    current = carry;
    chars = carry.reduce((sum, index) => sum + sentenceLength(sentences[index]), 0);
  };

  for (let i = 0; i < sentences.length; i += 1) {
    const sentence = sentences[i];
    const length = sentenceLength(sentence);

    if (length > maxChars) {
      flush(false);
      windows.push(...splitLongSentence(sentence, maxChars, windows.length));
      continue;
    }

    if (sentence.breakBefore && current.length > 0 && chars >= MIN_WINDOW_CHARS) {
      flush(false);
    }

    current.push(i);
    chars += length;
    if (chars >= targetChars) flush(true);
  }
  flush(false);

  return windows;
}

function sentenceLength(sentence: Sentence): number {
  return sentence.end - sentence.start;
}

function splitLongSentence(sentence: Sentence, maxChars: number, offset: number): Window[] {
  const words = sentence.text.split(/\s+/);
  const out: Window[] = [];
  let buffer: string[] = [];
  let chars = 0;
  let cursor = sentence.start;

  const flush = () => {
    if (buffer.length === 0) return;
    const text = buffer.join(' ');
    out.push({
      index: offset + out.length,
      start: cursor,
      end: Math.min(sentence.end, cursor + text.length),
      text,
      sentences: [],
    });
    cursor = Math.min(sentence.end, cursor + text.length + 1);
    buffer = [];
    chars = 0;
  };

  for (const word of words) {
    buffer.push(word);
    chars += word.length + 1;
    if (chars >= maxChars) flush();
  }
  flush();
  return out;
}

function textBetween(sentences: Sentence[], start: number, end: number): string {
  return sentences
    .filter((sentence) => sentence.end > start && sentence.start < end)
    .map((sentence) => sentence.text)
    .join(' ');
}