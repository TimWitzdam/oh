export interface Sentence {
  start: number;
  end: number;
  text: string;
  /** A line break precedes this sentence, so a paragraph most likely starts here. */
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

export function splitSentences(text: string): Sentence[] {
  const out: Sentence[] = [];
  let previousEnd = -1;

  for (const part of segmenter.segment(text)) {
    const raw = part.segment;
    const trimmed = raw.trim();
    if (!trimmed) continue;
    const start = part.index + (raw.length - raw.trimStart().length);
    const end = start + trimmed.length;
    const gap = previousEnd >= 0 ? text.slice(previousEnd, start) : '';
    out.push({
      start,
      end,
      text: trimmed,
      // A line break between two sentences means a paragraph boundary in
      // practice: hard-wrapped prose breaks lines *inside* a sentence, and the
      // sentence splitter has already joined those.
      breakBefore: previousEnd >= 0 && /\n/.test(gap),
    });
    previousEnd = end;
  }
  return out;
}

/**
 * Groups sentences into scoring windows.
 *
 * Two rules matter for accuracy: a window never spans a paragraph break, because
 * detectors score a paragraph as a unit, and consecutive windows overlap by one
 * sentence so a sentence sitting on a boundary gets scored in context twice
 * instead of once.
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

    if (sentence.breakBefore && current.length > 0) {
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