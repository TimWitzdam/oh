/**
 * Formatting-stripping for scoring.
 *
 * The detectors were trained on prose. Feeding them raw markdown pushes the
 * score down hard: a numbered list of five AI-written tips measures 94%
 * machine-written as plain text and 0% with the `1.`/`**`/`:-` markers left in.
 * So each window is normalised before it reaches a model, while the document
 * keeps its original characters for highlighting.
 *
 * Only syntax is removed: no wording changes, no sentence merging.
 */

const PATTERNS: Array<[RegExp, string]> = [
  // fenced code and inline code
  [/```[\s\S]*?```/g, ' '],
  [/`([^`]*)`/g, '$1'],
  // headings, block quotes
  [/^\s{0,3}#{1,6}\s+/gm, ''],
  [/^\s{0,3}>\s?/gm, ''],
  // Bullet and ordered list markers. These are matched inline as well as at
  // line starts: by the time a window reaches the model, its sentences have
  // been joined with spaces, so "- No latency - Works on a plane" arrives with
  // the markers in the middle of the line.
  [/(^|\s)(?:[-*+]\s+)+/g, '$1'],
  [/(^|\s)\d{1,3}[.)](?=\s)/g, '$1'],
  // images before links, so alt text survives
  [/!\[([^\]]*)\]\([^)]*\)/g, '$1'],
  [/\[([^\]]+)\]\([^)]*\)/g, '$1'],
  // emphasis
  [/\*\*\*([^*]+)\*\*\*/g, '$1'],
  [/\*\*([^*]+)\*\*/g, '$1'],
  [/(^|[^*\w])\*([^*\n]+)\*/g, '$1$2'],
  [/(^|[^\w_])_([^_\n]+)_/g, '$1$2'],
  [/~~([^~]+)~~/g, '$1'],
  // horizontal rules and table pipes
  [/^\s{0,3}([-*_]\s*){3,}$/gm, ' '],
  // setext underlines left over from the heading pass
  [/^\s{0,3}={3,}\s*$/gm, ' '],
];

export function stripFormatting(text: string): string {
  let out = text;
  for (const [pattern, replacement] of PATTERNS) {
    out = out.replace(pattern, replacement);
  }
  return out
    .replace(/\|/g, ' ')
    .replace(/[ \t]+/g, ' ')
    .replace(/ ?\n ?/g, ' ')
    .replace(/ {2,}/g, ' ')
    .trim();
}

/** True when the window carries markup worth removing. */
export function hasFormatting(text: string): boolean {
  return stripFormatting(text) !== text.replace(/\s*\n\s*/g, ' ').trim();
}