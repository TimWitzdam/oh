/**
 * Two passages to try a detector on, written for this app so nothing here is
 * someone else's text. They exist to show the shape of what the models read
 * rather than to be a benchmark: one paragraph in a personal register, one in
 * the flat, hedged register these detectors are trained to catch.
 */

export interface Sample {
  id: string;
  label: string;
  note: string;
  text: string;
}

export const SAMPLES: Sample[] = [
  {
    id: 'personal',
    label: 'A note in someone’s own voice',
    note: 'Specific, uneven, full of small useless details.',
    text: `The bus was late again, which made the third of four meetings feel like a formality. I had the numbers ready, but nobody asked for them until the chair finally read the agenda out loud. Afterwards Sanne said the room had been tense since nine, and she could tell because people had started typing. That is the sort of thing no summary captures, which is why I keep the minutes by hand instead of typing them up later. The coffee machine on the second floor has been broken since March. Someone taped a sign to it that says SORRY, and nobody has corrected the spelling.`,
  },
  {
    id: 'generic',
    label: 'A paragraph in the usual register',
    note: 'Round, hedged, and not really saying anything.',
    text: `It is important to understand that artificial intelligence is transforming the way we live and work. In this article, we will explore the many benefits that AI can bring to everyday life. From healthcare to education, there are countless opportunities for innovation. Additionally, businesses can use AI to improve efficiency and reduce costs. Overall, AI is a powerful tool that can help us solve some of the world's most pressing problems. In conclusion, it is clear that the future belongs to those who are willing to embrace this technology.`,
  },
];