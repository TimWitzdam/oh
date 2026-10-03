import type { Metadata, Viewport } from 'next';
import localFont from 'next/font/local';

import './globals.css';

const inter = localFont({
  src: './fonts/InterVariable.woff2',
  weight: '100 900',
  style: 'normal',
  variable: '--font-inter',
  display: 'swap',
});

const mono = localFont({
  src: './fonts/JetBrainsMonoVariable.woff2',
  weight: '100 800',
  style: 'normal',
  variable: '--font-mono-local',
  display: 'swap',
});

export const metadata: Metadata = {
  title: 'oh',
  description: 'Self-hosted AI text detector. Runs on CPU, offline, no accounts.',
  robots: { index: false, follow: false },
};

export const viewport: Viewport = {
  themeColor: '#f7f5f0',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${inter.variable} ${mono.variable}`}>
      <body className="min-h-screen antialiased">{children}</body>
    </html>
  );
}