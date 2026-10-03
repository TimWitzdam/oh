'use client';

/** Small inline glyphs; no icon dependency, no network request. */

export function CheckIcon({ className = '' }: { className?: string }) {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      aria-hidden
    >
      <path d="m3 8.5 3.2 3.2L13 5" />
    </svg>
  );
}

export function InfoIcon({ className = '' }: { className?: string }) {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      aria-hidden
    >
      <circle cx="12" cy="12" r="9" />
      <path d="M12 11.5V17" />
      <path d="M12 7.5h.01" />
    </svg>
  );
}

/** Brand mark, same artwork as public/logo.svg and src/app/icon.svg. Size it
 *  through className; the fills are fixed brand colours, not currentColor. */
export function Logo({ className = '' }: { className?: string }) {
  return (
    <svg viewBox="0 0 180 180" fill="none" className={className} aria-hidden>
      <path
        d="M90 180C139.706 180 180 139.706 180 90C180 40.2944 139.706 0 90 0C40.2944 0 0 40.2944 0 90C0 139.706 40.2944 180 90 180Z"
        fill="#FFC83B"
      />
      <path
        d="M58 88C65.732 88 72 79.0457 72 68C72 56.9543 65.732 48 58 48C50.268 48 44 56.9543 44 68C44 79.0457 50.268 88 58 88Z"
        fill="#2C2C2E"
      />
      <path
        d="M122 88C129.732 88 136 79.0457 136 68C136 56.9543 129.732 48 122 48C114.268 48 108 56.9543 108 68C108 79.0457 114.268 88 122 88Z"
        fill="#2C2C2E"
      />
      <path
        d="M90 146C101.046 146 110 137.046 110 126C110 114.954 101.046 106 90 106C78.9543 106 70 114.954 70 126C70 137.046 78.9543 146 90 146Z"
        fill="#2C2C2E"
      />
    </svg>
  );
}

export function DownloadIcon({ className = '' }: { className?: string }) {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      aria-hidden
    >
      <path d="M8 2v8m0 0 3-3m-3 3L5 7M2.5 12.5h11" />
    </svg>
  );
}