'use client';

import { useCallback, useEffect, useRef, useState } from 'react';

import { cancelDownload, fetchDownloads, startDownload, streamDownload } from '@/lib/client';
import type { DownloadJob } from '@/lib/downloads';

const ACTIVE: DownloadJob['status'][] = ['queued', 'downloading', 'verifying'];
const POLL_MS = 2500;
const STALE_MS = 4000;

export function isRunning(job: DownloadJob | undefined): boolean {
  return job !== undefined && ACTIVE.includes(job.status);
}

interface Subscription {
  controller: AbortController;
  poll: ReturnType<typeof setInterval>;
}

/**
 * Keeps one server-sent-events subscription per download job, so a download can
 * be followed across page reloads and resumed after a container restart.
 *
 * Streams do get dropped: proxies time them out, and browsers suspend them in
 * background tabs. So every subscription also polls the job list whenever the
 * stream goes quiet, which means the UI recovers on its own instead of freezing
 * on a stale progress bar.
 */
export function useDownloads(onSettled: (job: DownloadJob) => void) {
  const [jobs, setJobs] = useState<Record<string, DownloadJob>>({});
  const subs = useRef(new Map<string, Subscription>());
  const settled = useRef(onSettled);

  useEffect(() => {
    settled.current = onSettled;
  }, [onSettled]);

  const remember = useCallback((job: DownloadJob) => {
    setJobs((previous) => ({ ...previous, [job.id]: job }));
  }, []);

  const attach = useCallback(
    (jobId: string) => {
      if (subs.current.has(jobId)) return;

      const controller = new AbortController();
      let lastEventAt = Date.now();
      let closed = false;

      const close = () => {
        if (closed) return;
        closed = true;
        clearInterval(subscription.poll);
        subs.current.delete(jobId);
        controller.abort();
      };

      const settle = (job: DownloadJob | undefined) => {
        if (!job) return;
        remember(job);
        if (!ACTIVE.includes(job.status)) {
          close();
          settled.current(job);
        }
      };

      const subscription: Subscription = {
        controller,
        poll: setInterval(async () => {
          if (closed || Date.now() - lastEventAt < STALE_MS) return;
          const jobs = await fetchDownloads().catch(() => null);
          if (!jobs) return;
          settle(jobs.find((entry) => entry.id === jobId));
        }, POLL_MS),
      };
      subs.current.set(jobId, subscription);

      streamDownload(
        jobId,
        (job) => {
          lastEventAt = Date.now();
          settle(job);
        },
        controller.signal,
      ).catch(() => {
        // Stream ended without a terminal event; the poll above picks it up.
      });
    },
    [remember],
  );

  const rehydrate = useCallback(
    (existing: DownloadJob[]) => {
      for (const job of existing) {
        remember(job);
        if (ACTIVE.includes(job.status)) attach(job.id);
      }
    },
    [attach, remember],
  );

  const start = useCallback(
    async (modelId: string) => {
      const job = await startDownload(modelId);
      remember(job);
      attach(job.id);
      return job;
    },
    [attach, remember],
  );

  const cancel = useCallback(async (jobId: string) => {
    // The job stream pushes the terminal 'canceled' state, so there is nothing to
    // patch here beyond asking the server to abort.
    await cancelDownload(jobId).catch(() => undefined);
  }, []);

  useEffect(() => {
    const open = subs.current;
    return () => {
      for (const subscription of open.values()) {
        clearInterval(subscription.poll);
        subscription.controller.abort();
      }
      open.clear();
    };
  }, []);

  return { jobs, start, cancel, rehydrate };
}

export function latestJobForModel(
  jobs: Record<string, DownloadJob>,
  modelId: string,
): DownloadJob | undefined {
  return Object.values(jobs)
    .filter((job) => job.modelId === modelId)
    .sort((a, b) => b.createdAt.localeCompare(a.createdAt))[0];
}