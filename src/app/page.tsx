import { App } from '@/components/App';
import { loadState } from '@/lib/state';

export const dynamic = 'force-dynamic';

export default async function Page() {
  const state = await loadState();
  return <App initialState={state} />;
}