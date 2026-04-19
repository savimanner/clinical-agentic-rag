import { useCallback } from 'react';

import { useApiClient } from '../app/api-context';
import type { ThreadDetail } from '../types/api';
import { useAsyncResource } from './useAsyncResource';

export function useThreadDetail(threadId: string | null) {
  const client = useApiClient();

  const load = useCallback(
    async (signal: AbortSignal): Promise<ThreadDetail | null> => {
      if (!threadId || signal.aborted) {
        return null;
      }

      return client.getThread(threadId);
    },
    [client, threadId],
  );

  return useAsyncResource(load, [load]);
}
