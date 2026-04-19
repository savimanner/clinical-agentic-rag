import { useCallback } from 'react';

import { useApiClient } from '../app/api-context';
import { useAsyncResource } from './useAsyncResource';

export function useThreads() {
  const client = useApiClient();

  const load = useCallback(
    async (signal: AbortSignal) => {
      if (signal.aborted) {
        return [];
      }

      const threads = await client.listThreads();
      return [...threads].sort(
        (left, right) => new Date(right.updated_at).getTime() - new Date(left.updated_at).getTime(),
      );
    },
    [client],
  );

  return useAsyncResource(load, [load]);
}
