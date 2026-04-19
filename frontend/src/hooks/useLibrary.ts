import { useCallback } from 'react';

import { useApiClient } from '../app/api-context';
import { useAsyncResource } from './useAsyncResource';

export function useLibrary() {
  const client = useApiClient();

  const load = useCallback(
    async (signal: AbortSignal) => {
      if (signal.aborted) {
        return [];
      }
      return client.listLibrary();
    },
    [client],
  );

  return useAsyncResource(load, [load]);
}
