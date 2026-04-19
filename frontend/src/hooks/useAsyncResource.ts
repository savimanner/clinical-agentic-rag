import { useCallback, useEffect, useMemo, useState } from 'react';

interface AsyncResourceState<T> {
  data: T | null;
  error: Error | null;
  isLoading: boolean;
  refresh: () => Promise<T | null>;
  setData: React.Dispatch<React.SetStateAction<T | null>>;
}

export function useAsyncResource<T>(
  load: (signal: AbortSignal) => Promise<T>,
  deps: React.DependencyList,
): AsyncResourceState<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  const refresh = useCallback(async () => {
    const controller = new AbortController();
    setIsLoading(true);
    setError(null);

    try {
      const next = await load(controller.signal);
      setData(next);
      return next;
    } catch (caught) {
      const nextError = caught instanceof Error ? caught : new Error('Unknown error');
      setError(nextError);
      return null;
    } finally {
      setIsLoading(false);
    }
  }, deps);

  useEffect(() => {
    const controller = new AbortController();
    let ignore = false;

    async function run() {
      setIsLoading(true);
      setError(null);

      try {
        const next = await load(controller.signal);
        if (!ignore) {
          setData(next);
        }
      } catch (caught) {
        if (!ignore) {
          setError(caught instanceof Error ? caught : new Error('Unknown error'));
        }
      } finally {
        if (!ignore) {
          setIsLoading(false);
        }
      }
    }

    void run();

    return () => {
      ignore = true;
      controller.abort();
    };
  }, deps);

  return useMemo(
    () => ({
      data,
      error,
      isLoading,
      refresh,
      setData,
    }),
    [data, error, isLoading, refresh],
  );
}
