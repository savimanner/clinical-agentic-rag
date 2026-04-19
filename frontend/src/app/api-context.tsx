import { createContext, useContext } from 'react';

import type { AssistantApiClient } from '../lib/api';

const ApiClientContext = createContext<AssistantApiClient | null>(null);

export function ApiClientProvider({
  client,
  children,
}: React.PropsWithChildren<{ client: AssistantApiClient }>) {
  return <ApiClientContext.Provider value={client}>{children}</ApiClientContext.Provider>;
}

export function useApiClient(): AssistantApiClient {
  const client = useContext(ApiClientContext);

  if (!client) {
    throw new Error('Assistant API client is not configured.');
  }

  return client;
}
