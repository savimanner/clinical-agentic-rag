import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';

import { App } from './app/App';
import { createHttpApiClient } from './lib/api';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App client={createHttpApiClient()} />
  </StrictMode>,
);
