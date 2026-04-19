import type {
  AppendMessageRequest,
  CreateThreadRequest,
  DocumentSummary,
  ThreadDetail,
  ThreadSummary,
  UpdateThreadRequest,
} from '../types/api';

export class ApiError extends Error {
  status: number;
  detail: string;

  constructor(status: number, detail: string) {
    super(detail);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
  }
}

export interface AssistantApiClient {
  listLibrary(): Promise<DocumentSummary[]>;
  listThreads(): Promise<ThreadSummary[]>;
  createThread(payload: CreateThreadRequest): Promise<ThreadDetail>;
  getThread(threadId: string): Promise<ThreadDetail>;
  appendMessage(threadId: string, payload: AppendMessageRequest): Promise<ThreadDetail>;
  updateThread(threadId: string, payload: UpdateThreadRequest): Promise<ThreadDetail>;
  deleteThread(threadId: string): Promise<void>;
}

interface RequestOptions extends RequestInit {
  parseAs?: 'json' | 'void';
}

async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { parseAs = 'json', headers, ...rest } = options;
  const response = await fetch(`${import.meta.env.VITE_API_BASE_URL ?? ''}${path}`, {
    headers: {
      'Content-Type': 'application/json',
      ...headers,
    },
    ...rest,
  });

  if (!response.ok) {
    let detail = response.statusText || 'Request failed';
    try {
      const payload = (await response.json()) as { detail?: string };
      if (payload?.detail) {
        detail = payload.detail;
      }
    } catch {
      // Keep fallback detail when the response is not JSON.
    }
    throw new ApiError(response.status, detail);
  }

  if (parseAs === 'void' || response.status === 204) {
    return undefined as T;
  }

  return (await response.json()) as T;
}

export function createHttpApiClient(): AssistantApiClient {
  return {
    listLibrary: () => request<DocumentSummary[]>('/api/library'),
    listThreads: () => request<ThreadSummary[]>('/api/threads'),
    createThread: (payload) =>
      request<ThreadDetail>('/api/threads', {
        method: 'POST',
        body: JSON.stringify(payload),
      }),
    getThread: (threadId) => request<ThreadDetail>(`/api/threads/${threadId}`),
    appendMessage: (threadId, payload) =>
      request<ThreadDetail>(`/api/threads/${threadId}/messages`, {
        method: 'POST',
        body: JSON.stringify(payload),
      }),
    updateThread: (threadId, payload) =>
      request<ThreadDetail>(`/api/threads/${threadId}`, {
        method: 'PATCH',
        body: JSON.stringify(payload),
      }),
    deleteThread: (threadId) =>
      request<void>(`/api/threads/${threadId}`, {
        method: 'DELETE',
        parseAs: 'void',
      }),
  };
}
