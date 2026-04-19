import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { App } from '../app/App';
import type { AssistantApiClient } from '../lib/api';
import type {
  AppendMessageRequest,
  Citation,
  CreateThreadRequest,
  DocumentSummary,
  ThreadDetail,
  ThreadSummary,
  UpdateThreadRequest,
} from '../types/api';

const baseLibrary: DocumentSummary[] = [
  {
    doc_id: 'hypertension-guideline',
    title: 'Hypertension Guideline',
    language: 'en',
    chunk_count: 42,
    indexed: true,
    manifest_path: 'data/hypertension/manifest.json',
  },
  {
    doc_id: 'diabetes-guideline',
    title: 'Diabetes Guideline',
    language: 'en',
    chunk_count: 28,
    indexed: true,
    manifest_path: 'data/diabetes/manifest.json',
  },
];

const citation: Citation = {
  doc_id: 'hypertension-guideline',
  chunk_id: 'hypertension-guideline::chunk_0007',
  breadcrumbs: 'Treatment > First line',
  snippet: 'Begin with lifestyle adjustment and consider ACE inhibitors when blood pressure remains elevated.',
  source_path: 'guidelines/hypertension.md',
};

const threadOne: ThreadDetail = {
  id: 'thread-1',
  title: 'Initial blood pressure question',
  created_at: '2026-04-19T08:00:00.000Z',
  updated_at: '2026-04-19T08:05:00.000Z',
  message_count: 2,
  last_message_preview: 'Follow-up pending',
  doc_ids: ['hypertension-guideline'],
  messages: [
    {
      id: 'user-1',
      role: 'user',
      content: 'What is the first-line treatment?',
      created_at: '2026-04-19T08:00:00.000Z',
    },
    {
      id: 'assistant-1',
      role: 'assistant',
      content: 'First-line therapy generally starts with lifestyle changes and ACE inhibitors.',
      created_at: '2026-04-19T08:05:00.000Z',
      citations: [citation],
      used_doc_ids: ['hypertension-guideline'],
      debug_trace: [{ step: 'retrieve' }],
    },
  ],
};

const threadTwo: ThreadDetail = {
  id: 'thread-2',
  title: 'Diabetes comparison',
  created_at: '2026-04-19T09:00:00.000Z',
  updated_at: '2026-04-19T09:15:00.000Z',
  message_count: 1,
  last_message_preview: 'How does it compare?',
  doc_ids: [],
  messages: [
    {
      id: 'user-2',
      role: 'user',
      content: 'How does it compare?',
      created_at: '2026-04-19T09:00:00.000Z',
    },
  ],
};

function toSummary(thread: ThreadDetail): ThreadSummary {
  return {
    id: thread.id,
    title: thread.title,
    created_at: thread.created_at,
    updated_at: thread.updated_at,
    last_message_preview: thread.last_message_preview,
    message_count: thread.message_count,
    doc_ids: thread.doc_ids,
  };
}

class FakeClient implements AssistantApiClient {
  library = baseLibrary;
  threads = new Map<string, ThreadDetail>([
    [threadOne.id, structuredClone(threadOne)],
    [threadTwo.id, structuredClone(threadTwo)],
  ]);

  createThreadSpy = vi.fn<(payload: CreateThreadRequest) => Promise<ThreadDetail>>();
  appendMessageSpy = vi.fn<(threadId: string, payload: AppendMessageRequest) => Promise<ThreadDetail>>();
  updateThreadSpy = vi.fn<(threadId: string, payload: UpdateThreadRequest) => Promise<ThreadDetail>>();

  async listLibrary() {
    return this.library;
  }

  async listThreads() {
    return [...this.threads.values()].map(toSummary);
  }

  async createThread(payload: CreateThreadRequest) {
    const thread: ThreadDetail = {
      id: 'thread-created',
      title: payload.title ?? 'Untitled thread',
      created_at: '2026-04-19T10:00:00.000Z',
      updated_at: '2026-04-19T10:00:00.000Z',
      last_message_preview: null,
      message_count: 0,
      doc_ids: payload.doc_ids ?? [],
      messages: [],
    };
    this.threads.set(thread.id, thread);
    this.createThreadSpy(payload);
    return structuredClone(thread);
  }

  async getThread(threadId: string) {
    const thread = this.threads.get(threadId);
    if (!thread) {
      throw new Error('Missing thread');
    }
    return structuredClone(thread);
  }

  async appendMessage(threadId: string, payload: AppendMessageRequest) {
    const thread = this.threads.get(threadId);
    if (!thread) {
      throw new Error('Missing thread');
    }

    const updated: ThreadDetail = {
      ...thread,
      updated_at: '2026-04-19T10:05:00.000Z',
      message_count: thread.message_count + 2,
      last_message_preview: payload.content,
      messages: [
        ...thread.messages,
        {
          id: `user-${thread.messages.length + 1}`,
          role: 'user',
          content: payload.content,
          created_at: '2026-04-19T10:00:00.000Z',
        },
        {
          id: `assistant-${thread.messages.length + 2}`,
          role: 'assistant',
          content: `Answer for: ${payload.content}`,
          created_at: '2026-04-19T10:05:00.000Z',
          citations: [citation],
          used_doc_ids: ['hypertension-guideline'],
          debug_trace: [{ step: 'planner' }],
        },
      ],
    };

    this.threads.set(threadId, updated);
    this.appendMessageSpy(threadId, payload);
    return structuredClone(updated);
  }

  async updateThread(threadId: string, payload: UpdateThreadRequest) {
    const thread = this.threads.get(threadId);
    if (!thread) {
      throw new Error('Missing thread');
    }

    const updated: ThreadDetail = {
      ...thread,
      title: payload.title ?? thread.title,
      doc_ids: payload.doc_ids ?? thread.doc_ids,
      updated_at: '2026-04-19T10:06:00.000Z',
    };

    this.threads.set(threadId, updated);
    this.updateThreadSpy(threadId, payload);
    return structuredClone(updated);
  }

  async deleteThread(threadId: string) {
    this.threads.delete(threadId);
  }
}

function setMobile(matches: boolean) {
  Object.defineProperty(window, 'matchMedia', {
    writable: true,
    value: vi.fn().mockImplementation((query: string) => ({
      matches,
      media: query,
      onchange: null,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      addListener: vi.fn(),
      removeListener: vi.fn(),
      dispatchEvent: vi.fn(),
    })),
  });
}

describe('App', () => {
  beforeEach(() => {
    setMobile(false);
  });

  it('renders the thread rail and switches between threads', async () => {
    const client = new FakeClient();
    const user = userEvent.setup();

    render(<App client={client} initialEntries={['/threads/thread-1']} useMemoryRouter />);

    expect(await screen.findByDisplayValue('Initial blood pressure question')).toBeInTheDocument();
    expect(
      screen.getByText('First-line therapy generally starts with lifestyle changes and ACE inhibitors.', {
        selector: 'p',
      }),
    ).toBeInTheDocument();

    await user.click(screen.getByRole('link', { name: /Diabetes comparison/i }));

    await waitFor(() => {
      expect(screen.getByDisplayValue('Diabetes comparison')).toBeInTheDocument();
    });
    expect(screen.getByText('How does it compare?', { selector: 'p' })).toBeInTheDocument();
  });

  it('creates a thread from the welcome state and sends the first message', async () => {
    const client = new FakeClient();
    client.threads.clear();
    const user = userEvent.setup();

    render(<App client={client} initialEntries={['/']} useMemoryRouter />);

    await user.type(screen.getByRole('textbox', { name: 'Message' }), 'Summarize first-line therapy');
    await user.click(screen.getByRole('button', { name: 'Send' }));

    await waitFor(() => {
      expect(client.createThreadSpy).toHaveBeenCalled();
      expect(client.appendMessageSpy).toHaveBeenCalledWith('thread-created', {
        content: 'Summarize first-line therapy',
        debug: false,
      });
    });

    expect(
      await screen.findByText('Answer for: Summarize first-line therapy', {
        selector: 'p',
      }),
    ).toBeInTheDocument();
  });

  it('holds source scoping until a thread exists', async () => {
    const client = new FakeClient();
    client.threads.clear();

    render(<App client={client} initialEntries={['/']} useMemoryRouter />);

    expect(await screen.findByText('Source scope unlocks after the first message.')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Sources' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Edit sources' })).not.toBeInTheDocument();
  });

  it('opens evidence in the desktop pane when a citation chip is selected', async () => {
    const client = new FakeClient();
    const user = userEvent.setup();

    render(<App client={client} initialEntries={['/threads/thread-1']} useMemoryRouter />);

    await user.click(await screen.findByRole('button', { name: /hypertension-guideline treatment > first line/i }));

    expect(await screen.findByRole('heading', { name: 'Hypertension Guideline' })).toBeInTheDocument();
    expect(screen.getByText(citation.snippet)).toBeInTheDocument();
  });

  it('uses a mobile citation sheet and patches thread scope from the source drawer', async () => {
    setMobile(true);
    const client = new FakeClient();
    const user = userEvent.setup();

    render(<App client={client} initialEntries={['/threads/thread-1']} useMemoryRouter />);

    await user.click(await screen.findByRole('button', { name: /hypertension-guideline treatment > first line/i }));
    expect(await screen.findByRole('dialog')).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Sources' }));
    await user.click(screen.getByLabelText(/Diabetes Guideline/i));
    await user.click(screen.getByRole('button', { name: 'Apply sources' }));

    await waitFor(() => {
      expect(client.updateThreadSpy).toHaveBeenCalledWith('thread-1', {
        doc_ids: ['hypertension-guideline', 'diabetes-guideline'],
      });
    });

    expect(await screen.findByText('Hypertension Guideline and Diabetes Guideline')).toBeInTheDocument();
  });

  it('keeps the mobile thread rail closed on a fresh workspace', async () => {
    setMobile(true);
    const client = new FakeClient();
    client.threads.clear();

    const { container } = render(<App client={client} initialEntries={['/']} useMemoryRouter />);

    expect(await screen.findByRole('heading', { name: 'Open a new line of inquiry' })).toBeInTheDocument();
    expect(container.querySelector('.thread-rail')).not.toHaveClass('open');
  });
});
