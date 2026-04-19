import { startTransition, useEffect, useMemo, useState } from 'react';
import { BrowserRouter, MemoryRouter, Route, Routes, useNavigate, useParams } from 'react-router-dom';

import { ApiClientProvider, useApiClient } from './api-context';
import { ConversationPane } from '../components/ConversationPane';
import { EvidencePane } from '../components/EvidencePane';
import { MessageComposer } from '../components/MessageComposer';
import { SourceDrawer } from '../components/SourceDrawer';
import { ThreadHeader } from '../components/ThreadHeader';
import { ThreadRail } from '../components/ThreadRail';
import { useLibrary } from '../hooks/useLibrary';
import { useMediaQuery } from '../hooks/useMediaQuery';
import { useThreadDetail } from '../hooks/useThreadDetail';
import { useThreads } from '../hooks/useThreads';
import { buildThreadTitle } from '../lib/format';
import type { AssistantApiClient } from '../lib/api';
import type { Citation } from '../types/api';
import '../styles.css';

function Workspace() {
  const client = useApiClient();
  const navigate = useNavigate();
  const { threadId } = useParams();
  const activeThreadId = threadId ?? null;
  const isMobile = useMediaQuery('(max-width: 980px)');
  const [activeCitation, setActiveCitation] = useState<Citation | null>(null);
  const [isRailOpen, setIsRailOpen] = useState(false);
  const [isSourceDrawerOpen, setIsSourceDrawerOpen] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const [debugEnabled, setDebugEnabled] = useState(false);
  const [workspaceError, setWorkspaceError] = useState<string | null>(null);

  const library = useLibrary();
  const threads = useThreads();
  const threadDetail = useThreadDetail(activeThreadId);

  useEffect(() => {
    setActiveCitation(null);
  }, [activeThreadId]);

  useEffect(() => {
    if (!activeThreadId && !threads.isLoading && (threads.data?.length ?? 0) > 0) {
      navigate(`/threads/${threads.data?.[0].id}`, { replace: true });
    }
  }, [activeThreadId, navigate, threads.data, threads.isLoading]);

  const activeScopeDocIds = threadDetail.data?.doc_ids ?? [];
  const detailErrorMessage = threadDetail.error?.message ?? null;
  const canUseWorkspace = library.error == null && threads.error == null;

  async function refreshThreadList() {
    startTransition(() => {
      void threads.refresh();
    });
  }

  async function handleCreateThread() {
    setWorkspaceError(null);

    try {
      const thread = await client.createThread({ title: 'Untitled thread' });
      threadDetail.setData(thread);
      await refreshThreadList();
      navigate(`/threads/${thread.id}`);
      setIsRailOpen(false);
    } catch (caught) {
      setWorkspaceError(caught instanceof Error ? caught.message : 'Unable to create thread.');
    }
  }

  async function handleSendMessage(content: string, debug: boolean) {
    setWorkspaceError(null);
    setIsSending(true);

    try {
      let thread = threadDetail.data;

      if (!thread) {
        thread = await client.createThread({ title: buildThreadTitle(content) });
        threadDetail.setData(thread);
        await refreshThreadList();
        navigate(`/threads/${thread.id}`);
      }

      const nextThread = await client.appendMessage(thread.id, { content, debug });
      threadDetail.setData(nextThread);
      setActiveCitation(nextThread.messages.at(-1)?.citations?.[0] ?? null);
      await refreshThreadList();
    } catch (caught) {
      setWorkspaceError(caught instanceof Error ? caught.message : 'Unable to send message.');
    } finally {
      setIsSending(false);
    }
  }

  async function handleRenameThread(title: string) {
    if (!threadDetail.data) {
      return;
    }

    setWorkspaceError(null);
    try {
      const nextThread = await client.updateThread(threadDetail.data.id, { title });
      threadDetail.setData(nextThread);
      await refreshThreadList();
    } catch (caught) {
      setWorkspaceError(caught instanceof Error ? caught.message : 'Unable to update thread.');
    }
  }

  async function handleScopeUpdate(docIds: string[]) {
    if (!threadDetail.data) {
      setWorkspaceError('Create a thread before changing its source scope.');
      return;
    }

    setWorkspaceError(null);
    try {
      const nextThread = await client.updateThread(threadDetail.data.id, { doc_ids: docIds });
      threadDetail.setData(nextThread);
      setIsSourceDrawerOpen(false);
      await refreshThreadList();
    } catch (caught) {
      setWorkspaceError(caught instanceof Error ? caught.message : 'Unable to update sources.');
    }
  }

  async function handleDeleteThread(threadId: string) {
    setWorkspaceError(null);
    try {
      await client.deleteThread(threadId);
      threadDetail.setData(null);
      setActiveCitation(null);
      await refreshThreadList();
      navigate('/');
    } catch (caught) {
      setWorkspaceError(caught instanceof Error ? caught.message : 'Unable to delete thread.');
    }
  }

  const threadList = threads.data ?? [];
  const documents = library.data ?? [];

  const statusMessage = useMemo(() => {
    if (workspaceError) {
      return workspaceError;
    }
    if (threads.error) {
      return threads.error.message;
    }
    if (library.error) {
      return library.error.message;
    }
    if (detailErrorMessage) {
      return detailErrorMessage;
    }
    return null;
  }, [detailErrorMessage, library.error, threads.error, workspaceError]);

  const showMobileSheet = Boolean(isMobile && activeCitation);

  return (
    <div className="app-shell">
      <ThreadRail
        activeScopeDocIds={activeScopeDocIds}
        activeThreadId={activeThreadId}
        isLoading={threads.isLoading}
        isOpen={isRailOpen || (!activeThreadId && isMobile)}
        library={documents}
        onClose={() => setIsRailOpen(false)}
        onCreateThread={() => void handleCreateThread()}
        threads={threadList}
      />

      <div className="workspace">
        <ThreadHeader
          onDeleteThread={handleDeleteThread}
          onOpenRail={() => setIsRailOpen(true)}
          onOpenSources={() => setIsSourceDrawerOpen(true)}
          onRenameThread={handleRenameThread}
          thread={threadDetail.data}
        />

        {statusMessage ? (
          <div className="status-banner" role="status">
            <strong>Workspace status:</strong> {statusMessage}
          </div>
        ) : null}

        <div className="workspace__body">
          <main className="workspace__main">
            <ConversationPane
              activeCitation={activeCitation}
              isBusy={isSending}
              onSelectCitation={(citation) => setActiveCitation(citation)}
              thread={threadDetail.data}
            />

            <MessageComposer
              activeScopeDocIds={activeScopeDocIds}
              debugEnabled={debugEnabled}
              isDisabled={!canUseWorkspace}
              isSending={isSending}
              library={documents}
              onOpenSources={() => setIsSourceDrawerOpen(true)}
              onToggleDebug={() => setDebugEnabled((current) => !current)}
              onSubmit={handleSendMessage}
            />
          </main>

          <EvidencePane
            citation={activeCitation}
            library={documents}
            onCloseSheet={() => setActiveCitation(null)}
            showMobileSheet={showMobileSheet}
            thread={threadDetail.data}
          />
        </div>
      </div>

      <SourceDrawer
        isLoading={library.isLoading}
        isOpen={isSourceDrawerOpen}
        library={documents}
        onApply={handleScopeUpdate}
        onClose={() => setIsSourceDrawerOpen(false)}
        selectedDocIds={activeScopeDocIds}
      />
    </div>
  );
}

function AppRouter() {
  return (
    <Routes>
      <Route element={<Workspace />} path="/" />
      <Route element={<Workspace />} path="/threads/:threadId" />
    </Routes>
  );
}

interface AppProps {
  client: AssistantApiClient;
  initialEntries?: string[];
  useMemoryRouter?: boolean;
}

export function App({ client, initialEntries, useMemoryRouter = false }: AppProps) {
  if (useMemoryRouter) {
    return (
      <ApiClientProvider client={client}>
        <MemoryRouter
          future={{ v7_relativeSplatPath: true, v7_startTransition: true }}
          initialEntries={initialEntries}
        >
          <AppRouter />
        </MemoryRouter>
      </ApiClientProvider>
    );
  }

  return (
    <ApiClientProvider client={client}>
      <BrowserRouter future={{ v7_relativeSplatPath: true, v7_startTransition: true }}>
        <AppRouter />
      </BrowserRouter>
    </ApiClientProvider>
  );
}
