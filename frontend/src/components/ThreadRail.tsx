import { Link } from 'react-router-dom';

import { formatRelativeTime, summarizeScope } from '../lib/format';
import type { DocumentSummary, ThreadSummary } from '../types/api';

interface ThreadRailProps {
  activeThreadId: string | null;
  activeScopeDocIds: string[];
  isLoading: boolean;
  isOpen: boolean;
  library: DocumentSummary[];
  onClose: () => void;
  onCreateThread: () => void;
  threads: ThreadSummary[];
}

export function ThreadRail({
  activeThreadId,
  activeScopeDocIds,
  isLoading,
  isOpen,
  library,
  onClose,
  onCreateThread,
  threads,
}: ThreadRailProps) {
  return (
    <>
      <div
        aria-hidden={!isOpen}
        className={`shell-overlay ${isOpen ? 'visible' : ''}`}
        onClick={onClose}
      />
      <aside className={`thread-rail ${isOpen ? 'open' : ''}`}>
        <div className="thread-rail__header">
          <p className="eyebrow">Local-first retrieval studio</p>
          <h1>Guideline Studio</h1>
          <p className="thread-rail__lede">
            Editorial chat with persistent threads, evidence-first reading, and a quiet
            workspace for follow-up questions.
          </p>
        </div>

        <button className="primary-button primary-button--full" onClick={onCreateThread} type="button">
          New Thread
        </button>

        <section className="thread-rail__scope">
          <p className="section-label">Current scope</p>
          <strong>{summarizeScope(activeScopeDocIds, library)}</strong>
          <span>{activeScopeDocIds.length === 0 ? `${library.length} documents available` : `${activeScopeDocIds.length} selected documents`}</span>
        </section>

        <section className="thread-rail__list">
          <div className="thread-rail__section-heading">
            <p className="section-label">Recent threads</p>
            <span>{threads.length}</span>
          </div>

          {isLoading ? (
            <div className="thread-skeleton-list" aria-label="Loading threads">
              <div className="thread-skeleton" />
              <div className="thread-skeleton" />
              <div className="thread-skeleton" />
            </div>
          ) : threads.length === 0 ? (
            <div className="thread-empty">
              <p>No threads yet.</p>
              <span>Start with a question and the first thread will appear here.</span>
            </div>
          ) : (
            <nav aria-label="Recent threads" className="thread-nav">
              {threads.map((thread) => (
                <Link
                  className={`thread-link ${thread.id === activeThreadId ? 'active' : ''}`}
                  key={thread.id}
                  onClick={onClose}
                  to={`/threads/${thread.id}`}
                >
                  <span className="thread-link__title">{thread.title}</span>
                  <span className="thread-link__meta">
                    {thread.message_count} messages · {formatRelativeTime(thread.updated_at)}
                  </span>
                  {thread.last_message_preview ? (
                    <span className="thread-link__preview">{thread.last_message_preview}</span>
                  ) : null}
                </Link>
              ))}
            </nav>
          )}
        </section>
      </aside>
    </>
  );
}
