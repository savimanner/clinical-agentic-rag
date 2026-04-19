import { useEffect, useState, useTransition } from 'react';

import type { ThreadDetail } from '../types/api';

interface ThreadHeaderProps {
  thread: ThreadDetail | null;
  onDeleteThread: (threadId: string) => Promise<void>;
  onOpenRail: () => void;
  onOpenSources: () => void;
  onRenameThread: (title: string) => Promise<void>;
}

export function ThreadHeader({
  thread,
  onDeleteThread,
  onOpenRail,
  onOpenSources,
  onRenameThread,
}: ThreadHeaderProps) {
  const [draftTitle, setDraftTitle] = useState(thread?.title ?? 'New thread');
  const [isPending, startTransition] = useTransition();

  useEffect(() => {
    setDraftTitle(thread?.title ?? 'New thread');
  }, [thread?.id, thread?.title]);

  const canSave = thread && draftTitle.trim() && draftTitle.trim() !== thread.title;

  return (
    <header className="thread-header">
      <div className="thread-header__identity">
        <button
          aria-label="Open threads"
          className="icon-button mobile-only"
          onClick={onOpenRail}
          type="button"
        >
          Threads
        </button>

        <div className="thread-header__titles">
          <p className="section-label">
            {thread ? `${thread.message_count} messages in thread` : 'New conversation'}
          </p>
          {thread ? (
            <div className="thread-title-editor">
              <input
                aria-label="Thread title"
                className="thread-title-editor__input"
                disabled={isPending}
                onChange={(event) => setDraftTitle(event.target.value)}
                value={draftTitle}
              />
              <button
                className="ghost-button"
                disabled={!canSave || isPending}
                onClick={() => {
                  if (!canSave) {
                    return;
                  }
                  startTransition(() => {
                    void onRenameThread(draftTitle.trim());
                  });
                }}
                type="button"
              >
                {isPending ? 'Saving...' : 'Save'}
              </button>
            </div>
          ) : (
            <h2>Open a new line of inquiry</h2>
          )}
        </div>
      </div>

      <div className="thread-header__actions">
        <button className="ghost-button" onClick={onOpenSources} type="button">
          Sources
        </button>
        {thread ? (
          <button
            className="ghost-button ghost-button--danger"
            onClick={() => void onDeleteThread(thread.id)}
            type="button"
          >
            Delete
          </button>
        ) : null}
      </div>
    </header>
  );
}
