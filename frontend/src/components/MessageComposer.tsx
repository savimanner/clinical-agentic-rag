import { FormEvent, KeyboardEvent, useState, useTransition } from 'react';

import type { DocumentSummary } from '../types/api';

interface MessageComposerProps {
  activeScopeDocIds: string[];
  debugEnabled: boolean;
  isDisabled: boolean;
  isSending: boolean;
  library: DocumentSummary[];
  onOpenSources: () => void;
  onSubmit: (content: string, debug: boolean) => Promise<void>;
  onToggleDebug: () => void;
}

export function MessageComposer({
  activeScopeDocIds,
  debugEnabled,
  isDisabled,
  isSending,
  library,
  onOpenSources,
  onToggleDebug,
  onSubmit,
}: MessageComposerProps) {
  const [content, setContent] = useState('');
  const [isPending, startTransition] = useTransition();

  const selectedDocuments = activeScopeDocIds
    .map((docId) => library.find((document) => document.doc_id === docId))
    .filter((document): document is DocumentSummary => Boolean(document));

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const next = content.trim();
    if (!next || isDisabled || isSending || isPending) {
      return;
    }

    setContent('');
    startTransition(() => {
      void onSubmit(next, debugEnabled);
    });
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') {
      event.preventDefault();
      const form = event.currentTarget.form;
      form?.requestSubmit();
    }
  }

  return (
    <form className="composer" onSubmit={handleSubmit}>
      <div className="composer__toolbar">
        <div className="composer__scope">
          <p className="section-label">Scope</p>
          <div className="scope-chip-row">
            {selectedDocuments.length === 0 ? (
              <span className="scope-chip">All indexed documents</span>
            ) : (
              selectedDocuments.map((document) => (
                <span className="scope-chip" key={document.doc_id}>
                  {document.title}
                </span>
              ))
            )}
          </div>
        </div>

        <button className="ghost-button" onClick={onOpenSources} type="button">
          Edit sources
        </button>
      </div>

      <label className="composer__field">
        <span className="sr-only">Message</span>
        <textarea
          aria-label="Message"
          disabled={isDisabled || isSending || isPending}
          onChange={(event) => setContent(event.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask a question, refine the follow-up, or request a comparison between sections..."
          rows={4}
          value={content}
        />
      </label>

      <div className="composer__footer">
        <label className="composer__debug">
          <input checked={debugEnabled} onChange={onToggleDebug} type="checkbox" />
          <span>Include developer trace</span>
        </label>
        <p>Press Ctrl/Cmd + Enter to send.</p>
        <button
          className="primary-button"
          disabled={isDisabled || isSending || isPending || content.trim().length === 0}
          type="submit"
        >
          {isSending || isPending ? 'Sending...' : 'Send'}
        </button>
      </div>
    </form>
  );
}
