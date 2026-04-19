import { useDeferredValue, useEffect, useMemo, useState, useTransition } from 'react';

import type { DocumentSummary } from '../types/api';

interface SourceDrawerProps {
  isLoading: boolean;
  isOpen: boolean;
  library: DocumentSummary[];
  onApply: (docIds: string[]) => Promise<void>;
  onClose: () => void;
  selectedDocIds: string[];
}

export function SourceDrawer({
  isLoading,
  isOpen,
  library,
  onApply,
  onClose,
  selectedDocIds,
}: SourceDrawerProps) {
  const [draftSelection, setDraftSelection] = useState<string[]>(selectedDocIds);
  const [search, setSearch] = useState('');
  const [isPending, startTransition] = useTransition();
  const deferredSearch = useDeferredValue(search);

  useEffect(() => {
    setDraftSelection(selectedDocIds);
  }, [selectedDocIds]);

  const visibleDocuments = useMemo(() => {
    const query = deferredSearch.trim().toLowerCase();
    if (!query) {
      return library;
    }
    return library.filter((document) => {
      const haystack = `${document.title} ${document.doc_id}`.toLowerCase();
      return haystack.includes(query);
    });
  }, [deferredSearch, library]);

  function toggleSelection(docId: string) {
    setDraftSelection((current) =>
      current.includes(docId) ? current.filter((item) => item !== docId) : [...current, docId],
    );
  }

  return (
    <>
      <div
        aria-hidden={!isOpen}
        className={`shell-overlay ${isOpen ? 'visible visible--drawer' : ''}`}
        onClick={onClose}
      />
      <aside className={`source-drawer ${isOpen ? 'open' : ''}`}>
        <div className="source-drawer__header">
          <p className="eyebrow">Source scope</p>
          <h2>Decide what this thread can draw from.</h2>
          <p>
            Leave everything unchecked to search across the full indexed library, or narrow the
            thread to a smaller working set.
          </p>
        </div>

        <label className="source-drawer__search">
          <span className="sr-only">Search sources</span>
          <input
            onChange={(event) => setSearch(event.target.value)}
            placeholder="Search titles or doc IDs"
            value={search}
          />
        </label>

        <div className="source-drawer__actions">
          <button className="ghost-button" onClick={() => setDraftSelection([])} type="button">
            Use full library
          </button>
          <span>{draftSelection.length === 0 ? 'All documents' : `${draftSelection.length} selected`}</span>
        </div>

        <div className="source-list" role="list">
          {isLoading ? (
            <div className="thread-skeleton-list" aria-label="Loading sources">
              <div className="thread-skeleton" />
              <div className="thread-skeleton" />
              <div className="thread-skeleton" />
            </div>
          ) : (
            visibleDocuments.map((document) => {
              const checked = draftSelection.includes(document.doc_id);

              return (
                <label className="source-list__item" key={document.doc_id}>
                  <input
                    checked={checked}
                    onChange={() => toggleSelection(document.doc_id)}
                    type="checkbox"
                  />
                  <div>
                    <strong>{document.title}</strong>
                    <span>
                      {document.doc_id} · {document.chunk_count} chunks ·{' '}
                      {document.indexed ? 'Indexed' : 'Not indexed'}
                    </span>
                  </div>
                </label>
              );
            })
          )}
        </div>

        <div className="source-drawer__footer">
          <button className="ghost-button" onClick={onClose} type="button">
            Cancel
          </button>
          <button
            className="primary-button"
            disabled={isPending}
            onClick={() => {
              startTransition(() => {
                void onApply(draftSelection);
              });
            }}
            type="button"
          >
            {isPending ? 'Saving...' : 'Apply sources'}
          </button>
        </div>
      </aside>
    </>
  );
}
