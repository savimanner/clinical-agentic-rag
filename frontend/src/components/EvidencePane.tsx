import { formatTimestamp } from '../lib/format';
import type { Citation, DocumentSummary, ThreadDetail } from '../types/api';

interface EvidencePaneProps {
  citation: Citation | null;
  library: DocumentSummary[];
  onCloseSheet: () => void;
  showMobileSheet: boolean;
  thread: ThreadDetail | null;
}

function EvidenceContent({
  citation,
  library,
  thread,
}: {
  citation: Citation | null;
  library: DocumentSummary[];
  thread: ThreadDetail | null;
}) {
  if (!citation) {
    return (
      <div className="evidence-pane__empty">
        <p className="section-label">Evidence</p>
        <h3>Select a citation.</h3>
        <p>
          Each assistant response surfaces its source chips below the answer. Pick one to inspect
          the underlying snippet, breadcrumbs, and file path.
        </p>
      </div>
    );
  }

  const document = library.find((item) => item.doc_id === citation.doc_id);
  const lastTouchedMessage = thread?.messages
    .filter((message) => message.citations?.some((item) => item.chunk_id === citation.chunk_id))
    .at(-1);

  return (
    <div className="evidence-pane__body">
      <p className="section-label">Evidence</p>
      <h3>{document?.title ?? citation.doc_id}</h3>
      <div className="evidence-metadata">
        <span>{citation.breadcrumbs}</span>
        <span>{citation.chunk_id}</span>
      </div>
      <blockquote>{citation.snippet}</blockquote>
      <dl className="evidence-facts">
        <div>
          <dt>Document ID</dt>
          <dd>{citation.doc_id}</dd>
        </div>
        <div>
          <dt>Source path</dt>
          <dd>{citation.source_path}</dd>
        </div>
        {lastTouchedMessage ? (
          <div>
            <dt>Referenced in</dt>
            <dd>{formatTimestamp(lastTouchedMessage.created_at)}</dd>
          </div>
        ) : null}
      </dl>
    </div>
  );
}

export function EvidencePane({
  citation,
  library,
  onCloseSheet,
  showMobileSheet,
  thread,
}: EvidencePaneProps) {
  return (
    <>
      <aside className="evidence-pane desktop-only">
        <EvidenceContent citation={citation} library={library} thread={thread} />
      </aside>

      {showMobileSheet && citation ? (
        <div aria-modal="true" className="citation-sheet" role="dialog">
          <button
            aria-label="Close evidence"
            className="citation-sheet__dismiss"
            onClick={onCloseSheet}
            type="button"
          >
            Close
          </button>
          <EvidenceContent citation={citation} library={library} thread={thread} />
        </div>
      ) : null}
    </>
  );
}
