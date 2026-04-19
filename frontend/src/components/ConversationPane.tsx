import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

import { formatTimestamp } from '../lib/format';
import type { Citation, ThreadDetail, ThreadMessage } from '../types/api';

interface ConversationPaneProps {
  activeCitation: Citation | null;
  isBusy: boolean;
  onSelectCitation: (citation: Citation) => void;
  thread: ThreadDetail | null;
}

function MessageBubble({
  activeCitation,
  message,
  onSelectCitation,
}: {
  activeCitation: Citation | null;
  message: ThreadMessage;
  onSelectCitation: (citation: Citation) => void;
}) {
  const isAssistant = message.role === 'assistant';
  const citations = message.citations ?? [];

  return (
    <article className={`message-bubble ${message.role}`}>
      <div className="message-bubble__meta">
        <span className="message-bubble__role">{message.role}</span>
        <time dateTime={message.created_at}>{formatTimestamp(message.created_at)}</time>
      </div>

      <div className="message-bubble__content">
        {isAssistant ? (
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
        ) : (
          <p>{message.content}</p>
        )}
      </div>

      {isAssistant && citations.length > 0 ? (
        <div className="message-bubble__support">
          <div className="citation-row">
            {citations.map((citation) => {
              const isActive = activeCitation?.chunk_id === citation.chunk_id;
              return (
                <button
                  aria-label={`${citation.doc_id} ${citation.breadcrumbs}`}
                  className={`citation-chip ${isActive ? 'active' : ''}`}
                  key={citation.chunk_id}
                  onClick={() => onSelectCitation(citation)}
                  type="button"
                >
                  <span>{citation.doc_id}</span>
                  <strong>{citation.breadcrumbs}</strong>
                </button>
              );
            })}
          </div>

          {message.used_doc_ids?.length ? (
            <p className="message-bubble__source-summary">
              Grounded in {message.used_doc_ids.join(', ')}.
            </p>
          ) : null}
        </div>
      ) : null}

      {isAssistant && message.debug_trace?.length ? (
        <details className="developer-trace">
          <summary>Developer trace</summary>
          <pre>{JSON.stringify(message.debug_trace, null, 2)}</pre>
        </details>
      ) : null}
    </article>
  );
}

export function ConversationPane({
  activeCitation,
  isBusy,
  onSelectCitation,
  thread,
}: ConversationPaneProps) {
  if (!thread) {
    return (
      <section className="conversation conversation--empty">
        <div className="empty-hero">
          <p className="eyebrow">Quiet by default</p>
          <h2>Ask the corpus a concrete question.</h2>
          <p>
            The assistant keeps the thread intact, cites its evidence inline, and leaves the
            raw trace tucked away unless you ask for it.
          </p>
          <div className="empty-hero__prompts">
            <span>Summarize the first-line treatment recommendations.</span>
            <span>What changed between the screening sections?</span>
            <span>Which passage supports that contraindication?</span>
          </div>
        </div>
      </section>
    );
  }

  return (
    <section className="conversation" key={thread.id}>
      {thread.messages.length === 0 ? (
        <div className="empty-thread">
          <p className="section-label">Fresh thread</p>
          <h3>Start with a focused question.</h3>
          <p>
            Replies will land here with citations visible by default so the evidence is always
            one click away.
          </p>
        </div>
      ) : (
        <div className="message-stack">
          {thread.messages.map((message) => (
            <MessageBubble
              activeCitation={activeCitation}
              key={message.id}
              message={message}
              onSelectCitation={onSelectCitation}
            />
          ))}
          {isBusy ? (
            <div aria-label="Assistant is thinking" className="message-bubble assistant pending">
              <div className="message-bubble__meta">
                <span className="message-bubble__role">assistant</span>
                <span>thinking</span>
              </div>
              <div className="thinking-dots">
                <span />
                <span />
                <span />
              </div>
            </div>
          ) : null}
        </div>
      )}
    </section>
  );
}
