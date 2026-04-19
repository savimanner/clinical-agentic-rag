import type { DocumentSummary } from '../types/api';

const relativeFormatter = new Intl.RelativeTimeFormat('en', { numeric: 'auto' });

export function formatRelativeTime(value: string): string {
  const date = new Date(value);
  const delta = date.getTime() - Date.now();
  const minutes = Math.round(delta / 60000);

  if (Math.abs(minutes) < 60) {
    return relativeFormatter.format(minutes, 'minute');
  }

  const hours = Math.round(minutes / 60);
  if (Math.abs(hours) < 24) {
    return relativeFormatter.format(hours, 'hour');
  }

  const days = Math.round(hours / 24);
  return relativeFormatter.format(days, 'day');
}

export function formatTimestamp(value: string): string {
  return new Intl.DateTimeFormat('en', {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(new Date(value));
}

export function buildThreadTitle(content: string): string {
  const cleaned = content.replace(/\s+/g, ' ').trim();
  if (!cleaned) {
    return 'Untitled thread';
  }

  if (cleaned.length <= 60) {
    return cleaned;
  }

  return `${cleaned.slice(0, 57).trimEnd()}...`;
}

export function summarizeScope(docIds: string[], library: DocumentSummary[]): string {
  if (docIds.length === 0) {
    return 'All indexed documents';
  }

  const titles = docIds
    .map((docId) => library.find((document) => document.doc_id === docId)?.title ?? docId)
    .slice(0, 2);

  if (docIds.length <= 2) {
    return titles.join(' and ');
  }

  return `${titles.join(', ')} +${docIds.length - 2} more`;
}
