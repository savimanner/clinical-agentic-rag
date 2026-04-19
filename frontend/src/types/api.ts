export interface Citation {
  doc_id: string;
  chunk_id: string;
  breadcrumbs: string;
  snippet: string;
  source_path: string;
}

export interface ThreadMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  created_at: string;
  citations?: Citation[];
  used_doc_ids?: string[];
  debug_trace?: Array<Record<string, unknown>> | null;
}

export interface ThreadScope {
  doc_ids: string[];
}

export interface ThreadSummary extends ThreadScope {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  last_message_preview?: string | null;
  message_count: number;
}

export interface ThreadDetail extends ThreadSummary {
  messages: ThreadMessage[];
}

export interface CreateThreadRequest extends Partial<ThreadScope> {
  title?: string;
}

export interface AppendMessageRequest {
  content: string;
  debug?: boolean;
}

export interface UpdateThreadRequest extends Partial<ThreadScope> {
  title?: string;
}

export interface DocumentSummary {
  doc_id: string;
  title: string;
  language: string;
  source_pdf?: string | null;
  primary_markdown?: string | null;
  chunk_file?: string | null;
  chunk_count: number;
  indexed: boolean;
  manifest_path: string;
}
