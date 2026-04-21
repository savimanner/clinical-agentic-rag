export interface Citation {
  doc_id: string;
  chunk_id: string;
  breadcrumbs: string;
  snippet: string;
  source_path: string;
}

export interface RetrievalStageItem {
  doc_id: string;
  chunk_id: string;
  breadcrumbs: string;
  snippet: string;
  source_path: string;
  rank?: number | null;
  score?: number | null;
  source_modes?: Array<'lexical' | 'dense'>;
  cited_directly?: boolean | null;
}

export interface RetrievalStage {
  total_hits: number;
  omitted_hits: number;
  items: RetrievalStageItem[];
}

export interface RetrievalExplanation {
  query_used: string;
  refined_question_used?: string | null;
  lexical_hits: RetrievalStage;
  dense_hits: RetrievalStage;
  merged_candidates: RetrievalStage;
  reranked_top_chunks: RetrievalStage;
  final_supporting_chunks: RetrievalStage;
}

export interface ThreadMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  created_at: string;
  citations?: Citation[];
  used_doc_ids?: string[];
  retrieval_explanation?: RetrievalExplanation | null;
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
