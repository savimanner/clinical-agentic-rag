import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';

import { RetrievalExplanationPanel } from '../components/RetrievalExplanationPanel';
import type { RetrievalExplanation } from '../types/api';

const explanation: RetrievalExplanation = {
  query_used: 'What is first-line treatment for hypertension?',
  refined_question_used: 'ace inhibitor first line hypertension',
  lexical_hits: {
    total_hits: 0,
    omitted_hits: 0,
    items: [],
  },
  dense_hits: {
    total_hits: 1,
    omitted_hits: 0,
    items: [
      {
        doc_id: 'hypertension-guideline',
        chunk_id: 'hypertension-guideline::chunk_0007',
        breadcrumbs: 'Treatment > First line',
        snippet: 'ACE inhibitors are recommended when blood pressure remains elevated.',
        source_path: 'guidelines/hypertension.md',
        rank: 1,
        score: 0.88,
        source_modes: ['dense'],
      },
    ],
  },
  merged_candidates: {
    total_hits: 0,
    omitted_hits: 0,
    items: [],
  },
  reranked_top_chunks: {
    total_hits: 0,
    omitted_hits: 0,
    items: [],
  },
  final_supporting_chunks: {
    total_hits: 1,
    omitted_hits: 0,
    items: [
      {
        doc_id: 'hypertension-guideline',
        chunk_id: 'hypertension-guideline::chunk_0007',
        breadcrumbs: 'Treatment > First line',
        snippet: 'ACE inhibitors are recommended when blood pressure remains elevated.',
        source_path: 'guidelines/hypertension.md',
        rank: 1,
        cited_directly: true,
        source_modes: ['lexical', 'dense'],
      },
    ],
  },
};

describe('RetrievalExplanationPanel', () => {
  it('starts collapsed and renders the retrieval stages when opened', async () => {
    const user = userEvent.setup();

    render(<RetrievalExplanationPanel explanation={explanation} />);

    const toggle = screen.getByText('Why this answer');
    expect(toggle).toBeInTheDocument();
    expect(toggle.closest('details')).not.toHaveAttribute('open');

    await user.click(toggle);

    expect(screen.getByText('Meaning-based matches')).toBeVisible();
    expect(screen.getByText('Final evidence chosen')).toBeVisible();
    expect(screen.getByText('Refined retrieval query')).toBeVisible();
    expect(screen.queryByText('Exact wording matches')).not.toBeInTheDocument();
    expect(screen.queryByText('Best candidates after merging')).not.toBeInTheDocument();
    expect(screen.getByText('Directly cited in the final answer.')).toBeVisible();
  });
});
