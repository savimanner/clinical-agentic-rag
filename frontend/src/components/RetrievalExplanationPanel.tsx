import type { RetrievalExplanation, RetrievalStage, RetrievalStageItem } from '../types/api';

interface RetrievalExplanationPanelProps {
  explanation: RetrievalExplanation;
}

type StageConfig = {
  description: string;
  key: 'dense_hits' | 'final_supporting_chunks';
  title: string;
};

const stageConfigs: StageConfig[] = [
  {
    key: 'dense_hits',
    title: 'Meaning-based matches',
    description: 'Meaning-based search found chunks about the same idea, even when the wording differs.',
  },
  {
    key: 'final_supporting_chunks',
    title: 'Final evidence chosen',
    description: 'These are the chunks that directly support the final answer shown above.',
  },
];

function buildItemReason(stageKey: StageConfig['key'], item: RetrievalStageItem): string {
  if (stageKey === 'dense_hits') {
    return 'Matched the question by meaning rather than exact wording.';
  }
  if (stageKey === 'final_supporting_chunks') {
    return item.cited_directly ? 'Directly cited in the final answer.' : 'Kept as final supporting evidence.';
  }
  return 'Included in the answer evidence set.';
}

function formatScore(item: RetrievalStageItem): string | null {
  if (typeof item.score !== 'number') {
    return item.rank ? `Rank ${item.rank}` : null;
  }
  return `Rank ${item.rank ?? 'n/a'} • Score ${item.score.toFixed(3)}`;
}

function StageSection({
  config,
  stage,
}: {
  config: StageConfig;
  stage: RetrievalStage;
}) {
  if (stage.total_hits === 0) {
    return null;
  }

  return (
    <section className="retrieval-panel__stage">
      <div className="retrieval-panel__stage-header">
        <div>
          <p className="section-label">{config.title}</p>
          <p>{config.description}</p>
        </div>
        <div className="retrieval-panel__count">
          <strong>{stage.total_hits}</strong>
          <span>{stage.total_hits === 1 ? 'chunk' : 'chunks'}</span>
        </div>
      </div>

      <div className="retrieval-panel__items">
        {stage.items.map((item) => (
          <article className="retrieval-panel__item" key={`${config.key}-${item.chunk_id}`}>
            <div className="retrieval-panel__item-meta">
              <strong>{item.breadcrumbs}</strong>
              {formatScore(item) ? <span>{formatScore(item)}</span> : null}
            </div>
            <p className="retrieval-panel__item-doc">{item.doc_id}</p>
            <blockquote>{item.snippet}</blockquote>
            <p className="retrieval-panel__item-reason">{buildItemReason(config.key, item)}</p>
          </article>
        ))}
      </div>

      {stage.omitted_hits > 0 ? (
        <p className="retrieval-panel__overflow">Plus {stage.omitted_hits} more omitted from this summary.</p>
      ) : null}
    </section>
  );
}

export function RetrievalExplanationPanel({ explanation }: RetrievalExplanationPanelProps) {
  const refinedQuestionUsed =
    explanation.refined_question_used && explanation.refined_question_used !== explanation.query_used
      ? explanation.refined_question_used
      : null;

  return (
    <details className="retrieval-panel">
      <summary>Why this answer</summary>

      <div className="retrieval-panel__body">
        <div className="retrieval-panel__query">
          <div>
            <span className="section-label">Query used</span>
            <p>{explanation.query_used}</p>
          </div>
          {refinedQuestionUsed ? (
            <div>
              <span className="section-label">Refined retrieval query</span>
              <p>{refinedQuestionUsed}</p>
            </div>
          ) : null}
        </div>

        {stageConfigs.map((config) => (
          <StageSection config={config} key={config.key} stage={explanation[config.key]} />
        ))}
      </div>
    </details>
  );
}
