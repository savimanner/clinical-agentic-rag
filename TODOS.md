# TODOs

## Simplify Remaining Graph Orchestration

What: Revisit the remaining LangGraph orchestration after the new hybrid retrieval path has been stable in production-like use.

Why: The current implementation keeps a trimmed graph for scoped delivery, but the retrieval pipeline now does the real work. The graph may still contain transitional complexity that is no longer buying anything.

Pros: Reduces maintenance burden, makes future debugging simpler, and keeps the answer path explicit.

Cons: Needs real usage signal first, otherwise this turns into speculative cleanup.

Context: The retrieval rebuild intentionally kept the graph shape and trimmed it surgically instead of replacing it outright. Once the new pipeline is proven, this code should be reevaluated with fresh evidence rather than inertia.

Depends on / blocked by: Depends on the new retrieval pipeline, reranking, and tests landing first.

## Add Retrieval Benchmark Suite

What: Build a benchmark pack of curated medical-guideline questions with expected supporting chunks and key facts, and run it whenever chunking, embeddings, or retrieval change.

Why: This project’s main failure mode is missing key facts that are present in the corpus. A benchmark suite turns that from vague drift into measurable regressions.

Pros: Protects retrieval quality over time, makes corpus/index changes safer, and gives future tuning work an objective target.

Cons: Requires curation and occasional maintenance as the corpus evolves.

Context: The current change adds regression, integration, and API-level tests. A benchmark suite would sit above those tests and measure end-to-end retrieval quality against a stable question set.

Depends on / blocked by: Depends on the canonical schema and hybrid retrieval path being in place first.

## Add Retrieval Compare Mode

What: Add a future compare mode that runs multiple retrieval strategies for one question and shows how dense-only, lexical-only, and hybrid retrieval differ.

Why: The new retrieval explanation panel will teach one answer path well, but compare mode is the strongest follow-on for helping a learner understand why hybrid retrieval wins or fails on a given question.

Pros: Makes retrieval tradeoffs concrete, creates a strong teaching surface, and builds naturally on top of the explanation object introduced in the first pass.

Cons: Increases per-query runtime cost, expands the UI surface, and is easy to overbuild if added before the base explanation panel proves useful.

Context: The approved design for the retrieval explanation panel explicitly deferred compare mode from v1. It should remain deferred until the explanation object and `Why this answer` panel are live and validated.

Depends on / blocked by: Blocked by the explanation panel shipping first and getting some real usage or feedback.
