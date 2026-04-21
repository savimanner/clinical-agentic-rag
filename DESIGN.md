# Design

## Retrieval Explanation Panel

### Goal

The retrieval part of the RAG system should stop feeling like a black box.
The product goal is not only better answers. It is better answers plus a clear,
human-readable explanation of how retrieval produced those answers.

For this repo, that means a user should be able to open an assistant response and
see:

- what semantic search returned
- what final chunks actually supported the answer

The current version should teach how the dense-only baseline works without
building a separate debug app.

### In Scope

- Add a first-class `retrieval_explanation` object to assistant responses
- Render that explanation in the existing thread chat UI under a collapsed
  `Why this answer` panel
- Keep the explanation attached to the real answer artifact so reopening an old
  thread shows the same explanation that existed when the answer was generated
- Keep the backend payload factual and structured; let the frontend render the
  teaching copy

### Not In Scope

- Compare mode
- Animated replay
- Learning-mode toggle
- Reintroducing lexical retrieval, fusion, or reranking into the live path
- Whole-graph rewrite

### Existing Foundation

The system already has the hard retrieval pieces:

- section-aware chunking and lexical index artifacts
- dense retrieval from Chroma
- thread message persistence
- a hidden developer trace panel in the UI

This design keeps the dense retrieval path in the live answer flow and leaves
the lexical artifacts available for future compare or benchmarking work.

### Core Product Decision

The explanation panel is not a prettified `debug_trace`.

It is a first-class product output with its own typed schema. The retrieval
pipeline should construct the explanation object once. The graph should only add
answer-specific support information, such as which chunks directly supported the
final answer and whether a refined retrieval query was used.

### Data Model

Add a typed `retrieval_explanation` object to the shared assistant payload used
by both `/api/chat` and threaded assistant messages.

The explanation object should be structured around retrieval stages, for example:

- `query_used`
- `refined_question_used` when a retry actually happened
- `dense_hits`
- `final_supporting_chunks`

Rules:

- Use explicit Pydantic and TypeScript models end-to-end
- Persist the explanation with the assistant message
- Store structured facts only, not pre-rendered explanatory sentences
- Cap each stage to a small top-N summary set, around 3-5 items, and include
  counts for omitted items

### UI Shape

Inside each assistant message, add a collapsed `Why this answer` panel.

The panel should show stages in order:

1. Meaning-based matches
2. Final evidence chosen

Each stage should show:

- chunk breadcrumb
- short snippet
- why it was included
- score or rank only when it adds clarity

The frontend should generate the helper text from stage type. Example:

- `Meaning-based search found chunks about the same idea, even when the wording differs.`

The normal UI should stop depending on `debug_trace`. That field can remain for
engineering debugging, but it is no longer the product explanation surface.

### Persistence Rules

The explanation must be stored with the assistant message in the thread JSON.
It must not be recomputed when the thread is reopened.

Reason: this feature is partly educational. If the retrieval explanation changes
after embeddings, corpus content, or retrieval code change, the user would see a
story about an answer that never actually happened.

### Testing Requirements

This feature is complete only if these paths are covered:

- backend regression test for explanation persistence round-trip through thread storage
- frontend component tests for collapsed panel behavior and stage rendering
- targeted retry-path test proving refined-query usage is recorded
- one end-to-end thread-flow test covering send, answer, open panel, reload, and
  the same explanation showing again

### Success Criteria

- A smart non-expert can inspect one answer and explain the difference between
  dense retrieval and final cited evidence in their own words
- A learner can tell which chunks were considered and which ones actually
  supported the final answer
- The feature improves trust without turning the UI into a wall of raw data
- The implementation leaves a clean base for later compare mode and replay mode

### Implementation Order

1. Define the shared assistant payload schema, including `retrieval_explanation`
2. Make the retrieval pipeline return the structured explanation object
3. Persist the explanation through thread and chat responses
4. Render the collapsed `Why this answer` panel in the existing conversation UI
5. Add regression, component, retry, and integration tests

### Follow-On Direction

Once the base panel proves useful, the strongest next teaching feature is
compare mode: run dense-only, lexical-only, and hybrid retrieval side by side
for the same question and show why the results differ.
