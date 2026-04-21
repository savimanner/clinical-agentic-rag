# Changelog

All notable changes to this project should be recorded here.

## Unreleased

### Changed

- Simplified the live retrieval path to a dense-only baseline: query rewrite, semantic retrieval, answer generation, and deterministic citation fallback.
- Updated the retrieval explanation UI and debug trace to reflect the dense-only runtime path.
- Added regression coverage for dense-only retrieval, citation fallback behavior, and conservative fallback responses.
