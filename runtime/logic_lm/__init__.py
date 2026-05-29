"""logic_lm — symbolic-LLM hybrid pipeline.

Lives at `runtime/logic_lm/` (post-v5 refactor; previously `elite/` then
`src/logic_lm/`). Comprises:

- config/    — typed settings (paths, LLM, retrieval, Prolog, prompts)
- indexes/   — HNSW dense + keyword sparse stores
- knowledge/ — ontology + hybrid retrieval over BHXH corpus
- llm/       — OpenAI client + factory
- pipelines/ — program (Prolog) synthesis pipeline
- services/  — encoder service
- solvers/   — SWI Prolog runtime wrapper
- cli/       — answer/build/query entry points
"""
