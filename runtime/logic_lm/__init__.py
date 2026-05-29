"""logic_lm — symbolic-LLM hybrid pipeline.

Originally the top-level `elite/` package; flattened into `src/logic_lm/`
and renamed during the v5 refactor. Comprises:

- config/    — typed settings (paths, LLM, retrieval, Prolog, prompts)
- indexes/   — HNSW dense + keyword sparse stores
- knowledge/ — ontology + hybrid retrieval over BHXH corpus
- llm/       — OpenAI client + factory
- pipelines/ — program (Prolog) synthesis pipeline
- services/  — encoder service
- solvers/   — SWI Prolog runtime wrapper
- cli/       — answer/build/query entry points
"""
