"""Inference runtime.

Modules that execute at query time to answer a user question:

- rag_query (B7) — GraphRAG retrieval + GPT-4o-mini generation
- chat — interactive REPL on top of rag_query
- llm_only — LLM-only baseline (no retrieval)
- logic_lm — symbolic-LLM hybrid package (config, indexes, knowledge,
  llm, pipelines, services, solvers, cli)
- logic_lm_pipelines — 3 arm-aware wrappers around the logic_lm package
- graphrag_retriever_adapter — adapt RagPipeline as a logic-LM retriever
- run_inference / run_multimodel_inference — orchestrate inference batches
- rerender_plain_answer — backfill plain_answer on existing logic-LM records
"""
