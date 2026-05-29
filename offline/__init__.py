"""Offline (build-time) pipeline.

Contains the data-preparation steps that run once to populate the Neo4j KG and
auxiliary artifacts before any inference can happen:

- parse_docx (B1) — deterministic docx parser
- rule_extract (B2) — regex relation extraction
- llm_extract (B3) — OpenAI semantic extraction
- merge_normalize (B4) — dedup + validate
- embed (B5) — BGE-M3 embeddings
- load_neo4j (B6) — load nodes/edges/vectors into Neo4j
- build_logic_lm_corpus_2024 — build the logic-LM corpus + ontology files
"""
