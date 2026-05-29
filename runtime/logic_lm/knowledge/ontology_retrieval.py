import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Mapping, Optional, Set, Union

from src.logic_lm.config import settings
from src.logic_lm.indexes.keyword_inverted_index import KeywordInvertedIndex
from src.logic_lm.knowledge.bhxh_ontology import (
    extract_query_threshold_ids,
    match_query_concept_ids,
    normalize_for_matching,
)
from src.logic_lm.knowledge.hybrid_retrieval import RetrievedKnowledgeChunk, RetrievedKnowledgeContext


class OntologyRetrieval:
    def __init__(self, keyword_index: Optional[KeywordInvertedIndex] = None) -> None:
        self._keyword_index = keyword_index or KeywordInvertedIndex()
        self._chunks: Dict[str, RetrievedKnowledgeChunk] = {}
        self._chunk_concepts: Dict[str, Set[str]] = {}
        self._chunk_thresholds: Dict[str, Set[str]] = {}
        self._chunk_article_keys: Dict[str, Set[str]] = {}
        self._concept_to_chunks: DefaultDict[str, Set[str]] = defaultdict(set)
        self._threshold_to_chunks: DefaultDict[str, Set[str]] = defaultdict(set)
        self._article_to_chunks: DefaultDict[str, Set[str]] = defaultdict(set)
        self._node_labels: Dict[str, str] = {}
        self._concept_weights: Dict[str, float] = {}
        self._is_built = False

    def index_ontology(self, ontology_path: Union[str, Path]) -> None:
        with Path(ontology_path).open(
            settings.FILE_MODE_READ,
            encoding=settings.PATH_ENCODING,
        ) as ontology_file:
            ontology = json.load(ontology_file)
        self.index_ontology_data(ontology)

    def index_ontology_data(self, ontology: Mapping[str, Any]) -> None:
        self._reset()
        nodes = list(ontology.get("nodes") or [])
        for node in nodes:
            if not isinstance(node, Mapping):
                continue
            node_id = str(node.get("id") or settings.EMPTY_STRING)
            if not node_id:
                continue
            label_parts = [
                str(node.get("label") or settings.EMPTY_STRING),
                settings.SPACE.join(str(alias) for alias in node.get("aliases") or []),
            ]
            self._node_labels[node_id] = settings.SPACE.join(
                part for part in label_parts if part
            )
            if node.get("type") == "concept":
                parent_count = len(node.get("parents") or [])
                self._concept_weights[node_id] = settings.ONTOLOGY_CONCEPT_MATCH_WEIGHT * (
                    1.0 + 0.35 * parent_count
                )

        semantic_docs: Dict[str, str] = {}
        for chunk in list(ontology.get("chunks") or []):
            if not isinstance(chunk, Mapping):
                continue
            retrieved_chunk = self._chunk_from_ontology(chunk)
            if retrieved_chunk is None:
                continue

            self._chunks[retrieved_chunk.id] = retrieved_chunk

            concept_node_ids = {
                _concept_node_id(str(concept_id))
                for concept_id in chunk.get("concept_ids") or []
            }
            threshold_node_ids = {
                str(threshold_id)
                for threshold_id in chunk.get("threshold_ids") or []
                if str(threshold_id)
            }
            article_keys = _article_keys(
                retrieved_chunk.document,
                retrieved_chunk.article,
                retrieved_chunk.clause,
            )

            self._chunk_concepts[retrieved_chunk.id] = concept_node_ids
            self._chunk_thresholds[retrieved_chunk.id] = threshold_node_ids
            self._chunk_article_keys[retrieved_chunk.id] = article_keys

            for concept_node_id in concept_node_ids:
                self._concept_to_chunks[concept_node_id].add(retrieved_chunk.id)
            for threshold_node_id in threshold_node_ids:
                self._threshold_to_chunks[threshold_node_id].add(retrieved_chunk.id)
            for article_key in article_keys:
                self._article_to_chunks[article_key].add(retrieved_chunk.id)

            semantic_docs[retrieved_chunk.id] = self._semantic_doc(
                retrieved_chunk,
                concept_node_ids,
                threshold_node_ids,
            )

        self._keyword_index.index_docs(semantic_docs)
        self._is_built = True

    def retrieve(
        self,
        query: str,
        top_k: int = settings.DEFAULT_RETRIEVAL_TOP_K_IN_CODE,
    ) -> RetrievedKnowledgeContext:
        if not self._is_built or not self._chunks or not query or top_k <= 0:
            return RetrievedKnowledgeContext(chunks=[], scores={})

        inner_k = max(top_k * settings.RETRIEVAL_INNER_K_MULTIPLIER, top_k)
        scores: DefaultDict[str, float] = defaultdict(float)

        self._score_concepts(query, scores)
        self._score_thresholds(query, scores)
        self._score_legal_refs(query, scores)
        self._score_document_refs(query, scores)
        self._score_keywords(query, inner_k, scores)
        self._expand_same_article(scores)

        ranked = sorted(
            (
                (chunk_id, score)
                for chunk_id, score in scores.items()
                if chunk_id in self._chunks and score > 0
            ),
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]
        if not ranked:
            return RetrievedKnowledgeContext(chunks=[], scores={})

        max_score = max(score for _, score in ranked) or 1.0
        normalized = [(chunk_id, score / max_score) for chunk_id, score in ranked]
        return RetrievedKnowledgeContext(
            chunks=[self._chunks[chunk_id] for chunk_id, _ in normalized],
            scores={chunk_id: score for chunk_id, score in normalized},
        )

    def _reset(self) -> None:
        self._chunks = {}
        self._chunk_concepts = {}
        self._chunk_thresholds = {}
        self._chunk_article_keys = {}
        self._concept_to_chunks = defaultdict(set)
        self._threshold_to_chunks = defaultdict(set)
        self._article_to_chunks = defaultdict(set)
        self._node_labels = {}
        self._concept_weights = {}
        self._is_built = False

    def _chunk_from_ontology(
        self,
        chunk: Mapping[str, Any],
    ) -> Optional[RetrievedKnowledgeChunk]:
        chunk_id = str(chunk.get(settings.FIELD_ID) or settings.EMPTY_STRING)
        text = str(chunk.get(settings.FIELD_TEXT) or settings.EMPTY_STRING)
        if not chunk_id or not text:
            return None
        return RetrievedKnowledgeChunk(
            id=chunk_id,
            text=text,
            document=_optional_str(chunk.get(settings.FIELD_DOCUMENT)),
            article=_optional_str(chunk.get(settings.FIELD_ARTICLE)),
            clause=_optional_str(chunk.get(settings.FIELD_CLAUSE)),
            point=_optional_str(chunk.get(settings.FIELD_POINT)),
        )

    def _semantic_doc(
        self,
        chunk: RetrievedKnowledgeChunk,
        concept_node_ids: Iterable[str],
        threshold_node_ids: Iterable[str],
    ) -> str:
        parts = [
            chunk.text,
            chunk.document or settings.EMPTY_STRING,
            f"Điều {chunk.article}" if chunk.article else settings.EMPTY_STRING,
            f"Khoản {chunk.clause}" if chunk.clause else settings.EMPTY_STRING,
            chunk.point or settings.EMPTY_STRING,
        ]
        parts.extend(self._node_labels.get(node_id, node_id) for node_id in concept_node_ids)
        parts.extend(self._node_labels.get(node_id, node_id) for node_id in threshold_node_ids)
        return settings.SPACE.join(part for part in parts if part)

    def _score_concepts(self, query: str, scores: DefaultDict[str, float]) -> None:
        for concept_id in match_query_concept_ids(query):
            concept_node_id = _concept_node_id(concept_id)
            weight = self._concept_weights.get(
                concept_node_id,
                settings.ONTOLOGY_CONCEPT_MATCH_WEIGHT,
            )
            for chunk_id in self._concept_to_chunks.get(concept_node_id, set()):
                scores[chunk_id] += weight

    def _score_thresholds(self, query: str, scores: DefaultDict[str, float]) -> None:
        for threshold_node_id in extract_query_threshold_ids(query):
            for chunk_id in self._threshold_to_chunks.get(threshold_node_id, set()):
                scores[chunk_id] += settings.ONTOLOGY_THRESHOLD_MATCH_WEIGHT

    def _score_legal_refs(self, query: str, scores: DefaultDict[str, float]) -> None:
        for key in _query_article_keys(query):
            for chunk_id in self._article_to_chunks.get(key, set()):
                scores[chunk_id] += settings.ONTOLOGY_LEGAL_REF_MATCH_WEIGHT

    def _score_document_refs(self, query: str, scores: DefaultDict[str, float]) -> None:
        years = set(re.findall(r"\b(?:19|20)\d{2}\b", query))
        if not years:
            return
        for chunk_id, chunk in self._chunks.items():
            document = normalize_for_matching(chunk.document or settings.EMPTY_STRING)
            if any(year in document for year in years):
                scores[chunk_id] += settings.ONTOLOGY_DOCUMENT_MATCH_WEIGHT

    def _score_keywords(
        self,
        query: str,
        inner_k: int,
        scores: DefaultDict[str, float],
    ) -> None:
        results = self._keyword_index.search(query, top_k=inner_k)
        if not results:
            return
        max_keyword_score = max(score for _, score in results)
        if max_keyword_score <= 0:
            return
        for chunk_id, score in results:
            scores[chunk_id] += (
                settings.ONTOLOGY_KEYWORD_MATCH_WEIGHT * score / max_keyword_score
            )

    def _expand_same_article(self, scores: DefaultDict[str, float]) -> None:
        seeded = dict(scores)
        for chunk_id, score in seeded.items():
            article_keys = [
                key
                for key in self._chunk_article_keys.get(chunk_id, set())
                if key.startswith("article:")
            ]
            for article_key in article_keys:
                for neighbor_id in self._article_to_chunks.get(article_key, set()):
                    if neighbor_id == chunk_id:
                        continue
                    scores[neighbor_id] += score * settings.ONTOLOGY_ARTICLE_EXPANSION_WEIGHT


def _optional_str(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _concept_node_id(concept_id: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", normalize_for_matching(concept_id)).strip("_")
    return f"concept:{slug or 'unknown'}"


def _article_keys(
    document: Optional[str],
    article: Optional[str],
    clause: Optional[str],
) -> Set[str]:
    keys: Set[str] = set()
    article_slug = _norm_ref(article)
    clause_slug = _norm_ref(clause)
    document_slug = _norm_ref(document)
    if article_slug:
        keys.add(f"article:{article_slug}")
        if document_slug:
            keys.add(f"document_article:{document_slug}:{article_slug}")
    if article_slug and clause_slug:
        keys.add(f"clause:{article_slug}:{clause_slug}")
        if document_slug:
            keys.add(f"document_clause:{document_slug}:{article_slug}:{clause_slug}")
    return keys


def _query_article_keys(query: str) -> Set[str]:
    normalized = normalize_for_matching(query)
    articles = re.findall(r"\bdieu\s+([0-9]+[a-z]?)\b", normalized)
    clauses = re.findall(r"\bkhoan\s+([0-9]+[a-z]?)\b", normalized)
    keys: Set[str] = set()
    for article in articles:
        article_slug = _norm_ref(article)
        if article_slug:
            keys.add(f"article:{article_slug}")
        for clause in clauses:
            clause_slug = _norm_ref(clause)
            if article_slug and clause_slug:
                keys.add(f"clause:{article_slug}:{clause_slug}")
    return keys


def _norm_ref(value: Optional[str]) -> str:
    normalized = normalize_for_matching(value or settings.EMPTY_STRING)
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
    return normalized
