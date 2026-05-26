from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple, Union

from config import settings
from indexes.hnsw_embedding_index import EmbeddingHNSW
from indexes.keyword_inverted_index import KeywordInvertedIndex


@dataclass(frozen=True)
class RetrievedKnowledgeChunk:


    id: str
    text: str
    document: Optional[str] = None
    article: Optional[str] = None
    clause: Optional[str] = None
    point: Optional[str] = None

    @property
    def source_ref(self) -> Dict[str, Optional[str]]:
        return {
            settings.FIELD_DOCUMENT: self.document,
            settings.FIELD_ARTICLE: self.article,
            settings.FIELD_CLAUSE: self.clause,
            settings.FIELD_POINT: self.point,
        }


@dataclass(frozen=True)
class RetrievedKnowledgeContext:


    chunks: List[RetrievedKnowledgeChunk]
    scores: Dict[str, float]

    @property
    def source_refs(self) -> List[Dict[str, Optional[str]]]:
        return [chunk.source_ref for chunk in self.chunks]


class M1Retrieval:


    def __init__(
        self,
        keyword_index: Optional[KeywordInvertedIndex] = None,
        embedding_index: Optional[EmbeddingHNSW] = None,
        bm25_weight: float = 0.5,
        embedding_weight: float = 0.5,
    ) -> None:
        if bm25_weight < 0 or embedding_weight < 0:
            raise ValueError(settings.ERROR_RETRIEVAL_WEIGHTS_NON_NEGATIVE)
        if bm25_weight + embedding_weight == 0:
            raise ValueError(settings.ERROR_RETRIEVAL_WEIGHT_SUM_POSITIVE)

        self._keyword_index = keyword_index or KeywordInvertedIndex()
        self._embedding_index = embedding_index or EmbeddingHNSW()
        self._bm25_weight = bm25_weight
        self._embedding_weight = embedding_weight
        self._chunks: Dict[str, RetrievedKnowledgeChunk] = {}
        self._is_built = False

    def index_corpus(self, corpus_path: Union[str, Path]) -> None:


        chunks: Dict[str, RetrievedKnowledgeChunk] = {}
        docs: Dict[str, str] = {}

        path = Path(corpus_path)
        with path.open(settings.FILE_MODE_READ, encoding=settings.PATH_ENCODING) as corpus_file:
            for line_number, line in enumerate(corpus_file, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        settings.ERROR_INVALID_JSONL_TEMPLATE.format(
                            line_number=line_number,
                            error=exc,
                        )
                    ) from exc

                chunk = self._chunk_from_record(record, line_number)
                if chunk.id in chunks:
                    raise ValueError(
                        settings.ERROR_DUPLICATE_CORPUS_ID_TEMPLATE.format(
                            line_number=line_number,
                            chunk_id=chunk.id,
                        )
                    )

                chunks[chunk.id] = chunk
                docs[chunk.id] = chunk.text

        self._keyword_index.index_docs(docs)
        self._embedding_index.index_docs(docs)
        self._chunks = chunks
        self._is_built = True

    def retrieve(
        self,
        query: str,
        top_k: int = settings.DEFAULT_RETRIEVAL_TOP_K_IN_CODE,
    ) -> RetrievedKnowledgeContext:

        if not self._is_built or not self._chunks or not query or top_k <= 0:
            return RetrievedKnowledgeContext(chunks=[], scores={})

        inner_k = max(top_k * settings.RETRIEVAL_INNER_K_MULTIPLIER, top_k)
        combined: Dict[str, float] = {}

        bm25_results = self._keyword_index.search(query, top_k=inner_k)
        self._add_normalized_scores(combined, bm25_results, self._bm25_weight)

        embedding_results = self._embedding_index.query(query, top_k=inner_k)
        self._add_normalized_scores(combined, embedding_results, self._embedding_weight)

        weight_sum = self._bm25_weight + self._embedding_weight
        ranked = sorted(
            (
                (chunk_id, score / weight_sum)
                for chunk_id, score in combined.items()
                if chunk_id in self._chunks
            ),
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]

        chunks = [self._chunks[chunk_id] for chunk_id, _ in ranked]
        scores = {chunk_id: score for chunk_id, score in ranked}
        return RetrievedKnowledgeContext(chunks=chunks, scores=scores)

    def _chunk_from_record(
        self,
        record: Mapping[str, object],
        line_number: int,
    ) -> RetrievedKnowledgeChunk:
        if settings.FIELD_ID not in record:
            raise ValueError(
                settings.ERROR_MISSING_CORPUS_FIELD_TEMPLATE.format(
                    field=settings.FIELD_ID,
                    line_number=line_number,
                )
            )
        if settings.FIELD_TEXT not in record:
            raise ValueError(
                settings.ERROR_MISSING_CORPUS_FIELD_TEMPLATE.format(
                    field=settings.FIELD_TEXT,
                    line_number=line_number,
                )
            )

        chunk_id = str(record[settings.FIELD_ID])
        text = str(record[settings.FIELD_TEXT])
        if not chunk_id:
            raise ValueError(
                settings.ERROR_EMPTY_CORPUS_FIELD_TEMPLATE.format(
                    field=settings.FIELD_ID,
                    line_number=line_number,
                )
            )
        if not text:
            raise ValueError(
                settings.ERROR_EMPTY_CORPUS_FIELD_TEMPLATE.format(
                    field=settings.FIELD_TEXT,
                    line_number=line_number,
                )
            )

        return RetrievedKnowledgeChunk(
            id=chunk_id,
            text=text,
            document=self._optional_str(record.get(settings.FIELD_DOCUMENT)),
            article=self._optional_str(record.get(settings.FIELD_ARTICLE)),
            clause=self._optional_str(record.get(settings.FIELD_CLAUSE)),
            point=self._optional_str(record.get(settings.FIELD_POINT)),
        )

    def _add_normalized_scores(
        self,
        combined: Dict[str, float],
        results: List[Tuple[str, float]],
        weight: float,
    ) -> None:
        if not results or weight <= 0:
            return

        max_score = max(score for _, score in results)
        if max_score <= 0:
            return

        for chunk_id, raw_score in results:
            normalized = max(0.0, raw_score / max_score)
            combined[chunk_id] = combined.get(chunk_id, 0.0) + weight * normalized

    def _optional_str(self, value: object) -> Optional[str]:
        if value is None:
            return None
        return str(value)
