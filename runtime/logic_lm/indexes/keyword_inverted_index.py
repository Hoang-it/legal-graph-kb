from typing import Dict, List, Tuple
from collections import defaultdict
import re

from src.logic_lm.config import settings

try:
    from rank_bm25 import BM25Okapi
    HAS_RANK_BM25 = True
except ImportError:
    HAS_RANK_BM25 = False


class KeywordInvertedIndex:


    def __init__(self):

        self._docs: Dict[str, str] = {}
        self._tokenized_docs: Dict[str, List[str]] = {}
        self._bm25 = None
        self._is_built = False
        self._doc_ids: List[str] = []

    def _tokenize(self, text: str) -> List[str]:


        text = text.lower()

        tokens = re.findall(settings.REGEX_WORD, text)
        return tokens

    def index_docs(self, docs: Dict[str, str]) -> None:

        self._docs = dict(docs)
        self._doc_ids = list(docs.keys())


        self._tokenized_docs = {}
        for func_id, text in docs.items():
            self._tokenized_docs[func_id] = self._tokenize(text)


        if HAS_RANK_BM25:
            self._index_with_bm25()
        else:
            self._index_fallback()

        self._is_built = True

    def _index_with_bm25(self) -> None:


        tokenized_list = [self._tokenized_docs[func_id] for func_id in self._doc_ids]
        self._bm25 = BM25Okapi(tokenized_list)

    def _index_fallback(self) -> None:


        self._inverted_index = defaultdict(list)
        for func_id in self._doc_ids:
            tokens = self._tokenized_docs[func_id]
            for token in set(tokens):
                self._inverted_index[token].append(func_id)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:

        if not self._is_built:
            return []

        if not query:
            return []

        if HAS_RANK_BM25:
            return self._search_with_bm25(query, top_k)
        else:
            return self._search_fallback(query, top_k)

    def _search_with_bm25(self, query: str, top_k: int) -> List[Tuple[str, float]]:

        query_tokens = self._tokenize(query)

        if not query_tokens:
            return []


        scores = self._bm25.get_scores(query_tokens)


        results = []
        for doc_id, score in zip(self._doc_ids, scores):
            if score > 0:
                results.append((doc_id, float(score)))


        results.sort(key=lambda x: x[1], reverse=True)


        return results[:top_k]

    def _search_fallback(self, query: str, top_k: int) -> List[Tuple[str, float]]:

        query_tokens = self._tokenize(query)

        if not query_tokens:
            return []


        scores = {}
        for func_id in self._doc_ids:
            tokens = self._tokenized_docs[func_id]

            score = sum(1 for qt in query_tokens if qt in tokens)
            if score > 0:
                scores[func_id] = float(score)


        results = sorted(scores.items(), key=lambda x: x[1], reverse=True)


        return results[:top_k]





