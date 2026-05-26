from typing import Dict, List, Tuple
import math

from config import settings
from services.encoder import get_encoder

try:
    import hnswlib
    HAS_HNSW = True
except Exception:
    HAS_HNSW = False


class EmbeddingHNSW:


    def __init__(self, dim: int = None):
        self._func_ids: List[str] = []
        self._id_map: Dict[int, str] = {}
        self._embeddings = None
        self._is_built = False
        self._dim = dim
        self._index = None

    def _embed_texts(self, texts: List[str]):

        return get_encoder().encode(texts)

    def index_docs(self, docs: Dict[str, str]) -> None:

        if not docs:
            self._is_built = True
            return

        self._func_ids = list(docs.keys())
        texts = [docs[fid] for fid in self._func_ids]

        embs = self._embed_texts(texts)



        self._embeddings = embs
        self._id_map = {i: fid for i, fid in enumerate(self._func_ids)}


        if HAS_HNSW and embs and isinstance(embs[0], list):
            try:
                dim = len(embs[0])
                self._index = hnswlib.Index(space=settings.HNSW_SPACE, dim=dim)
                self._index.init_index(
                    max_elements=len(self._func_ids),
                    ef_construction=settings.HNSW_EF_CONSTRUCTION,
                    M=settings.HNSW_M,
                )

                vecs = []
                for v in embs:
                    norm = math.sqrt(sum(x * x for x in v))
                    if norm == 0:
                        vecs.append([0.0] * dim)
                    else:
                        vecs.append([float(x) / norm for x in v])
                self._index.add_items(vecs, list(range(len(vecs))))
                self._index.set_ef(settings.HNSW_EF)
            except Exception:
                self._index = None

        self._is_built = True

    def query(self, query_text: str, top_k: int = 10) -> List[Tuple[str, float]]:

        if not self._is_built:
            return []
        if not query_text:
            return []

        q_emb = self._embed_texts([query_text])
        if not q_emb:
            return []


        if self._index is not None and isinstance(q_emb[0], list):
            try:
                q = q_emb[0]
                norm = math.sqrt(sum(x * x for x in q))
                if norm == 0:
                    q_norm = [0.0 for _ in q]
                else:
                    q_norm = [float(x) / norm for x in q]
                labels, distances = self._index.knn_query([q_norm], k=min(top_k, len(self._func_ids)))
                labels = labels[0].tolist()
                distances = distances[0].tolist()
                results = []
                for lbl, sim in zip(labels, distances):
                    fid = self._id_map.get(int(lbl))
                    results.append((fid, float(sim)))
                return results
            except Exception:
                pass


        results = []
        for i, emb in enumerate(self._embeddings):
            score = 0.0
            if isinstance(emb, dict):
                qv = q_emb[0]
                if isinstance(qv, dict):

                    for k, v in qv.items():
                        if k in emb:
                            score += v * emb[k]
                else:

                    score = 0.0
            else:

                qv = q_emb[0]
                d = min(len(emb), len(qv))
                s = 0.0
                for j in range(d):
                    s += float(emb[j]) * float(qv[j])
                score = s
            results.append((self._func_ids[i], score))


        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]





