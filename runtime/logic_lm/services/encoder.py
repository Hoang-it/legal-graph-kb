import math
import re
from threading import Lock
from typing import Dict, List, Optional, Union

from runtime.logic_lm.config import settings


DenseVector = List[float]
SparseVector = Dict[str, float]
AnyVector = Union[DenseVector, SparseVector]


class EncoderService:


    def __init__(self, model_name: str = settings.ENCODER_MODEL) -> None:
        self._model_name = model_name
        self._model = None
        self._init_attempted = False





    def _ensure_model(self) -> None:
        if self._init_attempted:
            return
        self._init_attempted = True
        # Lazy import: `sentence_transformers` pulls in torch + transformers
        # (~15s). Deferring it to the first encode() keeps importing this module
        # (and everything that transitively imports it — the Streamlit UI, eval
        # routing) cheap, since those callers only need the retrieval dataclasses
        # and never the in-house encoder. Behaviour is unchanged: if the import
        # fails we fall back to the sparse bag-of-words encoding below.
        try:
            from sentence_transformers import SentenceTransformer
        except Exception:
            self._model = None
            return
        try:
            self._model = SentenceTransformer(self._model_name)
        except Exception:
            self._model = None

    def encode(self, texts: List[str]) -> List[AnyVector]:

        if not texts:
            return []

        self._ensure_model()
        if self._model is not None:
            try:
                embs = self._model.encode(texts, convert_to_numpy=True)
                return [list(vec.astype(float)) for vec in embs]
            except Exception:
                pass


        out: List[AnyVector] = []
        for text in texts:
            tokens = _tokenize(text)
            vec: SparseVector = {}
            for t in tokens:
                vec[t] = vec.get(t, 0.0) + 1.0
            norm = math.sqrt(sum(v * v for v in vec.values()))
            if norm > 0:
                for k in vec:
                    vec[k] /= norm
            out.append(vec)
        return out

def _tokenize(text: str) -> List[str]:
    return re.findall(settings.REGEX_WORD, (text or settings.EMPTY_STRING).lower())






_singleton: Optional[EncoderService] = None
_singleton_lock = Lock()


def get_encoder() -> EncoderService:

    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = EncoderService()
    return _singleton
