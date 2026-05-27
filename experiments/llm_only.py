"""LLM-only baseline pipeline (no RAG, no graph).

Cùng SYSTEM prompt yêu cầu citation inline rõ authority như GraphRAG, NHƯNG
KHÔNG inject context retrieved. Model phải trả lời dựa trên training data
của mình.

Format response giống RagAnswer để dễ so sánh.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

from dotenv import load_dotenv

from src.citations import format_citation, parse_displayed_citations

load_dotenv()
if not (os.environ.get("OPENAI_BASE_URL") or "").strip():
    os.environ.pop("OPENAI_BASE_URL", None)

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


# Prompt KHÔNG nhắc tới CONTEXT (vì không có). Vẫn yêu cầu citation canonical.
# Mục tiêu: fair comparison — cả 2 arm đều output cùng format.
SYSTEM_PROMPT_LLM_ONLY = """Bạn là trợ lý pháp lý chuyên về Luật Bảo hiểm xã hội Việt Nam.

QUY TẮC:
1. Trả lời câu hỏi bằng tiếng Việt.
2. Mọi khẳng định pháp lý PHẢI kèm citation inline ngay sau claim liên quan, theo format canonical `[Luật BHXH 2024 (41/2024/QH15), Điều X khoản Y]` hoặc `[Luật BHXH 2024 (41/2024/QH15), Điều X khoản Y điểm z]`. KHÔNG dùng citation mơ hồ như `[Điều X]` nếu thiếu tên văn bản.
3. Nếu không chắc chắn về quy định nào, hãy nói rõ "tôi không có đủ thông tin chính xác để trả lời câu hỏi này".
4. KHÔNG bịa số liệu, ngày tháng, mức tiền.
5. Ngắn gọn nhưng đầy đủ.
"""


@dataclass
class LlmOnlyAnswer:
    question: str
    answer: str
    citations: list[str] = field(default_factory=list)
    citation_ids: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0


def _parse_citations(answer: str) -> tuple[list[str], list[str]]:
    refs = parse_displayed_citations(answer)
    citations = [format_citation(ref) for ref in refs]
    ids = [ref.item_id for ref in refs]
    return list(dict.fromkeys(citations)), list(dict.fromkeys(ids))


class LlmOnlyPipeline:
    def __init__(self):
        self._openai = None

    @property
    def openai(self):
        if self._openai is None:
            from openai import OpenAI

            self._openai = OpenAI()
        return self._openai

    def ask(self, question: str) -> LlmOnlyAnswer:
        t0 = time.time()
        resp = self.openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_LLM_ONLY},
                {"role": "user", "content": question},
            ],
            temperature=0,
        )
        elapsed = time.time() - t0
        text = resp.choices[0].message.content or ""
        cits, ids = _parse_citations(text)
        return LlmOnlyAnswer(
            question=question,
            answer=text,
            citations=cits,
            citation_ids=ids,
            elapsed_s=round(elapsed, 3),
            prompt_tokens=resp.usage.prompt_tokens,
            completion_tokens=resp.usage.completion_tokens,
        )

    def close(self):
        # OpenAI client has no explicit close needed
        pass


if __name__ == "__main__":
    # Quick smoke test
    p = LlmOnlyPipeline()
    r = p.ask("Bảo hiểm xã hội là gì?")
    print(f"Answer: {r.answer[:300]}")
    print(f"Citations: {r.citations}")
    print(f"Time: {r.elapsed_s}s, tokens: {r.prompt_tokens}+{r.completion_tokens}")
