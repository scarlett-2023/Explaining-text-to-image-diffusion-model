import re
from typing import List
import spacy

# 懒加载 spaCy，避免多进程 map 时重复加载
_SPACY_NLP = None

EN_BASIC_ADV_LIST = {
    "very","really","extremely","highly","quite","fairly","rather","pretty","too",
    "so","just","almost","nearly","barely","hardly","scarcely","surely","clearly",
    "basically","literally","virtually","slightly","approximately","roughly","exactly",
    "truly","deeply","strongly","particularly","especially","significantly","notably"
}

def _load_en():
    global _SPACY_NLP
    if _SPACY_NLP is None:
        try:
            _SPACY_NLP = spacy.load("en_core_web_sm")
        except OSError as e:
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' not found. "
                "Run: python -m spacy download en_core_web_sm"
            ) from e
    return _SPACY_NLP

def remove_english_adverbs(text: str) -> str:
    """
    使用 spaCy POS 标注移除英文副词（ADV），并额外过滤一些高频副词与 -ly 词形启发式。
    """
    if not text or not text.strip():
        return text
    nlp = _load_en()
    doc = nlp(text)
    kept_tokens: List[str] = []
    for token in doc:
        # 保留标点及空白
        if token.is_punct or token.is_space:
            kept_tokens.append(token.text)
            continue
        low = token.text.lower()
        # 规则1：POS=ADV
        if token.pos_ == "ADV":
            continue
        # 规则2：常见副词表
        if low in EN_BASIC_ADV_LIST:
            continue
        # 规则3：以 -ly 结尾的副词启发式（排除 family, only 等常见非副词）
        if low.endswith("ly") and low not in {"family","only","early","reply","supply","belly","jelly"}:
            # 若该词在 spaCy 标记为形容词等，也可能不是副词，这里仍做启发式删除以保守去副词
            continue

        kept_tokens.append(token.text)

    # 粗略重建：用空格连接，再还原部分标点粘连
    result = " ".join(kept_tokens)
    # 修复空格与标点的常见问题
    result = re.sub(r"\s+([,.!?;:])", r"\1", result)
    result = re.sub(r"\(\s+", "(", result)
    result = re.sub(r"\s+\)", ")", result)
    return result.strip()