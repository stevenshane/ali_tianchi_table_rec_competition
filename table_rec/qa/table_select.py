"""根据 table_hint / 题目定位目标表格。"""
import re
from difflib import SequenceMatcher

import config
from .. import llm_client, table_model

STOP = re.compile(r"(表格?|中|的|，|,|。|情况|明细|\s)")


def _norm(s: str) -> str:
    return STOP.sub("", (s or "").lower())


def describe(t: dict) -> str:
    g = table_model.grid(t)
    head = " ".join(str(v) for row in g[: (t.get("header_rows") or 1) + 1] for v in row if v)
    first_col = " ".join(str(row[0]) for row in g if row and row[0])
    return f"{t.get('title') or ''} {head} {first_col}"


def _score(t: dict, key: str) -> float:
    if not key:
        return 0.0
    k = _norm(key)
    title = _norm(t.get("title") or "")
    if k and title and (k in title or title in k):
        return 1.0
    body = _norm(describe(t))
    if k and k in body:
        return 0.8
    return max(SequenceMatcher(None, k, title).ratio() if title else 0.0,
               0.6 * SequenceMatcher(None, k, body[: len(k) * 4]).ratio())


def rank(tables: list[dict], question: str, hint: str) -> list[tuple[float, dict]]:
    scored = []
    for t in tables:
        s = 2 * _score(t, hint) + _score(t, question[:30])
        scored.append((s, t))
    return sorted(scored, key=lambda x: -x[0])


def pick(tables: list[dict], question: str, hint: str) -> tuple[dict | None, bool]:
    """返回 (最佳表格, 是否明确)。"""
    if not tables:
        return None, False
    if len(tables) == 1:
        return tables[0], True
    ranked = rank(tables, question, hint)
    best, second = ranked[0][0], ranked[1][0]
    return ranked[0][1], best >= 1.0 and best - second >= 0.3


CHOOSE_SCHEMA = {
    "type": "object",
    "properties": {"table": {"type": "integer"}},
    "required": ["table"],
    "additionalProperties": False,
}


def choose_with_llm(tables: list[dict], question: str, hint: str) -> dict:
    """规则无法明确定位时，让模型根据标题与前两行选择表格（table 从 1 开始）。"""
    lines = []
    for i, t in enumerate(tables, 1):
        g = table_model.grid(t)
        lines.append(f"t{i}: 标题={t.get('title')!r} 页={t.get('pages')} 前两行={g[:2]} 首列={[r[0] for r in g[:12]]}")
    res = llm_client.call_json(
        config.REASONING_MODEL, "根据题目与表格提示，从候选表格中选出题目所指的表格，只返回表格编号。",
        [llm_client.text_block(f"题目: {question}\n表格提示: {hint or '无'}\n候选:\n" + "\n".join(lines))],
        CHOOSE_SCHEMA, "low")
    i = res["table"]
    return tables[i - 1] if 1 <= i <= len(tables) else tables[0]
