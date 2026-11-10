"""融合置信度（0~100）：模型自评 + OCR 一致性 + 结构/逻辑校验。"""
import re

from . import numbers as N


def _norm(s: str) -> str:
    return re.sub(r"[\s,，]", "", s or "")


def ocr_agreement(t: dict, ocr_text: str | None) -> float:
    """数字单元格在 OCR/文字层中能找到的比例；无数字时用文本单元格。"""
    if ocr_text is None:
        return 70.0
    pool = _norm(ocr_text)
    hdr = t.get("header_rows") or 1
    nums = [c for c in t["cells"] if c["row"] >= hdr and N.is_numeric(c.get("text"))]
    cells = nums or [c for c in t["cells"] if N.clean(c.get("text")) not in N.EMPTY_MARKS]
    if not cells:
        return 50.0
    hit = sum(1 for c in cells if _norm(c["text"]) and _norm(c["text"]) in pool)
    return 100.0 * hit / len(cells)


def score(t: dict, validation: dict, ocr_text: str | None) -> int:
    cells = t["cells"]
    unreadable = sum(1 for c in cells if c.get("text") is None)
    cell_confs = [c.get("cell_conf", 80) for c in cells if c.get("cell_conf") is not None]
    mean_cell = sum(cell_confs) / len(cell_confs) if cell_confs else 80
    self_conf = 0.5 * (t.get("self_confidence") or mean_cell) + 0.5 * mean_cell

    validity = 100.0
    validity -= 15 * len(validation["structure_issues"])
    validity -= 10 * len(validation["totals_failed"])
    validity -= 3 * len(validation["semantic_suspects"])
    validity -= 5 * len(validation["business_violations"])
    validity -= 2 * unreadable
    validity += min(10, 2 * validation["totals_passed"])   # 合计关系通过加分
    validity = max(0.0, min(100.0, validity))

    agree = ocr_agreement(t, ocr_text)
    conf = 0.45 * self_conf + 0.25 * agree + 0.30 * validity
    return int(round(max(0, min(100, conf))))
