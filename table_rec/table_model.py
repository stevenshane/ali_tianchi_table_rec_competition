"""表格 JSON 的通用操作：网格展开、结构校验、跨页合并。

表格 JSON 统一格式：
{
  "source_file", "pages", "table_index", "title", "unit", "language", "header_rows",
  "row_count", "col_count",
  "cells": [{"row","col","rowspan","colspan","text","value_type","style","cell_conf"}],
  "footnotes", "confidence", "recognized_by", "continues_from_prev",
  "quality", "preprocess", "context", "validation"
}
"""
import copy


def new_table(**kw) -> dict:
    t = {
        "source_file": None, "pages": [], "table_index": None, "title": None, "unit": None,
        "language": None, "header_rows": 1, "row_count": 0, "col_count": 0, "cells": [],
        "footnotes": [], "confidence": 0, "recognized_by": None, "continues_from_prev": False,
        "quality": {}, "preprocess": [], "context": {}, "validation": {},
    }
    t.update(kw)
    return t


def recount(t: dict) -> dict:
    cells = t["cells"]
    t["row_count"] = max((c["row"] + c["rowspan"] for c in cells), default=0)
    t["col_count"] = max((c["col"] + c["colspan"] for c in cells), default=0)
    return t


def grid(t: dict, fill_spans: bool = True) -> list[list]:
    """展开为二维文本网格；合并区域默认用左上角文本填充（便于按行列定位），否则为 None。"""
    g = [[None] * t["col_count"] for _ in range(t["row_count"])]
    for c in t["cells"]:
        for r in range(c["row"], min(c["row"] + c["rowspan"], t["row_count"])):
            for k in range(c["col"], min(c["col"] + c["colspan"], t["col_count"])):
                if fill_spans or (r == c["row"] and k == c["col"]):
                    g[r][k] = c["text"]
    return g


def cell_at(t: dict, r: int, k: int) -> dict | None:
    for c in t["cells"]:
        if c["row"] <= r < c["row"] + c["rowspan"] and c["col"] <= k < c["col"] + c["colspan"]:
            return c
    return None


def structure_issues(t: dict) -> list[str]:
    """检查网格覆盖：无重叠、无越界、无空洞。"""
    issues = []
    R, C = t["row_count"], t["col_count"]
    occ = [[0] * C for _ in range(R)]
    for c in t["cells"]:
        if c["rowspan"] < 1 or c["colspan"] < 1 or c["row"] < 0 or c["col"] < 0:
            issues.append(f"非法单元格 ({c['row']},{c['col']})")
            continue
        if c["row"] + c["rowspan"] > R or c["col"] + c["colspan"] > C:
            issues.append(f"单元格越界 ({c['row']},{c['col']})")
            continue
        for r in range(c["row"], c["row"] + c["rowspan"]):
            for k in range(c["col"], c["col"] + c["colspan"]):
                occ[r][k] += 1
    overlaps = sum(1 for row in occ for v in row if v > 1)
    holes = sum(1 for row in occ for v in row if v == 0)
    if overlaps:
        issues.append(f"单元格重叠 {overlaps} 处")
    if holes:
        issues.append(f"网格空洞 {holes} 处")
    return issues


def fill_holes(t: dict) -> dict:
    """为空洞补空单元格（无文字的真实单元格）。"""
    R, C = t["row_count"], t["col_count"]
    occ = [[False] * C for _ in range(R)]
    for c in t["cells"]:
        for r in range(c["row"], min(R, c["row"] + c["rowspan"])):
            for k in range(c["col"], min(C, c["col"] + c["colspan"])):
                occ[r][k] = True
    for r in range(R):
        for k in range(C):
            if not occ[r][k]:
                t["cells"].append({"row": r, "col": k, "rowspan": 1, "colspan": 1, "text": "",
                                   "value_type": "empty", "style": {}, "cell_conf": 100})
    t["cells"].sort(key=lambda c: (c["row"], c["col"]))
    return t


def row_texts(t: dict, r: int) -> list:
    return grid(t)[r] if 0 <= r < t["row_count"] else []


def merge_continuation(prev: dict, nxt: dict) -> dict:
    """把 nxt（跨页续表）追加到 prev 下方；若 nxt 重复了表头则去掉。"""
    out = copy.deepcopy(prev)
    hdr = prev.get("header_rows") or 0
    skip = 0
    if hdr and nxt["row_count"] > hdr:
        g_prev, g_next = grid(prev), grid(nxt)
        if all(g_prev[i] == g_next[i] for i in range(hdr)):
            skip = hdr
    offset = prev["row_count"]
    for c in nxt["cells"]:
        if c["row"] < skip:
            continue
        nc = dict(c)
        nc["row"] = c["row"] - skip + offset
        out["cells"].append(nc)
    out["pages"] = sorted(set(prev.get("pages", [])) | set(nxt.get("pages", [])))
    out["footnotes"] = (prev.get("footnotes") or []) + (nxt.get("footnotes") or [])
    return recount(out)


def merge_cross_page(tables: list[dict]) -> list[dict]:
    """按顺序合并标记为续表、且列数一致的表格。"""
    out: list[dict] = []
    for t in tables:
        if out and t.get("continues_from_prev") and out[-1]["col_count"] == t["col_count"]:
            conf = min(out[-1].get("confidence", 0), t.get("confidence", 0))
            out[-1] = merge_continuation(out[-1], t)
            out[-1]["confidence"] = conf
        else:
            out.append(t)
    return out
