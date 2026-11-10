"""结构恢复题：由缓存的表格 JSON 确定性生成答案（整表或局部范围）。"""
import re

CN_NUM = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
NUM = r"([一二两三四五六七八九十]+|\d+)"


def _to_int(s: str) -> int:
    if s.isdigit():
        return int(s)
    if s == "十":
        return 10
    if s.startswith("十"):
        return 10 + CN_NUM[s[1]]
    if s.endswith("十"):
        return CN_NUM[s[0]] * 10
    if "十" in s:
        a, b = s.split("十")
        return CN_NUM[a] * 10 + CN_NUM[b]
    return CN_NUM[s]


def parse_range(question: str) -> tuple[set[int] | None, set[int] | None, bool]:
    """返回 (行集合, 列集合, 是否局部)。None 表示该维度未指定。"""
    q = question.replace(" ", "")
    rows = cols = None
    for kind, unit in (("row", "行"), ("col", "列")):
        sel: set[int] = set()
        for m in re.finditer(r"前" + NUM + unit, q):
            sel |= set(range(_to_int(m.group(1))))
        for m in re.finditer(r"第" + NUM + unit, q):
            sel.add(_to_int(m.group(1)) - 1)
        if re.search("首" + unit, q):
            sel.add(0)
        if sel:
            if kind == "row":
                rows = sel
            else:
                cols = sel
    return rows, cols, rows is not None or cols is not None


def build(table: dict, question: str) -> dict:
    rows, cols, partial = parse_range(question)
    if re.search(r"表头", question):  # “表头/表头行”= 全部表头行（多级表头时不止第 0 行）
        rows = (rows or set()) | set(range(max(1, table.get("header_rows") or 1)))
        partial = True
    cells = []
    for c in sorted(table["cells"], key=lambda c: (c["row"], c["col"])):
        r_cover = set(range(c["row"], c["row"] + c["rowspan"]))
        c_cover = set(range(c["col"], c["col"] + c["colspan"]))
        if partial and not ((rows and r_cover & rows) or (cols and c_cover & cols)):
            continue
        cells.append({"text": re.sub(r"\s+", " ", c.get("text") or "").strip(), "row": c["row"], "col": c["col"],
                      "rowspan": c["rowspan"], "colspan": c["colspan"]})
    return {"row_count": table["row_count"], "col_count": table["col_count"], "cells": cells}
