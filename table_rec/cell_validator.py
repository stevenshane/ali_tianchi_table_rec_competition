"""单元格层逻辑校验：文字链路与结构链路合并后的正确性检查。

- 结构：覆盖/重叠/越界
- 语义：列类型（由表头推断）与单元格值是否匹配（反例：年份=上海、城市=2026、金额=18%）
- 合计：小计/合计 = 子项之和，不一致时反向定位可疑单元格
- 业务规则：数量>0、百分比 0~100、日期格式、功率单位
"""
import re

from . import numbers as N
from . import table_model

TOTAL_RE = re.compile(r"(合\s*计|总\s*计|小\s*计|共\s*计|^total|subtotal|grand total)", re.I)

PURE_TOTAL = re.compile(r"(合\s*计|总\s*计|小\s*计|共\s*计|total|subtotal|grand total)\s*[:：]?"
                        r"(\s*[（(][^）)]*[）)])?", re.I)

COL_RULES = [  # (类型, 表头关键词)
    ("year", re.compile(r"(年份|年度|^year$)", re.I)),
    ("date", re.compile(r"(日期|^date$|时间)", re.I)),
    ("percent", re.compile(r"(比例|比率|占比|百分比|%|percent|ratio|rate)", re.I)),
    ("quantity", re.compile(r"(数量|个数|件数|人数|台数|qty|quantity|count)", re.I)),
    ("amount", re.compile(r"(金额|余额|收入|成本|费用|价格|单价|总价|账面|原值|净值|amount|price|cost|revenue)", re.I)),
    ("text", re.compile(r"(城市|地区|名称|公司|单位名称|姓名|项目|类别|理由|原因|说明|备注|摘要|city|name|company|region)", re.I)),
]
NUMERIC_TYPES = {"year", "percent", "quantity", "amount"}


def column_headers(t: dict) -> list[str]:
    g = table_model.grid(t)
    hdr = max(1, t.get("header_rows") or 1)
    heads = []
    for k in range(t["col_count"]):
        parts = []
        for r in range(min(hdr, t["row_count"])):
            v = N.clean(g[r][k])
            if v and v not in parts:
                parts.append(v)
        heads.append("/".join(parts))
    return heads


def column_types(t: dict) -> list[str | None]:
    if t.get("header_rows") == 0:  # 键值对表单：首行不是表头，不按列推断类型
        return [None] * t["col_count"]
    heads = column_headers(t)
    g = table_model.grid(t, fill_spans=False)
    hdr = t.get("header_rows") or 1
    types = []
    for k, h in enumerate(heads):
        leaf = h.split("/")[-1] if h else ""
        # 先按最末级表头匹配，全部不中再看上级表头（避免“期末余额/计提理由”被判成金额列）
        typ = next((name for name, rx in COL_RULES if rx.search(leaf)), None)             or next((name for name, rx in COL_RULES if rx.search(h)), None)
        if typ is None:  # 按多数派推断
            vals = [g[r][k] for r in range(hdr, t["row_count"]) if N.clean(g[r][k]) not in N.EMPTY_MARKS]
            if vals and sum(N.is_numeric(v) for v in vals) / len(vals) >= 0.7:
                typ = "number"
        types.append(typ)
    return types


def _row_label(g, r) -> str:
    return " ".join(N.clean(v) for v in g[r][:2] if v)


def semantic_suspects(t: dict, types) -> list[dict]:
    out = []
    hdr = t.get("header_rows") or 1
    g = table_model.grid(t, fill_spans=False)
    # 表中间的子表头行（如 Mean/SD）：整行没有数字，不做类型检查
    subheader = {r for r in range(t["row_count"]) if not any(N.is_numeric(v) for v in g[r])}
    for c in t["cells"]:
        if c["row"] < hdr or c["row"] in subheader or (c["colspan"] > 1 and c["col"] == 0):
            continue
        typ = types[c["col"]] if c["col"] < len(types) else None
        v = N.clean(c.get("text"))
        if not typ or v in N.EMPTY_MARKS:
            continue
        reason = None
        if typ == "year" and not N.is_year(v):
            reason = f"年份列出现非年份值“{v}”"
        elif typ == "date" and not N.parse_date(v) and not N.is_year(v) and not re.search(r"\d", v):
            reason = f"日期列出现非日期值“{v}”"
        elif typ in ("amount", "quantity", "number") and not N.is_numeric(v):
            reason = f"数值列出现非数值“{v}”"
        elif typ == "amount" and N.is_percent(v):
            reason = f"金额列出现百分比“{v}”"
        elif typ == "text" and N.is_numeric(v) and not re.search(r"[一-鿿A-Za-z]", v):
            reason = f"文本列出现纯数字“{v}”"
        if reason:
            out.append({"row": c["row"], "col": c["col"], "reason": reason})
    return out


def business_violations(t: dict, types) -> list[dict]:
    out = []
    hdr = t.get("header_rows") or 1
    for c in t["cells"]:
        if c["row"] < hdr:
            continue
        v = N.clean(c.get("text"))
        typ = types[c["col"]] if c["col"] < len(types) else None
        num = N.parse_number(v)
        if typ == "quantity" and num is not None and num <= 0:
            out.append({"row": c["row"], "col": c["col"], "reason": f"数量应 > 0：{v}"})
        if typ == "percent" and num is not None and N.is_percent(v) and not (0 <= num <= 100):
            out.append({"row": c["row"], "col": c["col"], "reason": f"百分比超出 0~100%：{v}"})
        for u in N.bad_power_units(v):
            out.append({"row": c["row"], "col": c["col"], "reason": f"功率单位应为 W/kW/MW：{u}"})
    return out


def _approx(a: float, b: float) -> bool:
    return abs(a - b) <= max(0.011, abs(b) * 1e-6)


def total_checks(t: dict, types) -> tuple[list[dict], list[dict]]:
    """小计/合计 = 子项之和。返回 (通过的检查, 不一致项含可疑单元格)。"""
    g = table_model.grid(t, fill_spans=False)
    hdr = t.get("header_rows") or 1
    R, C = t["row_count"], t["col_count"]
    passed, failed = [], []
    total_rows = [r for r in range(hdr, R) if TOTAL_RE.search(_row_label(g, r))]
    num_cols = [k for k in range(1, C) if (types[k] in NUMERIC_TYPES | {"number"}) and types[k] != "percent"]
    prev_total = hdr - 1
    for r in total_rows:
        for k in num_cols:
            tv = N.parse_number(g[r][k])
            if tv is None:
                continue
            children = [(i, N.parse_number(g[i][k])) for i in range(prev_total + 1, r)
                        if i not in total_rows and N.parse_number(g[i][k]) is not None]
            if len(children) < 2:
                continue
            s = sum(v for _, v in children)
            if _approx(s, tv):
                passed.append({"row": r, "col": k})
                continue
            # 备选：总计 = 之前各小计之和
            subs = [N.parse_number(g[i][k]) for i in total_rows if i < r and N.parse_number(g[i][k]) is not None]
            if len(subs) >= 2 and _approx(sum(subs), tv):
                passed.append({"row": r, "col": k})
                continue
            # 备选：总计 = 表头以下全部非合计行
            allc = [N.parse_number(g[i][k]) for i in range(hdr, r)
                    if i not in total_rows and N.parse_number(g[i][k]) is not None]
            if _approx(sum(allc), tv):
                passed.append({"row": r, "col": k})
                continue
            suspects = []
            for i, v in children:
                expect = tv - (s - v)
                if N.digit_distance(v, expect) <= 1:
                    suspects.append({"row": i, "col": k, "expected": N.fmt_number(expect),
                                     "got": N.fmt_number(v)})
            if not suspects and not PURE_TOTAL.fullmatch(_row_label(g, r).strip()):
                continue  # “Total cholesterol”之类并非合计关系，且无单字误识的迹象 -> 不判错
            if not suspects:
                suspects.append({"row": r, "col": k, "expected": N.fmt_number(s), "got": N.fmt_number(tv)})
            failed.append({"total_cell": [r, k], "sum": N.fmt_number(s), "total": N.fmt_number(tv),
                           "suspects": suspects})
        prev_total = r
    return passed, failed


def validate(t: dict) -> dict:
    types = column_types(t)
    struct = table_model.structure_issues(t)
    sem = semantic_suspects(t, types)
    biz = business_violations(t, types)
    passed, failed = total_checks(t, types)
    suspects = {(s["row"], s["col"]) for s in sem}
    for f in failed:
        suspects |= {(s["row"], s["col"]) for s in f["suspects"]}
    return {
        "column_types": types,
        "structure_issues": struct,
        "semantic_suspects": sem,
        "business_violations": biz,
        "totals_passed": len(passed),
        "totals_failed": failed,
        "suspect_cells": sorted(suspects),
    }
