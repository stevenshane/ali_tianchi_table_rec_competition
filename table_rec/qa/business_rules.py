"""输出前的业务规则检查。不符合的返回违规说明，由调用方重新分析。

1. 数量必须 > 0
2. 百分比应在 0%～100%
3. 日期格式 yyyy-MM-dd
4. 功率单位仅 W / kW / MW
5. 小计/合计 = 子项之和（利用表格校验结果反向定位可疑单元格）
6. 比例类答案（比例、占比、比率、增长率等）一律带百分号
"""
import re

from .. import numbers as N

QTY_Q = re.compile(r"(数量|个数|多少个|几个|几项|几类|几家|几天|件数|人数|台数)")
PCT_Q = re.compile(r"(比例|占比|百分比|百分之)")
RATIO_Q = re.compile(r"(比例|占比|百分比|百分之|比率|增长率|同比|环比|毛利率|利率|计提比例)")
DATE_Q = re.compile(r"(日期|哪天|哪一天|何时|时间点)")
DATE_LIKE = re.compile(r"\d{4}\s*[年/.\-]\s*\d{1,2}\s*[月/.\-]\s*\d{1,2}")


def _scalars(value) -> list:
    if isinstance(value, list):
        return value
    return [value]


def check(task: dict, value, used_cells: list[dict], tables: list[dict]) -> list[str]:
    q = task["question"]
    fmt = (task.get("answer_format") or "").lower()
    issues = []
    vals = _scalars(value)

    if fmt == "number" and value is not None and not isinstance(value, list):
        if N.parse_number(value) is None:
            issues.append(f"answer_format=number，但答案“{value}”不是数字")

    if QTY_Q.search(q) and not re.search(r"(差|减少|增加|变动|变化)", q):
        for v in vals:
            n = N.parse_number(v)
            if n is not None and n <= 0:
                issues.append(f"数量必须 > 0，但答案为 {v}")

    # 比例类答案一律带百分号
    # 数组中同时要求金额/数量等非比例值时无法区分元素，跳过此项
    mixed = isinstance(value, list) and re.search(r"(金额|余额|数量|账款|价值|成本|收入|费用|额)", q)
    if RATIO_Q.search(q) and not mixed:
        for v in vals:
            if isinstance(v, (int, float, str)) and not isinstance(v, bool) \
                    and N.parse_number(v) is not None and not N.is_percent(v):
                issues.append(f"比例类答案应以百分数带百分号输出（如 59.03%），但答案为 {v}")

    if PCT_Q.search(q) and not re.search(r"(增长|变动|变化|同比|环比)", q):
        for v in vals:
            n = N.parse_number(v)
            if n is not None and N.is_percent(v) and not 0 <= n <= 100:
                issues.append(f"百分比应在 0%～100% 之间，但答案为 {v}")

    for v in vals:
        s = N.clean(v)
        if DATE_LIKE.search(s) and (DATE_Q.search(q) or N.parse_date(s)):
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
                issues.append(f"日期应为 yyyy-MM-dd 格式，但答案为 {s}")
        for u in N.bad_power_units(s):
            issues.append(f"功率单位只能是 W/kW/MW，答案中出现 {u}")

    # 合计关系：答案引用的单元格若在校验中被判为合计不一致的可疑单元格，提示复核
    for uc in used_cells or []:
        ti = uc.get("table", 1) - 1
        if not 0 <= ti < len(tables):
            continue
        for f in tables[ti].get("validation", {}).get("totals_failed", []):
            for s in f["suspects"]:
                if s["row"] == uc.get("row") and s["col"] == uc.get("col"):
                    issues.append(
                        f"引用的单元格 t{ti + 1}({s['row']},{s['col']})={s['got']} 与合计关系不一致"
                        f"（按合计反推应为 {s['expected']}，合计行 {f['total_cell']}），请结合合计关系复核")
    return issues
