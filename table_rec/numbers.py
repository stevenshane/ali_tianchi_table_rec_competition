"""数值/日期/单位解析工具。"""
import re
from datetime import datetime

_NUM = re.compile(r"^[-+]?\d+(?:\.\d+)?$")
EMPTY_MARKS = {"", "-", "—", "--", "——", "/", "－", "–", "N/A", "n/a", "NA"}


def clean(text) -> str:
    if text is None:
        return ""
    return str(text).strip().replace("　", " ")


def parse_number(text) -> float | None:
    """解析金额/数量/百分比文本：去千分位，括号或前置负号视为负数，% 返回百分数值本身。"""
    s = clean(text)
    if s in EMPTY_MARKS:
        return None
    s = s.replace(",", "").replace("，", "").replace(" ", "").replace("¥", "").replace("￥", "").replace("$", "")
    neg = False
    if (s.startswith("(") and s.endswith(")")) or (s.startswith("（") and s.endswith("）")):
        neg, s = True, s[1:-1]
    s = s.rstrip("%％")
    s = s.replace("−", "-").replace("－", "-")
    if not _NUM.match(s):
        return None
    v = float(s)
    return -v if neg else v


def is_numeric(text) -> bool:
    return parse_number(text) is not None


def is_percent(text) -> bool:
    s = clean(text)
    return s.endswith("%") or s.endswith("％")


DATE_PATTERNS = ["%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%Y年%m月%d日", "%Y%m%d", "%d/%m/%Y", "%m/%d/%Y"]


def parse_date(text) -> str | None:
    """解析为 yyyy-MM-dd，失败返回 None。"""
    s = clean(text).replace(" ", "")
    for p in DATE_PATTERNS:
        try:
            return datetime.strptime(s, p).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def is_year(text) -> bool:
    return bool(re.fullmatch(r"(19|20)\d{2}(年|年度)?", clean(text).replace(" ", "")))


POWER_RE = re.compile(r"(\d[\d.,]*)\s*([a-zA-Z]*[wW])\b")
VALID_POWER = {"W", "kW", "MW"}


def bad_power_units(text) -> list[str]:
    """返回文本中不合规的功率单位（仅允许 W / kW / MW）。"""
    return [u for _, u in POWER_RE.findall(clean(text)) if u not in VALID_POWER and u.lower() in
            {"w", "kw", "mw", "gw"}]


def fmt_number(v: float) -> str:
    """输出数字：去千分位，整数不带小数点，浮点去除多余 0。"""
    if v is None:
        return ""
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return ("%.10f" % v).rstrip("0").rstrip(".")


def digit_distance(a: float, b: float) -> int:
    """两个数字字符串的编辑距离（用于判断是否为 OCR 误识别）。"""
    x, y = fmt_number(a), fmt_number(b)
    dp = list(range(len(y) + 1))
    for i, cx in enumerate(x, 1):
        prev, dp[0] = dp[0], i
        for j, cy in enumerate(y, 1):
            prev, dp[j] = dp[j], min(dp[j] + 1, dp[j - 1] + 1, prev + (cx != cy))
    return dp[-1]
