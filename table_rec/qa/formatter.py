"""答案格式归一：按 answer_format 输出最终字符串。"""
import json
import re

from .. import numbers as N

THOUSANDS = re.compile(r"^[-+]?\(?\d{1,3}(,\d{3})+(\.\d+)?\)?$")


def _norm_scalar(v):
    """数组元素：None -> ""；带千分位的纯数字文本 -> 数字；其它文本去多余空白。"""
    if v is None:
        return ""
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, (int, float)):
        return float(N.fmt_number(v)) if "." in N.fmt_number(v) else int(N.fmt_number(v))
    s = re.sub(r"\s+", " ", str(v)).strip()
    if THOUSANDS.match(s):
        n = N.parse_number(s)
        if n is not None:
            return float(N.fmt_number(n)) if "." in N.fmt_number(n) else int(N.fmt_number(n))
    return s


def _number(v) -> str:
    if v is None:
        return ""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return N.fmt_number(v)
    s = str(v).strip()
    if N.is_percent(s):  # 百分比保留百分号，仅去千分位
        return s.replace(",", "").replace("，", "")
    n = N.parse_number(s)
    return N.fmt_number(n) if n is not None else s


def to_answer(value, answer_format: str, question_type: str) -> str:
    fmt = (answer_format or "").lower()
    if question_type == "structure" or fmt == "json":
        return json.dumps(value, ensure_ascii=False) if value is not None else ""
    if fmt == "json_array" or isinstance(value, list):
        arr = value if isinstance(value, list) else ([] if value is None else [value])
        return json.dumps([_norm_scalar(x) for x in arr], ensure_ascii=False)
    if fmt == "number":
        return _number(value)
    if value is None:
        return ""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return N.fmt_number(value)
    return re.sub(r"\s+", " ", str(value)).strip()
