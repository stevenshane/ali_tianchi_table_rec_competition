"""读取 tests.xlsx 题目清单，按 ids 过滤。"""
from openpyxl import load_workbook

import config

FIELDS = ["id", "file_name", "question_type", "question", "table_hint", "answer_format"]


def parse_ids(spec: str | None) -> set[str] | None:
    """'1,5,10-20' -> {'1','5','10',...,'20'}；None 表示全部。"""
    if not spec:
        return None
    out = set()
    for part in spec.replace("，", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out |= {str(i) for i in range(int(a), int(b) + 1)}
        else:
            out.add(part)
    return out


def load_tasks(ids: set[str] | None = None) -> list[dict]:
    wb = load_workbook(config.TESTS_XLSX, read_only=True, data_only=True)
    ws = wb.worksheets[0]
    rows = ws.iter_rows(values_only=True)
    header = [str(h).strip() if h is not None else "" for h in next(rows)]
    tasks = []
    for row in rows:
        rec = {h: row[i] if i < len(row) else None for i, h in enumerate(header)}
        if rec.get("id") is None or rec.get("file_name") is None:
            continue
        t = {k: (str(rec.get(k)).strip() if rec.get(k) is not None else "") for k in FIELDS}
        if isinstance(rec["id"], float) and rec["id"].is_integer():
            t["id"] = str(int(rec["id"]))
        if ids is None or t["id"] in ids:
            tasks.append(t)
    wb.close()
    return tasks
