"""写出 submission_yyyyMMdd_HHmmss.xlsx（id, answer）；指定 ids 时与最近一次结果合并。"""
from datetime import datetime
from pathlib import Path

from openpyxl import Workbook, load_workbook

import config


def _latest() -> Path | None:
    files = sorted(config.SUBMISSION_DIR.glob("submission_????????_??????.xlsx"))
    return files[-1] if files else None


def _read_existing() -> dict[str, str]:
    path = _latest()
    if path is None:
        return {}
    wb = load_workbook(path, read_only=True)
    rows = wb.worksheets[0].iter_rows(values_only=True)
    header = list(next(rows))
    out = {}
    for r in rows:
        rec = dict(zip(header, r))
        if rec.get("id") is not None:
            out[str(rec["id"])] = rec.get("answer") or ""
    wb.close()
    return out


def write(results: list[dict], all_ids: list[str]) -> Path:
    data = _read_existing()
    for r in results:
        data[r["id"]] = r["answer"]
    wb = Workbook()
    ws = wb.active
    ws.title = "submission"
    ws.append(["id", "answer"])
    for i in all_ids:  # 按题目清单顺序，每题一行
        ws.append([int(i) if i.isdigit() else i, data.get(i, "")])
    path = config.SUBMISSION_DIR / f"submission_{datetime.now():%Y%m%d_%H%M%S}.xlsx"
    wb.save(path)
    return path
