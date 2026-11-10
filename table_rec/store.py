"""识别结果存储：files_json/{原文件名}_{置信度}_{序号}.json + index.json 映射，便于复用。"""
import json
import threading
from datetime import datetime

import config

INDEX = config.FILES_JSON_DIR / "index.json"
_lock = threading.Lock()


def json_name(file_name: str, conf: int, idx: int) -> str:
    return f"{file_name}_{int(conf)}_{idx}.json"


def load_index() -> dict:
    if INDEX.exists():
        return json.loads(INDEX.read_text(encoding="utf-8"))
    return {}


def _write_index(idx: dict):
    config.FILES_JSON_DIR.mkdir(parents=True, exist_ok=True)
    tmp = INDEX.with_suffix(".tmp")
    tmp.write_text(json.dumps(idx, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(INDEX)


def has(file_name: str) -> bool:
    entry = load_index().get(file_name)
    return bool(entry) and all((config.FILES_JSON_DIR / t["json"]).exists() for t in entry["tables"])


def save(file_name: str, tables: list[dict], pages_context: list[dict]) -> list[str]:
    """保存一个文件的全部表格；返回 json 文件名列表。会清理该文件的旧结果。"""
    config.FILES_JSON_DIR.mkdir(parents=True, exist_ok=True)
    with _lock:
        idx = load_index()
        for old in (idx.get(file_name) or {}).get("tables", []):
            (config.FILES_JSON_DIR / old["json"]).unlink(missing_ok=True)
        names, entries = [], []
        for i, t in enumerate(tables, 1):
            t["table_index"] = i
            name = json_name(file_name, t["confidence"], i)
            (config.FILES_JSON_DIR / name).write_text(json.dumps(t, ensure_ascii=False, indent=1), encoding="utf-8")
            names.append(name)
            entries.append({"json": name, "table_index": i, "title": t.get("title"), "pages": t.get("pages"),
                            "confidence": t["confidence"], "shape": [t["row_count"], t["col_count"]]})
        idx[file_name] = {"tables": entries, "pages_context": pages_context,
                          "updated": datetime.now().isoformat(timespec="seconds")}
        _write_index(idx)
    return names


def load_tables(file_name: str) -> list[dict]:
    entry = load_index().get(file_name) or {}
    return [json.loads((config.FILES_JSON_DIR / t["json"]).read_text(encoding="utf-8"))
            for t in entry.get("tables", [])]


def file_entry(file_name: str) -> dict:
    return load_index().get(file_name) or {}
