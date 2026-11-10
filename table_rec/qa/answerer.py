"""单题作答：读取 files_json 缓存 -> 选表 -> 结构题确定性生成 / 提取推理题模型作答 -> 格式化。"""
import logging

from .. import store
from . import formatter, reasoning_answer, structure_answer, table_select

log = logging.getLogger(__name__)


def answer_task(task: dict) -> dict:
    tables = store.load_tables(task["file_name"])
    entry = store.file_entry(task["file_name"])
    issues: list[str] = []
    try:
        if task["question_type"] == "structure":
            best, sure = table_select.pick(tables, task["question"], task.get("table_hint", ""))
            if best is not None and not sure:
                best = table_select.choose_with_llm(tables, task["question"], task.get("table_hint", ""))
            value = structure_answer.build(best, task["question"]) if best else None
        else:
            ranked = [t for _, t in table_select.rank(tables, task["question"], task.get("table_hint", ""))]
            value, issues = reasoning_answer.answer(task, ranked, entry)
        answer = formatter.to_answer(value, task.get("answer_format", ""), task["question_type"])
    except Exception as e:
        log.exception("题 %s 作答失败: %s", task["id"], e)
        answer, issues = "", [f"error: {e}"]
    return {"id": task["id"], "answer": answer, "issues": issues}
