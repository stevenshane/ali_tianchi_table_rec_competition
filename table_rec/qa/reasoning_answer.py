"""内容提取/推理题：基于 files_json 中的表格 JSON，由模型（可调用 SQL 工具）作答。"""
import json
import logging

import config
from .. import llm_client, table_model
from . import business_rules, sql_tool

log = logging.getLogger(__name__)

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer_json": {"type": "string",
                        "description": "最终答案的 JSON 编码：数字、字符串、数组或 null"},
        "used_cells": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"table": {"type": "integer"}, "row": {"type": "integer"}, "col": {"type": "integer"}},
                "required": ["table", "row", "col"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["answer_json", "used_cells"],
    "additionalProperties": False,
}

SYSTEM = """你是严谨的表格问答助手。只能依据给出的表格识别结果（JSON 网格与页面上下文）作答，不要使用外部知识补全数据。

作答规则：
1. answer_json 只放最终答案的 JSON 编码，不要写“根据表格可知”“答案是”等说明。
2. answer_format=number：输出数字（去千分位逗号；表中为百分比时保留百分号字符串，如 "18%"）；
   计算得出的比例、占比、比率、增长率等一律以百分数形式并带百分号输出，如 0.5903 应写成 "59.03%"；
   answer_format=string：输出字符串；answer_format=json_array：输出 JSON 数组，数组元素直接是题目要求的值，
   顺序与题目所列顺序一致；某值在表格中为空则填空字符串 ""。
3. 文本答案保留原始含义与原文写法，可去除多余空格和换行；数值保持表中精度，计算结果不要无故四舍五入。
4. 日期统一为 yyyy-MM-dd；功率单位只用 W/kW/MW。
5. 定位数据时注意多级表头、合并单元格（已在网格中向下/向右填充）、字段继承、单位说明与脚注；区分“本期/上期”“期末/期初”等列。
6. 需要筛选、汇总、排序、计数时，可调用 run_sql 工具在 SQLite 中计算并复核；涉及合计时可用“合计=子项之和”交叉验证。
7. 单元格为 null 表示识别时看不清：若答案依赖它，answer_json 返回 null，禁止脑补。
8. used_cells 列出答案所依据的单元格（table 为表格编号 t 后的数字，row/col 为网格行列号）。"""


def render_table(i: int, t: dict) -> str:
    lines = [f"### 表格 t{i}（文件 {t.get('source_file')}，页 {t.get('pages')}，识别置信度 {t.get('confidence')}）",
             f"标题: {t.get('title')}；单位: {t.get('unit')}；语言: {t.get('language')}；"
             f"表头行数: {t.get('header_rows')}；逻辑行列: {t['row_count']} x {t['col_count']}"]
    merges = [f"({c['row']},{c['col']}) {c['rowspan']}x{c['colspan']} {c.get('text')!r}"
              for c in t["cells"] if c["rowspan"] > 1 or c["colspan"] > 1]
    if merges:
        lines.append("合并单元格(行,列 行跨度x列跨度): " + "; ".join(merges))
    styles = [f"({c['row']},{c['col']}) {c['style']}" for c in t["cells"] if c.get("style")]
    if styles:
        lines.append("单元格颜色: " + "; ".join(styles))
    lines.append("网格（行号: 各列文本；null=看不清）:")
    for r, row in enumerate(table_model.grid(t)):
        lines.append(f"r{r}: " + json.dumps(row, ensure_ascii=False))
    if t.get("footnotes"):
        lines.append("脚注: " + " | ".join(t["footnotes"]))
    v = t.get("validation") or {}
    if v.get("totals_failed"):
        lines.append("校验提示（合计不一致）: " + json.dumps(v["totals_failed"], ensure_ascii=False))
    return "\n".join(lines)


def render_context(file_entry: dict, tables: list[dict]) -> str:
    ctxs = file_entry.get("pages_context") or [t.get("context") for t in tables[:1]]
    parts = []
    for c in ctxs:
        if not c:
            continue
        parts.append(f"[页 {c.get('page', '?')}] 页面标题: {c.get('page_title')}；语言: {c.get('language')}；"
                     f"图片: {c.get('figures')}；页眉页脚: {c.get('header_footer')}\n表外正文:\n"
                     f"{(c.get('body_text') or '')[:3000]}")
    return "\n\n".join(parts)


def _ask(task: dict, tables: list[dict], file_entry: dict, feedback: str | None) -> tuple:
    con = sql_tool.build(tables)
    body = "\n\n".join(render_table(i, t) for i, t in enumerate(tables, 1)) if tables else "（未识别到表格）"
    prompt = (f"文件: {task['file_name']}\n题型: {task['question_type']}\n"
              f"表格提示(table_hint): {task.get('table_hint') or '无'}\n答案格式(answer_format): {task.get('answer_format')}\n"
              f"题目: {task['question']}\n\n{body}\n\n## 页面上下文\n{render_context(file_entry, tables)}")
    if feedback:
        prompt += f"\n\n## 上一次作答未通过业务规则检查，请重新分析\n{feedback}"
    res = llm_client.call_with_tools(config.REASONING_MODEL, SYSTEM, [llm_client.text_block(prompt)],
                                     ANSWER_SCHEMA, config.REASONING_EFFORT, [sql_tool.TOOL], sql_tool.handler(con))
    try:
        value = json.loads(res["answer_json"])
    except json.JSONDecodeError:
        value = res["answer_json"]
    return value, res.get("used_cells", [])


def answer(task: dict, tables: list[dict], file_entry: dict) -> tuple:
    """返回 (答案值, 违规说明列表)。违规时重新分析一次，取违规更少的结果。"""
    value, used = _ask(task, tables, file_entry, None)
    issues = business_rules.check(task, value, used, tables)
    if not issues:
        return value, []
    log.info("题 %s 业务规则不通过，重新分析: %s", task["id"], issues)
    fb = f"上次答案: {json.dumps(value, ensure_ascii=False)}\n问题:\n- " + "\n- ".join(issues)
    value2, used2 = _ask(task, tables, file_entry, fb)
    issues2 = business_rules.check(task, value2, used2, tables)
    return (value2, issues2) if len(issues2) <= len(issues) else (value, issues)
