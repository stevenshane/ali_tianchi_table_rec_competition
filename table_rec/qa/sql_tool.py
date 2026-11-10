"""把表格 JSON 装入内存 SQLite，供模型通过 run_sql 工具做筛选、汇总、排序。"""
import json
import re
import sqlite3

from .. import numbers as N
from .. import table_model
from ..cell_validator import column_headers

TOOL = {
    "name": "run_sql",
    "description": (
        "在内存 SQLite 中执行只读 SQL 查询（SELECT），返回最多 200 行 JSON。"
        "每张表格对应一个表 t{序号}，列 row_idx 为原表行号，列 c0..cN 为原表各列文本（合并单元格已向下/向右填充），"
        "is_header=1 为表头行。可用函数 num(x) 把金额/百分比文本转为数值（去千分位、括号负数）。"
        "表与列的含义见 meta 表（table_name, col_name, header）。"
    ),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "SELECT 语句"}},
        "required": ["query"],
        "additionalProperties": False,
    },
}


def build(tables: list[dict]) -> sqlite3.Connection:
    con = sqlite3.connect(":memory:", check_same_thread=False)
    con.create_function("num", 1, N.parse_number, deterministic=True)
    con.execute("CREATE TABLE meta (table_name TEXT, col_name TEXT, header TEXT)")
    for i, t in enumerate(tables, 1):
        name = f"t{i}"
        cols = [f"c{k}" for k in range(t["col_count"])]
        con.execute(f"CREATE TABLE {name} (row_idx INTEGER, is_header INTEGER, "
                    + ", ".join(f"{c} TEXT" for c in cols) + ")")
        hdr = t.get("header_rows") or 1
        for r, row in enumerate(table_model.grid(t)):
            con.execute(f"INSERT INTO {name} VALUES ({', '.join(['?'] * (len(cols) + 2))})",
                        [r, int(r < hdr)] + [v for v in row])
        for c, h in zip(cols, column_headers(t)):
            con.execute("INSERT INTO meta VALUES (?,?,?)", (name, c, h))
    return con


def handler(con: sqlite3.Connection):
    def run(name: str, args: dict) -> str:
        if name != "run_sql":
            raise ValueError(f"未知工具 {name}")
        q = args["query"].strip().rstrip(";")
        if not re.match(r"^(select|with)\b", q, re.I):
            raise ValueError("只允许 SELECT 查询")
        cur = con.execute(q)
        cols = [d[0] for d in cur.description or []]
        rows = [dict(zip(cols, r)) for r in cur.fetchmany(200)]
        return json.dumps(rows, ensure_ascii=False)
    return run
