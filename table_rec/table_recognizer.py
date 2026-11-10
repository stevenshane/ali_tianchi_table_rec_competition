"""视觉模型表格识别：表格区域图（必要时切片）+ OCR/文字层提示 -> 单元格级 JSON。"""
import logging

import cv2
import numpy as np

import config
from . import llm_client, table_model

log = logging.getLogger(__name__)

VALUE_TYPES = ["text", "number", "amount", "percent", "date", "year", "id", "unit", "empty", "other"]

CELL_SCHEMA = {
    "type": "object",
    "properties": {
        "row": {"type": "integer"}, "col": {"type": "integer"},
        "rowspan": {"type": "integer"}, "colspan": {"type": "integer"},
        "text": {"type": ["string", "null"]},
        "value_type": {"type": "string", "enum": VALUE_TYPES},
        "font_color": {"type": ["string", "null"]},
        "bg_color": {"type": ["string", "null"]},
        "cell_conf": {"type": "integer"},
    },
    "required": ["row", "col", "rowspan", "colspan", "text", "value_type", "font_color", "bg_color", "cell_conf"],
    "additionalProperties": False,
}

TABLE_SCHEMA = {
    "type": "object",
    "properties": {
        "tables": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": ["string", "null"]},
                    "unit": {"type": ["string", "null"]},
                    "language": {"type": ["string", "null"]},
                    "header_rows": {"type": "integer"},
                    "row_count": {"type": "integer"},
                    "col_count": {"type": "integer"},
                    "cells": {"type": "array", "items": CELL_SCHEMA},
                    "footnotes": {"type": "array", "items": {"type": "string"}},
                    "continues_from_prev": {"type": "boolean"},
                    "self_confidence": {"type": "integer"},
                },
                "required": ["title", "unit", "language", "header_rows", "row_count", "col_count", "cells",
                             "footnotes", "continues_from_prev", "self_confidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["tables"],
    "additionalProperties": False,
}

SYSTEM = """你是严谨的表格结构识别引擎，把图片中的表格还原为逻辑网格 JSON。

规则：
1. 逻辑网格：row/col 为单元格左上角所在行列（从 0 开始），rowspan/colspan 为真实合并范围。每个真实单元格只输出一次，被合并覆盖的位置不输出；网格中每个位置必须恰好被一个单元格覆盖（空白单元格输出 text=""）。
2. 文字逐字照录，保留原始数字格式（千分位、负号、括号、百分号、小数位），不要换算单位、不要补全。看不清的单元格 text 返回 null，cell_conf 给低分；禁止脑补。
3. 多级表头：header_rows 为表头所占行数。字段继承（上方/左侧合并单元格的含义）只体现在 rowspan/colspan，不要把文字复制到被覆盖的格子。
4. 页眉、页脚、页码、水印不属于表格；表格标题写入 title，表格外的“单位：元”等写入 unit，表下注释写入 footnotes。
5. font_color / bg_color 用简单颜色名（black/red/green/blue/orange/yellow/gray/purple 等），默认黑字白底时填 null。
6. 若表格顶部没有表头、明显是上一页表格的延续，continues_from_prev=true。
7. value_type 描述单元格值类型；cell_conf(0-100) 为该单元格识别把握；self_confidence(0-100) 为整表结构与文字的总体把握。
8. 提供的 OCR/文字层提示仅供参考：文字层（PDF 原生文字）是精确的，OCR 可能有误，以图片为准。"""


def _tiles(img: np.ndarray) -> list[np.ndarray]:
    """超出 MAX_VLM_EDGE 时按长边方向切成带重叠的条带。"""
    h, w = img.shape[:2]
    m = config.MAX_VLM_EDGE
    if max(h, w) <= m:
        return [img]
    tiles = []
    if h >= w:
        step = m - config.TILE_OVERLAP
        for y in range(0, h, step):
            tiles.append(img[y:min(h, y + m)])
            if y + m >= h:
                break
    else:
        step = m - config.TILE_OVERLAP
        for x in range(0, w, step):
            tiles.append(img[:, x:min(w, x + m)])
            if x + m >= w:
                break
    return tiles


def _overview(img: np.ndarray, edge: int = 1568) -> np.ndarray:
    h, w = img.shape[:2]
    s = edge / max(h, w)
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else img


def _to_table(raw: dict) -> dict:
    cells = []
    for c in raw["cells"]:
        cells.append({
            "row": max(0, c["row"]), "col": max(0, c["col"]),
            "rowspan": max(1, c["rowspan"]), "colspan": max(1, c["colspan"]),
            "text": c["text"], "value_type": c["value_type"],
            "style": {k: c[k] for k in ("font_color", "bg_color") if c.get(k)},
            "cell_conf": c["cell_conf"],
        })
    t = table_model.new_table(
        title=raw["title"], unit=raw["unit"], language=raw["language"], header_rows=raw["header_rows"],
        cells=sorted(cells, key=lambda c: (c["row"], c["col"])), footnotes=raw["footnotes"],
        continues_from_prev=raw["continues_from_prev"], recognized_by="vlm",
    )
    t["self_confidence"] = raw["self_confidence"]
    t["declared_shape"] = [raw["row_count"], raw["col_count"]]
    table_model.recount(t)
    return t


def recognize(img: np.ndarray, hint_text: str | None = None, hint_kind: str = "OCR",
              caption: str | None = None, extra: str | None = None,
              enhanced: np.ndarray | None = None) -> list[dict]:
    """img 为仅做几何/分辨率修复的图（颜色保真）；enhanced 为光照/对比度增强后的辅助视图。"""
    tiles = _tiles(img)
    content = []
    if len(tiles) > 1:
        content.append(llm_client.text_block("下图为整张表格的缩略总览："))
        content.append(llm_client.image_block(_overview(img)))
        content.append(llm_client.text_block(
            f"以下是同一张表格按顺序切分的 {len(tiles)} 个高清切片（相邻切片有少量重叠，请去重），"
            "请合并为一张完整表格："))
    for tile in tiles:
        content.append(llm_client.image_block(tile))
    if enhanced is not None:
        content.append(llm_client.text_block("同一表格经光照/对比度增强后的辅助视图（仅用于看清文字，颜色以前面的原色图为准）："))
        content.append(llm_client.image_block(_overview(enhanced, config.MAX_VLM_EDGE)))
    prompt = "请识别图中的表格，输出逻辑网格 JSON。"
    if caption:
        prompt += f"\n版面分析给出的表格标题：{caption}"
    if hint_text:
        prompt += f"\n\n{hint_kind}提示（按行）：\n{hint_text[:12000]}"
    if extra:
        prompt += "\n\n" + extra
    content.append(llm_client.text_block(prompt))
    res = llm_client.call_json(config.VISION_MODEL, SYSTEM, content, TABLE_SCHEMA, config.VISION_EFFORT)
    tables = []
    for raw in res["tables"]:
        t = _to_table(raw)
        if t["row_count"] and t["col_count"]:
            tables.append(t)
    return tables
