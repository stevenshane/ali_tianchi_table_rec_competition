"""低置信度二次识别：关键区域放大后复核关键单元格；仍低于阈值则记录到 low_confidence.txt。"""
import copy
import json
import logging
import re
from datetime import datetime

import cv2
import numpy as np

import config
from . import cell_validator, confidence, llm_client, table_model
from .preprocess import upscale
from .quality import to_gray

log = logging.getLogger(__name__)

KEY_HEADER = re.compile(r"(金额|数量|日期|比例|百分比|%|率|小数|编号|号码|身份证|合同|设备|单位|价|额|余额|"
                        r"amount|qty|quantity|date|percent|rate|no\.|id|unit)", re.I)
KEY_TYPES = {"amount", "number", "percent", "date", "year", "id", "unit"}

FIX_SCHEMA = {
    "type": "object",
    "properties": {
        "fixes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "row": {"type": "integer"}, "col": {"type": "integer"},
                    "text": {"type": ["string", "null"]}, "cell_conf": {"type": "integer"},
                },
                "required": ["row", "col", "text", "cell_conf"],
                "additionalProperties": False,
            },
        },
        "self_confidence": {"type": "integer"},
    },
    "required": ["fixes", "self_confidence"],
    "additionalProperties": False,
}

SYSTEM = ("你是表格单元格复核员。给你表格放大后的局部高清图和当前识别结果，请逐个核对指定的关键单元格"
          "（金额、数量、日期、百分比、小数、编号、单位等），给出图中真实文字。逐字照录，保留原始格式；"
          "看不清返回 null，禁止脑补。self_confidence 为复核后整表的把握（0-100）。")


def key_cells(t: dict, validation: dict, limit: int = 80) -> list[tuple[int, int]]:
    heads = cell_validator.column_headers(t)
    hdr = t.get("header_rows") or 1
    keys = list(dict.fromkeys(tuple(x) for x in validation["suspect_cells"]))
    for c in t["cells"]:
        if c["row"] < hdr or (c["row"], c["col"]) in keys:
            continue
        key_col = c["col"] < len(heads) and KEY_HEADER.search(heads[c["col"]] or "")
        if (c.get("value_type") in KEY_TYPES or key_col) and (c.get("cell_conf", 100) < 90 or c.get("text") is None):
            keys.append((c["row"], c["col"]))
    if len(keys) < 5:  # 少量时把关键列全部纳入
        for c in t["cells"]:
            if c["row"] >= hdr and c.get("value_type") in KEY_TYPES and (c["row"], c["col"]) not in keys:
                keys.append((c["row"], c["col"]))
    return keys[:limit]


def row_bands(img: np.ndarray, row_count: int) -> list[tuple[int, int]] | None:
    """有边框表格：用横线位置估计每一行的 y 范围；线数不匹配时返回 None。"""
    gray = to_gray(img)
    h, w = gray.shape
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10)
    hor = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, w // 3), 1)))
    ys = np.where(hor.sum(axis=1) > 0)[0]
    lines = []
    for y in ys:
        if not lines or y - lines[-1] > 3:
            lines.append(int(y))
    if len(lines) != row_count + 1:
        return None
    return [(lines[i], lines[i + 1]) for i in range(row_count)]


def _zoom_regions(img: np.ndarray, t: dict, keys) -> list[np.ndarray]:
    bands = row_bands(img, t["row_count"])
    h = img.shape[0]
    if bands:
        rows = sorted({r for r, _ in keys})
        groups, cur = [], []
        for r in rows:  # 相邻关键行合并成一个区域
            if cur and r - cur[-1] > 2:
                groups.append(cur)
                cur = []
            cur.append(r)
        if cur:
            groups.append(cur)
        crops = []
        hdr = t.get("header_rows") or 1
        head = img[0:bands[hdr - 1][1]] if hdr <= len(bands) else None
        for g in groups:
            y0, y1 = bands[g[0]][0], bands[g[-1]][1]
            band = img[max(0, y0 - 4):min(h, y1 + 4)]
            if head is not None:  # 带上表头便于对齐列
                band = np.vstack([head, band]) if head.shape[1] == band.shape[1] else band
            crops.append(band)
    else:  # 无法定位行：整表切成重叠的上下两半/多块
        n = max(2, min(6, t["row_count"] // 12 + 1))
        step = h // n
        crops = [img[max(0, i * step - 30):min(h, (i + 1) * step + 30)] for i in range(n)]
    out = []
    for c in crops:
        z = config.REFINE_ZOOM if max(c.shape[:2]) * config.REFINE_ZOOM <= 6000 else 2
        out.append(upscale(c, z))
    return out


def refine(t: dict, geometric_crop: np.ndarray, ocr_text: str | None) -> dict:
    """返回置信度最高的版本（可能为原表）。"""
    validation = t["validation"]
    keys = key_cells(t, validation)
    if not keys:
        return t
    crops = _zoom_regions(geometric_crop, t, keys)
    current = {f"{r},{k}": (table_model.cell_at(t, r, k) or {}).get("text") for r, k in keys}
    content = [llm_client.image_block(c) for c in crops[:8]]
    content.append(llm_client.text_block(
        "以上为表格局部放大图（按从上到下顺序）。当前整表网格（行列从 0 开始）：\n"
        + json.dumps(table_model.grid(t), ensure_ascii=False)
        + "\n\n需复核的关键单元格（\"行,列\": 当前文字）：\n" + json.dumps(current, ensure_ascii=False)
        + "\n\n校验发现的问题：\n" + json.dumps({"合计不一致": validation["totals_failed"],
                                             "语义可疑": validation["semantic_suspects"]}, ensure_ascii=False)
    ))
    try:
        res = llm_client.call_json(config.VISION_MODEL, SYSTEM, content, FIX_SCHEMA, config.VISION_EFFORT)
    except Exception as e:
        log.warning("二次识别失败: %s", e)
        return t
    t2 = copy.deepcopy(t)
    for f in res["fixes"]:
        c = table_model.cell_at(t2, f["row"], f["col"])
        if c is not None:
            if c.get("text") != f["text"]:
                c.setdefault("refined_from", c.get("text"))
            c["text"], c["cell_conf"] = f["text"], f["cell_conf"]
    t2["self_confidence"] = res["self_confidence"]
    t2["validation"] = cell_validator.validate(t2)
    t2["confidence"] = confidence.score(t2, t2["validation"], ocr_text)
    t2["refined"] = True
    log.info("二次识别：置信度 %s -> %s", t["confidence"], t2["confidence"])
    return t2 if t2["confidence"] >= t["confidence"] else t


def record_low_confidence(t: dict, name: str):
    with open(config.LOW_CONF_TXT, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "time": datetime.now().isoformat(timespec="seconds"),
            "file": t["source_file"], "json": name, "table_index": t["table_index"],
            "pages": t["pages"], "title": t["title"], "confidence": t["confidence"],
            "suspect_cells": t["validation"].get("suspect_cells"),
            "totals_failed": t["validation"].get("totals_failed"),
            "structure_issues": t["validation"].get("structure_issues"),
        }, ensure_ascii=False) + "\n")
