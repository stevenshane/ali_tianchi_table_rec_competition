"""有文字层的 PDF：用 pdfplumber 直接抽取表格（含合并单元格），并抽取页面文字与单词坐标。"""
import re

import pdfplumber

from . import table_model

LINE_SETTINGS = {"vertical_strategy": "lines", "horizontal_strategy": "lines",
                 "snap_tolerance": 3, "join_tolerance": 3, "intersection_tolerance": 3}


def _join_lines(s: str | None) -> str:
    if not s:
        return ""
    parts = [p.strip() for p in s.split("\n") if p.strip()]
    out = ""
    for p in parts:
        if out and (out[-1].isascii() and out[-1].isalnum()) and (p[0].isascii() and p[0].isalnum()):
            out += " " + p
        else:
            out += p
    return out


def _cluster(vals: list[float], tol: float = 2.0) -> list[float]:
    vals = sorted(vals)
    out: list[float] = []
    for v in vals:
        if not out or v - out[-1] > tol:
            out.append(v)
    return out


def _index(edges: list[float], v: float, tol: float = 2.0) -> int:
    for i, e in enumerate(edges):
        if abs(e - v) <= tol:
            return i
    return min(range(len(edges)), key=lambda i: abs(edges[i] - v))


def _table_from_cells(page, cell_boxes) -> dict:
    xs = _cluster([b[0] for b in cell_boxes] + [b[2] for b in cell_boxes])
    ys = _cluster([b[1] for b in cell_boxes] + [b[3] for b in cell_boxes])
    cells = []
    for b in cell_boxes:
        r0, r1 = _index(ys, b[1]), _index(ys, b[3])
        c0, c1 = _index(xs, b[0]), _index(xs, b[2])
        if r1 <= r0 or c1 <= c0:
            continue
        text = _join_lines(page.crop(b, strict=False).extract_text(x_tolerance_ratio=0.15))
        cells.append({"row": r0, "col": c0, "rowspan": r1 - r0, "colspan": c1 - c0, "text": text,
                      "value_type": None, "style": {}, "cell_conf": 98})
    t = table_model.new_table(cells=sorted(cells, key=lambda c: (c["row"], c["col"])))
    return table_model.fill_holes(table_model.recount(t))


NUMERIC = re.compile(r"^[-+(（]?[\d,，.]+%?[)）]?$")


def _guess_header_rows(t: dict) -> int:
    g = table_model.grid(t)
    for i, row in enumerate(g):
        if any(v and NUMERIC.match(v.replace(" ", "")) for v in row[1:]):
            return max(1, i)
    return 1


def _title_above(page, bbox, max_gap=40) -> str | None:
    x0, top = bbox[0], bbox[1]
    region = page.crop((0, max(0, top - max_gap), page.width, top), strict=False)
    text = (region.extract_text() or "").strip().split("\n")
    return text[-1].strip() if text and text[-1].strip() else None


NUM_TOKEN = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+\.\d+")


def _norm_text(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def check_extraction(page, found: list, tables: list[dict]) -> list[str]:
    """文字层抽取质量闸门：返回问题列表，为空表示可直接采用。

    - 页面上含 ≥2 个数字的文本行必须落在某个表格内（否则说明有表格未被线框检测到）
    - 表格框内的单词必须被单元格文本覆盖
    - 空单元格比例不能过高（线框断裂会产生大量碎片格）
    """
    issues = []
    boxes = [ft.bbox for ft in found]
    for line in page.extract_text_lines():
        if len(NUM_TOKEN.findall(line["text"])) >= 2:
            cy = (line["top"] + line["bottom"]) / 2
            if not any(b[1] - 2 <= cy <= b[3] + 2 and line["x0"] >= b[0] - 5 and line["x1"] <= b[2] + 5
                       for b in boxes):
                issues.append(f"表格外存在数据行: {line['text'][:40]}")
                break
    for ft, t in zip(found, tables):
        cell_text = _norm_text("".join(c["text"] for c in t["cells"]))
        words = page.crop(ft.bbox, strict=False).extract_words()
        if words:
            miss = sum(1 for w in words if _norm_text(w["text"]) not in cell_text)
            if miss / len(words) > 0.05:
                issues.append(f"表格框内 {miss}/{len(words)} 个词未进入单元格")
        empty = sum(1 for c in t["cells"] if not c["text"])
        if t["cells"] and empty / len(t["cells"]) > 0.35:
            issues.append(f"空单元格比例过高 {empty}/{len(t['cells'])}")
    return issues


def extract_page(pdf_path, page_no: int) -> tuple[list[dict], dict]:
    """返回 (表格列表, 页面上下文)。表格坐标信息用于判断跨页续表；ctx["issues"] 为抽取质量问题。"""
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_no - 1]
        tables, found = [], []
        for ft in page.find_tables(LINE_SETTINGS):
            if len(ft.cells) < 4:
                continue
            t = _table_from_cells(page, ft.cells)
            if t["row_count"] < 2 or t["col_count"] < 2:
                continue
            t["header_rows"] = _guess_header_rows(t)
            t["title"] = _title_above(page, ft.bbox)
            t["pages"] = [page_no]
            t["bbox_rel"] = [ft.bbox[0] / page.width, ft.bbox[1] / page.height,
                             ft.bbox[2] / page.width, ft.bbox[3] / page.height]
            tables.append(t)
            found.append(ft)
        text = page.extract_text() or ""
        lines = [s.strip() for s in text.split("\n") if s.strip()]
        # 页眉页脚候选：页首 3 行、页尾 2 行中的短行
        hf = [s for s in lines[:3] + lines[-2:] if len(s) <= 30]
        ctx = {"body_text": text, "header_footer": hf, "issues": check_extraction(page, found, tables)}
        return tables, ctx


def page_words(pdf_path, page_no: int) -> str:
    """页面文字层（按行），作为 VLM 识别的精确文字提示。"""
    with pdfplumber.open(pdf_path) as pdf:
        return pdf.pages[page_no - 1].extract_text(layout=False) or ""
