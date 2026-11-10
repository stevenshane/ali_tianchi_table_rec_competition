"""版面分区：表格 / 正文 / 图片 / 页眉页脚。

线框检测（OpenCV）+ OCR 文本框 + 视觉模型版面分析（复杂页面）三者结合。
"""
import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

import config
from . import llm_client, ocr
from .quality import to_gray

log = logging.getLogger(__name__)


@dataclass
class Region:
    kind: str                                  # table/text/figure/header/footer/title
    box: tuple[int, int, int, int]             # 像素坐标 x0,y0,x1,y1
    caption: str | None = None
    summary: str | None = None


@dataclass
class Layout:
    tables: list[Region] = field(default_factory=list)
    page_title: str | None = None
    language: str | None = None
    body_text: str = ""
    header_footer: list[str] = field(default_factory=list)
    figures: list[str] = field(default_factory=list)
    source: str = "lines"

    def context(self) -> dict:
        return {"page_title": self.page_title, "language": self.language, "body_text": self.body_text,
                "header_footer": self.header_footer, "figures": self.figures, "layout_source": self.source}


def line_tables(img: np.ndarray) -> list[tuple[int, int, int, int]]:
    """形态学提取横竖线，找出有边框的表格区域。"""
    gray = to_gray(img)
    h, w = gray.shape
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10)
    hor = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, w // 30), 1)))
    ver = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, h // 40))))
    grid = cv2.dilate(cv2.add(hor, ver), np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x, y, bw_, bh = cv2.boundingRect(c)
        if bw_ > w * 0.2 and bh > h * 0.03 and bw_ * bh > w * h * 0.02:
            boxes.append((x, y, x + bw_, y + bh))
    return sorted(boxes, key=lambda b: (b[1], b[0]))


LAYOUT_SCHEMA = {
    "type": "object",
    "properties": {
        "page_title": {"type": ["string", "null"]},
        "language": {"type": "string"},
        "regions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["table", "text", "figure", "header", "footer", "title"]},
                    "bbox": {"type": "array", "items": {"type": "integer"}},
                    "caption": {"type": ["string", "null"]},
                    "is_decorative": {"type": "boolean"},
                    "summary": {"type": ["string", "null"]},
                },
                "required": ["type", "bbox", "caption", "is_decorative", "summary"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["page_title", "language", "regions"],
    "additionalProperties": False,
}

LAYOUT_SYSTEM = (
    "你是文档版面分析器。识别页面中的区域：table（表格，包括无边框/弱边框表格、日程网格等二维结构）、"
    "text（正文段落）、figure（图片/图表/logo）、header（页眉）、footer（页脚/页码）、title（标题）。"
    "bbox 使用相对整张图片的归一化坐标 [x0,y0,x1,y1]，取值 0-1000，宜略宽松以完整包含区域。"
    "同一张表跨越的区域只输出一个 table；表格的标题写在 caption。figure 若仅为装饰/水印/logo 设 is_decorative=true，"
    "否则用 summary 一句话描述其内容。language 填页面主要语言或语言组合（如“中文”“英文”“中英文”）。"
    "看不清的文字返回 null，禁止脑补。"
)


def _vlm_layout(img: np.ndarray) -> dict:
    h, w = img.shape[:2]
    s = min(1.0, 1600 / max(h, w))
    small = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else img
    return llm_client.call_json(
        config.VISION_MODEL, LAYOUT_SYSTEM,
        [llm_client.image_block(small), llm_client.text_block("请输出该页面的版面分区。")],
        LAYOUT_SCHEMA, config.LAYOUT_EFFORT,
    )


def _to_px(b, w, h, margin=0.02):
    x0, y0, x1, y1 = [max(0, min(1000, int(v))) for v in (list(b) + [0, 0, 1000, 1000])[:4]]
    mx, my = int(w * margin), int(h * margin)
    return (max(0, x0 * w // 1000 - mx), max(0, y0 * h // 1000 - my),
            min(w, x1 * w // 1000 + mx), min(h, y1 * h // 1000 + my))


def _overlap(a, b) -> float:
    """a 与 b 的交集占 a 的比例。"""
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    area = (a[2] - a[0]) * (a[3] - a[1])
    return inter / area if area else 0.0


def _lines_text(tokens: list[ocr.OcrToken]) -> str:
    rows: list[list[ocr.OcrToken]] = []
    for t in sorted(tokens, key=lambda t: (t.box[1], t.box[0])):
        if rows and abs(rows[-1][0].box[1] - t.box[1]) < max(8, t.height * 0.6):
            rows[-1].append(t)
        else:
            rows.append([t])
    return "\n".join(" ".join(x.text for x in sorted(r, key=lambda t: t.box[0])) for r in rows)


def is_simple(img, line_boxes, tokens) -> bool:
    """单个有框表格占据页面主体，且表外文字很少 -> 简单页面，不调用 VLM 版面分析。"""
    h, w = img.shape[:2]
    if len(line_boxes) != 1:
        return False
    b = line_boxes[0]
    if (b[2] - b[0]) * (b[3] - b[1]) < 0.5 * w * h:
        return False
    outside = [t for t in tokens if _overlap(t.box, b) < 0.5]
    return len(outside) <= 4


def analyze(img: np.ndarray, tokens: list[ocr.OcrToken], use_vlm: bool = True) -> Layout:
    h, w = img.shape[:2]
    boxes = line_tables(img)
    lay = Layout()
    if not use_vlm or is_simple(img, boxes, tokens):
        lay.tables = [Region("table", b) for b in boxes] or [Region("table", (0, 0, w, h))]
        lay.source = "lines"
    else:
        try:
            res = _vlm_layout(img)
            lay.source = "vlm+lines"
            lay.page_title = res.get("page_title")
            lay.language = res.get("language")
            for r in res["regions"]:
                box = _to_px(r["bbox"], w, h)
                if r["type"] == "table":
                    # 用线框检测结果吸附（更精确）
                    for lb in boxes:
                        if _overlap(lb, box) > 0.7:
                            box = (min(box[0], lb[0]), min(box[1], lb[1]), max(box[2], lb[2]), max(box[3], lb[3]))
                    lay.tables.append(Region("table", box, caption=r.get("caption")))
                elif r["type"] in ("header", "footer"):
                    lay.header_footer.append(_lines_text([t for t in tokens if _overlap(t.box, box) > 0.5]))
                elif r["type"] == "figure" and not r.get("is_decorative") and r.get("summary"):
                    lay.figures.append(r["summary"])
            hf_boxes = [_to_px(r["bbox"], w, h, 0) for r in res["regions"] if r["type"] in ("header", "footer")]
        except Exception as e:
            log.warning("VLM 版面分析失败，退回线框检测: %s", e)
            lay.tables = [Region("table", b) for b in boxes]
            hf_boxes = []
        if not lay.tables:
            lay.tables = [Region("table", (0, 0, w, h))]
        lay.tables.sort(key=lambda r: (r.box[1], r.box[0]))
        tokens = [t for t in tokens if all(_overlap(t.box, hb) < 0.5 for hb in hf_boxes)]
    body = [t for t in tokens if all(_overlap(t.box, r.box) < 0.5 for r in lay.tables)]
    lay.body_text = _lines_text(body)
    return lay


def crop(img: np.ndarray, box) -> np.ndarray:
    x0, y0, x1, y1 = box
    return img[y0:y1, x0:x1].copy()
