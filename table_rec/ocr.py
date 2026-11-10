"""RapidOCR 封装：文本框、字符高度、方向判断、原图+二值图双路 OCR。"""
import re
import threading
from dataclasses import dataclass

import cv2
import numpy as np

_engine = None
_lock = threading.RLock()


def engine():
    global _engine
    with _lock:
        if _engine is None:
            from rapidocr_onnxruntime import RapidOCR
            _engine = RapidOCR()
        return _engine


@dataclass
class OcrToken:
    text: str
    score: float
    box: tuple[int, int, int, int]   # x0, y0, x1, y1

    @property
    def height(self) -> int:
        return self.box[3] - self.box[1]

    @property
    def width(self) -> int:
        return self.box[2] - self.box[0]


def run(img: np.ndarray, use_cls: bool = True) -> list[OcrToken]:
    result, _ = engine()(img, use_cls=use_cls)
    tokens = []
    for box, text, score in result or []:
        pts = np.array(box, dtype=float)
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        tokens.append(OcrToken(str(text), float(score), (int(x0), int(y0), int(x1), int(y1))))
    return tokens


def _downscale(img: np.ndarray, max_edge: int = 1600) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    s = min(1.0, max_edge / max(h, w))
    if s < 1.0:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img, s


def median_char_height(tokens: list[OcrToken], scale: float = 1.0) -> float | None:
    hs = [t.height for t in tokens if t.score > 0.5 and len(t.text) >= 2 and t.width > t.height]
    if not hs:
        return None
    return float(np.median(hs)) / scale


def _rot(img: np.ndarray, k: int) -> np.ndarray:
    """逆时针旋转 k*90°。"""
    return np.ascontiguousarray(np.rot90(img, k))


def _ocr_quality(tokens: list[OcrToken]) -> float:
    """只统计横向排布的文本框：RapidOCR 会自动旋转竖长框再识别，竖排框计入会让 90°/270° 无法区分。"""
    return sum(t.score * len(t.text) for t in tokens if t.score > 0.5 and len(t.text) >= 2 and t.width > t.height)


def detect_orientation(img: np.ndarray) -> tuple[int, float, list[OcrToken], float]:
    """返回 (逆时针旋转次数 k 使文字水平, 置信度, 缩略图OCR结果, 缩放比)。

    关闭方向分类器(use_cls=False)做 OCR：方向错误的文字得分会显著下降。
    先测 0°，若与开启分类器时的得分接近则直接判定为 0°；否则比较 4 个方向。
    """
    small, s = _downscale(img)
    ref = _ocr_quality(run(small, use_cls=True))
    tokens0 = run(small, use_cls=False)
    scores = {0: _ocr_quality(tokens0)}
    token_by_k = {0: tokens0}
    if ref and scores[0] / ref >= 0.8:
        return 0, min(1.0, scores[0] / ref), tokens0, s
    for k in (1, 2, 3):
        tk = run(_rot(small, k), use_cls=False)
        scores[k] = _ocr_quality(tk)
        token_by_k[k] = tk
    best = max(scores, key=scores.get)
    rest = sorted((v for k, v in scores.items() if k != best), reverse=True)
    conf = 1 - rest[0] / scores[best] if scores[best] else 0.0
    return best, conf, token_by_k[best], s


NUM_RE = re.compile(r"[-+]?\(?\d[\d,，]*(?:\.\d+)?%?\)?")


def numeric_tokens(tokens: list[OcrToken]) -> list[str]:
    out = []
    for t in tokens:
        for m in NUM_RE.findall(t.text):
            out.append(m.replace(",", "").replace("，", ""))
    return out


def dual_path(img: np.ndarray, binary: np.ndarray) -> list[OcrToken]:
    """原图 + 二值图双路 OCR：按框 IoU 合并，同位置取高分结果，另一路独有的框补入。"""
    a = run(img)
    b = run(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR) if binary.ndim == 2 else binary)
    merged = list(a)
    for tb in b:
        hit = None
        for i, ta in enumerate(merged):
            if iou(ta.box, tb.box) > 0.5:
                hit = i
                break
        if hit is None:
            merged.append(tb)
        elif tb.score > merged[hit].score + 0.05:
            merged[hit] = tb
    return merged


def iou(a, b) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    if inter == 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua else 0.0
