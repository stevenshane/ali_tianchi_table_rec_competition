"""图像修复：按质量路由执行动作。所有步骤返回新图，原图另存保留。"""
import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import config
from . import ocr
from .quality import QualityReport, to_gray

log = logging.getLogger(__name__)


def save(img: np.ndarray, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".png", img)[1].tofile(str(path))  # 兼容中文路径
    path = path.resolve()
    return str(path.relative_to(config.ROOT)) if path.is_relative_to(config.ROOT) else str(path)


# ---------------- 几何 ----------------
def rotate90k(img, k):
    return np.ascontiguousarray(np.rot90(img, k))


def deskew(img, angle):
    """反向旋转相同角度（angle 为逆时针为正的倾斜角）。"""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), -angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    border = (255, 255, 255) if img.ndim == 3 else 255
    return cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_CUBIC, borderValue=border)


def perspective(img, quad):
    pts = np.array(quad, dtype=np.float32)
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    tl, br = pts[np.argmin(s)], pts[np.argmax(s)]
    tr, bl = pts[np.argmin(d)], pts[np.argmax(d)]
    src = np.array([tl, tr, br, bl], dtype=np.float32)
    w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h), flags=cv2.INTER_CUBIC)


# ---------------- 分辨率 ----------------
def upscale(img, factor=2):
    """普通插值放大：Pillow Lanczos。"""
    pil = Image.fromarray(img[:, :, ::-1] if img.ndim == 3 else img)
    pil = pil.resize((pil.width * factor, pil.height * factor), Image.LANCZOS)
    arr = np.array(pil)
    return arr[:, :, ::-1].copy() if arr.ndim == 3 else arr


# ---------------- 光照 ----------------
def clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def contrast_stretch(img):
    gray = to_gray(img)
    p2, p98 = np.percentile(gray, [2, 98])
    if p98 - p2 < 1:
        return img
    out = (img.astype(np.float32) - p2) * (255.0 / (p98 - p2))
    return np.clip(out, 0, 255).astype(np.uint8)


def bg_normalize(img):
    """估计缓慢变化的背景光照（大核闭运算+模糊），再做除法归一化。"""
    k = max(15, (min(img.shape[:2]) // 30) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    out = []
    for ch in cv2.split(img):
        bg = cv2.morphologyEx(ch, cv2.MORPH_CLOSE, kernel)
        bg = cv2.GaussianBlur(bg, (k, k), 0)
        out.append(cv2.divide(ch, bg, scale=255))
    return cv2.merge(out)


# ---------------- 慎用项 ----------------
def light_denoise(img):
    return cv2.fastNlMeansDenoisingColored(img, None, 3, 3, 7, 21)


def binarize(img):
    """自适应阈值二值化，仅用于双路 OCR，不替换原图。"""
    gray = to_gray(img)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)


def numbers_preserved(before, after) -> bool:
    """关键数字前后对比：去噪后 OCR 出的数字集合不应少于去噪前。"""
    a = set(ocr.numeric_tokens(ocr.run(before)))
    b = set(ocr.numeric_tokens(ocr.run(after)))
    lost = a - b
    return len(lost) <= max(1, len(a) // 50)


@dataclass
class Processed:
    image: np.ndarray                 # 修复后的工作图
    geometric: np.ndarray             # 仅做几何/分辨率处理的图（无损信息，用于二次放大）
    original_path: str
    steps: list = field(default_factory=list)


def apply_route(img: np.ndarray, report: QualityReport, work_dir: Path, tag: str) -> Processed:
    """按路由执行动作。几何类先执行，再分辨率，再光照，最后去噪。"""
    orig_path = save(img, work_dir / f"{tag}_orig.png")
    steps = []
    cur = img
    order = ["perspective", "rotate90k", "deskew", "upscale", "bg_normalize", "clahe", "light_denoise"]
    actions = sorted(report.actions, key=lambda a: order.index(a[0]) if a[0] in order else 99)
    geometric = img
    for name, params in actions:
        try:
            if name == "perspective":
                cur = perspective(cur, params["quad"])
            elif name == "rotate90k":
                cur = rotate90k(cur, params["k"])
            elif name == "deskew":
                cur = deskew(cur, params["angle"])
            elif name == "upscale":
                cur = upscale(cur, params.get("factor", 2))
            elif name == "bg_normalize":
                cur = bg_normalize(cur)
            elif name == "clahe":
                cand = clahe(cur)
                # 对比度仍不足时再做拉伸
                cur = contrast_stretch(cand) if to_gray(cand).std() < 40 else cand
            elif name == "light_denoise":
                cand = light_denoise(cur)
                if not numbers_preserved(cur, cand):
                    steps.append({"step": name, "skipped": "关键数字前后不一致，保留去噪前图像"})
                    continue
                cur = cand
            else:
                continue
        except Exception as e:  # 单步失败不影响整体流程
            log.warning("预处理 %s 失败: %s", name, e)
            steps.append({"step": name, "error": str(e)})
            continue
        if name in ("perspective", "rotate90k", "deskew", "upscale"):
            geometric = cur
        steps.append({"step": name, "params": {k: v for k, v in params.items() if k != "quad"},
                      "image": save(cur, work_dir / f"{tag}_{len(steps) + 1}_{name}.png")})
    return Processed(image=cur, geometric=geometric, original_path=orig_path, steps=steps)
