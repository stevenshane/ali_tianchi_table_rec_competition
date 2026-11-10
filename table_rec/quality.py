"""质量检测与路由：计算图像指标、打分，并给出需要执行的修复动作列表。"""
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

import config
from . import ocr


@dataclass
class QualityReport:
    score: int = 100
    metrics: dict = field(default_factory=dict)
    issues: list = field(default_factory=list)
    actions: list = field(default_factory=list)   # [(name, params)]

    def to_dict(self):
        return asdict(self)


def to_gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def estimate_skew(gray: np.ndarray) -> float:
    """利用表格横线/文字基线的 Hough 直线估计平均倾斜角（度，逆时针为正）。"""
    h, w = gray.shape
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 1800, threshold=120,
                            minLineLength=max(80, w // 5), maxLineGap=10)
    angles = []
    if lines is not None:
        for x0, y0, x1, y1 in lines.reshape(-1, 4):
            ang = np.degrees(np.arctan2(y0 - y1, x1 - x0))  # 图像 y 轴向下，取反得到逆时针角
            if abs(ang) <= config.MAX_SKEW_DEG:
                angles.append(ang)
    if len(angles) >= 3:
        return float(np.median(angles))
    # 无足够直线时，用文字像素的最小外接矩形估计
    bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    bw = cv2.dilate(bw, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3)))
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        (_, _), (rw, rh), a = cv2.minAreaRect(c)
        if rw < 60 and rh < 60:
            continue
        if rw < rh:
            a -= 90
        a = -a
        if abs(a) <= config.MAX_SKEW_DEG:
            angles.append(a)
    return float(np.median(angles)) if angles else 0.0


def contrast_metrics(gray: np.ndarray) -> dict:
    p5, p95 = np.percentile(gray, [5, 95])
    t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    fg = gray[gray <= t]
    bg = gray[gray > t]
    gap = float(bg.mean() - fg.mean()) if fg.size and bg.size else 0.0
    return {"p5": float(p5), "p95": float(p95), "range": float(p95 - p5), "fg_bg_gap": gap}


def illumination_std(img: np.ndarray, grid: int = 8) -> float:
    """分块背景亮度（HSV 的 V 通道块内 90 分位）标准差，反映光照不均；用 V 通道避免彩色底纹误判。"""
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2]
    h, w = gray.shape
    vals = []
    for i in range(grid):
        for j in range(grid):
            blk = gray[i * h // grid:(i + 1) * h // grid, j * w // grid:(j + 1) * w // grid]
            if blk.size:
                vals.append(np.percentile(blk, 90))
    return float(np.std(vals)) if vals else 0.0


def noise_metrics(gray: np.ndarray) -> dict:
    bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    n, _, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    tiny = int((areas <= 4).sum())
    total = int(len(areas))
    return {"tiny": tiny, "total": total, "tiny_frac": tiny / total if total else 0.0}


def detect_document_quad(img: np.ndarray) -> tuple[np.ndarray | None, float]:
    """检测手机拍照中纸张四角；返回 (4x2 角点, 置信度)。"""
    gray = to_gray(img)
    h, w = gray.shape
    small_s = 1000 / max(h, w)
    small = cv2.resize(gray, (int(w * small_s), int(h * small_s))) if small_s < 1 else gray
    s = small_s if small_s < 1 else 1.0
    blur = cv2.GaussianBlur(small, (5, 5), 0)
    edges = cv2.dilate(cv2.Canny(blur, 50, 150), np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_img = small.shape[0] * small.shape[1]
    best, best_conf = None, 0.0
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:5]:
        area = cv2.contourArea(c)
        if not (0.3 * area_img < area < 0.97 * area_img):
            continue
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        fit = area / max(cv2.contourArea(approx), 1)
        mask = np.zeros_like(small)
        cv2.fillConvexPoly(mask, approx.reshape(-1, 2), 255)
        outside = small[mask == 0]
        inside = small[mask > 0]
        # 背景与纸面区分明显才认为是可靠的四角
        sep = min(1.0, abs(float(inside.mean()) - float(outside.mean())) / 60) if outside.size else 0.0
        conf = float(min(fit, 1.0) * sep)
        if conf > best_conf:
            best, best_conf = approx.reshape(4, 2).astype(np.float32) / s, conf
    return best, best_conf


def is_phone_photo(img: np.ndarray, is_pdf: bool) -> bool:
    """非 PDF，四周边框不是纯白、且不是高饱和的电子图形色块 -> 视为手机拍照。"""
    if is_pdf or img.ndim == 2:
        return False
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    b = max(5, min(h, w) // 40)
    ring = np.concatenate([hsv[:b].reshape(-1, 3), hsv[-b:].reshape(-1, 3),
                           hsv[:, :b].reshape(-1, 3), hsv[:, -b:].reshape(-1, 3)])
    v, sat = ring[:, 2].astype(float), ring[:, 1].astype(float)
    not_white = v.mean() < 200 or v.std() > 40
    graphic = np.median(sat) > 120          # 边框是饱和纯色 -> 电子表格截图
    return not_white and not graphic


def quad_tilt(quad: np.ndarray) -> float:
    """四边相对水平/竖直方向的最大偏角（度）；接近 0 说明无需透视矫正。"""
    q = np.asarray(quad, dtype=float)
    worst = 0.0
    for i in range(4):
        dx, dy = q[(i + 1) % 4] - q[i]
        ang = abs(np.degrees(np.arctan2(dy, dx))) % 90
        worst = max(worst, min(ang, 90 - ang))
    return worst


def assess(img: np.ndarray, is_pdf: bool = False, dpi: int | None = None) -> QualityReport:
    rep = QualityReport()
    m = rep.metrics
    h, w = img.shape[:2]
    m["size"] = [w, h]

    # 1) 手机拍照 + 透视（慎用：仅四角置信度高时）
    if is_phone_photo(img, is_pdf):
        quad, qconf = detect_document_quad(img)
        m["phone_photo"] = True
        m["quad_conf"] = round(qconf, 3)
        m["quad_tilt"] = round(quad_tilt(quad), 2) if quad is not None else None
        if quad is not None and qconf >= config.PERSPECTIVE_MIN_CONF and m["quad_tilt"] > config.SKEW_DEG:
            rep.issues.append("perspective")
            rep.actions.append(("perspective", {"quad": quad.tolist()}))

    # 2) 方向（默认执行）
    k, oconf, tokens, s = ocr.detect_orientation(img)
    m["orientation_k"] = k
    m["orientation_conf"] = round(oconf, 3)
    if k != 0 and oconf > 0.6:
        rep.issues.append(f"rotated_{k * 90}")
        rep.actions.append(("rotate90k", {"k": k}))
    work = np.ascontiguousarray(np.rot90(img, k)) if k else img
    gray = to_gray(work)

    # 3) 倾斜（默认执行）
    skew = estimate_skew(gray)
    m["skew_deg"] = round(skew, 2)
    if abs(skew) > config.SKEW_DEG:
        rep.issues.append("skew")
        rep.actions.append(("deskew", {"angle": skew}))

    # 4) 分辨率（默认执行）
    char_h = ocr.median_char_height(tokens, s)
    m["char_height"] = round(char_h, 1) if char_h else None
    short = min(work.shape[:2])
    m["short_edge"] = short
    small_text = char_h is not None and char_h < config.MIN_CHAR_H
    if is_pdf:
        if small_text and (dpi or 0) < config.PDF_DPI_SMALL_FONT:
            rep.issues.append("small_font")
            rep.actions.append(("rerender_dpi", {"dpi": config.PDF_DPI_SMALL_FONT}))
    elif short < config.MIN_SHORT_EDGE or small_text:
        rep.issues.append("low_resolution")
        rep.actions.append(("upscale", {"factor": config.UPSCALE_FACTOR}))

    # 5) 光照：仅由指标触发
    illum = illumination_std(work)
    m["illumination_std"] = round(illum, 2)
    if illum > config.UNEVEN_LIGHT:
        rep.issues.append("uneven_light")
        rep.actions.append(("bg_normalize", {}))
    cm = contrast_metrics(gray)
    m["contrast"] = {k2: round(v, 1) for k2, v in cm.items()}
    if cm["fg_bg_gap"] < config.LOW_CONTRAST:
        rep.issues.append("low_contrast")
        rep.actions.append(("clahe", {}))

    # 6) 噪声：轻度去噪，关键数字前后对比
    nm = noise_metrics(gray)
    m["noise"] = nm
    if nm["tiny_frac"] > config.NOISE_TINY_FRAC and nm["tiny"] > config.NOISE_MIN_COUNT:
        rep.issues.append("noise")
        rep.actions.append(("light_denoise", {}))

    penalty = {"perspective": 15, "skew": 8, "low_resolution": 15, "small_font": 10,
               "uneven_light": 10, "low_contrast": 12, "noise": 10}
    rep.score = max(0, 100 - sum(penalty.get(i, 10) for i in rep.issues))
    return rep
