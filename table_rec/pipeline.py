"""识别主流程：加载 -> 质量路由 -> 修复 -> 分区 -> 表格识别 -> 校验/置信度 -> 二次识别 -> 存储。"""
import logging
import re
from pathlib import Path

import config
from . import (cell_validator, confidence, layout, loader, ocr, pdf_text, preprocess, quality,
               refine, store, table_model, table_recognizer)

log = logging.getLogger(__name__)

CONT_TITLE = re.compile(r"^\s*(续|接上[页表])")


def guess_language(text: str) -> str | None:
    cjk = len(re.findall(r"[一-鿿]", text or ""))
    latin = len(re.findall(r"[A-Za-z]", text or ""))
    if not cjk and not latin:
        return None
    if cjk and latin / max(cjk, 1) > 1.5 and cjk > 20:
        return "中英文"
    return "中文" if cjk >= latin * 0.3 else "英文"


def _finish(t: dict, ocr_text: str | None, geometric_crop=None) -> dict:
    """校验 + 置信度 + 低置信度二次识别。"""
    issues = table_model.structure_issues(t)
    if issues and all("空洞" in s for s in issues):
        table_model.fill_holes(t)
    t["validation"] = cell_validator.validate(t)
    t["confidence"] = confidence.score(t, t["validation"], ocr_text)
    if t["confidence"] < config.CONF_THRESHOLD and geometric_crop is not None:
        t = refine.refine(t, geometric_crop, ocr_text)
    return t


# ---------------- 有文字层的 PDF 页 ----------------
def _text_layer_page(path: Path, page: loader.Page) -> tuple[list[dict], dict]:
    plumber_tables, ctx = pdf_text.extract_page(path, page.page_no)
    page_text = ctx["body_text"]
    head_lines = ctx["header_footer"]
    context = {"page_title": None, "language": guess_language(page_text), "body_text": page_text[:6000],
               "header_footer": head_lines, "figures": [], "layout_source": "pdf_text_layer"}
    good = []
    for i, t in enumerate(plumber_tables):
        # 标题实为页眉 -> 置空；“续：”或页顶无标题的第一张表 -> 续表候选（合并时还要求列数一致）
        if t["title"] in head_lines:
            t["title"] = None
        is_cont_title = bool(t["title"] and CONT_TITLE.match(t["title"]))
        t["continues_from_prev"] = is_cont_title or (i == 0 and t["title"] is None and t["bbox_rel"][1] < 0.25)
        t.update(recognized_by="pdf_text", self_confidence=95)
        t = _finish(t, page_text)
        good.append(t)
    if ctx["issues"]:
        log.info("%s p%d 文字层表格抽取不可靠，转视觉识别: %s", page.file_name, page.page_no, ctx["issues"])
    elif not good:
        return [], context  # 页面无表格
    elif all(t["confidence"] >= 80 and not t["validation"]["structure_issues"] for t in good):
        return good, context
    # 无线框表格 / 抽取质量差：用视觉模型 + 精确文字层提示识别
    try:
        vlm = table_recognizer.recognize(page.image, hint_text=page_text, hint_kind="PDF文字层（精确）")
    except Exception as e:
        log.warning("%s p%d VLM 识别失败: %s", page.file_name, page.page_no, e)
        return good, context
    vlm = [_finish(t, page_text, page.image) for t in vlm]
    if ctx["issues"]:  # 闸门已判定文字层抽取不可靠，其高置信度不可信
        return (vlm or good), context
    best = vlm if sum(t["confidence"] for t in vlm) / max(1, len(vlm)) >= \
        sum(t["confidence"] for t in good) / max(1, len(good)) else good
    return best, context


# ---------------- 位图 / 扫描页 ----------------
def _image_page(path: Path, page: loader.Page) -> tuple[list[dict], dict]:
    work_dir = config.WORK_DIR / page.file_name
    tag = f"p{page.page_no}"
    rep = quality.assess(page.image, is_pdf=page.is_pdf, dpi=page.dpi)
    for name, params in list(rep.actions):
        if name == "rerender_dpi":  # PDF 小字体：重新渲染更高 DPI，而非插值放大
            page.image = loader.rerender(path, page.page_no, params["dpi"])
            page.dpi = params["dpi"]
            rep.actions.remove((name, params))
    proc = preprocess.apply_route(page.image, rep, work_dir, tag)
    tokens = ocr.run(proc.image)
    lay = layout.analyze(proc.image, tokens)
    context = lay.context()
    if not context.get("language"):
        context["language"] = guess_language(" ".join(t.text for t in tokens))
    tables = []
    for region in lay.tables:
        crop = layout.crop(proc.image, region.box)
        geo = layout.crop(proc.geometric, region.box) if proc.geometric.shape == proc.image.shape else crop
        crop_tokens = ocr.dual_path(crop, preprocess.binarize(crop))
        ocr_text = layout._lines_text(crop_tokens)
        enhanced = any(st["step"] in ("bg_normalize", "clahe", "light_denoise") and "image" in st for st in proc.steps)
        try:
            found = table_recognizer.recognize(geo, hint_text=ocr_text, hint_kind="OCR", caption=region.caption,
                                               enhanced=crop if enhanced else None)
        except Exception as e:
            log.warning("%s p%d 表格识别失败: %s", page.file_name, page.page_no, e)
            continue
        for t in found:
            t["bbox"] = list(region.box)
            t["title"] = t["title"] or region.caption
            tables.append(_finish(t, ocr_text, geo))
    for t in tables:
        t["quality"] = rep.to_dict()
        t["preprocess"] = {"original": proc.original_path, "steps": proc.steps}
    return tables, context


def recognize_file(path: Path, force: bool = False) -> list[str]:
    name = path.name
    if not force and store.has(name):
        log.info("复用已有识别结果: %s", name)
        return [t["json"] for t in store.file_entry(name)["tables"]]
    log.info("开始识别: %s", name)
    all_tables, contexts = [], []
    for batch in loader.iter_pages(path):
        for page in batch:
            if page.has_text_layer:
                tables, ctx = _text_layer_page(path, page)
            else:
                tables, ctx = _image_page(path, page)
            for t in tables:
                t["source_file"] = name
                t["pages"] = t.get("pages") or [page.page_no]
                t["context"] = ctx
            contexts.append({"page": page.page_no, **ctx})
            all_tables.extend(tables)
    merged = table_model.merge_cross_page(all_tables)
    for t in merged:  # 合并后重新校验
        t["validation"] = cell_validator.validate(t)
    names = store.save(name, merged, contexts)
    for t, n in zip(merged, names):
        if t["confidence"] < config.CONF_THRESHOLD:
            refine.record_low_confidence(t, n)
    log.info("完成识别: %s -> %s", name, names)
    return names
