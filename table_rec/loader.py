"""文件加载：图片 -> BGR 数组；PDF 按批（每批最多 20 页）渲染并判断文字层。"""
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import pymupdf as fitz
import numpy as np
from PIL import Image, ImageOps

import config

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass
class Page:
    file_name: str
    page_no: int                 # 从 1 开始
    image: np.ndarray            # BGR
    dpi: int | None = None       # PDF 渲染 DPI；位图为 None
    has_text_layer: bool = False
    is_pdf: bool = False
    meta: dict = field(default_factory=dict)


def load_image(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            im = Image.alpha_composite(bg, im)
        im = im.convert("RGB")
        return np.array(im)[:, :, ::-1].copy()


def render_pdf_page(doc: fitz.Document, index: int, dpi: int) -> np.ndarray:
    pix = doc[index].get_pixmap(dpi=dpi, alpha=False)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    return arr[:, :, :3][:, :, ::-1].copy()


_NORMAL_CHAR = re.compile(r"[一-鿿　-〿＀-￯A-Za-z0-9\s.,;:%()\-+/]")


def text_layer_ok(text: str) -> bool:
    """文字层可用：字符数足够，且不是字体编码错乱产生的乱码（正常字符占比要高）。"""
    text = text.strip()
    if len(text) < config.PDF_TEXT_MIN_CHARS:
        return False
    return len(_NORMAL_CHAR.findall(text)) / len(text) >= config.PDF_TEXT_MIN_NORMAL_RATIO


def pdf_has_text_layer(page: fitz.Page) -> bool:
    return text_layer_ok(page.get_text("text"))


def iter_pdf_batches(path: Path, dpi: int = config.PDF_DPI) -> Iterator[list[Page]]:
    """按批产出页面，每批最多 PDF_PAGE_BATCH 页。"""
    with fitz.open(path) as doc:
        n = doc.page_count
        for start in range(0, n, config.PDF_PAGE_BATCH):
            batch = []
            for i in range(start, min(start + config.PDF_PAGE_BATCH, n)):
                batch.append(Page(
                    file_name=path.name, page_no=i + 1,
                    image=render_pdf_page(doc, i, dpi), dpi=dpi,
                    has_text_layer=pdf_has_text_layer(doc[i]), is_pdf=True,
                ))
            yield batch


def rerender(path: Path, page_no: int, dpi: int) -> np.ndarray:
    with fitz.open(path) as doc:
        return render_pdf_page(doc, page_no - 1, dpi)


def iter_pages(path: Path) -> Iterator[list[Page]]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        yield from iter_pdf_batches(path)
    elif ext in IMAGE_EXTS:
        yield [Page(file_name=path.name, page_no=1, image=load_image(path))]
    else:
        raise ValueError(f"不支持的文件类型: {path}")
