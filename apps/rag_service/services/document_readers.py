from __future__ import annotations
import io, os, pdfplumber, fitz, pytesseract
from typing import Dict, Optional

from PIL import Image
from pptx import Presentation
from docx import Document

from .text_normalizer import normalize_json, format_blocks, normalize_text


def read_pdf(path: str) -> Optional[str]:
    text_by_page: Dict[int, str] = {}
    images_for_ocr = []

    # 1. Try PyMuPDF
    try:
        doc = fitz.open(path)
        for idx, page in enumerate(doc):
            page_num = idx + 1
            txt = page.get_text().strip()
            text_by_page[page_num] = txt

            # OCR fallback for image-only pages
            if len(txt) < 50:
                for img in page.get_images():
                    pix = fitz.Pixmap(doc, img[0])
                    if pix.n - pix.alpha < 4:
                        images_for_ocr.append(
                            {"page": page_num, "data": pix.tobytes("png")}
                        )
        doc.close()
    except Exception as exc:
        print(f"[PDF] PyMuPDF extraction failed: {exc}")

    # OCR pass
    for img in images_for_ocr:
        try:
            pil_img = Image.open(io.BytesIO(img["data"]))
            ocr_txt = pytesseract.image_to_string(
                pil_img, config="--oem 3 --psm 6"
            ).strip()
            if ocr_txt:
                merged = f"{text_by_page.get(img['page'], '')}\n{ocr_txt}".strip()
                text_by_page[img["page"]] = merged
        except Exception as exc:
            print(f"[PDF] OCR failed (page {img['page']}): {exc}")

    # 2. pdfplumber (second opinion)
    try:
        with pdfplumber.open(path) as pdf:
            for idx, page in enumerate(pdf.pages):
                page_num = idx + 1
                extra = (page.extract_text() or "").strip()
                if extra and extra not in text_by_page.get(page_num, ""):
                    text_by_page[page_num] = (
                        f"{text_by_page.get(page_num, '')}\n{extra}".strip()
                    )
    except Exception as exc:
        print(f"[PDF] pdfplumber extraction failed: {exc}")

    if text_by_page:
        full = "\n".join(
            text_by_page[p] for p in sorted(text_by_page) if text_by_page[p]
        )
        return normalize_text(full) if full else None
    return None


def read_txt(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return normalize_text(f.read())
    except Exception:
        return None


def read_json(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            cleaned_json = normalize_json(f.read())
            return format_blocks(cleaned_json, show_list_index=False)
    except Exception:
        return None

def read_pptx(path: str) -> Optional[str]:
    try:
        prs = Presentation(path)
        text = []
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text.append(shape.text)
        return "\n".join(text)
    except Exception:
        return None


def read_docx(path: str) -> Optional[str]:
    try:
        doc = Document(path)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception:
        return None


# def read_doc(path: str) -> Optional[str]:
#     try:
#         return textract.process(path).decode("utf-8", errors="ignore")
#     except Exception:
#         return None


# ---------- helpers --------- #

def normalize_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\x00", "").replace("\xa0", " ")
    s = s.encode("utf-8", "ignore").decode("utf-8")
    return "\n".join([line.strip() for line in s.splitlines() if line.strip()])