import fitz  # PyMuPDF
import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import stringWidth
from utils.translate import Translate

FONT_PATH = "NotoSans-Regular.ttf"
FONT_NAME = "NotoSans"

if not os.path.exists(FONT_PATH):
    raise FileNotFoundError(f"Font file not found: {FONT_PATH}")
pdfmetrics.registerFont(TTFont(FONT_NAME, FONT_PATH))


def extract_pdf_cells(pdf_path: str):
    doc = fitz.open(pdf_path)

    cells = []
    for page in doc:
        page_number = page.number
        page_dict = page.get_text("dict")
        for block in page_dict["blocks"]:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    bbox = span["bbox"]
                    text = span["text"]
                    font_size = span["size"]
                    font_name = span["font"]
                    col = span.get("color", 0)
                    r = (col >> 16) & 0xFF
                    g = (col >> 8) & 0xFF
                    b = col & 0xFF

                    cell = {
                        "page": page_number,
                        "bbox": [
                            round(bbox[0], 6),
                            round(bbox[1], 6),
                            round(bbox[2] - bbox[0], 6),
                            round(bbox[3] - bbox[1], 6),
                        ],
                        "text": text,
                        "font": {
                            "color": [r, g, b, 255],
                            "name": font_name,
                            "size": font_size,
                        },
                        "text_vi": None,
                    }

                    if text.strip():
                        try:
                            translator = Translate(text)
                            res_str = translator.translate(src_lang="en", tgt_lang="vi")
                            cell["text_vi"] = res_str
                        except Exception:
                            cell["text_vi"] = text
                    else:
                        cell["text_vi"] = text

                    cells.append(cell)

    return {"cells": cells}


def create_pdf_from_json(data, output_path):
    c = canvas.Canvas(output_path, pagesize=A4)
    pages = {}

    for cell in data["cells"]:
        page_num = cell["page"]
        pages.setdefault(page_num, []).append(cell)

    for page_num in sorted(pages.keys()):
        for cell in pages[page_num]:
            x, y, w, h = cell["bbox"]
            font_size = cell["font"]["size"]
            r, g, b, _ = cell["font"]["color"]
            text = cell["text_vi"]

            text_width = stringWidth(text, FONT_NAME, font_size)
            if text_width > w:
                font_size *= w / text_width
            if font_size > h:
                font_size = h

            c.setFillColorRGB(r / 255, g / 255, b / 255)
            c.setFont(FONT_NAME, font_size)
            adjusted_y = A4[1] - y
            c.drawString(x, adjusted_y, text)

        c.showPage()
    c.save()
