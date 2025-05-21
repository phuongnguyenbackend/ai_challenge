import fitz  # PyMuPDF
import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import stringWidth
from google import genai
import logging
from PIL import Image, ImageDraw
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

FONT_PATH = "NotoSans-Regular.ttf"
FONT_NAME = "NotoSans"

if not os.path.exists(FONT_PATH):
    raise FileNotFoundError(f"Font file not found: {FONT_PATH}")
pdfmetrics.registerFont(TTFont(FONT_NAME, FONT_PATH))

# Initialize the Gemini client once (developer API)
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

def translate_with_gemini_batched(text: str) -> str:
    prompt = (
        "You are a professional translator. "
        "First, carefully read the entire English text below—each line is separated by newline characters—to fully grasp its context, style, and terminology. "
        "Then translate it into Vietnamese, producing exactly one Vietnamese line for each English line, and preserve the original line order without skipping, merging, or reordering any lines. "
        "Return only the translated lines in the exact same order they appeared, separated by newlines, with no additional commentary:\n\n"
        f"{text}"
    )
    response = client.models.generate_content(
        model="gemini-2.0-flash",  # or whichever Gemini model you prefer
        contents=prompt
    )
    return response.text  # the full newline-delimited translation


def extract_pdf_cells(pdf_path: str, translate: bool = False):
    doc = fitz.open(pdf_path)
    cells = []

    # 1) Extract every text span (no per-span translation here)
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    x0, y0, x1, y1 = span["bbox"]
                    col = span.get("color", 0)
                    r = (col >> 16) & 0xFF
                    g = (col >> 8) & 0xFF
                    b = col & 0xFF

                    cells.append({
                        "page": page.number,
                        "bbox": [
                            round(x0, 6),
                            round(y0, 6),
                            round(x1 - x0, 6),
                            round(y1 - y0, 6),
                        ],
                        "text": span["text"],
                        "font": {
                            "color": [r, g, b, 255],
                            "name": span["font"],
                            "size": span["size"],
                        },
                        "text_vi": None,   # to fill in next
                    })
    doc.close()

    # 2) Batch-translate all lines in one shot
    if translate and cells:
        english_lines = [cell["text"] for cell in cells]
        batch_text = "\n".join(english_lines)

        try:
            translated = translate_with_gemini_batched(batch_text)
            vi_lines = translated.splitlines()
        except Exception as e:
            logging.error(f"Gemini translation failed: {e}")
            vi_lines = english_lines

        # 3) Map each translated line back into its cell
        for cell, vi in zip(cells, vi_lines):
            cell["text_vi"] = vi

    else:
        for cell in cells:
            cell["text_vi"] = cell["text"]

    return {"cells": cells}


def mask_text_in_image(image_path, cells, page_width, page_height, zoom=2, border_thickness=5):
    """Mask text areas in the PNG using the average color of a border around each bbox."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    
    for cell in cells:
        x, y, w, h = cell["bbox"]
        # Integer pixel coordinates
        left = int(x * zoom)
        upper = int(y * zoom)
        right = int((x + w) * zoom)
        lower = int((y + h) * zoom)

        # Avoid empty or invalid regions
        if right <= left or lower <= upper:
            continue

        # Define a border around the text box (e.g., 5 pixels wide)
        border_left = max(left - border_thickness, 0)
        border_upper = max(upper - border_thickness, 0)
        border_right = min(right + border_thickness, img.width)
        border_lower = min(lower + border_thickness, img.height)

        # Create regions for the border (top, bottom, left, right strips)
        regions = []
        if border_upper < upper:  # Top border
            regions.append(img.crop((border_left, border_upper, border_right, upper)))
        if lower < border_lower:  # Bottom border
            regions.append(img.crop((border_left, lower, border_right, border_lower)))
        if border_left < left:  # Left border
            regions.append(img.crop((border_left, border_upper, left, border_lower)))
        if right < border_right:  # Right border
            regions.append(img.crop((right, border_upper, border_right, border_lower)))

        # Combine pixel data from border regions
        pixels = []
        for region in regions:
            for x in range(region.width):
                for y in range(region.height):
                    pixels.append(region.getpixel((x, y)))

        # Calculate average color of the border
        if pixels:
            avg_color = tuple(
                int(sum(channel) / len(pixels)) for channel in zip(*pixels)
            )
        else:
            # Fallback to a default color if no border pixels are available
            avg_color = (255, 255, 255)  # White as default

        # Fill the text box with the average border color
        draw.rectangle([left, upper, right, lower], fill=avg_color)

    # Save masked image
    masked_image_path = image_path.replace(".png", "_masked.png")
    img.save(masked_image_path)
    return masked_image_path

def extract_background_images(pdf_path, temp_dir, cells_by_page, zoom=2):
    """Extract each page as a PNG and mask text areas."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Original PDF not found: {pdf_path}")
    
    os.makedirs(temp_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    image_paths = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))  # High resolution
        image_path = os.path.join(temp_dir, f"page_{page_num}.png")
        pix.save(image_path)
        
        # Mask text for this page
        page_cells = cells_by_page.get(page_num, [])
        if page_cells:
            masked_image_path = mask_text_in_image(
                image_path, page_cells, page.rect.width, page.rect.height, zoom=zoom
            )
        else:
            masked_image_path = image_path
        
        image_paths.append((masked_image_path, page.rect.width, page.rect.height))
        if masked_image_path != image_path:
            os.remove(image_path)  # Remove unmasked image
    
    doc.close()
    return image_paths

def create_pdf_from_json(data, pdf_path, output_path, temp_dir):
    # Group cells by page for masking and rendering
    cells_by_page = {}
    for cell in data.get("cells", []):
        if not all(key in cell for key in ["page", "bbox", "font", "text_vi"]):
            logging.warning(f"Skipping invalid cell: {cell}")
            continue
        page_num = cell["page"]
        cells_by_page.setdefault(page_num, []).append(cell)

    # Extract and mask background images
    try:
        image_paths = extract_background_images(pdf_path, temp_dir, cells_by_page)
    except Exception as e:
        logging.error(f"Failed to extract background images: {str(e)}")
        raise

    # Initialize canvas with first page size
    doc = fitz.open(pdf_path)
    first_page_size = (doc[0].rect.width, doc[0].rect.height)
    c = canvas.Canvas(output_path, pagesize=first_page_size)
    doc.close()

    # Render pages
    for page_num in tqdm(sorted(cells_by_page.keys()), desc="Rendering pages"):
        if page_num >= len(image_paths):
            logging.warning(f"Skipping page {page_num}: No background image available")
            continue

        image_path, page_width, page_height = image_paths[page_num]
        # Draw background image, scaled to page size
        c.drawImage(image_path, 0, 0, width=page_width, height=page_height, preserveAspectRatio=True)

        for cell in cells_by_page.get(page_num, []):
            x, y, w, h = cell["bbox"]
            text = cell.get("text_vi", "") or cell.get("text", "")
            if not text:
                continue
            original_font_size = cell["font"]["size"]
            r, g, b, a = cell["font"]["color"]

            # Adjust font size
            font_size = original_font_size
            text_width = stringWidth(text, FONT_NAME, font_size)
            if text_width > w:
                font_size *= 0.95 * w / text_width  # 5% margin
            if font_size > h:
                font_size = h

            # Set font and color
            c.setFillColorRGB(r / 255, g / 255, b / 255, alpha=a / 255)
            c.setFont(FONT_NAME, font_size)

            # Adjust y-coordinate (PyMuPDF: top-left, ReportLab: bottom-left)
            adjusted_y = page_height - (y + h)
            try:
                c.drawString(x, adjusted_y, text)
            except Exception as e:
                logging.warning(f"Failed to render text '{text}' at ({x}, {adjusted_y}): {str(e)}")

        c.showPage()

    c.save()
    logging.info(f"✅ PDF generated: {output_path}")

    # Cleanup temporary images
    try:
        for image_path, _, _ in image_paths:
            if os.path.exists(image_path):
                os.remove(image_path)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)
        logging.info(f"Cleaned up temporary directory: {temp_dir}")
    except Exception as e:
        logging.warning(f"Failed to clean up temporary files: {str(e)}")
