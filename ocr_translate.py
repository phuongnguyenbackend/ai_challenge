import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import cv2
import numpy as np
import pytesseract
from basenet.vgg16_bn import vgg16_bn, init_weights
import imgproc
import craft_utils
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import google.generativeai as genai
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv
load_dotenv()   # this reads .env from your CWD
import os

class double_conv(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch+mid_ch, mid_ch, kernel_size=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class CRAFT(nn.Module):
    def __init__(self):
        super(CRAFT, self).__init__()
        self.basenet = vgg16_bn(pretrained=False, freeze=False)
        self.upconv1 = double_conv(1024, 512, 256)
        self.upconv2 = double_conv(512, 256, 128)
        self.upconv3 = double_conv(256, 128, 64)
        self.upconv4 = double_conv(128, 64, 32)
        self.conv_cls = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, kernel_size=1)
        )
        init_weights(self.upconv1.modules())
        init_weights(self.upconv2.modules())
        init_weights(self.upconv3.modules())
        init_weights(self.upconv4.modules())
        init_weights(self.conv_cls.modules())
    def forward(self, x):
        sources = self.basenet(x)
        y = torch.cat([sources[0], sources[1]], dim=1)
        y = self.upconv1(y)
        y = F.interpolate(y, size=sources[2].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[2]], dim=1)
        y = self.upconv2(y)
        y = F.interpolate(y, size=sources[3].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[3]], dim=1)
        y = self.upconv3(y)
        y = F.interpolate(y, size=sources[4].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[4]], dim=1)
        feature = self.upconv4(y)
        y = self.conv_cls(feature)
        return y.permute(0,2,3,1), feature

def load_weights(model, weight_path):
    state_dict = torch.load(weight_path, map_location='cpu')
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

def get_text_color(roi):
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg_color = np.mean(roi[binary == 0], axis=0)
    return fg_color

def noname_for_this_function(words, y_thresh=0.4, x_thres=5.7):
    words = [w for w in words if not (w["h"] > 1.5 * w["w"])]
    y_centers = np.array([ (w["y"] + w["y"] + w["h"]) / 2 for w in words ]).reshape(-1, 1)
    h = np.array([w["h"] for w in words])
    med_h = np.median(h)
    actual_y_thresh = med_h * y_thresh

    if len(words) > 1:
        y_clust = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=actual_y_thresh,
            linkage="single",
            compute_full_tree=True
        )
        y_labels = y_clust.fit_predict(y_centers)
    else:
        y_labels = np.array([0])

    rows = {}
    for w, lab in zip(words, y_labels):
        rows.setdefault(lab, []).append(w)

    row_order = sorted(
        rows.keys(),
        key=lambda l: np.mean([ (w["y"] + w["y"] + w["h"]) / 2 for w in rows[l] ])
    )

    sentences = []
    for lab in row_order:
        row = sorted(rows[lab], key=lambda w: w["x"])
        if len(row) == 1:
            sentences.append({"words": row, "text": row[0]["text"]})
            continue

        gaps = [curr["x"] - (prev["x"] + prev["w"])
                for prev, curr in zip(row, row[1:])]
        positive_gaps = [g for g in gaps if g > 0]
        median_gap = np.median(positive_gaps) if positive_gaps else np.median(gaps)
        mean_width = np.mean([w["w"] for w in row[:-1]])
        threshold_x = max(min(median_gap * x_thres, mean_width * 2), 1)

        segment = [row[0]]
        for prev, curr, gap in zip(row, row[1:], gaps):
            if gap > threshold_x:
                sentences.append({"words": segment, "text": " ".join(word["text"] for word in segment)})
                segment = [curr]
            else:
                segment.append(curr)
        sentences.append({"words": segment, "text": " ".join(word["text"] for word in segment)})

    return sentences

if __name__=='__main__':
    model = CRAFT()
    load_weights(model, "craft_mlt_25k.pth")
    model.cuda()
    model.eval()
    image = cv2.imread("image.png")
    if image is None:
        raise FileNotFoundError("Could not load image.png. Please ensure the file exists.")
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, 1280, interpolation=cv2.INTER_LINEAR, mag_ratio=1.5)
    ratio_h = ratio_w = 1/target_ratio
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2,0,1).unsqueeze(0).cuda()
    with torch.no_grad():
        output, _ = model(x)
    score_text = output[0,:,:,0].cpu().data.numpy()
    score_link = output[0,:,:,1].cpu().data.numpy()
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, 0.7, 0.4, 0.4, False)
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    words = []
    for box in boxes:
        x_min, y_min = int(min(box, key=lambda p: p[0])[0]), int(min(box, key=lambda p: p[1])[1])
        x_max, y_max = int(max(box, key=lambda p: p[0])[0]), int(max(box, key=lambda p: p[1])[1])
        roi = image[y_min:y_max, x_min:x_max]
        # Use Tesseract with hocr output to attempt font detection
        custom_config = r'--oem 1 --psm 6 -c tessedit_create_hocr=1'
        hocr_data = pytesseract.image_to_pdf_or_hocr(roi, extension='hocr', config=custom_config)
        text = pytesseract.image_to_string(roi, config="--psm 6").strip()
        if text:
            color = get_text_color(roi)
            # Parse hocr to infer font (simplified approximation)
            font_name = "arial"  # Default fallback
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(hocr_data, 'html.parser')
                for span in soup.find_all('span', class_='ocr_word'):
                    if 'style' in span.attrs:
                        style = span['style']
                        if 'font-family' in style:
                            font_name = style.split('font-family:')[1].split(';')[0].strip().lower()
                            # Map to available fonts (simplified)
                            font_map = {
                                'times': 'times.ttf',
                                'arial': 'arial.ttf',
                                'helvetica': 'helvetica.ttf',
                                'courier': 'courier.ttf'
                            }
                            font_name = font_map.get(font_name, 'arial.ttf')
                            break
            except Exception as e:
                print(f"Font detection failed: {e}. Using default font (arial.ttf).")
            words.append({"text": text, "x": x_min, "y": y_min, "w": x_max-x_min, "h": y_max-y_min, "color": color, "font": font_name})
    sentences = noname_for_this_function(words)

    # Calculate overall bounding box for each sentence
    for sentence in sentences:
        if sentence["words"]:
            min_x = min(word["x"] for word in sentence["words"])
            min_y = min(word["y"] for word in sentence["words"])
            max_x = max(word["x"] + word["w"] for word in sentence["words"])
            max_y = max(word["y"] + word["h"] for word in sentence["words"])
            sentence["bbox"] = (min_x, min_y, max_x, max_y)

    # Collect sentence texts and translate using Gemini API
    sentence_texts = [sentence["text"] for sentence in sentences]
    full_text = "\n".join(sentence_texts)

    # Set up Gemini API
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Please set the GEMINI_API_KEY environment variable.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')

    # Create translation prompt
    prompt = (
        "You are a professional translator. "
        "First, carefully read the entire English text below—each line is separated by newline characters—to fully grasp its context, style, and terminology. "
        "Then translate it into Vietnamese, producing exactly one Vietnamese line for each English line, and preserve the original line order without skipping, merging, or reordering any lines. "
        "Return only the translated lines in the exact same order they appeared, separated by newlines, with no additional commentary:\n\n"
        f"{full_text}"
    )

    # Translate with error handling
    try:
        response = model.generate_content(prompt)
        translated_text = response.text
        translated_sentences = translated_text.split("\n")
        if len(translated_sentences) != len(sentence_texts):
            print("Translation mismatch. Using original texts as fallback.")
            translated_sentences = sentence_texts.copy()
    except Exception as e:
        print(f"Translation failed: {e}. Using original texts as fallback.")
        translated_sentences = sentence_texts.copy()

    # Create mask for all word bounding boxes
    all_word_bboxes = []
    for sentence in sentences:
        for word in sentence["words"]:
            x, y, w, h = word["x"], word["y"], word["w"], word["h"]
            all_word_bboxes.append(((x, y), (x + w, y + h)))

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for (x1, y1), (x2, y2) in all_word_bboxes:
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    # Inpaint to remove original text
    inpainted_image = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)

    # Convert to PIL Image and overlay translated text
    pil_image = Image.fromarray(cv2.cvtColor(inpainted_image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    # Load font with fallback and apply first word's font
    font_base_path = os.path.dirname(__file__)  # Assume fonts are in the same directory
    font_files = {
        'times.ttf': 'times.ttf',
        'arial.ttf': 'arial.ttf',
        'helvetica.ttf': 'helvetica.ttf',
        'courier.ttf': 'courier.ttf'
    }

    for i, sentence in enumerate(sentences):
        if "bbox" in sentence and sentence["words"]:
            min_x, min_y, max_x, max_y = sentence["bbox"]
            translated_sentence = translated_sentences[i]

            # Calculate bounding box dimensions
            bbox_width = max_x - min_x
            bbox_height = max_y - min_y

            # Start with font size based on 80% of bounding box height
            font_size = int(bbox_height * 0.8)
            font_size = max(font_size, 10)  # Minimum font size

            # Use the font of the first word
            first_word = sentence["words"][0]
            font_file = first_word.get("font", "arial.ttf")
            font_path = os.path.join(font_base_path, font_file)

            # Load font with the initial size
            try:
                font = ImageFont.truetype(font_path, font_size)
            except Exception as e:
                print(f"Failed to load font {font_file}: {e}. Using arial.ttf as fallback.")
                font = ImageFont.truetype(os.path.join(font_base_path, 'arial.ttf'), font_size)

            # Measure text dimensions and adjust font size to fit both width and height
            text_bbox = draw.textbbox((0, 0), translated_sentence, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            # Reduce font size until text fits within the bounding box
            while (text_width > bbox_width or text_height > bbox_height) and font_size > 10:
                font_size -= 1
                try:
                    font = ImageFont.truetype(font_path, font_size)
                except:
                    font = ImageFont.truetype(os.path.join(font_base_path, 'arial.ttf'), font_size)
                text_bbox = draw.textbbox((0, 0), translated_sentence, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]

            # Set text position to left margin (min_x)
            text_x = min_x
            text_y = min_y + (bbox_height - text_height) / 2  # Keep vertical centering for balance

            # Use the color of the first word for the translated text
            first_word_color = first_word["color"]
            text_color = tuple(int(c) for c in first_word_color)  # Convert to tuple of integers for PIL

            # Draw the translated text
            draw.text((text_x, text_y), translated_sentence, font=font, fill=text_color)

    # Save the translated image
    pil_image.save("translated_image.png")

    # Save original output
    img_out = image.copy()
    with open("xxyy.txt", "w", encoding="utf-8") as f:
        for sentence in sentences:
            text = sentence["text"]
            min_x, min_y, max_x, max_y = sentence["bbox"]
            f.write(f"{min_x},{min_y},{max_x},{max_y}: {text}\n")
            cv2.rectangle(img_out, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
    cv2.imwrite("result.png", img_out)

    # Save translated text to file
    with open("translated_xxyy.txt", "w", encoding="utf-8") as f:
        for i, sentence in enumerate(sentences):
            if "bbox" in sentence:
                min_x, min_y, max_x, max_y = sentence["bbox"]
                translated_sentence = translated_sentences[i]
                f.write(f"{min_x},{min_y},{max_x},{max_y}: {translated_sentence}\n")

    print("Translation complete. Output saved as 'translated_image.png' and 'translated_xxyy.txt'.")