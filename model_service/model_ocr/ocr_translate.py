
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import cv2
import numpy as np
import pytesseract
from .basenet.vgg16_bn import vgg16_bn, init_weights
from model_ocr import imgproc
from model_ocr import craft_utils
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import google.generativeai as genai
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv
load_dotenv()   # this reads .env from your CWD
import os
import sys
import langid
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def detect_language(text):
    lang, _ = langid.classify(text)
    return lang

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


def load_craft_model(weights_path: str):
    model = CRAFT()
    load_weights(model, weights_path)
    model.cuda()
    model.eval()
    return model

def detect_text(model, image):
    img_resized, target_ratio, _ = imgproc.resize_aspect_ratio(image, 1280, interpolation=cv2.INTER_LINEAR, mag_ratio=1.5)
    ratio_h = ratio_w = 1 / target_ratio
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).cuda()
    
    with torch.no_grad():
        output, _ = model(x)
    
    score_text = output[0, :, :, 0].cpu().data.numpy()
    score_link = output[0, :, :, 1].cpu().data.numpy()
    boxes, _ = craft_utils.getDetBoxes(score_text, score_link, 0.7, 0.4, 0.4, False)
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    
    words = []
    for box in boxes:
        x_min = int(min(box, key=lambda p: p[0])[0])
        y_min = int(min(box, key=lambda p: p[1])[1])
        x_max = int(max(box, key=lambda p: p[0])[0])
        y_max = int(max(box, key=lambda p: p[1])[1])
        roi = image[y_min:y_max, x_min:x_max]
        text = pytesseract.image_to_string(roi, config="--psm 6").strip()
        if text:
            color = get_text_color(roi)
            words.append({"text": text, "x": x_min, "y": y_min, "w": x_max - x_min, "h": y_max - y_min, "color": color})
    return words


def translate_with_gemini(sentences,tgt_lang):
    sentence_texts = [s["text"] for s in sentences]
    full_text = "\n".join(sentence_texts)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Please set the GEMINI_API_KEY environment variable.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')
    src_lang = detect_language(full_text)
    prompt = (
        "You are a professional translator. "
        f"First, carefully read the entire {src_lang} text below—each line is separated by newline characters—to fully grasp its context, style, and terminology. "
        f"Then translate it into {tgt_lang}, producing exactly one {tgt_lang} line for each {src_lang} line, and preserve the original line order without skipping, merging, or reordering any lines. "
        "Return only the translated lines in the exact same order they appeared, separated by newlines, with no additional commentary:\n\n"
        f"{full_text}"
    )
    response = model.generate_content(prompt)
    return response.text.split("\n")

def calculate_sentence_bboxes(sentences):
    for s in sentences:
        if s["words"]:
            min_x = min(w["x"] for w in s["words"])
            min_y = min(w["y"] for w in s["words"])
            max_x = max(w["x"] + w["w"] for w in s["words"])
            max_y = max(w["y"] + w["h"] for w in s["words"])
            s["bbox"] = (min_x, min_y, max_x, max_y)
    return sentences

def inpaint_text(image, sentences):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for s in sentences:
        for w in s["words"]:
            x1, y1, x2, y2 = w["x"], w["y"], w["x"] + w["w"], w["y"] + w["h"]
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)

def draw_translated_text(pil_img, sentences, translated_sentences, font_path="arial.ttf"):
    draw = ImageDraw.Draw(pil_img)
    for i, s in enumerate(sentences):
        if "bbox" in s:
            min_x, min_y, max_x, max_y = s["bbox"]
            translated = translated_sentences[i]
            bbox_w, bbox_h = max_x - min_x, max_y - min_y
            font_size = max(int(bbox_h * 0.8), 10)
            try:
                font = ImageFont.truetype(font_path, font_size)
            except:
                font = ImageFont.load_default()
            while True:
                bbox = draw.textbbox((0, 0), translated, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                if (text_w <= bbox_w and text_h <= bbox_h) or font_size <= 10:
                    break
                font_size -= 1
                try:
                    font = ImageFont.truetype(font_path, font_size)
                except:
                    font = ImageFont.load_default()
            text_x = min_x
            text_y = min_y + (bbox_h - text_h) / 2
            color = tuple(int(c) for c in s["words"][0]["color"])
            draw.text((text_x, text_y), translated, font=font, fill=color)

def save_debug_files(image, sentences, translated_sentences):
    with open("xxyy.txt", "w", encoding="utf-8") as f:
        for s in sentences:
            min_x, min_y, max_x, max_y = s["bbox"]
            f.write(f"{min_x},{min_y},{max_x},{max_y}: {s['text']}\n")
    img_out = image.copy()
    for s in sentences:
        min_x, min_y, max_x, max_y = s["bbox"]
        cv2.rectangle(img_out, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
    cv2.imwrite("result.png", img_out)
    with open("translated_xxyy.txt", "w", encoding="utf-8") as f:
        for i, s in enumerate(sentences):
            min_x, min_y, max_x, max_y = s["bbox"]
            f.write(f"{min_x},{min_y},{max_x},{max_y}: {translated_sentences[i]}\n")

def process_image(input_path, tgt_lang ,output_path="translated_image_1.png"):
    image = cv2.imread(input_path)
    model = load_craft_model("model_ocr/craft_mlt_25k.pth")
    words = detect_text(model, image)
    sentences = noname_for_this_function(words)
    sentences = calculate_sentence_bboxes(sentences)
    translated_sentences = translate_with_gemini(sentences,tgt_lang)
    inpainted = inpaint_text(image, sentences)
    pil_img = Image.fromarray(cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB))
    draw_translated_text(pil_img, sentences, translated_sentences)
    pil_img.save(output_path)
    save_debug_files(image, sentences, translated_sentences)
