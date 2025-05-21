from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from model_service.utils.translate import Translate, TranslationRequest
import uuid
import os
from model_service.utils.pdf_utils import (
    extract_pdf_cells,
    create_pdf_from_json
)


UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
TEMP_DIR = "temp_backgrounds"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

router = APIRouter()

@router.post("/translate")
async def translate(request : TranslationRequest):
    try:
        translate = Translate(request.text)
        corrected = translate.gemini_correct()
        translate = Translate(corrected)
        src_lang = translate.detect_language()

        if src_lang == request.tgt_lang:
            translated = corrected
        else:
            translated = translate.translate(src_lang, request.tgt_lang)
        
        return {
            "src_lang": src_lang,
            "corrected": corrected,
            "translated": translated
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    

@router.post("/translate-pdf")
async def translate_pdf(file: UploadFile = File(...)):
    # Save uploaded file
    file_id = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_FOLDER, f"{file_id}.pdf")
    output_path = os.path.join(OUTPUT_FOLDER, f"{file_id}_vi.pdf")

    with open(input_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Extract, translate, and generate output PDF
    json_data = extract_pdf_cells(
        pdf_path=input_path,
        translate=True
    )
    create_pdf_from_json(
        data=json_data,
        pdf_path=input_path,
        output_path=output_path,
        temp_dir=TEMP_DIR
    )

    # Return file as response
    return FileResponse(output_path, media_type="application/pdf", filename="translated_vi.pdf")