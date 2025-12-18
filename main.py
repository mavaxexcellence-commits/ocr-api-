from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
import pytesseract
from PIL import Image
import cv2
import os
from googletrans import Translator
from reportlab.pdfgen import canvas

app = FastAPI()
translator = Translator()

@app.post("/ocr")
async def ocr_translate(
    file: UploadFile = File(...),
    lang: str = "fr"
):
    print("en cours d'exécution")

    image_path = "temp.jpg"
    pdf_path = "resultat.pdf"

    with open(image_path, "wb") as f:
        f.write(await file.read())

    img = cv2.imread(image_path)
    if img is None:
        return JSONResponse(
            status_code=400,
            content={"error": "erreur d'exécution, image flou ou non lisible"}
        )

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()

    if blur < 100:
        return JSONResponse(
            status_code=400,
            content={"error": "erreur d'exécution, image flou ou non lisible"}
        )

    text = pytesseract.image_to_string(Image.open(image_path))
    if not text.strip():
        return JSONResponse(
            status_code=400,
            content={"error": "erreur d'exécution, image flou ou non lisible"}
        )

    translated = translator.translate(text, dest=lang).text

    c = canvas.Canvas(pdf_path)
    y = 800
    for line in translated.split("\n"):
        c.drawString(40, y, line)
        y -= 14
    c.save()

    print("félicitation pour votre design")
    return FileResponse(pdf_path, filename="resultat.pdf")
