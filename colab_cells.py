# Colab : Conversion photo / vidéo / audio -> TXT + PDF
# Colle chaque bloc comme cellule dans Google Colab.

# ---- Cell 1 : Installer dépendances système et Python ----
# Exécuter une fois au début
!apt-get update -qq
!apt-get install -y -qq ffmpeg
# Installer paquets Python
!pip install -q easyocr opencv-python-headless moviepy pydub openai-whisper torch torchaudio Pillow numpy reportlab ffmpeg-python

# ---- Cell 2 : Monter Google Drive (optionnel) ----
from google.colab import drive
drive.mount('/content/drive')
# Exemple d'emplacement de travail : /content/drive/MyDrive/IA_convert

# ---- Cell 3 : Imports utiles ----
import os
import io
import math
import tempfile
from PIL import Image
import numpy as np
import easyocr
import whisper
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# ---- Cell 4 : Utilitaires de sauvegarde ----
def save_text(txt, out_path):
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(txt)

def save_text_as_pdf(txt, pdf_path, title=None):
    # Simple pagination: on écrit du texte brute sur pages A4
    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4
    margin = 40
    y = height - margin
    lines = txt.splitlines()
    max_chars_per_line = 100  # ajuster selon police/taille
    # chunker pour éviter coupage trop long
    buffer_lines = []
    for line in lines:
        while len(line) > max_chars_per_line:
            buffer_lines.append(line[:max_chars_per_line])
            line = line[max_chars_per_line:]
        buffer_lines.append(line)
    lines = buffer_lines
    line_height = 12
    if title:
        c.setFont("Helvetica-Bold", 14)
        c.drawString(margin, y, title)
        y -= 20
    c.setFont("Helvetica", 10)
    for line in lines:
        if y < margin + line_height:
            c.showPage()
            c.setFont("Helvetica", 10)
            y = height - margin
        c.drawString(margin, y, line)
        y -= line_height
    c.save()

# ---- Cell 5 : OCR sur image (EasyOCR) ----
# Exemples d'utilisation :
reader = easyocr.Reader(['en','fr'], gpu=False)  # gpu=True si GPU dispo

def ocr_image_file(image_path, reader=reader):
    # Retourne texte brut
    result = reader.readtext(image_path, detail=0, paragraph=True)
    txt = "\n".join(result)
    return txt

# Usage:
# img_txt = ocr_image_file('/content/drive/MyDrive/IA_convert/example.jpg')
# save_text(img_txt, '/content/drive/MyDrive/IA_convert/example.txt')
# save_text_as_pdf(img_txt, '/content/drive/MyDrive/IA_convert/example.pdf', title='OCR image')

# ---- Cell 6 : Extraction frames d'une vidéo + OCR ----
def ocr_video(video_path, frame_interval_seconds=1.0, reader=reader):
    clip = VideoFileClip(video_path)
    duration = clip.duration
    texts = []
    t = 0.0
    while t < duration:
        frame = clip.get_frame(t)  # numpy array HxWxC (RGB)
        pil = Image.fromarray(frame)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpf:
            pil.save(tmpf.name)
            frame_txt = reader.readtext(tmpf.name, detail=0, paragraph=True)
        if frame_txt:
            texts.append(f"--- t={t:.2f}s ---\n" + "\n".join(frame_txt))
        t += frame_interval_seconds
    clip.reader.close()
    clip.audio = None
    aggregated = "\n\n".join(texts)
    return aggregated

# Usage:
# vid_txt = ocr_video('/content/drive/MyDrive/IA_convert/sample.mp4', frame_interval_seconds=1.0)
# save_text(vid_txt, '/content/drive/MyDrive/IA_convert/sample_video.txt')
# save_text_as_pdf(vid_txt, '/content/drive/MyDrive/IA_convert/sample_video.pdf', title='OCR video')

# ---- Cell 7 : Transcription audio (Whisper) ----
# Charger modèle Whisper (taille selon GPU et temps)
model = whisper.load_model("small")  # tiny / base / small / medium / large

def transcribe_audio(input_path, model=model, language=None):
    # input_path peut être mp3/wav/m4a...
    # Whisper gère conversion interne, mais on peut forcer wav:
    # audio = AudioSegment.from_file(input_path)
    # tmp_wav = '/content/tmp_audio.wav'
    # audio.export(tmp_wav, format='wav')
    result = model.transcribe(input_path, language=language)
    # result['text'] contient la transcription
    return result['text']

# Usage:
# audio_txt = transcribe_audio('/content/drive/MyDrive/IA_convert/sample.wav', language='fr')
# save_text(audio_txt, '/content/drive/MyDrive/IA_convert/sample_audio.txt')
# save_text_as_pdf(audio_txt, '/content/drive/MyDrive/IA_convert/sample_audio.pdf', title='Transcription audio')

# ---- Cell 8 : Pipeline complet exemple ----
def process_image_to_outputs(image_path, out_prefix, reader=reader):
    txt = ocr_image_file(image_path, reader)
    save_text(txt, out_prefix + '.txt')
    save_text_as_pdf(txt, out_prefix + '.pdf', title=f'OCR {os.path.basename(image_path)}')
    return txt

def process_video_to_outputs(video_path, out_prefix, frame_interval_seconds=1.0, reader=reader):
    txt = ocr_video(video_path, frame_interval_seconds, reader)
    save_text(txt, out_prefix + '.txt')
    save_text_as_pdf(txt, out_prefix + '.pdf', title=f'OCR video {os.path.basename(video_path)}')
    return txt

def process_audio_to_outputs(audio_path, out_prefix, model=model, language=None):
    txt = transcribe_audio(audio_path, model, language)
    save_text(txt, out_prefix + '.txt')
    save_text_as_pdf(txt, out_prefix + '.pdf', title=f'Transcription {os.path.basename(audio_path)}')
    return txt

# Exemple d'appel:
# process_image_to_outputs('/content/drive/MyDrive/IA_convert/example.jpg', '/content/drive/MyDrive/IA_convert/example_out')
# process_video_to_outputs('/content/drive/MyDrive/IA_convert/sample.mp4', '/content/drive/MyDrive/IA_convert/sample_out', frame_interval_seconds=1.0)
# process_audio_to_outputs('/content/drive/MyDrive/IA_convert/sample.wav', '/content/drive/MyDrive/IA_convert/sample_audio_out', language='fr')
