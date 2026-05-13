"""
Japan AI Project — Full Backend
Features:
  1. Character Chat  (Tokugawa Ieyasu & Toyotomi Hideyoshi)
  2. Audio Transcription (single + bulk)
  3. Manga Storyboard Generator
"""

import os, time, tempfile, pathlib, json, re
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional, List
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set in .env file!")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

app = FastAPI(title="Japan AI Project", version="3.0.0")

# ═══════════════════════════════════════════════════════════════════════════════
#  CHARACTER PROMPTS
# ═══════════════════════════════════════════════════════════════════════════════

CHARACTER_PROMPTS = {
    "ieyasu": """You are Tokugawa Ieyasu, the great shogun of Japan, reborn in modern times.
You are stern, disciplined, and deeply committed to LAW and ORDER.
Your role: Teach foreigners living in Japan about Japanese LAWS and RULES.
Topics: trash disposal rules, smoking regulations, traffic laws, work permits,
illegal labor, noise ordinances, recycling categories, jaywalking, etc.

Personality: Firm but fair. You speak with authority. You sometimes reference your
historical battles and governance as metaphors. You do NOT tolerate lawbreaking.
Always respond in the same language the user writes in (English or Japanese).
Keep responses concise (2-4 sentences). End with a relevant Japanese law tip.""",

    "hideyoshi": """You are Toyotomi Hideyoshi, the charming unifier of Japan, reborn in modern times.
You are warm, sociable, and a master of MANNERS and SOCIAL HARMONY.
Your role: Teach foreigners living in Japan about Japanese MANNERS and SOCIAL SKILLS.
Topics: greetings (ojigi/bowing), neighborhood relations, gift-giving etiquette,
removing shoes indoors, quiet hours, train etiquette, business card exchange, etc.

Personality: Cheerful, encouraging, uses humor and warmth. You celebrate small improvements.
Always respond in the same language the user writes in (English or Japanese).
Keep responses concise (2-4 sentences). End with a practical manner tip."""
}

CHARACTER_NAMES = {
    "ieyasu":   "Tokugawa Ieyasu ⚔️",
    "hideyoshi":"Toyotomi Hideyoshi 🌸"
}

# ═══════════════════════════════════════════════════════════════════════════════
#  REQUEST MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    character: str
    message: str
    history: Optional[list] = []

class MangaRequest(BaseModel):
    situation: str
    character: str
    language: str = "en"
    panels: int = 4

# ═══════════════════════════════════════════════════════════════════════════════
#  1. CHAT ENDPOINT
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/api/chat")
async def chat(req: ChatRequest):
    if req.character not in CHARACTER_PROMPTS:
        raise HTTPException(400, "Character must be 'ieyasu' or 'hideyoshi'")
    if not req.message.strip():
        raise HTTPException(400, "Message cannot be empty")

    history_text = ""
    for turn in req.history[-6:]:
        role = "User" if turn.get("role") == "user" else CHARACTER_NAMES[req.character]
        history_text += f"{role}: {turn.get('content','')}\n"

    prompt = f"""{CHARACTER_PROMPTS[req.character]}

--- Conversation History ---
{history_text}
--- Current Message ---
User: {req.message}
{CHARACTER_NAMES[req.character]}:"""

    try:
        resp = model.generate_content(prompt)
        return {
            "character": req.character,
            "character_name": CHARACTER_NAMES[req.character],
            "reply": resp.text.strip()
        }
    except Exception as e:
        raise HTTPException(500, f"Gemini error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
#  2. AUDIO TRANSCRIPTION — single + bulk
# ═══════════════════════════════════════════════════════════════════════════════

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".ogg", ".m4a", ".flac", ".webm", ".mp4", ".aac"}

TRANSCRIBE_PROMPT = """Transcribe this audio completely and accurately.

CRITICAL LANGUAGE RULE:
- Write each word/sentence in the EXACT script of the language being spoken
- Japanese speech → write in Japanese script (kanji + hiragana + katakana)
  Example: do NOT write "Sumimasen" — write "すみません"
  Example: do NOT write "Haru-san House wa doko desu ka" — write "ハルさんハウスはどこですか？"
- English speech → write in English
- Do NOT romanize any language — use the native script
- Do NOT translate — keep each language exactly as spoken
- If speakers switch languages, switch scripts accordingly

Format rules:
- Multiple speakers: label as [Speaker 1], [Speaker 2], etc.
- Include timestamps every ~1 minute: [00:00], [01:00], etc.
- Unclear parts: write [unclear]
- Line breaks between speakers/segments"""


async def _transcribe_one(file: UploadFile) -> dict:
    """Core transcription logic. Returns a result dict, never raises."""
    ext = pathlib.Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return {
            "filename": file.filename, "status": "error",
            "error": f"Unsupported format '{ext}'", "transcript": "", "size_mb": 0
        }

    content = await file.read()
    size_mb = round(len(content) / 1024 / 1024, 1)

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        gemini_file = genai.upload_file(tmp_path, mime_type=f"audio/{ext.lstrip('.')}")

        for _ in range(30):  # wait max 60s
            if gemini_file.state.name != "PROCESSING":
                break
            time.sleep(2)
            gemini_file = genai.get_file(gemini_file.name)

        if gemini_file.state.name == "FAILED":
            return {"filename": file.filename, "status": "error",
                    "error": "Gemini processing failed", "transcript": "", "size_mb": size_mb}

        resp = model.generate_content([TRANSCRIBE_PROMPT, gemini_file])
        try: gemini_file.delete()
        except: pass

        return {
            "filename": file.filename, "status": "ok",
            "transcript": resp.text.strip(), "size_mb": size_mb, "error": ""
        }
    except Exception as e:
        return {"filename": file.filename, "status": "error",
                "error": str(e), "transcript": "", "size_mb": size_mb}
    finally:
        try: os.unlink(tmp_path)
        except: pass


# Single file
@app.post("/api/transcribe")
async def transcribe_single(file: UploadFile = File(...)):
    result = await _transcribe_one(file)
    if result["status"] == "error":
        raise HTTPException(500, result["error"])
    return {"filename": result["filename"], "transcript": result["transcript"],
            "duration_note": f"File: {result['size_mb']} MB."}


# Bulk — many files at once, processed one by one
@app.post("/api/transcribe/bulk")
async def transcribe_bulk(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(400, "No files uploaded")
    if len(files) > 50:
        raise HTTPException(400, "Maximum 50 files per request")

    results = []
    for i, file in enumerate(files):
        print(f"[Bulk] {i+1}/{len(files)}: {file.filename}")
        result = await _transcribe_one(file)
        results.append(result)
        if i < len(files) - 1:
            time.sleep(1)  # avoid rate limits between files

    success = sum(1 for r in results if r["status"] == "ok")
    return {
        "total": len(results),
        "success": success,
        "failed": len(results) - success,
        "results": results
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  3. MANGA STORYBOARD ENDPOINT
# ═══════════════════════════════════════════════════════════════════════════════

MANGA_PROMPT_TEMPLATE = """You are a manga storyboard writer for a Japanese life-guide comic.
The comic teaches foreigners living in Japan about rules and manners.
The main character teaching in this story is: {char_name}

Create a {panels}-panel manga storyboard for this situation:
"{situation}"

IMPORTANT: Your entire response must be ONLY a valid JSON object. No explanation, no markdown, no code fences.
Start your response with {{ and end with }}

JSON structure:
{{
  "title": "Short manga chapter title",
  "moral": "One-sentence lesson for foreigners",
  "panels": [
    {{
      "panel": 1,
      "scene": "Setting description for artist",
      "characters": "Who is in the panel and what they are doing",
      "dialogue": [{{"speaker": "Character name", "text": "What they say"}}],
      "expression": "Emotional tone / art direction",
      "sfx": "Sound effect if any, empty string if none"
    }}
  ]
}}

Dialogue language: {language_instruction}
Make it engaging, educational, and slightly dramatic — manga style!
{char_name} must appear and deliver the key lesson.
Remember: output ONLY the JSON object, nothing else."""


@app.post("/api/manga")
async def manga_storyboard(req: MangaRequest):
    if req.character not in CHARACTER_PROMPTS:
        raise HTTPException(400, "Character must be 'ieyasu' or 'hideyoshi'")
    if not req.situation.strip():
        raise HTTPException(400, "Situation cannot be empty")
    panels = max(3, min(6, req.panels))
    lang_instruction = "Japanese with English translation in brackets" if req.language == "ja" else "English"

    prompt = MANGA_PROMPT_TEMPLATE.format(
        char_name=CHARACTER_NAMES[req.character],
        panels=panels, situation=req.situation,
        language_instruction=lang_instruction
    )

    try:
        resp = model.generate_content(prompt)
        raw = resp.text.strip()
        raw = re.sub(r'^```(?:json)?\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw).strip()
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            raw = match.group(0)
        return json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(500, "Gemini returned invalid JSON. Please try again.")
    except Exception as e:
        raise HTTPException(500, f"Manga generation error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
#  HEALTH + FRONTEND
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/health")
async def health():
    return {"status": "ok", "model": "gemini-2.5-flash", "version": "3.0.0"}

@app.get("/", response_class=HTMLResponse)
async def frontend():
    p = pathlib.Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(content=p.read_text(encoding="utf-8"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)