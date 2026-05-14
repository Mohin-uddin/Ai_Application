"""
Japan AI Project — Full Backend v6
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Transcription : faster-whisper   (FREE, 4-5x faster than whisper, parallel chunks)
Voice Profile : librosa          (pitch, gender, speed — 6-stage)
Speaker Split : pyannote.audio   (optional, needs HF_TOKEN)
Storage/Search: SQLite           (date folders, keyword search)
Chat + Manga  : Gemini API
"""

import os, time, tempfile, pathlib, json, re, math, subprocess
import sqlite3, datetime, asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
from dotenv import load_dotenv
import numpy as np

load_dotenv()

# ── Gemini (Chat + Manga only) ────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
gemini_model = None
if GEMINI_API_KEY:
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-2.5-flash")
    print("✅ Gemini ready (Chat + Manga)")
else:
    print("⚠️  GEMINI_API_KEY not set — Chat & Manga disabled")

# ── faster-whisper (FREE, local, 4-5x faster) ─────────────────────────────────
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")
# Options: tiny | base | small | medium | large-v3
# larger = more accurate but slower & needs more RAM
# Recommended: base (fast) or small (balanced)
_whisper_model = None
_whisper_lock = asyncio.Lock()  # prevent concurrent model loads

def get_whisper():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        print(f"⏳ Loading faster-whisper '{WHISPER_MODEL_SIZE}' model...")
        # device="cpu", compute_type="int8" → fastest on CPU, minimal RAM
        # int8 not supported on all CPUs (esp. Mac Apple Silicon)
        # Try int8 first, fall back to float32
        try:
            _whisper_model = WhisperModel(
                WHISPER_MODEL_SIZE,
                device="cpu",
                compute_type="int8"
            )
        except Exception:
            _whisper_model = WhisperModel(
                WHISPER_MODEL_SIZE,
                device="cpu",
                compute_type="float32"
            )
        print("✅ faster-whisper ready")
    return _whisper_model

# ── pyannote speaker diarization (optional) ───────────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN", "")
_pyannote_pipeline = None

def get_pyannote():
    global _pyannote_pipeline
    if not HF_TOKEN:
        return None
    if _pyannote_pipeline is None:
        try:
            from pyannote.audio import Pipeline
            _pyannote_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=HF_TOKEN
            )
            print("✅ pyannote speaker diarization ready")
        except Exception as e:
            print(f"⚠️  pyannote failed: {e}")
    return _pyannote_pipeline

# ── SQLite Database ───────────────────────────────────────────────────────────
DB_PATH = pathlib.Path("transcripts.db")
TRANSCRIPT_DIR = pathlib.Path("transcript_files")
TRANSCRIPT_DIR.mkdir(exist_ok=True)

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS transcripts (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                filename        TEXT,
                date_str        TEXT,
                time_str        TEXT,
                folder          TEXT,
                transcript      TEXT,
                speakers_json   TEXT    DEFAULT '[]',
                voice_profile   TEXT    DEFAULT '{}',
                keywords        TEXT    DEFAULT '',
                file_size_mb    REAL    DEFAULT 0,
                chunks_count    INTEGER DEFAULT 1,
                created_at      TEXT    DEFAULT (datetime('now','localtime'))
            )
        """)
        conn.commit()

init_db()

# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(title="Japan AI Project", version="6.0.0")

CHUNK_SECONDS  = 30
ALLOWED_EXT    = {".mp3",".wav",".ogg",".m4a",".flac",".webm",".mp4",".aac"}
MAX_PARALLEL   = int(os.getenv("MAX_PARALLEL_CHUNKS", "4"))
# How many chunks to process simultaneously
# 4 is safe for most machines; increase if you have more CPU cores

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

class RenameRequest(BaseModel):
    transcript_id: int
    old_label: str
    new_name: str

# ═══════════════════════════════════════════════════════════════════════════════
#  CHARACTER PROMPTS
# ═══════════════════════════════════════════════════════════════════════════════

CHARACTER_PROMPTS = {
    "ieyasu": """You are Tokugawa Ieyasu, the great shogun of Japan, reborn in modern times.
Stern, disciplined, committed to LAW and ORDER.
Teach foreigners about Japanese LAWS: trash disposal, smoking, traffic laws, work permits, noise rules.
Personality: firm but fair. Reference historical battles as metaphors.
Respond in the same language as the user. Keep concise (2-4 sentences). End with a law tip.""",
    "hideyoshi": """You are Toyotomi Hideyoshi, charming unifier of Japan, reborn in modern times.
Warm, sociable, master of MANNERS and SOCIAL HARMONY.
Teach foreigners about Japanese MANNERS: greetings, neighborhood relations, train etiquette, gift-giving.
Personality: cheerful, encouraging, warm.
Respond in the same language as the user. Keep concise (2-4 sentences). End with a manner tip."""
}
CHARACTER_NAMES = {
    "ieyasu":   "Tokugawa Ieyasu ⚔️",
    "hideyoshi":"Toyotomi Hideyoshi 🌸"
}

# ═══════════════════════════════════════════════════════════════════════════════
#  HELPER: ffmpeg
# ═══════════════════════════════════════════════════════════════════════════════

def _ffmpeg_ok() -> bool:
    try:
        subprocess.run(['ffmpeg','-version'],capture_output=True,check=True,timeout=5)
        return True
    except Exception:
        return False

def _get_duration(path: str) -> float:
    r = subprocess.run(
        ['ffprobe','-v','quiet','-print_format','json','-show_format', path],
        capture_output=True, text=True, timeout=30
    )
    return float(json.loads(r.stdout)['format']['duration'])

def _split_audio(input_path: str, chunk_sec: int = 30) -> List[str]:
    dur = _get_duration(input_path)
    chunks = []
    for i in range(math.ceil(dur / chunk_sec)):
        # Always export as WAV — Whisper reads WAV reliably
        out = input_path + f'_c{i:04d}.wav'
        subprocess.run([
            'ffmpeg','-y','-i', input_path,
            '-ss', str(i * chunk_sec), '-t', str(chunk_sec),
            '-ar', '16000',      # 16kHz — Whisper native sample rate
            '-ac', '1',          # mono
            '-acodec', 'pcm_s16le',  # standard WAV encoding
            out
        ], capture_output=True, timeout=60)
        if os.path.exists(out) and os.path.getsize(out) > 0:
            chunks.append(out)
    return chunks

# ═══════════════════════════════════════════════════════════════════════════════
#  HELPER: Voice profiling (librosa — 6-stage)
# ═══════════════════════════════════════════════════════════════════════════════

def _voice_profile(audio_path: str) -> dict:
    try:
        import librosa
        y, sr = librosa.load(audio_path, sr=None, duration=120, mono=True)
        if len(y) == 0:
            return {"status": "no_audio"}
        f0, voiced, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'),
                                      fmax=librosa.note_to_hz('C7'), sr=sr)
        valid = f0[voiced] if voiced is not None else np.array([])
        avg_pitch = float(np.nanmean(valid)) if len(valid) > 0 else 0.0
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(tempo) if np.isscalar(tempo) else float(tempo[0])
        rms = librosa.feature.rms(y=y)
        avg_energy = float(np.mean(rms)) * 1000
        return {
            "status":    "ok",
            "pitch_hz":  round(avg_pitch, 1),
            "pitch":     "High" if avg_pitch > 200 else "Low",
            "gender":    "Female" if avg_pitch > 165 else "Male",
            "tempo_bpm": round(tempo, 1),
            "speed":     "Fast" if tempo > 130 else ("Normal" if tempo > 90 else "Slow"),
            "energy":    round(avg_energy, 2)
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

# ═══════════════════════════════════════════════════════════════════════════════
#  HELPER: Speaker diarization (pyannote — optional)
# ═══════════════════════════════════════════════════════════════════════════════

def _diarize(audio_path: str) -> List[dict]:
    pipe = get_pyannote()
    if not pipe:
        return []
    try:
        dz = pipe(audio_path)
        return [{"start": round(t.start, 2), "end": round(t.end, 2), "speaker": spk}
                for t, _, spk in dz.itertracks(yield_label=True)]
    except Exception as e:
        print(f"Diarization error: {e}")
        return []

# ═══════════════════════════════════════════════════════════════════════════════
#  HELPER: Keyword extraction
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_keywords(text: str) -> List[str]:
    if gemini_model and len(text) > 50:
        try:
            prompt = f"""Extract important keywords, names, and topics from this text.
Return ONLY a JSON array of strings. No explanation, no markdown.
Text: {text[:3000]}"""
            r = gemini_model.generate_content(prompt)
            raw = r.text.strip()
            raw = re.sub(r'^```(?:json)?\s*', '', raw)
            raw = re.sub(r'\s*```$', '', raw).strip()
            return [str(k) for k in json.loads(raw)][:30]
        except:
            pass
    words = re.findall(r'\b[A-Z][a-z]{2,}\b', text)
    return list(dict.fromkeys(words))[:20]

# ═══════════════════════════════════════════════════════════════════════════════
#  HELPER: faster-whisper transcription of ONE chunk (runs in thread)
# ═══════════════════════════════════════════════════════════════════════════════

def _transcribe_chunk(chunk_idx: int, path: str, time_offset: float) -> dict:
    """
    Transcribe one audio chunk using faster-whisper.
    Returns: {idx, status, text, segments, language}
    faster-whisper is thread-safe — multiple chunks can run simultaneously.
    """
    try:
        wm = get_whisper()
        segments_gen, info = wm.transcribe(
            path,
            beam_size=5,
            language=None,       # auto-detect
            vad_filter=False,    # disabled — was filtering real speech
        )
        segments = []
        full_text_parts = []
        for seg in segments_gen:
            # Adjust timestamps by chunk offset
            segments.append({
                "start": round(seg.start + time_offset, 2),
                "end":   round(seg.end   + time_offset, 2),
                "text":  seg.text.strip()
            })
            full_text_parts.append(seg.text.strip())

        return {
            "idx":      chunk_idx,
            "status":   "ok",
            "text":     " ".join(full_text_parts),
            "segments": segments,
            "language": info.language
        }
    except Exception as e:
        import traceback
        print(f"[Chunk {chunk_idx}] ERROR: {e}")
        traceback.print_exc()
        return {
            "idx":      chunk_idx,
            "status":   "error",
            "error":    str(e),
            "text":     "",
            "segments": [],
            "language": "?"
        }

# ═══════════════════════════════════════════════════════════════════════════════
#  HELPER: merge speaker labels into transcript
# ═══════════════════════════════════════════════════════════════════════════════

def _merge_speakers(whisper_segments: List[dict], diarization: List[dict]) -> str:
    if not diarization:
        lines = [f"[{int(s['start']//60):02d}:{int(s['start']%60):02d}] {s['text']}"
                 for s in whisper_segments]
        return "\n".join(lines)
    lines = []
    for seg in whisper_segments:
        ts = f"[{int(seg['start']//60):02d}:{int(seg['start']%60):02d}]"
        speaker = "?"
        for d in diarization:
            if d["start"] <= seg["start"] <= d["end"]:
                speaker = d["speaker"]; break
        lines.append(f"{ts} [{speaker}] {seg['text']}")
    return "\n".join(lines)

# ═══════════════════════════════════════════════════════════════════════════════
#  HELPER: save to SQLite + date folder
# ═══════════════════════════════════════════════════════════════════════════════

def _save_transcript(filename, transcript, speakers, voice_profile, keywords, size_mb, chunks):
    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M")
    folder = TRANSCRIPT_DIR / date_str
    folder.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r'[^\w\-.]', '_', pathlib.Path(filename).stem)
    txt_path = folder / f"{safe_name}_{now.strftime('%H%M%S')}.txt"
    header = (f"{'━'*40}\nJapan AI Guide — Transcript\n"
              f"Date : {date_str}\nTime : {time_str}\nFile : {filename}\n"
              f"Size : {size_mb} MB | Chunks: {chunks}\n"
              f"Keywords: {', '.join(keywords)}\n{'━'*40}\n\n")
    txt_path.write_text(header + transcript, encoding="utf-8")
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            """INSERT INTO transcripts
               (filename,date_str,time_str,folder,transcript,
                speakers_json,voice_profile,keywords,file_size_mb,chunks_count)
               VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (filename, date_str, time_str, str(txt_path), transcript,
             json.dumps(speakers, ensure_ascii=False),
             json.dumps(voice_profile, ensure_ascii=False),
             ",".join(keywords), size_mb, chunks)
        )
        conn.commit()
        return cur.lastrowid

def _sse(d: dict) -> str:
    return f"data: {json.dumps(d, ensure_ascii=False)}\n\n"

# ═══════════════════════════════════════════════════════════════════════════════
#  1. CHAT
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/api/chat")
async def chat(req: ChatRequest):
    if not gemini_model:
        raise HTTPException(503, "Gemini API key not configured")
    if req.character not in CHARACTER_PROMPTS:
        raise HTTPException(400, "Invalid character")
    hist = ""
    for t in req.history[-6:]:
        role = "User" if t.get("role") == "user" else CHARACTER_NAMES[req.character]
        hist += f"{role}: {t.get('content','')}\n"
    prompt = (f"{CHARACTER_PROMPTS[req.character]}\n"
              f"--- History ---\n{hist}"
              f"--- Now ---\nUser: {req.message}\n{CHARACTER_NAMES[req.character]}:")
    try:
        r = gemini_model.generate_content(prompt)
        return {"character": req.character,
                "character_name": CHARACTER_NAMES[req.character],
                "reply": r.text.strip()}
    except Exception as e:
        raise HTTPException(500, f"Gemini error: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
#  2. TRANSCRIPTION — faster-whisper + PARALLEL chunks + SSE progress
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/api/transcribe/stream")
async def transcribe_stream(files: List[UploadFile] = File(...)):
    if len(files) > 50:
        raise HTTPException(400, "Max 50 files")

    async def generate():
        loop = asyncio.get_event_loop()
        all_results = []
        ffmpeg = await loop.run_in_executor(None, _ffmpeg_ok)

        for fi, file in enumerate(files):
            fn = file.filename
            ext = pathlib.Path(fn).suffix.lower()
            yield _sse({"type":"file_start","file":fn,
                        "file_idx":fi,"total_files":len(files)})

            if ext not in ALLOWED_EXT:
                err = f"Unsupported format '{ext}'"
                yield _sse({"type":"file_error","file":fn,"error":err})
                all_results.append({"filename":fn,"status":"error",
                                    "error":err,"transcript":""})
                continue

            content = await file.read()
            size_mb = round(len(content) / 1024 / 1024, 1)

            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            chunk_paths = []
            try:
                # ── Split audio ──────────────────────────────────────────
                if ffmpeg and len(content) > 300_000:
                    yield _sse({"type":"splitting","file":fn,
                                "message":f"Splitting into {CHUNK_SECONDS}s chunks..."})
                    try:
                        chunk_paths = await loop.run_in_executor(
                            None, _split_audio, tmp_path, CHUNK_SECONDS)
                    except Exception:
                        chunk_paths = [tmp_path]
                else:
                    chunk_paths = [tmp_path]

                total = len(chunk_paths)
                yield _sse({"type":"chunks_ready","file":fn,
                            "total_chunks":total,"size_mb":size_mb,
                            "parallel":MAX_PARALLEL})

                # ── PARALLEL transcription ───────────────────────────────
                # Prepare jobs: (chunk_idx, path, time_offset)
                jobs = [(i, cp, i * CHUNK_SECONDS) for i, cp in enumerate(chunk_paths)]

                # Results dict (order-preserving after parallel run)
                chunk_results = {}
                completed_count = 0

                # Use ThreadPoolExecutor — faster-whisper is thread-safe
                with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as executor:
                    # Submit all at once
                    future_to_idx = {
                        executor.submit(_transcribe_chunk, idx, path, offset): idx
                        for idx, path, offset in jobs
                    }

                    # As each chunk finishes, yield SSE progress
                    for future in as_completed(future_to_idx):
                        res = future.result()
                        chunk_results[res["idx"]] = res
                        completed_count += 1
                        pct = round(completed_count / total * 100)

                        yield _sse({
                            "type":         "chunk_done",
                            "file":         fn,
                            "chunk":        res["idx"] + 1,
                            "total_chunks": total,
                            "percent":      pct,
                            "status":       res["status"]
                        })

                        # Clean up chunk file
                        cp = chunk_paths[res["idx"]]
                        if cp != tmp_path:
                            try: os.unlink(cp)
                            except: pass

                # ── Merge chunks in ORDER ───────────────────────────────
                all_segs = []
                all_text_parts = []
                detected_lang = "?"

                for i in range(total):
                    r = chunk_results.get(i, {})
                    if r.get("status") == "ok":
                        detected_lang = r.get("language", "?")
                        all_segs.extend(r.get("segments", []))
                        header = f"\n[── Chunk {i+1}/{total} | {i*CHUNK_SECONDS//60:02d}:{i*CHUNK_SECONDS%60:02d} ──]\n"
                        all_text_parts.append(header + r.get("text", ""))

                # ── Speaker diarization (optional) ───────────────────────
                speakers = []
                if HF_TOKEN and all_segs:
                    yield _sse({"type":"status","file":fn,
                                "message":"Running speaker diarization..."})
                    try:
                        speakers = await loop.run_in_executor(None, _diarize, tmp_path)
                    except: pass

                # ── Build final transcript ───────────────────────────────
                if speakers:
                    full_transcript = _merge_speakers(all_segs, speakers)
                else:
                    full_transcript = "\n".join(all_text_parts)

                # ── Voice profiling ──────────────────────────────────────
                yield _sse({"type":"status","file":fn,
                            "message":"Analyzing voice profile..."})
                vprofile = await loop.run_in_executor(None, _voice_profile, tmp_path)

                # ── Keyword extraction ───────────────────────────────────
                yield _sse({"type":"status","file":fn,
                            "message":"Extracting keywords..."})
                keywords = await loop.run_in_executor(None, _extract_keywords, full_transcript)

                # ── Save to SQLite + folder ──────────────────────────────
                rec_id = await loop.run_in_executor(
                    None, _save_transcript, fn, full_transcript,
                    speakers, vprofile, keywords, size_mb, total
                )

                unique_speakers = list(dict.fromkeys(s["speaker"] for s in speakers))
                result = {
                    "filename":     fn,
                    "status":       "ok",
                    "id":           rec_id,
                    "transcript":   full_transcript,
                    "chunks":       total,
                    "size_mb":      size_mb,
                    "language":     detected_lang,
                    "voice_profile":vprofile,
                    "keywords":     keywords,
                    "speakers":     unique_speakers
                }
                all_results.append(result)
                yield _sse({"type":"file_done","file":fn,"status":"ok","result":result})

            except Exception as e:
                all_results.append({"filename":fn,"status":"error",
                                    "error":str(e),"transcript":""})
                yield _sse({"type":"file_error","file":fn,"error":str(e)})
            finally:
                try: os.unlink(tmp_path)
                except: pass
                for cp in chunk_paths:
                    if cp != tmp_path:
                        try: os.unlink(cp)
                        except: pass

        ok = sum(1 for r in all_results if r["status"] == "ok")
        yield _sse({"type":"complete","total":len(all_results),
                    "success":ok,"failed":len(all_results)-ok,
                    "results":all_results})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"}
    )

# ── Speaker rename ─────────────────────────────────────────────────────────────
@app.post("/api/transcript/rename-speaker")
async def rename_speaker(req: RenameRequest):
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("SELECT transcript FROM transcripts WHERE id=?",
                           (req.transcript_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Transcript not found")
        new_text = row[0].replace(f"[{req.old_label}]", f"[{req.new_name}]")
        conn.execute("UPDATE transcripts SET transcript=? WHERE id=?",
                     (new_text, req.transcript_id))
        conn.commit()
    return {"ok": True, "message": f"Replaced '{req.old_label}' → '{req.new_name}'"}

# ═══════════════════════════════════════════════════════════════════════════════
#  3. SEARCH
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/search")
async def search(q: str = Query(""), date: str = Query(""), limit: int = 20):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        base = ("SELECT id,filename,date_str,time_str,keywords,"
                "file_size_mb,chunks_count,voice_profile FROM transcripts WHERE 1=1")
        params = []
        if q:
            base += " AND (transcript LIKE ? OR keywords LIKE ? OR filename LIKE ?)"
            params += [f"%{q}%", f"%{q}%", f"%{q}%"]
        if date:
            base += " AND date_str=?"
            params.append(date)
        base += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(base, params).fetchall()
    return {"results": [dict(r) for r in rows]}

@app.get("/api/transcript/{id}")
async def get_transcript(id: int):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM transcripts WHERE id=?", (id,)).fetchone()
        if not row:
            raise HTTPException(404, "Not found")
        return dict(row)

@app.get("/api/dates")
async def get_dates():
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT DISTINCT date_str, COUNT(*) as cnt FROM transcripts "
            "GROUP BY date_str ORDER BY date_str DESC"
        ).fetchall()
    return {"dates": [{"date": r[0], "count": r[1]} for r in rows]}

# ═══════════════════════════════════════════════════════════════════════════════
#  4. MANGA
# ═══════════════════════════════════════════════════════════════════════════════

MANGA_PROMPT = """You are a manga storyboard writer for a Japanese life-guide comic.
Main character: {char_name}
Situation: "{situation}"
Panels: {panels}

IMPORTANT: Respond ONLY with valid JSON (no markdown, no explanation).
{{
  "title": "chapter title",
  "moral": "one-sentence lesson",
  "panels": [
    {{
      "panel": 1,
      "scene": "setting description for artist",
      "characters": "who is in panel and what they do",
      "dialogue": [{{"speaker": "name", "text": "line"}}],
      "expression": "art direction / emotion",
      "sfx": "sound effect or empty string"
    }}
  ]
}}
Dialogue language: {lang}
{char_name} must deliver the key lesson."""

@app.post("/api/manga")
async def manga(req: MangaRequest):
    if not gemini_model:
        raise HTTPException(503, "Gemini API key not configured")
    if req.character not in CHARACTER_PROMPTS:
        raise HTTPException(400, "Invalid character")
    panels = max(3, min(6, req.panels))
    lang = "Japanese with English translation in brackets" if req.language == "ja" else "English"
    prompt = MANGA_PROMPT.format(
        char_name=CHARACTER_NAMES[req.character],
        situation=req.situation, panels=panels, lang=lang
    )
    try:
        r = gemini_model.generate_content(prompt)
        raw = r.text.strip()
        raw = re.sub(r'^```(?:json)?\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw).strip()
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m: raw = m.group(0)
        return json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(500, "Invalid JSON. Try again.")
    except Exception as e:
        raise HTTPException(500, f"Manga error: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
#  HEALTH + FRONTEND
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/test-whisper")
async def test_whisper():
    """Quick test — transcribe a 1-second silent WAV to check if Whisper loads."""
    import wave, struct
    tmp = tempfile.mktemp(suffix=".wav")
    try:
        # Write 1 second of silence at 16kHz
        with wave.open(tmp, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(struct.pack("<16000h", *([0]*16000)))
        wm = get_whisper()
        segs, info = wm.transcribe(tmp, beam_size=1, vad_filter=False)
        list(segs)  # consume generator
        return {"status": "ok", "model": WHISPER_MODEL_SIZE,
                "language": info.language, "message": "Whisper loaded and working"}
    except Exception as e:
        import traceback
        return {"status": "error", "error": str(e),
                "traceback": traceback.format_exc()}
    finally:
        try: os.unlink(tmp)
        except: pass

@app.get("/api/health")
async def health():
    return {
        "status":         "ok",
        "version":        "6.0.0",
        "whisper_model":  WHISPER_MODEL_SIZE,
        "parallel_chunks":MAX_PARALLEL,
        "gemini":         bool(GEMINI_API_KEY),
        "ffmpeg":         _ffmpeg_ok(),
        "pyannote":       bool(HF_TOKEN),
        "db":             str(DB_PATH)
    }

@app.get("/", response_class=HTMLResponse)
async def frontend():
    p = pathlib.Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(content=p.read_text(encoding="utf-8"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)