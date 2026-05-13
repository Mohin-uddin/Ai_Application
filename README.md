# 🏯 Japan AI Guide — Full Project

## তিনটো Feature আছে:
| Feature | কী করে |
|---|---|
| 🗾 Character Chat | Ieyasu (আইন) বা Hideyoshi (শিষ্টাচার)-এর সাথে চ্যাট |
| 🎙️ Audio Transcription | যেকোনো অডিও ফাইল → টেক্সট |
| 📖 Manga Storyboard | পরিস্থিতি দিলে AI manga panel script বানাবে |

---

## Setup — ৩ স্টেপ

### Step 1: Install
```bash
pip install -r requirements.txt
```

### Step 2: API Key সেট করো
```bash
cp .env.example .env
# .env খুলে GEMINI_API_KEY=তোমার_key বসাও
# Key পাবে: https://aistudio.google.com/app/apikey
```

### Step 3: Run
```bash
python main.py
```
→ Browser: **http://localhost:8000**
→ API Docs: **http://localhost:8000/docs**

---

## API Endpoints

| Method | URL | কী করে |
|---|---|---|
| GET | `/` | Frontend UI |
| POST | `/api/chat` | Character chat |
| POST | `/api/transcribe` | Audio → Text |
| POST | `/api/manga` | Manga storyboard generate |
| GET | `/api/health` | Health check |

---

## Manga Storyboard কীভাবে কাজ করে?

1. Tab → "Manga Storyboard" যাও
2. পরিস্থিতি লেখো (বাংলায়ও চলবে, তবে English best)
3. Character বেছে নাও (Ieyasu = আইন, Hideyoshi = শিষ্টাচার)
4. ভাষা ও panel সংখ্যা ঠিক করো
5. Generate চাপো → JSON থেকে সুন্দর panel-by-panel storyboard দেখাবে
6. Export বাটন দিয়ে `.txt` ফাইলে save করো (artist কে দেওয়ার জন্য)

---

## Project Structure
```
japan_ai_full/
├── main.py              ← FastAPI backend (সব logic)
├── requirements.txt     ← Python dependencies
├── .env.example         ← API key template
├── README.md
└── static/
    └── index.html       ← Frontend UI (সব tab একসাথে)
```
