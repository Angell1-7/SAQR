"""
SAQR API — receives an audio/video file and returns the deepfake analysis as JSON.

Run:
    pip install -r requirements.txt        (ffmpeg must be installed on the machine)
    SAQR_MODEL_DIR=/path/to/saqr_v2_final uvicorn app:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health    -> {"status": "ok", "model_version": "..."}
    POST /analyze   -> multipart/form-data with field "file"
"""
import os
import sqlite3
import tempfile
import time
from datetime import datetime

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from saqr_core import AnalysisError, SaqrDetector

MODEL_DIR = os.environ.get("SAQR_MODEL_DIR", "saqr_v2_final")
MAX_MB = 25
ALLOWED = {".wav", ".mp3", ".ogg", ".opus", ".m4a", ".aac", ".amr", ".flac", ".webm",
           ".mp4", ".mov", ".3gp", ".mkv"}

app = FastAPI(title="SAQR API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

detector = SaqrDetector(MODEL_DIR)

# ---- قاعدة البيانات: تحفظ نتيجة كل تحليل فقط (بدون الملف الصوتي وبدون اسم الملف)
DB_PATH = os.environ.get("SAQR_DB", "saqr.db")


def init_db():
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""CREATE TABLE IF NOT EXISTS analyses (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at       TEXT    NOT NULL,
            status           TEXT    NOT NULL CHECK (status IN ('success', 'error')),
            verdict          TEXT    CHECK (verdict IN ('real', 'suspicious', 'fake')),
            fake_probability REAL,
            audio_quality    TEXT    CHECK (audio_quality IN ('normal', 'call')),
            has_warning      INTEGER CHECK (has_warning IN (0, 1)),
            error_code       TEXT,
            duration_sec     REAL,
            file_type        TEXT,
            file_size_kb     INTEGER,
            processing_sec   REAL,
            model_version    TEXT    NOT NULL)""")


def save_analysis(**row):
    try:
        row["created_at"] = datetime.now().isoformat(sep=" ", timespec="seconds")
        row["model_version"] = detector.version
        cols, marks = ", ".join(row), ", ".join("?" * len(row))
        with sqlite3.connect(DB_PATH) as con:
            con.execute(f"INSERT INTO analyses ({cols}) VALUES ({marks})", list(row.values()))
    except Exception as e:                      # فشل الحفظ لا يوقف التحليل
        print(f"[db] save failed: {e}", flush=True)


init_db()


def error(code, message_ar, http_status):
    return JSONResponse(status_code=http_status,
                        content={"status": "error", "error_code": code, "message_ar": message_ar})


TEST_PAGE = """<!doctype html><html lang="ar" dir="rtl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>SAQR — اختبار</title>
<style>body{font-family:system-ui,sans-serif;max-width:640px;margin:40px auto;padding:0 16px;background:#111;color:#eee}
button{padding:10px 18px;font-size:16px}pre{background:#222;padding:12px;border-radius:8px;white-space:pre-wrap;direction:ltr;text-align:left}
#v{font-size:28px;margin:16px 0}</style></head><body>
<h2>صقر — صفحة اختبار الـ API</h2><p>هذه الصفحة للفريق فقط، والواجهة الحقيقية يبنيها الفرونت إند.</p>
<input type="file" id="f" accept="audio/*,video/*,.opus,.ogg,.m4a,.aac,.amr,.wav,.mp3,.flac,.webm,.mp4,.mov,.3gp,.mkv"> <button onclick="go()">تحليل</button>
<div id="v"></div><pre id="out"></pre>
<script>
async function go(){const f=document.getElementById('f').files[0];if(!f)return;
document.getElementById('v').textContent='جاري التحليل...';const t=performance.now();
const fd=new FormData();fd.append('file',f);
const r=await fetch('/analyze',{method:'POST',body:fd});const j=await r.json();
document.getElementById('v').textContent=(j.verdict_ar||j.message_ar||'')+(j.fake_probability!==undefined?'  —  '+j.fake_probability+'%':'')
+'   ('+((performance.now()-t)/1000).toFixed(1)+' ث)';
document.getElementById('out').textContent=JSON.stringify(j,null,2);}
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def test_page():
    return TEST_PAGE


@app.get("/health")
def health():
    return {"status": "ok", "model_version": detector.version}


@app.post("/analyze")
def analyze(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED:
        save_analysis(status="error", error_code="unsupported_format", file_type=ext.lstrip(".") or None)
        return error("unsupported_format", "صيغة الملف غير مدعومة", 415)

    data = file.file.read(MAX_MB * 1024 * 1024 + 1)
    size_kb = len(data) // 1024
    if len(data) > MAX_MB * 1024 * 1024:
        save_analysis(status="error", error_code="file_too_large", file_type=ext[1:], file_size_kb=size_kb)
        return error("file_too_large", f"حجم الملف أكبر من {MAX_MB} ميجابايت", 413)
    if not data:
        save_analysis(status="error", error_code="invalid_file", file_type=ext[1:], file_size_kb=0)
        return error("invalid_file", "الملف فارغ", 400)

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(data)
        path = tmp.name
    t0 = time.time()
    try:
        result = detector.analyze(path)
        secs = round(time.time() - t0, 2)
        print(f"[analyze] {file.filename}: {result['verdict']} {result['fake_probability']}% in {secs}s", flush=True)
        save_analysis(status="success", verdict=result["verdict"], fake_probability=result["fake_probability"],
                      audio_quality=result["audio_quality"], has_warning=int(result["warning"] is not None),
                      duration_sec=result["duration_sec"], file_type=ext[1:], file_size_kb=size_kb,
                      processing_sec=secs)
        return result
    except AnalysisError as e:
        save_analysis(status="error", error_code=e.code, file_type=ext[1:], file_size_kb=size_kb,
                      processing_sec=round(time.time() - t0, 2))
        return error(e.code, e.message_ar, e.http_status)
    except Exception:
        save_analysis(status="error", error_code="server_error", file_type=ext[1:], file_size_kb=size_kb,
                      processing_sec=round(time.time() - t0, 2))
        return error("server_error", "حدث خطأ أثناء التحليل، حاول مرة أخرى", 500)
    finally:
        os.remove(path)
