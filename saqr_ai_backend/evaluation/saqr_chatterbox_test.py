# ===== SAQR — اختبار v1.1 و v2 النهائي على أصوات Chatterbox (استنساخ حديث من نص) =====
import os, glob, re, zipfile, subprocess, shutil, time
import numpy as np, pandas as pd, torch, librosa, json
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor, pipeline
from sklearn.metrics import roc_curve
T0 = time.time(); SR, WIN = 16000, 96000; INP, TMP = "/kaggle/input", "/kaggle/temp"; os.makedirs(TMP, exist_ok=True)
sh = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True)
if not shutil.which("ffmpeg"): sh("apt-get -qq update && apt-get -qq install -y ffmpeg")
def find_dir(name):
    for p in glob.glob(f"{INP}/**/{name}", recursive=True):
        if os.path.isdir(p): return p
    z = glob.glob(f"{INP}/**/{name}.zip", recursive=True)
    dst = f"{TMP}/{name}"
    if not os.path.exists(dst): zipfile.ZipFile(z[0]).extractall(dst)
    return os.path.join(dst, name) if os.path.isdir(os.path.join(dst, name)) else dst

# ---------------------------------------------------------------- 1) المقاطع: 90 مستنسخ + الحقيقي السعودي من الاختبار الشامل
CB = os.path.dirname(glob.glob(f"{INP}/**/chatterbox_clones/meta.csv", recursive=True)[0])
meta = pd.read_csv(f"{CB}/meta.csv"); meta["path"] = CB + "/" + meta.file
BAT = find_dir("battery")
man = pd.read_csv(f"{BAT}/manifest.csv"); man["path"] = man.path.map(lambda p: str(p).replace("/content/drive/MyDrive/saqr_battery", BAT))
reals = man[(man.cond == "clean") & (man.group == "SADA-real")].path.tolist()
print(f"clones {len(meta)} | Saudi reals {len(reals)}")

# ---------------------------------------------------------------- 2) فحص الجودة: هل المستنسخ قال الجملة فعلًا؟ (Whisper)
asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo", torch_dtype=torch.float16, device="cuda")
def clean(s):
    s = re.sub(r"[ً-ْـ]", "", s); s = re.sub(r"[إأآا]", "ا", s); s = s.replace("ة", "ه").replace("ى", "ي")
    return "".join(re.sub(r"[^ء-ي ]", "", s).split())
def cer(r, h):
    if not r: return 1.0
    d = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        prev, d[0] = d.copy(), i
        for j in range(1, len(h) + 1): d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (r[i - 1] != h[j - 1]))
    return min(1.0, d[len(h)] / len(r))
meta["asr"] = [asr({"raw": librosa.load(p, sr=16000)[0], "sampling_rate": 16000},
                   generate_kwargs={"language": "arabic", "task": "transcribe"})["text"] for p in meta.path]
meta["CER"] = [cer(clean(t), clean(a)) for t, a in zip(meta.text, meta.asr)]
meta["dur"] = [librosa.get_duration(path=p) for p in meta.path]
meta.to_csv("/kaggle/working/chatterbox_quality.csv", index=False)
print("CER:", meta.CER.describe()[["25%", "50%", "75%"]].round(2).to_dict(), "| duration (s):", meta.dur.describe()[["min", "50%", "max"]].round(1).to_dict())
for i in [0, 30, 60]: print(f"  {meta.file[i]}  CER={meta.CER[i]:.2f}\n     text: {meta.text[i]}\n     heard: {meta.asr[i]}")
del asr; torch.cuda.empty_cache()

# ---------------------------------------------------------------- 3) نفس الظروف الست على الحقيقي والمستنسخ
def load16(p):
    sr = int(sh(f'ffprobe -v error -select_streams a:0 -show_entries stream=sample_rate -of csv=p=0 "{p}"').stdout.strip())
    raw = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", p, "-vn", "-ac", "1", "-f", "f32le", "pipe:1"], capture_output=True).stdout
    a = np.frombuffer(raw, np.float32).copy()
    return librosa.resample(a, orig_sr=sr, target_sr=SR, res_type="soxr_hq") if sr != SR else a
def ff(src, args, ext):
    o = f"{TMP}/c.{ext}"; r = sh(f'ffmpeg -nostdin -v error -y -i "{src}" {args} "{o}"')
    if r.returncode: raise RuntimeError(r.stderr[-200:])
    return load16(o)
def pink(x, seed):
    g = np.random.default_rng(seed); X = np.fft.rfft(g.standard_normal(len(x))); f = np.arange(len(X)); f[0] = 1
    z = np.fft.irfft(X / np.sqrt(f), len(x)); z = z / (np.sqrt(np.mean(z ** 2)) + 1e-9) * np.sqrt(np.mean(x ** 2) / 10)
    return (x + z).astype(np.float32)
def gsm(p):
    try: return ff(p, "-ar 8000 -ac 1 -c:a libgsm -f gsm", "gsm")
    except Exception: return ff(p, "-ar 8000 -ac 1 -c:a libopencore_amrnb -b:a 12.2k", "amr")
CONDS = {"clean": lambda p, k: load16(p),
         "whatsapp": lambda p, k: ff(p, "-ac 1 -ar 16000 -c:a libopus -b:a 16k", "ogg"),
         "noise": lambda p, k: pink(load16(p), k),
         "call_g711": lambda p, k: ff(p, "-af highpass=f=300,lowpass=f=3400 -ar 8000 -ac 1 -c:a pcm_mulaw", "wav"),
         "call_gsm": lambda p, k: gsm(p),
         "call_voip": lambda p, k: ff(p, "-af lowpass=f=3400 -ar 8000 -ac 1 -c:a libopus -b:a 12k", "ogg")}
AUD = []
for c, fn in CONDS.items():
    for k, p in enumerate(reals): AUD.append((c, 0, os.path.basename(p), fn(p, k)))
    for k, p in enumerate(meta.path): AUD.append((c, 1, os.path.basename(p), fn(p, 1000 + k)))
print(f"audio prepared: {len(AUD)} ({(time.time() - T0) / 60:.0f} min)")

# ---------------------------------------------------------------- 4) التقييم بنفس إعدادات الـ API لكل موديل
def speech_only(a):
    iv = librosa.effects.split(a, top_db=30); return np.concatenate([a[s:e] for s, e in iv]) if len(iv) else a[:0]
def windows(sp, shifts, overlap):
    n = len(sp); L = min(WIN, n)
    st = [0] if n <= WIN else list(range(0, n - WIN + 1, WIN // 2 if overlap else WIN))
    if n > WIN and st[-1] != n - WIN: st.append(n - WIN)
    out = []
    for sft in shifts:
        for s in st:
            x = sp[s + sft:s + sft + L]; out.append(np.pad(x, (0, L - len(x))))
    return out
MODELS = {"v1.1": find_dir("model_v11"),
          "v2_final": os.path.dirname(glob.glob(f"{INP}/**/saqr_v2_final/thresholds.json", recursive=True)[0])}
D = pd.DataFrame([{"cond": c, "label": y, "file": f} for c, y, f, _ in AUD]); TH = {}
for name, d in MODELS.items():
    th = json.load(open(f"{d}/thresholds.json")) if os.path.exists(f"{d}/thresholds.json") else {"T_LOW": 0.355, "T_HIGH": 0.999}
    TH[name] = th
    score, shifts, ov = th.get("score", "mean_prob"), th.get("tta_shifts", [0, 52, 104, 156]), th.get("overlap", True)
    m = AutoModelForAudioClassification.from_pretrained(d).cuda().eval(); fe = AutoFeatureExtractor.from_pretrained(d)
    FK = int(m.config.label2id["fake"]); RL = 1 - FK; S = []
    for _, _, _, a in AUD:
        sp = speech_only(a); sp = sp if len(sp) >= SR else a
        if th.get("max_speech_sec"): sp = sp[:int(th["max_speech_sec"] * SR)]
        ws = windows(sp, shifts, ov); z = []
        with torch.no_grad():
            for i in range(0, len(ws), 16):
                iv = fe(ws[i:i + 16], sampling_rate=SR, return_tensors="pt").input_values.cuda()
                lg = m(iv).logits.float().cpu().numpy(); z += list(lg[:, FK] - lg[:, RL])
        z = np.array(z); S.append(float(z.mean()) if score == "mean_logit" else float(np.mean(1 / (1 + np.exp(-z)))))
    D[name] = S; del m; torch.cuda.empty_cache(); print(f"scored {name} ({(time.time() - T0) / 60:.0f} min)")
D = D.merge(meta[["file", "CER"]], on="file", how="left"); D.to_csv("/kaggle/working/chatterbox_scores.csv", index=False)

def eer(y, s):
    fpr, tpr, _ = roc_curve(y, s); return float(fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]) * 100
pd.set_option("display.width", 250)
for title, ok in [("ALL clones", D.label == 1), ("INTELLIGIBLE clones only (CER < 0.35)", (D.label == 1) & (D.CER < 0.35))]:
    sub = D[(D.label == 0) | ok]
    rows = []
    for c in CONDS:
        g = sub[sub.cond == c]; r = {"condition": c, "n_clones": int((g.label == 1).sum())}
        for n in MODELS:
            tl, thh = TH[n]["T_LOW"], TH[n]["T_HIGH"]
            r[f"EER {n}"] = round(eer(g.label, g[n]), 1)
            r[f"clone flagged {n}"] = f"{np.mean(g[g.label == 1][n] >= tl) * 100:.0f}%"
            r[f"clone FAKE {n}"] = f"{np.mean(g[g.label == 1][n] > thh) * 100:.0f}%"
            r[f"real flagged {n}"] = f"{np.mean(g[g.label == 0][n] >= tl) * 100:.0f}%"
        rows.append(r)
    print(f"\n================ Chatterbox — {title} ================"); print(pd.DataFrame(rows).to_string(index=False))
print(f"\n✅ done in {(time.time() - T0) / 60:.0f} min")
