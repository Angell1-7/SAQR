# ===== SAQR — استخراج مكالمات حقيقية (متصلين على الهواء) من SADA test واختبار الموديلات عليها =====
import os, glob, sys, time, zipfile, shutil, subprocess
import numpy as np, pandas as pd, soundfile as sf, torch, librosa
from transformers import AutoModelForAudioClassification

T0 = time.time(); SR = 16000; MAX_CLIPS = 300
INP, TMP, OUT = "/kaggle/input", "/kaggle/temp", "/kaggle/working"; os.makedirs(TMP, exist_ok=True)
CALLS = f"{OUT}/sada_calls"; os.makedirs(CALLS, exist_ok=True)
if not shutil.which("ffmpeg"): subprocess.run("apt-get -qq update && apt-get -qq install -y ffmpeg", shell=True)
def find_dir(name):
    for p in glob.glob(f"{INP}/**/{name}", recursive=True):
        if os.path.isdir(p): return p
    z = glob.glob(f"{INP}/**/{name}.zip", recursive=True)
    if not z: raise FileNotFoundError(name)
    dst = f"{TMP}/{name}"
    if not os.path.exists(dst): zipfile.ZipFile(z[0]).extractall(dst)
    return os.path.join(dst, name) if os.path.isdir(os.path.join(dst, name)) else dst
sys.path.insert(0, os.path.dirname(glob.glob(f"{INP}/**/saqr_core.py", recursive=True)[0]))
import saqr_core as core

# ---------------------------------------------------------------- 1) قراءة SADA test
csv = [p for p in glob.glob(f"{INP}/**/test.csv", recursive=True) if "sada" in p.lower()][0]
ROOT = os.path.dirname(csv); df = pd.read_csv(csv)
print(csv, "| segments:", len(df)); print("columns:", list(df.columns))
col = lambda *keys: next(c for c in df.columns if any(k in c.lower() for k in keys))
FILE, START, END = col("filename", "file"), col("segmentstart", "start"), col("segmentend", "end")
for c in [c for c in df.columns if any(k in c.lower() for k in ["environment", "category", "show"])][:3]:
    print(f"\n{c}:", df[c].value_counts().head(12).to_dict())
idx = {}
for dp, _, fs in os.walk(ROOT):
    for f in fs:
        if f.lower().endswith((".wav", ".flac", ".mp3")): idx[f] = os.path.join(dp, f)
df["dur"] = df[END] - df[START]
cand = df[(df.dur >= 3) & (df.dur <= 20)].sample(frac=1, random_state=0).reset_index(drop=True)
print(f"\naudio files found: {len(idx)} | candidate segments (3-20 s): {len(cand)}")

# ---------------------------------------------------------------- 2) كشف المكالمات (نفس كاشف الـ API: ما فيه طاقة فوق 4200 هرتز)
found, checked = [], 0
for r in cand.itertuples():
    p = idx.get(os.path.basename(str(getattr(r, FILE))))
    if p is None: continue
    try:
        sr = sf.info(p).samplerate
        x, _ = sf.read(p, start=int(getattr(r, START) * sr), stop=int(getattr(r, END) * sr), dtype="float32")
    except Exception: continue
    if x.ndim > 1: x = x.mean(1)
    if sr != SR: x = librosa.resample(x, orig_sr=sr, target_sr=SR)
    checked += 1
    sp = core.speech_only(x)
    if len(sp) < 2 * SR: continue
    h = core.hf_ratio(sp)
    if h < core.CALL_HF_RATIO:
        name = f"call_{len(found):03d}.wav"; sf.write(f"{CALLS}/{name}", x, SR)
        found.append({"file": name, "src": os.path.basename(p), "start": getattr(r, START), "dur": round(r.dur, 1), "hf": h,
                      **{c: getattr(r, c) for c in ["Speaker", "Environment", "ShowName"] if c in df.columns}})
    if checked % 1000 == 0: print(f"  checked {checked} → calls {len(found)}  ({(time.time() - T0) / 60:.0f} min)")
    if len(found) >= MAX_CLIPS or time.time() - T0 > 50 * 60: break
C = pd.DataFrame(found); C.to_csv(f"{CALLS}/calls.csv", index=False)
print(f"\nchecked {checked} segments → {len(C)} call-quality segments ({len(C) / max(checked, 1) * 100:.1f}%)")
if len(C) == 0: raise SystemExit("ما لقينا مقاطع بجودة مكالمة")
for c in ["Environment", "ShowName"]:
    if c in C: print(f"{c} of detected calls:", C[c].value_counts().head(10).to_dict())
print("different episodes:", C.src.nunique())

# ---------------------------------------------------------------- 3) اختبار الموديلات الثلاثة على المكالمات الحقيقية
def norm(x): return ((x - x.mean()) / (x.std() + 1e-7)).astype(np.float32)
sig = lambda z: 1 / (1 + np.exp(-z))
MODELS = {"v1.1": find_dir("model_v11"),
          "v2_base": os.path.dirname(glob.glob(f"{INP}/**/saqr_v2_base/config.json", recursive=True)[0]),
          "v2_xlsr": os.path.dirname(glob.glob(f"{INP}/**/saqr_v2_xlsr/config.json", recursive=True)[0])}
AUD = [core.decode_audio(f"{CALLS}/{f}") for f in C.file]
rows = []
for name, d in MODELS.items():
    m = AutoModelForAudioClassification.from_pretrained(d).cuda().eval()
    FK = int(m.config.label2id["fake"]); RL = 1 - FK
    try: t = pd.read_json(f"{d}/thresholds.json", typ="series"); tl, th = float(t.T_LOW), float(t.T_HIGH)
    except Exception: tl, th = 0.355, 0.999
    P, Z = [], []
    for a in AUD:
        sp = core.speech_only(a)
        if len(sp) < SR: sp = a
        ws = [w for s in core.TTA_SHIFTS for w in core.make_windows(sp, s)]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            lg = m(torch.from_numpy(np.stack([norm(w) for w in ws])).cuda()).logits.float().cpu().numpy()
        z = lg[:, FK] - lg[:, RL]; P.append(float(np.mean(sig(z)))); Z.append(float(z.mean()))
    C[f"{name}_p"], C[f"{name}_z"] = P, Z; P = np.array(P)
    rows.append({"model": name, "thresholds": f"{tl:.3f} / {th:.3f}", "judged REAL": f"{np.mean(P < tl) * 100:.0f}%",
                 "SUSPICIOUS": f"{np.mean((P >= tl) & (P <= th)) * 100:.0f}%", "FAKE (wrong)": f"{np.mean(P > th) * 100:.0f}%"})
    del m; torch.cuda.empty_cache()
C.to_csv(f"{CALLS}/calls.csv", index=False)
print(f"\n================ REAL CALLS from SADA radio/TV ({len(C)} segments, {C.src.nunique()} episodes) ================")
print(pd.DataFrame(rows).to_string(index=False))
print(f"\n✅ done in {(time.time() - T0) / 60:.0f} min — اسمعي 5 مقاطع من Output/sada_calls وتأكدي إنها مكالمات فعلًا")
