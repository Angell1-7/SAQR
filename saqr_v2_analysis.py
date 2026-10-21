# ===== SAQR v2 — تحليل العتبات وطريقة تجميع النوافذ لموديل xlsr (بدون تدريب) =====
import os, glob, sys, time, zipfile, shutil, subprocess, pickle
import numpy as np, pandas as pd, soundfile as sf, torch, librosa
from datasets import load_from_disk
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
from sklearn.metrics import roc_curve

T0 = time.time(); SR, WIN = 16000, 96000
INP, TMP, OUT = "/kaggle/input", "/kaggle/temp", "/kaggle/working"; os.makedirs(TMP, exist_ok=True)
if not shutil.which("ffmpeg"):
    subprocess.run("apt-get -qq update && apt-get -qq install -y ffmpeg", shell=True)

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

# ---------------------------------------------------------------- 1) الموديل
XDIR = os.path.dirname(glob.glob(f"{INP}/**/saqr_v2_xlsr/config.json", recursive=True)[0])
model = AutoModelForAudioClassification.from_pretrained(XDIR).cuda().eval()
FAKE = int(model.config.label2id["fake"]); REAL = 1 - FAKE
print("model:", XDIR, "| feature extractor do_normalize =", AutoFeatureExtractor.from_pretrained(XDIR).do_normalize)

def norm(x): return ((x - x.mean()) / (x.std() + 1e-7)).astype(np.float32)
def load_wav(p):
    x, sr = sf.read(p, dtype="float32")
    if x.ndim > 1: x = x.mean(1)
    return x if sr == SR else librosa.resample(x, orig_sr=sr, target_sr=SR)
def win_logits(ws):
    out = np.zeros(len(ws), np.float32); by = {}
    for i, w in enumerate(ws): by.setdefault(len(w), []).append(i)
    with torch.no_grad():
        for L, idx in by.items():
            for j in range(0, len(idx), 16):
                ii = idx[j:j + 16]
                iv = torch.from_numpy(np.stack([norm(ws[k]) for k in ii])).cuda()
                with torch.autocast("cuda", dtype=torch.float16):
                    lg = model(iv).logits.float().cpu().numpy()
                out[ii] = lg[:, FAKE] - lg[:, REAL]
    return out
def clip_W(a):                                   # مصفوفة: الإزاحات × النوافذ
    sp = core.speech_only(a)
    if len(sp) < SR: sp = a
    nW = len(core.window_starts(len(sp)))
    ws = [w for s in core.TTA_SHIFTS for w in core.make_windows(sp, s)]
    return win_logits(ws).reshape(len(core.TTA_SHIFTS), nW)

sig = lambda z: 1 / (1 + np.exp(-z))
AGGS = ["mean_p", "mean_z", "top25_z", "max_z"]
def agg(W):
    zw = W.mean(0); k = max(1, int(np.ceil(len(zw) * 0.25)))
    return {"mean_p": float(np.mean(sig(W))),            # الطريقة الحالية
            "mean_z": float(W.mean()),                   # متوسط بدون تشبع
            "top25_z": float(np.sort(zw)[-k:].mean()),   # أعلى ربع النوافذ (للمدموج)
            "max_z": float(zw.max())}                    # أعلى نافذة

# ---------------------------------------------------------------- 2) التحقق (لضبط العتبات)
val_ds = load_from_disk(find_dir("data_val")).with_format("numpy")
meta_p = glob.glob(f"{INP}/**/vc_train/meta.csv", recursive=True)[0]; VCD = os.path.dirname(meta_p)
meta = pd.read_csv(meta_p); meta["target_speaker"] = meta.target_speaker.astype(str)
import random
spk = sorted(meta.target_speaker.unique()); random.Random(0).shuffle(spk)
vc_va = meta[meta.target_speaker.isin(set(spk[:max(1, len(spk) // 10)]))]
val_rows = []
for i, l in enumerate(val_ds["label"]):
    val_rows.append({**agg(clip_W(np.asarray(val_ds[i]["input_values"], np.float32))), "label": int(int(l) == FAKE)})
for r in vc_va.itertuples():
    val_rows.append({**agg(clip_W(load_wav(f"{VCD}/{r.clone}"))), "label": 1})
    val_rows.append({**agg(clip_W(load_wav(f"{VCD}/{r.real}"))), "label": 0})
V = pd.DataFrame(val_rows)
print(f"val scored: {len(V)}  ({(time.time() - T0) / 60:.0f} min)")

# ---------------------------------------------------------------- 3) الاختبار الشامل + المستنسخ + المكالمات الحقيقية
BAT, BATC = find_dir("battery"), find_dir("battery_cloned")
fix = lambda p: str(p).replace("/content/drive/MyDrive/saqr_battery", BAT).replace("/content/drive/MyDrive/cloned", BATC)
man = pd.read_csv(f"{BAT}/manifest.csv", dtype={"id": str}); man["src"] = "battery"
cln = pd.read_csv(f"{BAT}/scores_clones_v11_tta.csv", dtype={"id": str}).drop(columns="p", errors="ignore")
cln["src"] = "clone"; cln["label"] = 1; cln["group"] = "kNN-VC"
parts = [man, cln]
ext_p = f"{BAT}/scores_extra_v11_tta.csv"
if os.path.exists(ext_p):
    ext = pd.read_csv(ext_p, dtype=str).drop(columns="p", errors="ignore")
    print("extras columns:", list(ext.columns), "| rows:", len(ext))
    ext["src"] = "extra"; ext["label"] = ext["label"].astype(int) if "label" in ext else 0
    if "group" not in ext: ext["group"] = ext["cond"] if "cond" in ext else "extra"
    if "cond" not in ext: ext["cond"] = "extra"
    parts.append(ext)
else:
    print("⚠️ no extras file (InfoBay calls) in battery")
B = pd.concat(parts, ignore_index=True); B["path"] = B.path.map(fix)
B = B[B.path.map(os.path.exists)].reset_index(drop=True)
rows, WS = [], []
for i, r in B.iterrows():
    try:
        W = clip_W(core.decode_audio(r.path)); rows.append(agg(W)); WS.append(W)
    except Exception as e:
        rows.append({a: np.nan for a in AGGS}); WS.append(None); print("skip", r.path, e)
    if i % 500 == 0: print(f"  {i}/{len(B)}  ({(time.time() - T0) / 60:.0f} min)")
B = pd.concat([B, pd.DataFrame(rows)], axis=1)
pickle.dump({"val": V, "battery": B, "windows": WS}, open(f"{OUT}/xlsr_window_scores.pkl", "wb"))

# ---------------------------------------------------------------- 4) مقارنة الطرق
def eer(y, s):
    fpr, tpr, _ = roc_curve(y, s); return float(fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]) * 100
core_b = B[(B.src == "battery") & (B.cond != "partial")]
part, sada = B[(B.src == "battery") & (B.cond == "partial")], B[(B.src == "battery") & (B.cond == "clean") & (B.group == "SADA-real")]
clo, ext_b = B[B.src == "clone"], B[B.src == "extra"]
res = []
for a in AGGS:
    yv, sv = V.label.values, V[a].values; R, F = sv[yv == 0], sv[yv == 1]
    for pol in ["old", "real95", "real90"]:
        th = np.quantile(R, 0.99)
        tl = min(np.quantile(F, 0.05), th) if pol == "old" else np.quantile(R, 0.95 if pol == "real95" else 0.90)
        r_, f_ = core_b[core_b.label == 0][a], core_b[core_b.label == 1][a]
        rn = core_b[(core_b.label == 0) & (core_b.cond == "noise")][a]
        rc = core_b[(core_b.label == 0) & core_b.cond.str.startswith("call")][a]
        v = lambda s: "F" if s > th else ("R" if s < tl else "S")
        ev = "".join(v(s) for s in ext_b[a]) if len(ext_b) else "-"
        res.append({"agg": a, "policy": pol, "EER": round(eer(core_b.label, core_b[a]), 2),
                    "EER_partial": round(eer(np.r_[np.ones(len(part)), np.zeros(len(sada))], np.r_[part[a], sada[a]]), 2),
                    "real→FAKE%": round(np.mean(r_ > th) * 100, 1), "real→SUSP%": round(np.mean((r_ >= tl) & (r_ <= th)) * 100, 1),
                    "noise real flagged%": round(np.mean(rn >= tl) * 100, 1), "call real flagged%": round(np.mean(rc >= tl) * 100, 1),
                    "fake flagged%": round(np.mean(f_ >= tl) * 100, 1), "partial flagged%": round(np.mean(part[a] >= tl) * 100, 1),
                    "clones flagged%": round(np.mean(clo[a] >= tl) * 100, 1), "real calls": ev})
pd.set_option("display.width", 250)
print("\n================ xlsr: aggregation × threshold policy ================")
print(pd.DataFrame(res).to_string(index=False))
if len(ext_b): print("\nreal calls order:", list(ext_b.group.astype(str) + ":" + ext_b.path.map(os.path.basename)))

# ---------------------------------------------------------------- 5) سرعة الـ CPU (للـ API)
torch.set_num_threads(4); x = torch.randn(1, WIN)
for name, d in [("xlsr", XDIR), ("base(v1.1 size)", find_dir("model_v11"))]:
    m = AutoModelForAudioClassification.from_pretrained(d).eval()
    with torch.no_grad():
        m(x); t = time.time()
        for _ in range(3): m(x)
    s = (time.time() - t) / 3; n30 = len(core.window_starts(30 * SR)) * len(core.TTA_SHIFTS)
    print(f"CPU {name}: {s:.2f} s / window  →  30 s clip ≈ {s * n30:.0f} s with TTA ({n30} windows)")
print(f"\n✅ done in {(time.time() - T0) / 60:.0f} min — scores saved to {OUT}/xlsr_window_scores.pkl")
