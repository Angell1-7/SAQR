# ===== SAQR — فرز جودة أصوات OpenVoice تلقائيًا (Whisper) وإعادة الحساب على المفهوم منها فقط =====
# الفكرة: نفرّغ كلام المقطع الأصلي وكلام النسخة المستنسخة، ولو النص اختلف كثير → النسخة مخربة/غير مفهومة
import os, glob, re, sys, zipfile, time
import numpy as np, pandas as pd, torch, librosa
from transformers import pipeline
from sklearn.metrics import roc_curve

T0 = time.time(); INP, TMP = "/kaggle/input", "/kaggle/temp"; os.makedirs(TMP, exist_ok=True)
def find_dir(name):
    for p in glob.glob(f"{INP}/**/{name}", recursive=True):
        if os.path.isdir(p): return p
    z = glob.glob(f"{INP}/**/{name}.zip", recursive=True)
    if not z: raise FileNotFoundError(name)
    dst = f"{TMP}/{name}"
    if not os.path.exists(dst): zipfile.ZipFile(z[0]).extractall(dst)
    return os.path.join(dst, name) if os.path.isdir(os.path.join(dst, name)) else dst

# ---------------------------------------------------------------- 1) نفس أزواج المصدر/الهدف اللي استخدمها اختبار OpenVoice
BAT = find_dir("battery")
man = pd.read_csv(f"{BAT}/manifest.csv", dtype={"id": str})
man["path"] = man.path.map(lambda p: str(p).replace("/content/drive/MyDrive/saqr_battery", BAT))
pool = man[(man.cond == "clean") & (man.label == 0) & man.group.isin(["SADA-real", "ArFake-real"])].reset_index(drop=True)
CLD = os.path.dirname(glob.glob(f"{INP}/**/openvoice_clones/ov_000.wav", recursive=True)[0])
SC = pd.read_csv(glob.glob(f"{INP}/**/openvoice_scores.csv", recursive=True)[0])
pairs = pd.DataFrame({"clone": [f"ov_{i:03d}.wav" for i in range(100)]})
pairs["src"] = [pool.path[i % len(pool)] for i in range(100)]
pairs["src_group"] = [pool.group[i % len(pool)] for i in range(100)]
pairs = pairs[pairs.clone.map(lambda f: os.path.exists(f"{CLD}/{f}"))].reset_index(drop=True)

# ---------------------------------------------------------------- 2) التفريغ بـ Whisper
asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo",
               torch_dtype=torch.float16, device="cuda")
def text(p):
    a, _ = librosa.load(p, sr=16000)
    return asr({"raw": a, "sampling_rate": 16000}, generate_kwargs={"language": "arabic", "task": "transcribe"})["text"]
def clean(s):
    s = re.sub(r"[ً-ْـ]", "", s)                     # التشكيل والتطويل
    s = re.sub(r"[إأآا]", "ا", s); s = s.replace("ة", "ه").replace("ى", "ي")
    return re.sub(r"[^ء-ي ]", "", s).split()
def cer(ref, hyp):
    r, h = "".join(ref), "".join(hyp)
    if not r: return 1.0
    d = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        prev, d[0] = d.copy(), i
        for j in range(1, len(h) + 1):
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (r[i - 1] != h[j - 1]))
    return min(1.0, d[len(h)] / len(r))
pairs["src_text"] = [text(p) for p in pairs.src]
pairs["clone_text"] = [text(f"{CLD}/{f}") for f in pairs.clone]
pairs["CER"] = [cer(clean(a), clean(b)) for a, b in zip(pairs.src_text, pairs.clone_text)]
pairs.to_csv("/kaggle/working/openvoice_quality.csv", index=False)
print(f"transcribed {len(pairs)} pairs ({(time.time() - T0) / 60:.0f} min)")
print("CER distribution:", pairs.CER.describe()[["min", "25%", "50%", "75%", "max"]].round(2).to_dict())
for lo, hi in [(0, .2), (.2, .35), (.35, .5), (.5, .75), (.75, 1.01)]:
    g = pairs[(pairs.CER >= lo) & (pairs.CER < hi)]
    print(f"  CER {lo:.2f}-{hi:.2f}: {len(g):3d} clones   e.g. {list(g.clone[:4])}")

# ---------------------------------------------------------------- 3) إعادة الحساب على النسخ المفهومة فقط
def eer(y, s):
    fpr, tpr, _ = roc_curve(y, s); return float(fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]) * 100
MODELS = [m[:-2] for m in SC.columns if m.endswith("_z")]
for cut in [0.35, 0.5]:
    good = set(pairs.clone[pairs.CER < cut])
    print(f"\n================ OpenVoice — only intelligible clones (CER < {cut}): {len(good)} clones vs all reals ================")
    rows = []
    for c in SC.cond.unique():
        g = SC[(SC.cond == c) & ((SC.label == 0) | SC.file.isin(good))]; r = {"condition": c}
        for m in MODELS: r[f"EER {m}"] = round(eer(g.label, g[f"{m}_z"]), 1)
        rows.append(r)
    print(pd.DataFrame(rows).to_string(index=False))
print("\nby source group (CER < 0.35):", pairs[pairs.CER < .35].src_group.value_counts().to_dict(),
      "| all:", pairs.src_group.value_counts().to_dict())
print(f"\n✅ done in {(time.time() - T0) / 60:.0f} min — التفاصيل في openvoice_quality.csv")
