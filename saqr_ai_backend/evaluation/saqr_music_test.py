# ===== SAQR — اختبار الموسيقى الخلفية على v2.1 (CPU، حوالي 30 دقيقة) =====
import os, glob, json, tarfile, zipfile, subprocess, shutil, time
import numpy as np, pandas as pd, librosa, torch
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
from huggingface_hub import HfApi, hf_hub_download
from sklearn.metrics import roc_curve
T0 = time.time(); SR, WIN, N = 16000, 96000, 60; INP, TMP = "/kaggle/input", "/kaggle/temp"; os.makedirs(TMP, exist_ok=True)
torch.set_num_threads(os.cpu_count())
sh = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True)
if not shutil.which("ffmpeg"): sh("apt-get -qq update && apt-get -qq install -y ffmpeg")
def find_dir(name):
    for p in glob.glob(f"{INP}/**/{name}", recursive=True):
        if os.path.isdir(p): return p
    z = glob.glob(f"{INP}/**/{name}.zip", recursive=True)
    if not z: raise FileNotFoundError(f"{name} — تأكدي من الـ Input")
    dst = f"{TMP}/{name}"
    if not os.path.exists(dst): zipfile.ZipFile(z[0]).extractall(dst)
    return os.path.join(dst, name) if os.path.isdir(os.path.join(dst, name)) else dst

# ---------------------------------------------------------------- 1) الموديل v2.1 + حدوده
M = find_dir("saqr_v21_final"); th = json.load(open(f"{M}/thresholds.json")); TL, TH = th["T_LOW"], th["T_HIGH"]
model = AutoModelForAudioClassification.from_pretrained(M).eval(); fe = AutoFeatureExtractor.from_pretrained(M)
FAKE = int(model.config.label2id["fake"]); REAL = 1 - FAKE

def load16(p):
    sr = int(sh(f'ffprobe -v error -select_streams a:0 -show_entries stream=sample_rate -of csv=p=0 "{p}"').stdout.strip() or 16000)
    raw = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", p, "-vn", "-ac", "1", "-f", "f32le", "pipe:1"], capture_output=True).stdout
    a = np.frombuffer(raw, np.float32).copy(); return librosa.resample(a, orig_sr=sr, target_sr=SR, res_type="soxr_hq") if sr != SR else a
def speech_only(a):
    iv = librosa.effects.split(a, top_db=30); return np.concatenate([a[s:e] for s, e in iv]) if len(iv) else a[:0]
def windows(sp):
    n = len(sp); L = min(WIN, n); st = [0] if n <= WIN else list(range(0, n - WIN + 1, WIN))
    if n > WIN and st[-1] != n - WIN: st.append(n - WIN)
    return [np.pad(sp[s + f:s + f + L], (0, L - len(sp[s + f:s + f + L]))) for f in (0, 52) for s in st]
def score(a):                                                  # نفس طريقة الـ API
    sp = speech_only(a); sp = (sp if len(sp) >= SR else a)[:60 * SR]
    with torch.no_grad():
        lg = model(fe(windows(sp), sampling_rate=SR, return_tensors="pt").input_values).logits.numpy()
    return float(np.mean(lg[:, FAKE] - lg[:, REAL]))

# ---------------------------------------------------------------- 2) المقاطع: حقيقي (Common Voice test، جوالات) + مستنسخ (Chatterbox، أداة ما تدرب عليها)
REPO = "fsicoli/common_voice_16_0"; cvdir = f"{TMP}/cvtest"; os.makedirs(cvdir, exist_ok=True)
for f in [f for f in HfApi().list_repo_files(REPO, repo_type="dataset") if f.startswith("audio/ar/test/") and f.endswith(".tar")]:
    tarfile.open(hf_hub_download(REPO, f, repo_type="dataset")).extractall(cvdir, filter="data")
cvt = pd.read_csv(hf_hub_download(REPO, "transcript/ar/test.tsv", repo_type="dataset"), sep="\t", quoting=3)
mp3 = {os.path.basename(p): p for p in glob.glob(f"{cvdir}/**/*.mp3", recursive=True)}
reals = [mp3[p] for p in cvt[cvt.path.isin(mp3)].sample(frac=1, random_state=7).groupby("client_id").head(1).head(N).path]
CB = os.path.dirname(glob.glob(f"{INP}/**/chatterbox_clones/meta.csv", recursive=True)[0])
fakes = list(np.random.default_rng(7).permutation(sorted(glob.glob(f"{CB}/cb_*.wav")))[:N])

# ---------------------------------------------------------------- 3) الموسيقى (مقاطع موسيقى مرخصة من مكتبة librosa) والضجة
MUS = []
for k in ["brahms", "choice", "fishin", "nutcracker", "pistachio", "sweetwaltz", "trumpet", "vibeace"]:
    try: MUS.append(librosa.load(librosa.example(k), sr=SR)[0])
    except Exception as e: print("music skip", k, repr(e)[:80])
print(f"music tracks: {len(MUS)} | reals {len(reals)} | fakes {len(fakes)}")
def mix(x, bg, snr, seed):                                      # snr: الكلام أعلى من الخلفية بكم dB
    g = np.random.default_rng(seed)
    if len(bg) < len(x): bg = np.tile(bg, len(x) // len(bg) + 1)
    s = g.integers(0, len(bg) - len(x) + 1); b = bg[s:s + len(x)]
    b = b / (np.sqrt(np.mean(b ** 2)) + 1e-9) * np.sqrt(np.mean(x ** 2)) / 10 ** (snr / 20)
    return (x + b).astype(np.float32)
def pink(n, seed):
    g = np.random.default_rng(seed); X = np.fft.rfft(g.standard_normal(n)); f = np.arange(len(X)); f[0] = 1
    return np.fft.irfft(X / np.sqrt(f), n).astype(np.float32)
CONDS = {"clean": lambda x, k: x,
         "noise 10dB (للمقارنة)": lambda x, k: mix(x, pink(len(x), k), 10, k),
         "music 10dB (خلفية)": lambda x, k: mix(x, MUS[k % len(MUS)], 10, k),
         "music 0dB (عالية)": lambda x, k: mix(x, MUS[k % len(MUS)], 0, k)}

# ---------------------------------------------------------------- 4) التقييم
import soundfile as sf; SMP = "/kaggle/working/music_samples"; os.makedirs(SMP, exist_ok=True)
rows = []
for y, files in [(0, reals), (1, fakes)]:
    for k, p in enumerate(files):
        x = load16(p)
        for j, (c, fn) in enumerate(CONDS.items()):
            a = fn(x, k); s = score(a)
            rows.append({"label": y, "cond": c, "file": os.path.basename(p), "score": s})
            if k < 5:                                           # نحفظ 5 حقيقي + 5 مزيف بكل حالة عشان تسمعينها
                v = "FAKE" if s > TH else ("real" if s < TL else "SUSP")
                sf.write(f"{SMP}/{'fake' if y else 'real'}{k}_{j}_{c.split()[0]}{c.split()[1] if len(c.split()) > 1 else ''}_{v}.wav", a / max(1.0, np.abs(a).max()), SR)
    print(f"  {'fake' if y else 'real'} done ({(time.time() - T0) / 60:.0f} min)", flush=True)
D = pd.DataFrame(rows); D.to_csv("/kaggle/working/music_scores.csv", index=False)
out = []
for c in CONDS:
    g = D[D.cond == c]; R, F = g[g.label == 0].score, g[g.label == 1].score
    fpr, tpr, _ = roc_curve(g.label, g.score)
    out.append({"الحالة": c, "حقيقي انشك فيه": f"{np.mean(R >= TL):.0%}", "مزيف انكشف (مشكوك+مزيف)": f"{np.mean(F >= TL):.0%}",
                "مزيف = مزيف": f"{np.mean(F > TH):.0%}", "EER": f"{fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]:.1%}"})
pd.set_option("display.width", 250)
print("\n================ v2.1 — موسيقى خلفية ================"); print(pd.DataFrame(out).to_string(index=False))
print(f"\n✅ done in {(time.time() - T0) / 60:.0f} min")
