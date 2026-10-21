# ===== SAQR — كم مرة يظلم الموديل أصوات ناس عاديين مسجلة من جوالاتهم؟ (Common Voice العربي) =====
import os, glob, json, tarfile, zipfile, subprocess, shutil, time, random
import numpy as np, pandas as pd, torch, librosa
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
T0 = time.time(); SR, WIN = 16000, 96000; INP, TMP, OUT = "/kaggle/input", "/kaggle/temp", "/kaggle/working"
os.makedirs(TMP, exist_ok=True); sh = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True)
if not shutil.which("ffmpeg"): sh("apt-get -qq update && apt-get -qq install -y ffmpeg")
def find_dir(name):
    for p in glob.glob(f"{INP}/**/{name}", recursive=True):
        if os.path.isdir(p): return p
    z = glob.glob(f"{INP}/**/{name}.zip", recursive=True); dst = f"{TMP}/{name}"
    if not os.path.exists(dst): zipfile.ZipFile(z[0]).extractall(dst)
    return os.path.join(dst, name) if os.path.isdir(os.path.join(dst, name)) else dst

# ---------------------------------------------------------------- 1) تنزيل قسم الاختبار العربي من Common Voice (نسخة على Hugging Face، رخصة CC0)
REPO = "fsicoli/common_voice_16_0"
files = HfApi().list_repo_files(REPO, repo_type="dataset")
ar = [f for f in files if "/ar/" in f or f.endswith("/ar.tsv") or "/ar_" in f]
print("arabic files:", [f for f in ar if "test" in f or "dev" in f][:20])
tars = [f for f in ar if f.endswith(".tar") and "test" in f]
tsv = [f for f in ar if f.endswith(".tsv") and "test" in f]
CV = f"{TMP}/cv"; os.makedirs(CV, exist_ok=True)
for f in tars: tarfile.open(hf_hub_download(REPO, f, repo_type="dataset")).extractall(CV)
mp3 = {os.path.basename(p): p for p in glob.glob(f"{CV}/**/*.mp3", recursive=True)}
meta = pd.read_csv(hf_hub_download(REPO, tsv[0], repo_type="dataset"), sep="\t", quoting=3) if tsv else pd.DataFrame({"path": list(mp3)})
meta = meta[meta.path.isin(mp3)]
if "client_id" in meta:                                            # متحدثين مختلفين: 2 مقاطع لكل شخص كحد أقصى
    meta = meta.sample(frac=1, random_state=0).groupby("client_id").head(2)
meta = meta.head(400).reset_index(drop=True); meta["file"] = meta.path.map(mp3.get)
print(f"Common Voice clips: {len(meta)} from {meta.client_id.nunique() if 'client_id' in meta else '?'} speakers ({(time.time() - T0) / 60:.0f} min)")

# ---------------------------------------------------------------- 2) الصوت: الأصلي + نسخة واتساب
def load16(p):
    raw = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", p, "-vn", "-ac", "1", "-ar", "48000", "-f", "f32le", "pipe:1"], capture_output=True).stdout
    return librosa.resample(np.frombuffer(raw, np.float32).copy(), orig_sr=48000, target_sr=SR, res_type="soxr_hq")
def whatsapp(p):
    o = f"{TMP}/w.ogg"; sh(f'ffmpeg -nostdin -v error -y -i "{p}" -ac 1 -ar 16000 -c:a libopus -b:a 16k "{o}"'); return load16(o)
AUD = [("clean", load16(p)) for p in meta.file] + [("whatsapp", whatsapp(p)) for p in meta.file]
meta["dur"] = [len(a) / SR for c, a in AUD[:len(meta)]]

# ---------------------------------------------------------------- 3) التقييم بنفس إعدادات الـ API
def speech_only(a):
    iv = librosa.effects.split(a, top_db=30); return np.concatenate([a[s:e] for s, e in iv]) if len(iv) else a[:0]
def windows(sp, shifts, overlap):
    n = len(sp); L = min(WIN, n); st = [0] if n <= WIN else list(range(0, n - WIN + 1, WIN // 2 if overlap else WIN))
    if n > WIN and st[-1] != n - WIN: st.append(n - WIN)
    return [np.pad(sp[s + f:s + f + L], (0, L - len(sp[s + f:s + f + L]))) for f in shifts for s in st]
MODELS = {"v1.1": find_dir("model_v11"),
          "v2_final": os.path.dirname(glob.glob(f"{INP}/**/saqr_v2_final/thresholds.json", recursive=True)[0])}
rows = []
for name, d in MODELS.items():
    th = json.load(open(f"{d}/thresholds.json")) if os.path.exists(f"{d}/thresholds.json") else {"T_LOW": .355, "T_HIGH": .999}
    m = AutoModelForAudioClassification.from_pretrained(d).cuda().eval(); fe = AutoFeatureExtractor.from_pretrained(d)
    FK = int(m.config.label2id["fake"]); RL = 1 - FK; S = []
    for _, a in AUD:
        sp = speech_only(a); sp = sp if len(sp) >= SR else a
        ws = windows(sp, th.get("tta_shifts", [0, 52, 104, 156]), th.get("overlap", True))
        with torch.no_grad():
            lg = m(fe(ws, sampling_rate=SR, return_tensors="pt").input_values.cuda()).logits.float().cpu().numpy()
        z = lg[:, FK] - lg[:, RL]; S.append(float(z.mean()) if th.get("score") == "mean_logit" else float(np.mean(1 / (1 + np.exp(-z)))))
    S = np.array(S); n = len(meta)
    for cond, s in [("clean", S[:n]), ("whatsapp", S[n:])]:
        meta[f"{name}_{cond}"] = s
        rows.append({"model": name, "cond": cond, "REAL ✅": f"{np.mean(s < th['T_LOW']) * 100:.1f}%",
                     "SUSPICIOUS ⚠️": f"{np.mean((s >= th['T_LOW']) & (s <= th['T_HIGH'])) * 100:.1f}%",
                     "FAKE ❌": f"{np.mean(s > th['T_HIGH']) * 100:.1f}%"})
    if name == "v2_final":
        sus = (S[:n] >= th["T_LOW"])
        meta["dur_bin"] = pd.cut(meta.dur, [0, 3, 5, 8, 100], labels=["<3s", "3-5s", "5-8s", ">8s"])
        print("v2 flagged (clean) by clip length:", meta.assign(f=sus).groupby("dur_bin", observed=True).f.mean().mul(100).round(1).to_dict())
    del m; torch.cuda.empty_cache()
meta.to_csv(f"{OUT}/cv_real_scores.csv", index=False)
pd.set_option("display.width", 200)
print(f"\n================ REAL people, own phones/laptops (Common Voice Arabic, {len(meta)} clips) ================")
print(pd.DataFrame(rows).to_string(index=False))
print(f"\n✅ done in {(time.time() - T0) / 60:.0f} min")
