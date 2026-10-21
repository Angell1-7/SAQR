# ===== SAQR — اختبار الاستنساخ بأداة ما شافها الموديل (OpenVoice V2) =====
# المصدر (الكلام) والهدف (الصوت) كلها أصوات حقيقية من الاختبار الشامل (متحدثين ما تدرب عليهم الموديل)
import os, glob, sys, time, zipfile, shutil, subprocess, random, pickle, importlib
import numpy as np, pandas as pd, soundfile as sf, torch, librosa
from transformers import AutoModelForAudioClassification
from sklearn.metrics import roc_curve

T0 = time.time(); SR, WIN, N_CLONES = 16000, 96000, 100
INP, TMP, OUT = "/kaggle/input", "/kaggle/temp", "/kaggle/working"; os.makedirs(TMP, exist_ok=True)
CL = f"{OUT}/openvoice_clones"; os.makedirs(CL, exist_ok=True)
sh = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True)
if not shutil.which("ffmpeg"): sh("apt-get -qq update && apt-get -qq install -y ffmpeg")

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

# ---------------------------------------------------------------- 1) تركيب OpenVoice (بدون تخريب مكتبات Kaggle)
if not os.path.exists(f"{TMP}/OpenVoice"):
    print(sh(f"git clone -q https://github.com/myshell-ai/OpenVoice.git {TMP}/OpenVoice").stderr[-500:])
sys.path.insert(0, f"{TMP}/OpenVoice")
for _ in range(12):                                   # يركب أي مكتبة ناقصة وحدة وحدة
    try:
        from openvoice.api import ToneColorConverter; break
    except ModuleNotFoundError as e:
        print("installing", e.name); sh(f"pip install -q {e.name}")
        importlib.invalidate_caches()
from huggingface_hub import hf_hub_download
try:
    cfg = hf_hub_download("myshell-ai/OpenVoiceV2", "converter/config.json")
    ckpt = hf_hub_download("myshell-ai/OpenVoiceV2", "converter/checkpoint.pth")
except Exception as e:
    print("HF failed, trying S3:", e)
    sh(f"cd {TMP} && wget -q https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip && unzip -q -o checkpoints_v2_0417.zip")
    cfg, ckpt = f"{TMP}/checkpoints_v2/converter/config.json", f"{TMP}/checkpoints_v2/converter/checkpoint.pth"
try:
    tcc = ToneColorConverter(cfg, device="cuda", enable_watermark=False)
except TypeError:
    sh("pip install -q wavmark"); tcc = ToneColorConverter(cfg, device="cuda")
tcc.load_ckpt(ckpt); tcc.watermark_model = None       # بدون علامة مائية (المهاجم الحقيقي يقدر يشيلها)

# ---------------------------------------------------------------- 2) توليد 100 صوت مستنسخ
BAT = find_dir("battery")
fix = lambda p: str(p).replace("/content/drive/MyDrive/saqr_battery", BAT)
man = pd.read_csv(f"{BAT}/manifest.csv", dtype={"id": str}); man["path"] = man.path.map(fix)
pool = man[(man.cond == "clean") & (man.label == 0) & man.group.isin(["SADA-real", "ArFake-real"])].reset_index(drop=True)
print("real pool:", len(pool), pool.group.value_counts().to_dict())
rng = random.Random(7); pairs = []
for i in range(N_CLONES):
    s = i % len(pool); t = rng.randrange(len(pool))
    while t == s: t = rng.randrange(len(pool))
    pairs.append((s, t))
se = {}
def get_se(k):
    if k not in se: se[k] = tcc.extract_se([pool.path[k]])
    return se[k]
clones = []
for i, (s, t) in enumerate(pairs):
    out = f"{CL}/ov_{i:03d}.wav"
    try:
        tcc.convert(audio_src_path=pool.path[s], src_se=get_se(s), tgt_se=get_se(t), output_path=out)
        clones.append(out)
    except Exception as e: print("skip", i, e)
print(f"clones made: {len(clones)}  ({(time.time() - T0) / 60:.0f} min)")
reals = sorted(set(pool.path[s] for s, _ in pairs))

# ---------------------------------------------------------------- 3) نفس الظروف على الحقيقي والمستنسخ
def ff(src, args, ext):
    o = f"{TMP}/c.{ext}"; r = sh(f'ffmpeg -nostdin -v error -y -i "{src}" {args} "{o}"')
    if r.returncode: raise RuntimeError(r.stderr[-200:])
    return core.decode_audio(o)
def pink(x, seed):
    g = np.random.default_rng(seed); X = np.fft.rfft(g.standard_normal(len(x))); f = np.arange(len(X)); f[0] = 1
    z = np.fft.irfft(X / np.sqrt(f), len(x)); z = z / (np.sqrt(np.mean(z ** 2)) + 1e-9) * np.sqrt(np.mean(x ** 2) / 10)
    return (x + z).astype(np.float32)
def gsm(p):
    try: return ff(p, "-ar 8000 -ac 1 -c:a libgsm -f gsm", "gsm")
    except Exception: return ff(p, "-ar 8000 -ac 1 -c:a libopencore_amrnb -b:a 12.2k", "amr")   # بديل لو GSM مو موجود
CONDS = {
    "clean":     lambda p, k: core.decode_audio(p),
    "whatsapp":  lambda p, k: ff(p, "-ac 1 -ar 16000 -c:a libopus -b:a 16k", "ogg"),
    "noise":     lambda p, k: pink(core.decode_audio(p), k),
    "call_g711": lambda p, k: ff(p, "-af highpass=f=300,lowpass=f=3400 -ar 8000 -ac 1 -c:a pcm_mulaw", "wav"),
    "call_gsm":  lambda p, k: gsm(p),
    "call_voip": lambda p, k: ff(p, "-af lowpass=f=3400 -ar 8000 -ac 1 -c:a libopus -b:a 12k", "ogg"),
}
AUD = []                                               # (cond, label, اسم, الصوت)
for c, fn in CONDS.items():
    for k, p in enumerate(reals):  AUD.append((c, 0, os.path.basename(p), fn(p, k)))
    for k, p in enumerate(clones): AUD.append((c, 1, os.path.basename(p), fn(p, 1000 + k)))
print(f"audio prepared: {len(AUD)}  ({(time.time() - T0) / 60:.0f} min)")
del tcc; torch.cuda.empty_cache()

# ---------------------------------------------------------------- 4) التقييم بالموديلات الثلاثة
def norm(x): return ((x - x.mean()) / (x.std() + 1e-7)).astype(np.float32)
sig = lambda z: 1 / (1 + np.exp(-z))
MODELS = {"v1.1": find_dir("model_v11"),
          "v2_base": os.path.dirname(glob.glob(f"{INP}/**/saqr_v2_base/config.json", recursive=True)[0]),
          "v2_xlsr": os.path.dirname(glob.glob(f"{INP}/**/saqr_v2_xlsr/config.json", recursive=True)[0])}
TH = {}
for name, d in MODELS.items():
    m = AutoModelForAudioClassification.from_pretrained(d).cuda().eval()
    FK = int(m.config.label2id["fake"]); RL = 1 - FK
    try: t = pd.read_json(f"{d}/thresholds.json", typ="series"); TH[name] = (float(t.T_LOW), float(t.T_HIGH))
    except Exception: TH[name] = (0.355, 0.999)
    def W_of(a):
        sp = core.speech_only(a)
        if len(sp) < SR: sp = a
        ws = [w for s in core.TTA_SHIFTS for w in core.make_windows(sp, s)]
        out = np.zeros(len(ws), np.float32); by = {}
        for i, w in enumerate(ws): by.setdefault(len(w), []).append(i)
        with torch.no_grad():
            for L, idx in by.items():
                for j in range(0, len(idx), 16):
                    ii = idx[j:j + 16]; iv = torch.from_numpy(np.stack([norm(ws[q]) for q in ii])).cuda()
                    with torch.autocast("cuda", dtype=torch.float16): lg = m(iv).logits.float().cpu().numpy()
                    out[ii] = lg[:, FK] - lg[:, RL]
        return out.reshape(len(core.TTA_SHIFTS), -1)
    Ws = [W_of(a) for _, _, _, a in AUD]
    globals()[f"W_{name.replace('.', '')}"] = Ws
    print(f"scored with {name}  thresholds={TH[name]}  ({(time.time() - T0) / 60:.0f} min)")
    del m; torch.cuda.empty_cache()
D = pd.DataFrame([{"cond": c, "label": y, "file": f} for c, y, f, _ in AUD])
for name in MODELS:
    Ws = globals()[f"W_{name.replace('.', '')}"]
    D[f"{name}_p"] = [float(np.mean(sig(W))) for W in Ws]; D[f"{name}_z"] = [float(W.mean()) for W in Ws]
D.to_csv(f"{OUT}/openvoice_scores.csv", index=False)
pickle.dump({n: globals()[f"W_{n.replace('.', '')}"] for n in MODELS} | {"meta": D[["cond", "label", "file"]]}, open(f"{OUT}/openvoice_windows.pkl", "wb"))

def eer(y, s):
    fpr, tpr, _ = roc_curve(y, s); return float(fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]) * 100
rows = []
for c in CONDS:
    g = D[D.cond == c]; r = {"condition": c}
    for name in MODELS:
        tl, th = TH[name]; P = g[f"{name}_p"]
        r[f"EER {name}"] = round(eer(g.label, g[f"{name}_z"]), 1)
        r[f"clone flagged {name}"] = f"{np.mean(P[g.label == 1] >= tl) * 100:.0f}%"
        r[f"real flagged {name}"] = f"{np.mean(P[g.label == 0] >= tl) * 100:.0f}%"
    rows.append(r)
pd.set_option("display.width", 250)
R = pd.DataFrame(rows)
print(f"\n================ UNSEEN TOOL: OpenVoice V2  ({len(clones)} clones vs {len(reals)} real) ================")
print(R[["condition"] + [f"EER {n}" for n in MODELS]].to_string(index=False))
print("\nflagged (fake or suspicious) with each model's own thresholds:")
print(R[["condition"] + [f"clone flagged {n}" for n in MODELS] + [f"real flagged {n}" for n in MODELS]].to_string(index=False))
print(f"\n✅ done in {(time.time() - T0) / 60:.0f} min — clones in {CL} (اسمعي كم واحد من Output)")
