# ===== SAQR v2.1 — يكمل تدريب v2_xlsr مع "صوت الجوال" (Common Voice) + تقييم شامل مقابل v2 =====
SMOKE = True          # True = تجربة سريعة (~15 دقيقة) للتأكد إن الكود سليم   |   False = التدريب الكامل

import os, glob, json, random, zipfile, tarfile, time, shutil, subprocess, sys, re
import numpy as np, pandas as pd, soundfile as sf, torch, librosa
from datasets import load_from_disk
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor, TrainingArguments, Trainer
from scipy.signal import resample_poly, butter, sosfilt, fftconvolve
from sklearn.metrics import roc_curve
from sklearn.isotonic import IsotonicRegression
from huggingface_hub import HfApi, hf_hub_download

T0 = time.time(); SR, WIN = 16000, 96000
INP, TMP, OUT = "/kaggle/input", "/kaggle/temp", "/kaggle/working/saqr_v21_final"
os.makedirs(TMP, exist_ok=True); os.makedirs(OUT, exist_ok=True)
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
mdir = lambda pat: os.path.dirname(glob.glob(f"{INP}/**/{pat}", recursive=True)[0])

# ---------------------------------------------------------------- 1) البيانات
V2 = mdir("saqr_v2_final/thresholds.json")                       # نبدأ من الموديل المعتمد v2
train_ds = load_from_disk(find_dir("data_train")).with_format("numpy")
val_ds = load_from_disk(find_dir("data_val")).with_format("numpy")
model = AutoModelForAudioClassification.from_pretrained(V2); fe = AutoFeatureExtractor.from_pretrained(V2)
FAKE = int(model.config.label2id["fake"]); REAL = 1 - FAKE
VCD, CVT, CVD = mdir("vc_train/meta.csv"), mdir("cv_train/meta.csv"), mdir("cv_dev/meta.csv")
vc = pd.read_csv(f"{VCD}/meta.csv"); vc["target_speaker"] = vc.target_speaker.astype(str)
spk = sorted(vc.target_speaker.unique()); random.Random(0).shuffle(spk); val_spk = set(spk[:max(1, len(spk) // 10)])
vc_tr, vc_va = vc[~vc.target_speaker.isin(val_spk)], vc[vc.target_speaker.isin(val_spk)]   # نفس تقسيم v2
cv_tr, cv_dv = pd.read_csv(f"{CVT}/meta.csv"), pd.read_csv(f"{CVD}/meta.csv")
lab = lambda l: FAKE if int(l) == FAKE else REAL
pairs = lambda df, d: [("wav", f"{d}/{r.clone}", FAKE) for r in df.itertuples()] + [("wav", f"{d}/{r.real}", REAL) for r in df.itertuples()]
tr_items = [("tr", i, lab(l)) for i, l in enumerate(train_ds["label"])] + pairs(vc_tr, VCD) + pairs(cv_tr, CVT)
va_items = [("va", i, lab(l)) for i, l in enumerate(val_ds["label"])] + pairs(vc_va, VCD) + pairs(cv_dv, CVD)
n_v11 = len(val_ds); n_vc = 2 * len(vc_va)
BABBLE = [f"{VCD}/{r.real}" for r in vc_tr.itertuples()] + [f"{CVT}/{r.real}" for r in cv_tr.itertuples()]
if SMOKE:
    rr = random.Random(1); rr.shuffle(tr_items); tr_items = tr_items[:400]
    va_items = va_items[:60] + va_items[n_v11:n_v11 + 40] + va_items[n_v11 + n_vc:n_v11 + n_vc + 60]; n_v11, n_vc = 60, 40
print(f"train {len(tr_items)} (CV pairs {len(cv_tr)}) | val {len(va_items)} (CV-dev pairs {len(cv_dv)})")

# ---------------------------------------------------------------- 2) التشويه: نفس v2 + جديد: صدى الغرفة ومايك الجوال (للصنفين)
SOS8 = butter(4, [300, 3400], btype="band", fs=8000, output="sos")
def load_wav(p):
    x, sr = sf.read(p, dtype="float32")
    if x.ndim > 1: x = x.mean(1)
    return x if sr == SR else librosa.resample(x, orig_sr=sr, target_sr=SR)
def norm(x): return ((x - x.mean()) / (x.std() + 1e-7)).astype(np.float32)
def phone(x, rng):
    y = x / (np.abs(x).max() + 1e-9) * 0.9
    y = np.clip(sosfilt(SOS8, resample_poly(y, 1, 2)), -1, 1); mu = 255
    q = np.sign(y) * np.log1p(mu * np.abs(y)) / np.log1p(mu); q = np.round((q + 1) / 2 * mu) / mu * 2 - 1
    y = resample_poly(np.sign(q) * ((1 + mu) ** np.abs(q) - 1) / mu, 2, 1)[:len(x)]
    return np.pad(y, (0, len(x) - len(y))).astype(np.float32)
def lowpass(x, rng): return sosfilt(butter(6, rng.uniform(3500, 7000), btype="low", fs=SR, output="sos"), x).astype(np.float32)
def noise(x, rng):
    kind = rng.choice(["pink", "white", "babble"], p=[0.35, 0.15, 0.5]); n = len(x)
    if kind == "white": z = rng.standard_normal(n)
    elif kind == "pink":
        X = np.fft.rfft(rng.standard_normal(n)); f = np.arange(len(X)); f[0] = 1; z = np.fft.irfft(X / np.sqrt(f), n)
    else:
        z = np.zeros(n)
        for _ in range(rng.integers(2, 5)):
            b = np.resize(load_wav(BABBLE[rng.integers(len(BABBLE))]), n); z += b / (np.sqrt(np.mean(b ** 2)) + 1e-9)
    z = z / (np.sqrt(np.mean(z ** 2)) + 1e-9) * np.sqrt(np.mean(x ** 2) / 10 ** (rng.uniform(5, 25) / 10))
    return (x + z).astype(np.float32)
def room(x, rng):                                             # صدى غرفة صناعي
    rt = rng.uniform(0.15, 0.7); t = np.arange(int(rt * SR)) / SR
    ir = rng.standard_normal(len(t)) * np.exp(-6.9 * t / rt); ir[0] = 1.0
    y = fftconvolve(x, ir)[:len(x)]; return (y / (np.abs(y).max() + 1e-9) * (np.abs(x).max() + 1e-9)).astype(np.float32)
def mic(x, rng):                                              # مايك رخيص: قص ترددات منخفضة + ميل في الترددات
    y = sosfilt(butter(2, rng.uniform(80, 350), btype="high", fs=SR, output="sos"), x)
    if rng.random() < 0.5: y = y + rng.uniform(-0.4, 0.4) * sosfilt(butter(2, [2000, 6000], btype="band", fs=SR, output="sos"), y)
    return y.astype(np.float32)
def augment(x, rng):
    if rng.random() < 0.25: x = room(x, rng)
    if rng.random() < 0.30: x = mic(x, rng)
    if rng.random() < 0.40: x = noise(x, rng)
    r = rng.random()
    if r < 0.25: x = phone(x, rng)
    elif r < 0.40: x = lowpass(x, rng)
    return x
class DS(torch.utils.data.Dataset):
    def __init__(self, items, train): self.items, self.train = items, train
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        kind, ref, y = self.items[i]; rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,))))
        x = np.asarray((train_ds if kind == "tr" else val_ds)[int(ref)]["input_values"], np.float32) if kind in ("tr", "va") else load_wav(ref)
        if len(x) > WIN:
            s = rng.integers(0, len(x) - WIN + 1) if self.train else (len(x) - WIN) // 2; x = x[s:s + WIN]
        if self.train:
            k = int(rng.integers(0, 1600))
            if k: x = np.concatenate([x[k:], np.zeros(k, np.float32)])
            x = augment(x, rng)
        return {"input_values": norm(x), "labels": y}
def collate(b):
    L = max(len(z["input_values"]) for z in b); iv = np.zeros((len(b), L), np.float32)
    for i, z in enumerate(b): iv[i, :len(z["input_values"])] = z["input_values"]
    return {"input_values": torch.from_numpy(iv), "labels": torch.tensor([z["labels"] for z in b])}

# ---------------------------------------------------------------- 3) التدريب (يكمل من v2، تعلّم هادي عشان ما ينسى)
def eer_of(y, s):
    if len(np.unique(y)) < 2: return float("nan")
    fpr, tpr, _ = roc_curve(y, s); return float(fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))])
def metrics(p):
    lg = p.predictions.astype(np.float32); z = lg[:, FAKE] - lg[:, REAL]; y = (p.label_ids == FAKE).astype(int)
    a, b = n_v11, n_v11 + n_vc
    return {"eer": eer_of(y, z), "eer_v11val": eer_of(y[:a], z[:a]), "eer_vcval": eer_of(y[a:b], z[a:b]), "eer_cvdev": eer_of(y[b:], z[b:])}
model.freeze_feature_encoder()
args = TrainingArguments(output_dir=f"{TMP}/ckpt", seed=42, per_device_train_batch_size=4, per_device_eval_batch_size=8,
    learning_rate=1e-5, warmup_steps=10 if SMOKE else 200, num_train_epochs=1 if SMOKE else 2, fp16=True,
    eval_strategy="epoch", save_strategy="epoch", save_total_limit=1, load_best_model_at_end=True,
    metric_for_best_model="eer", greater_is_better=False, logging_steps=100, report_to="none",
    remove_unused_columns=False, dataloader_num_workers=2)
trainer = Trainer(model=model, args=args, train_dataset=DS(tr_items, True), eval_dataset=DS(va_items, False),
                  data_collator=collate, compute_metrics=metrics)
trainer.train(); trainer.save_model(OUT); fe.save_pretrained(OUT); shutil.rmtree(f"{TMP}/ckpt", ignore_errors=True)
print(f"✅ trained in {(time.time() - T0) / 60:.0f} min")

# ---------------------------------------------------------------- 4) التحليل بنفس إعدادات الـ API (إزاحتين، بدون تداخل، أول 60 ثانية، متوسط الـ logit)
def speech_only(a):
    iv = librosa.effects.split(a, top_db=30); return np.concatenate([a[s:e] for s, e in iv]) if len(iv) else a[:0]
def windows(sp):
    n = len(sp); L = min(WIN, n); st = [0] if n <= WIN else list(range(0, n - WIN + 1, WIN))
    if n > WIN and st[-1] != n - WIN: st.append(n - WIN)
    return [np.pad(sp[s + f:s + f + L], (0, L - len(sp[s + f:s + f + L]))) for f in (0, 52) for s in st]
def scorer(m):
    m = m.cuda().eval()
    def score(a):
        sp = speech_only(a); sp = (sp if len(sp) >= SR else a)[:60 * SR]; ws = windows(sp); z = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for i in range(0, len(ws), 16):
                lg = m(torch.from_numpy(np.stack([norm(w) for w in ws[i:i + 16]])).cuda()).logits.float().cpu().numpy()
                z += list(lg[:, FAKE] - lg[:, REAL])
        return float(np.mean(z))
    return score
S21 = scorer(trainer.model)
va_audio = [np.asarray(val_ds[int(r)]["input_values"], np.float32) if k == "va" else load_wav(r) for k, r, _ in va_items]
yv = np.array([y == FAKE for _, _, y in va_items]).astype(int); zv = np.array([S21(a) for a in va_audio])
R = zv[yv == 0]; T_LOW, T_HIGH = float(np.quantile(R, 0.90)), float(np.quantile(R, 0.99))
json.dump({"T_LOW": T_LOW, "T_HIGH": T_HIGH, "score": "mean_logit", "tta_shifts": [0, 52], "overlap": False,
           "max_speech_sec": 60, "cap_fake_on_call": False, "policy": "real90 on validation (incl. Common Voice dev)"},
          open(f"{OUT}/thresholds.json", "w"), indent=2)
iso = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99).fit(zv, yv)
x = np.unique(np.r_[np.quantile(zv, np.linspace(0, 1, 101)), T_LOW, T_HIGH])
json.dump({"x": x.round(4).tolist(), "y": iso.predict(x).round(4).tolist()}, open(f"{OUT}/calibration.json", "w"))
print(f"v2.1 thresholds: T_LOW={T_LOW:.3f} T_HIGH={T_HIGH:.3f} | val EER {eer_of(yv, zv) * 100:.2f}%")

# ---------------------------------------------------------------- 5) التقييم الشامل: v2 مقابل v2.1 على نفس الملفات
def load16(p):
    sr = int(sh(f'ffprobe -v error -select_streams a:0 -show_entries stream=sample_rate -of csv=p=0 "{p}"').stdout.strip() or 16000)
    raw = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", p, "-vn", "-ac", "1", "-f", "f32le", "pipe:1"], capture_output=True).stdout
    a = np.frombuffer(raw, np.float32).copy(); return librosa.resample(a, orig_sr=sr, target_sr=SR, res_type="soxr_hq") if sr != SR else a
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
CONDS = {"clean": lambda p, k: load16(p), "whatsapp": lambda p, k: ff(p, "-ac 1 -ar 16000 -c:a libopus -b:a 16k", "ogg"),
         "noise": lambda p, k: pink(load16(p), k),
         "call_g711": lambda p, k: ff(p, "-af highpass=f=300,lowpass=f=3400 -ar 8000 -ac 1 -c:a pcm_mulaw", "wav"),
         "call_gsm": lambda p, k: gsm(p), "call_voip": lambda p, k: ff(p, "-af lowpass=f=3400 -ar 8000 -ac 1 -c:a libopus -b:a 12k", "ogg")}
BAT, BATC = find_dir("battery"), find_dir("battery_cloned")
fix = lambda p: str(p).replace("/content/drive/MyDrive/saqr_battery", BAT).replace("/content/drive/MyDrive/cloned", BATC)
man = pd.read_csv(f"{BAT}/manifest.csv", dtype={"id": str}); man["path"] = man.path.map(fix)
cln = pd.read_csv(f"{BAT}/scores_clones_v11_tta.csv", dtype={"id": str}); cln["path"] = cln.path.map(fix)
pool = man[(man.cond == "clean") & (man.label == 0) & man.group.isin(["SADA-real", "ArFake-real"])].path.tolist()
OV = mdir("openvoice_clones/ov_000.wav"); CB = mdir("chatterbox_clones/meta.csv")
ov, cb = sorted(glob.glob(f"{OV}/ov_*.wav")), sorted(glob.glob(f"{CB}/cb_*.wav"))
if SMOKE: man = man.groupby("cond").head(8); cln = cln.groupby("cond").head(5); pool, ov, cb = pool[:10], ov[:10], cb[:10]
SETS = []                                                     # (المجموعة, الحالة, label, الصوت)
for r in man.itertuples(): SETS.append(("battery-" + r.cond, r.cond, int(r.label), load16(r.path)))
for r in cln.itertuples(): SETS.append(("kNN-clones", r.cond, 1, load16(r.path)))
for c, fn in CONDS.items():
    for k, p in enumerate(pool): SETS.append(("reals(SADA+ArFake)", c, 0, fn(p, k)))
    for k, p in enumerate(ov): SETS.append(("OpenVoice", c, 1, fn(p, 1000 + k)))
    for k, p in enumerate(cb): SETS.append(("Chatterbox:" + os.path.basename(p), c, 1, fn(p, 2000 + k)))
REPO = "fsicoli/common_voice_16_0"; cvdir = f"{TMP}/cvtest"; os.makedirs(cvdir, exist_ok=True)
for f in [f for f in HfApi().list_repo_files(REPO, repo_type="dataset") if f.startswith("audio/ar/test/") and f.endswith(".tar")]:
    tarfile.open(hf_hub_download(REPO, f, repo_type="dataset")).extractall(cvdir)
cvt = pd.read_csv(hf_hub_download(REPO, "transcript/ar/test.tsv", repo_type="dataset"), sep="\t", quoting=3)
mp3 = {os.path.basename(p): p for p in glob.glob(f"{cvdir}/**/*.mp3", recursive=True)}
cvt = cvt[cvt.path.isin(mp3)].sample(frac=1, random_state=0).groupby("client_id").head(2).head(40 if SMOKE else 400)
for p in cvt.path:
    SETS.append(("CommonVoice-test REAL", "clean", 0, load16(mp3[p])))
    SETS.append(("CommonVoice-test REAL", "whatsapp", 0, ff(mp3[p], "-ac 1 -ar 16000 -c:a libopus -b:a 16k", "ogg")))
print(f"eval clips: {len(SETS)} ({(time.time() - T0) / 60:.0f} min)")
D = pd.DataFrame([{"set": s.split(":")[0], "file": (s.split(":") + [""])[1], "cond": c, "label": y} for s, c, y, _ in SETS])
D["v2.1"] = [S21(a) for *_, a in SETS]
TH = {"v2.1": (T_LOW, T_HIGH)}
t2 = json.load(open(f"{V2}/thresholds.json")); TH["v2"] = (t2["T_LOW"], t2["T_HIGH"])
S2 = scorer(AutoModelForAudioClassification.from_pretrained(V2)); D["v2"] = [S2(a) for *_, a in SETS]
D.to_csv(f"/kaggle/working/v21_eval_scores.csv", index=False)

pd.set_option("display.width", 250)
def summary(g):
    r = {}
    for n in ["v2", "v2.1"]:
        tl, th = TH[n]; R, F = g[g.label == 0][n], g[g.label == 1][n]
        if len(R) and len(F):
            fpr, tpr, _ = roc_curve(g.label, g[n]); r[f"EER {n}"] = round(float(fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]) * 100, 1)
        if len(R): r[f"real→SUSP/FAKE {n}"] = f"{np.mean(R >= tl) * 100:.0f}%"
        if len(F): r[f"fake flagged {n}"] = f"{np.mean(F >= tl) * 100:.0f}%"
    return pd.Series(r)
print("\n================ REAL PEOPLE (Common Voice test) — v2 vs v2.1 ================")
print(D[D.set.str.startswith("Common")].groupby("cond").apply(summary).to_string())
print("\n================ BATTERY (same as before) ================")
print(D[D.set.str.startswith("battery")].groupby("cond").apply(summary).to_string())
print("\n================ CLONES (each vs the same reals) ================")
for s in ["kNN-clones", "OpenVoice", "Chatterbox"]:
    reals = D[D.set == "reals(SADA+ArFake)"] if s != "kNN-clones" else D[D.set.str.startswith("battery") & (D.label == 0)]
    g = pd.concat([D[D.set == s], reals])                     # نفس الحالة للحقيقي والمستنسخ
    print(f"\n--- {s}"); print(g.groupby("cond").apply(summary).to_string())
v = lambda x, n: "FAKE" if x > TH[n][1] else ("real" if x < TH[n][0] else "SUSP")
for f in ["cb_036.wav", "cb_038.wav"]:                     # أخطر مقطعين (اللهجة البيضاء)
    g = D[D.file == f]
    if len(g): print(f"\n{f}: " + " | ".join(f"{r.cond}: v2 {v(r['v2'], 'v2')} → v2.1 {v(r['v2.1'], 'v2.1')}" for _, r in g.iterrows()))
print(f"\n✅ all done in {(time.time() - T0) / 60:.0f} min — model in {OUT}")
