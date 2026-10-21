# ===== SAQR v2 — تدريب + معايرة + تقييم على الاختبار الشامل (Kaggle) =====
VARIANT = "base"      # "base" = يكمل من v1.1   |   "xlsr" = XLS-R 300M متعدد اللغات
SMOKE   = False       # True = تجربة سريعة (5-10 دقايق) للتأكد إن الكود سليم   |   False = التدريب الكامل

import os, glob, json, random, zipfile, time, shutil, subprocess, sys
import numpy as np, pandas as pd, soundfile as sf, torch, librosa
from datasets import load_from_disk
from transformers import (AutoModelForAudioClassification, AutoFeatureExtractor, AutoConfig,
                          TrainingArguments, Trainer, EarlyStoppingCallback)
from scipy.signal import resample_poly, butter, sosfilt
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_curve

T0 = time.time()
SR, WIN = 16000, 96000
INP, TMP = "/kaggle/input", "/kaggle/temp"; os.makedirs(TMP, exist_ok=True)
OUT = f"/kaggle/working/saqr_v2_{VARIANT}"; os.makedirs(OUT, exist_ok=True)
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

# ---------------------------------------------------------------- 1) البيانات
train_ds = load_from_disk(find_dir("data_train")).with_format("numpy")
val_ds   = load_from_disk(find_dir("data_val")).with_format("numpy")
V11 = find_dir("model_v11")
cfg11 = AutoConfig.from_pretrained(V11)
FAKE = int(cfg11.label2id["fake"]); REAL = 1 - FAKE

meta_p = glob.glob(f"{INP}/**/vc_train/meta.csv", recursive=True)[0]; VCD = os.path.dirname(meta_p)
meta = pd.read_csv(meta_p); meta["target_speaker"] = meta.target_speaker.astype(str)
spk = sorted(meta.target_speaker.unique()); random.Random(0).shuffle(spk)
val_spk = set(spk[:max(1, len(spk) // 10)])                      # 10% من المتحدثين للتحقق
vc_tr, vc_va = meta[~meta.target_speaker.isin(val_spk)], meta[meta.target_speaker.isin(val_spk)]
print(f"v11 train {len(train_ds)} | v11 val {len(val_ds)} | VC pairs train {len(vc_tr)} / val {len(vc_va)}")

lab = lambda l: FAKE if int(l) == FAKE else REAL
tr_items = [("tr", i, lab(l)) for i, l in enumerate(train_ds["label"])]
tr_items += [("wav", f"{VCD}/{r.clone}", FAKE) for r in vc_tr.itertuples()] + \
            [("wav", f"{VCD}/{r.real}", REAL) for r in vc_tr.itertuples()]
va_items = [("va", i, lab(l)) for i, l in enumerate(val_ds["label"])]
n_va_v11 = len(va_items)
va_items += [("wav", f"{VCD}/{r.clone}", FAKE) for r in vc_va.itertuples()] + \
            [("wav", f"{VCD}/{r.real}", REAL) for r in vc_va.itertuples()]
BABBLE = [f"{VCD}/{r.real}" for r in vc_tr.itertuples()]
if SMOKE:
    rr = random.Random(1); rr.shuffle(tr_items); tr_items = tr_items[:300]
    a, b = va_items[:n_va_v11], va_items[n_va_v11:]; rr.shuffle(a); rr.shuffle(b)
    va_items = a[:100] + b[:100]; n_va_v11 = 100
print("train items:", len(tr_items), "| val items:", len(va_items))

# ---------------------------------------------------------------- 2) التشويه العشوائي
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
def lowpass(x, rng):
    return sosfilt(butter(6, rng.uniform(3500, 7000), btype="low", fs=SR, output="sos"), x).astype(np.float32)
def noise(x, rng):
    kind = rng.choice(["pink", "white", "babble"], p=[0.35, 0.15, 0.5]); n = len(x)
    if kind == "white": z = rng.standard_normal(n)
    elif kind == "pink":
        X = np.fft.rfft(rng.standard_normal(n)); f = np.arange(len(X)); f[0] = 1; z = np.fft.irfft(X / np.sqrt(f), n)
    else:
        z = np.zeros(n)
        for _ in range(rng.integers(2, 5)):
            b = np.resize(load_wav(BABBLE[rng.integers(len(BABBLE))]), n); z += b / (np.sqrt(np.mean(b ** 2)) + 1e-9)
    snr = rng.uniform(5, 25)
    z = z / (np.sqrt(np.mean(z ** 2)) + 1e-9) * np.sqrt(np.mean(x ** 2) / 10 ** (snr / 10))
    return (x + z).astype(np.float32)
def augment(x, rng):
    if rng.random() < 0.4: x = noise(x, rng)                 # بيئة: ضجة
    r = rng.random()
    if r < 0.25: x = phone(x, rng)                            # قناة: مكالمة
    elif r < 0.40: x = lowpass(x, rng)                        # قناة: ضغط/نطاق ضيق
    return x

class DS(torch.utils.data.Dataset):
    def __init__(self, items, train): self.items, self.train = items, train
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        kind, ref, y = self.items[i]
        rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,))))   # عشوائية قابلة للإعادة
        if kind == "tr": x = np.asarray(train_ds[int(ref)]["input_values"], np.float32)
        elif kind == "va": x = np.asarray(val_ds[int(ref)]["input_values"], np.float32)
        else: x = load_wav(ref)
        if len(x) > WIN:
            s = rng.integers(0, len(x) - WIN + 1) if self.train else (len(x) - WIN) // 2
            x = x[s:s + WIN]
        if self.train:
            k = int(rng.integers(0, 1600))                    # إزاحة عشوائية حتى 100 ms
            if k: x = np.concatenate([x[k:], np.zeros(k, np.float32)])
            x = augment(x, rng)
        return {"input_values": norm(x), "labels": y}

def collate(b):
    L = max(len(z["input_values"]) for z in b); iv = np.zeros((len(b), L), np.float32)
    for i, z in enumerate(b): iv[i, :len(z["input_values"])] = z["input_values"]
    return {"input_values": torch.from_numpy(iv), "labels": torch.tensor([z["labels"] for z in b])}

# ---------------------------------------------------------------- 3) الموديل والتدريب
if VARIANT == "base":
    model = AutoModelForAudioClassification.from_pretrained(V11)
    fe = AutoFeatureExtractor.from_pretrained(V11); LR, EPOCHS = 1e-5, 3
else:
    model = AutoModelForAudioClassification.from_pretrained(
        "facebook/wav2vec2-xls-r-300m", num_labels=2,
        label2id={k: int(v) for k, v in cfg11.label2id.items()}, id2label={int(k): v for k, v in cfg11.id2label.items()})
    fe = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-xls-r-300m"); LR, EPOCHS = 2e-5, 4
model.freeze_feature_encoder()
if SMOKE: EPOCHS = 1

def eer_of(y, s):
    if len(np.unique(y)) < 2: return float("nan")              # لازم الصنفين موجودين
    fpr, tpr, _ = roc_curve(y, s); return float(fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))])
def metrics(p):
    lg = p.predictions.astype(np.float32); z = lg[:, FAKE] - lg[:, REAL]; y = (p.label_ids == FAKE).astype(int)
    return {"eer": eer_of(y, z), "eer_v11val": eer_of(y[:n_va_v11], z[:n_va_v11]),
            "eer_vcval": eer_of(y[n_va_v11:], z[n_va_v11:])}

args = TrainingArguments(
    output_dir=f"{TMP}/ckpt_{VARIANT}", seed=42, per_device_train_batch_size=4, per_device_eval_batch_size=8,
    learning_rate=LR, warmup_steps=10 if SMOKE else 300, num_train_epochs=EPOCHS, fp16=True,
    eval_strategy="epoch", save_strategy="epoch", save_total_limit=1, load_best_model_at_end=True,
    metric_for_best_model="eer", greater_is_better=False, logging_steps=100, report_to="none",
    remove_unused_columns=False, dataloader_num_workers=2)
trainer = Trainer(model=model, args=args, train_dataset=DS(tr_items, True), eval_dataset=DS(va_items, False),
                  data_collator=collate, compute_metrics=metrics,
                  callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])
trainer.train()
trainer.save_model(OUT); fe.save_pretrained(OUT)
print(f"✅ trained in {(time.time() - T0) / 60:.0f} min")

# ---------------------------------------------------------------- 4) المعايرة (على التحقق فقط)
model = trainer.model.eval(); DEV = next(model.parameters()).device
sig = lambda z: 1 / (1 + np.exp(-z))
def logits_batch(ws):
    out = np.zeros(len(ws)); by = {}
    for i, w in enumerate(ws): by.setdefault(len(w), []).append(i)
    with torch.no_grad():
        for L, idx in by.items():
            for j in range(0, len(idx), 16):
                ii = idx[j:j + 16]
                iv = torch.from_numpy(np.stack([norm(ws[k]) for k in ii])).to(DEV)
                with torch.autocast("cuda", dtype=torch.float16):
                    lg = model(iv).logits.float().cpu().numpy()
                out[ii] = lg[:, FAKE] - lg[:, REAL]
    return out
def score_audio(a):
    sp = core.speech_only(a)
    if len(sp) < SR: sp = a
    ws = [w for s in core.TTA_SHIFTS for w in core.make_windows(sp, s)]
    return float(np.mean(sig(logits_batch(ws))))

pred = trainer.predict(DS(va_items, False)); lg = pred.predictions.astype(np.float32)
zv, yv = lg[:, FAKE] - lg[:, REAL], (pred.label_ids == FAKE).astype(int)
def nll(t):
    p = np.clip(sig(zv / np.exp(t)), 1e-7, 1 - 1e-7); return -np.mean(yv * np.log(p) + (1 - yv) * np.log(1 - p))
T = float(np.exp(minimize_scalar(nll, bounds=(-3, 4), method="bounded").x))
pv = np.array([score_audio(DS(va_items, False)[i]["input_values"]) for i in range(len(va_items))])
T_HIGH = float(np.quantile(pv[yv == 0], 0.99))               # حقيقي يُتهم ≤ 1% على التحقق
T_LOW = float(min(np.quantile(pv[yv == 1], 0.05), T_HIGH))   # مزيف يفلت ≤ 5% على التحقق
json.dump({"T_LOW": T_LOW, "T_HIGH": T_HIGH}, open(f"{OUT}/thresholds.json", "w"), indent=2)
json.dump({"T": T, "val_eer": eer_of(yv, zv), "n_val": int(len(yv))}, open(f"{OUT}/temperature.json", "w"), indent=2)
print(f"T={T:.3f} | T_LOW={T_LOW:.4f} | T_HIGH={T_HIGH:.4f} | val EER {eer_of(yv, zv) * 100:.2f}% "
      f"(v11-val {eer_of(yv[:n_va_v11], zv[:n_va_v11]) * 100:.2f}% | VC-val {eer_of(yv[n_va_v11:], zv[n_va_v11:]) * 100:.2f}%)")

# ---------------------------------------------------------------- 5) الاختبار الشامل
BAT, BATC = find_dir("battery"), find_dir("battery_cloned")
fix = lambda p: p.replace("/content/drive/MyDrive/saqr_battery", BAT).replace("/content/drive/MyDrive/cloned", BATC)
man = pd.read_csv(f"{BAT}/manifest.csv", dtype={"id": str})
cln = pd.read_csv(f"{BAT}/scores_clones_v11_tta.csv", dtype={"id": str}).drop(columns="p")
for d in (man, cln): d["path"] = d.path.map(fix)
if SMOKE: man = man.groupby("cond").head(10); cln = cln.groupby("cond").head(5)
ps, shift_ps = [], {}
for i, r in man.iterrows():
    a = core.decode_audio(r.path); ps.append(score_audio(a))
    if r.cond == "clean": shift_ps[r.id] = score_audio(a[104:])
man["p"] = ps; cln["p"] = [score_audio(core.decode_audio(p)) for p in cln.path]
man.to_csv(f"{OUT}/battery_scores.csv", index=False); cln.to_csv(f"{OUT}/battery_clones.csv", index=False)

base = pd.read_csv(f"{BAT}/scores_v11_tta.csv", dtype={"id": str})
base_c = pd.read_csv(f"{BAT}/scores_clones_v11_tta.csv", dtype={"id": str})
pct = lambda x: f"{x * 100:.1f}%"
def eer_pct(g):
    if g.label.nunique() < 2: return float("nan")
    return eer_of(g.label.values, g.p.values) * 100
def cond_df(df, cond):
    g = df[df.cond == cond]
    if cond == "partial": g = pd.concat([g, df[(df.cond == "clean") & (df.group == "SADA-real")]])
    return g

print(f"\n================ REPORT: v2_{VARIANT}  (T_LOW={T_LOW:.3f}, T_HIGH={T_HIGH:.3f}) vs v1.1 ================")
print(f"{'condition':11}{'EER v1.1':>10}{'EER v2':>9}{'real→FAKE':>11}{'real→SUSP':>11}{'fake flagged':>14}{'fake→REAL':>11}")
for cond in ["clean", "whatsapp", "mp3_low", "video_aac", "noise", "call_g711", "call_gsm", "call_voip", "partial"]:
    g = cond_df(man, cond)
    if not len(g): continue
    R, F = g.p[g.label == 0].values, g.p[g.label == 1].values
    print(f"{cond:11}{eer_pct(cond_df(base, cond)):>9.2f}%{eer_pct(g):>8.2f}%{pct(np.mean(R > T_HIGH)):>11}"
          f"{pct(np.mean((R >= T_LOW) & (R <= T_HIGH))):>11}{pct(np.mean(F >= T_LOW)):>14}{pct(np.mean(F < T_LOW)):>11}")
v = lambda p: 2 if p > T_HIGH else (0 if p < T_LOW else 1)
cl = man[man.cond == "clean"]
print(f"\nshift stability (clean, 6.5 ms): verdict changed in {np.mean([v(r.p) != v(shift_ps[r.id]) for r in cl.itertuples()]) * 100:.1f}%")

print("\nCLONES (kNN-VC, test speakers) — flagged (fake or suspicious):   v1.1  →  v2")
for cond in ["clean", "whatsapp", "noise", "call_g711", "call_gsm", "call_voip"]:
    a, b = base_c[base_c.cond == cond].p, cln[cln.cond == cond].p
    if len(b): print(f"   {cond:10} {pct(np.mean(a >= 0.355)):>7}  →  {pct(np.mean(b >= T_LOW)):>7}   (missed {pct(np.mean(b < T_LOW))})")

print("\nPer group — real: % judged REAL | fake: % FLAGGED")
tab = {}
for (grp, cond), g in man[man.cond != "partial"].groupby(["group", "cond"]):
    tab.setdefault(grp, {})[cond] = np.mean(g.p < T_LOW) if g.label.iloc[0] == 0 else np.mean(g.p >= T_LOW)
cols = [c for c in ["clean", "whatsapp", "mp3_low", "video_aac", "noise", "call_g711", "call_gsm", "call_voip"] if c in man.cond.unique()]
print((pd.DataFrame(tab).T[cols] * 100).round(0).to_string())
shutil.rmtree(f"{TMP}/ckpt_{VARIANT}", ignore_errors=True)
print(f"\n✅ all done in {(time.time() - T0) / 60:.0f} min — model + thresholds + temperature + scores in {OUT}")
