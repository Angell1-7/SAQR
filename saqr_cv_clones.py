# ===== SAQR v2.1 — بيانات "صوت الجوال" (نسخة 2): أصوات حقيقية من Common Voice + نسخ مستنسخة منها بنفس الأجهزة =====
import os, glob, tarfile, subprocess, shutil, time, random
import numpy as np, pandas as pd, torch, soundfile as sf
from huggingface_hub import HfApi, hf_hub_download
T0 = time.time(); TMP, OUT = "/kaggle/temp", "/kaggle/working"; os.makedirs(TMP, exist_ok=True)
sh = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True)
if not shutil.which("ffmpeg"): sh("apt-get -qq update && apt-get -qq install -y ffmpeg")
REPO = "fsicoli/common_voice_16_0"
files = HfApi().list_repo_files(REPO, repo_type="dataset")
print("arabic tsv:", [f for f in files if f.startswith("transcript/ar/")])
tsv = lambda s: pd.read_csv(hf_hub_download(REPO, f"transcript/ar/{s}.tsv", repo_type="dataset"), sep="\t", quoting=3)
HELD_OUT = set(tsv("test").client_id) | set(tsv("dev").client_id)          # متحدثين الاختبار والتحقق ما يدخلون التدريب
knn = torch.hub.load("bshall/knn-vc", "knn_vc", prematched=True, trust_repo=True, pretrained=True, device="cuda")

def build(name, splits, n_pairs, n_targets, seed, exclude=()):
    out = f"{OUT}/cv_{name}"; os.makedirs(out, exist_ok=True); raw = f"{TMP}/cv_{name}"; os.makedirs(raw, exist_ok=True)
    dfs = []
    for s in splits:
        tars = [f for f in files if f.startswith(f"audio/ar/{s}/") and f.endswith(".tar")]
        for f in tars: tarfile.open(hf_hub_download(REPO, f, repo_type="dataset")).extractall(raw)
        if tars: dfs.append(tsv(s))
    mp3 = {os.path.basename(p): p for p in glob.glob(f"{raw}/**/*.mp3", recursive=True)}
    df = pd.concat(dfs).drop_duplicates("path"); df = df[df.path.isin(mp3) & ~df.client_id.isin(exclude)].reset_index(drop=True)
    print(f"[{name}] clips {len(df)} | speakers {df.client_id.nunique()}")
    wav = {}
    def to16(n):                                          # mp3 → wav 16k، ويرجع None لو الملف فاضي أو قصير
        if n not in wav:
            o = f"{raw}/{n[:-4]}.wav"; sh(f'ffmpeg -nostdin -v error -y -i "{mp3[n]}" -ac 1 -ar 16000 "{o}"')
            wav[n] = o if os.path.exists(o) and sf.info(o).duration > 1.0 else None
        return wav[n]
    cnt = df.client_id.value_counts(); rng = random.Random(seed)
    targets = list(cnt[cnt >= 5].index); rng.shuffle(targets); targets = targets[:n_targets]
    srcs = df.sample(frac=1, random_state=seed)
    srcs = srcs.groupby("client_id").head(max(15, n_pairs // max(1, df.client_id.nunique()) + 1)).reset_index(drop=True)
    per_t = max(1, n_pairs // max(1, len(targets))); meta, si = [], 0
    for t in targets:
        refs = [w for w in (to16(p) for p in df[df.client_id == t].path.head(12)) if w]
        if len(refs) < 3: continue
        try: ms = knn.get_matching_set(refs)
        except Exception as e: print("skip target", repr(e)[:100]); continue
        made, tries = 0, 0
        while made < per_t and tries < per_t * 3 and si < len(srcs):
            r = srcs.iloc[si]; si += 1; tries += 1
            if r.client_id == t: continue                         # المصدر لازم يكون شخص غير الهدف
            src = to16(r.path)
            if not src: continue
            try:
                y = knn.match(knn.get_features(src), ms, topk=4).cpu().numpy()
                x, _ = sf.read(src); y = y * (np.sqrt(np.mean(x ** 2)) / (np.sqrt(np.mean(y ** 2)) + 1e-9))
                k = len(meta); sf.write(f"{out}/clone_{k:04d}.wav", y.astype(np.float32), 16000); shutil.copy(src, f"{out}/real_{k:04d}.wav")
                meta.append({"id": k, "clone": f"clone_{k:04d}.wav", "real": f"real_{k:04d}.wav",
                             "src_speaker": r.client_id[:12], "target_speaker": t[:12], "dur": len(x) / 16000}); made += 1
            except Exception as e: print("skip", repr(e)[:100])
        if si >= len(srcs): srcs = srcs.sample(frac=1, random_state=si); si = 0      # نعيد الدورة لو خلصت المصادر
    pd.DataFrame(meta).to_csv(f"{out}/meta.csv", index=False); shutil.rmtree(raw, ignore_errors=True)
    print(f"[{name}] {len(meta)} pairs | {len(set(m['target_speaker'] for m in meta))} target voices | "
          f"{len(set(m['src_speaker'] for m in meta))} source speakers ({(time.time() - T0) / 60:.0f} min)")

build("train", ["train", "other"], n_pairs=2000, n_targets=150, seed=0, exclude=HELD_OUT)
build("dev", ["dev"], n_pairs=300, n_targets=20, seed=1)
print(f"\n✅ done in {(time.time() - T0) / 60:.0f} min — cv_train / cv_dev في Output")
