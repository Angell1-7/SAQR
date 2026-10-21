# ===== SAQR — استنساخ حديث (Chatterbox) من فويسات جوال حقيقية (Common Voice) — للتدريب وللاختبار (نسخة 2) =====
# سيناريو المحتال: ياخذ تسجيل جوال للضحية ويستنسخ صوته يقول أي جملة
import torch; assert torch.cuda.is_available(), "❌ مافيه GPU — Settings ← Accelerator ← GPU T4 x2"
import os, sys, glob, tarfile, subprocess, importlib, random, time, shutil
T0 = time.time(); TMP, OUT = "/kaggle/temp", "/kaggle/working"; os.makedirs(TMP, exist_ok=True)
LIMIT_MIN = 230                                       # يوقف التوليد بعد 230 دقيقة ويحفظ اللي سواه
sh = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True)
if not shutil.which("ffmpeg"): sh("apt-get -qq update && apt-get -qq install -y ffmpeg")

# ---------------------------------------------------------------- 1) تركيب Chatterbox (نفس الطريقة اللي نجحت)
print(sh("pip install -q --no-deps chatterbox-tts").stderr[-300:])
print(sh("pip install -q transformers==4.46.3 diffusers==0.29.0").stderr[-300:])
PIPNAME = {"perth": "resemble-perth", "s3tokenizer": "s3tokenizer", "pkuseg": "spacy-pkuseg", "spacy_pkuseg": "spacy-pkuseg"}
for _ in range(15):
    try:
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS; break
    except ModuleNotFoundError as e:
        name = e.name.split(".")[0]; print("installing", name); sh(f"pip install -q {PIPNAME.get(name, name)}")
        importlib.invalidate_caches()
import torchaudio as ta, numpy as np, pandas as pd, soundfile as sf
from huggingface_hub import HfApi, hf_hub_download
model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
print(f"model loaded ({(time.time() - T0) / 60:.0f} min)")

# ---------------------------------------------------------------- 2) أصوات Common Voice (مرجع الاستنساخ) + جمل متنوعة
REPO = "fsicoli/common_voice_16_0"; files = HfApi().list_repo_files(REPO, repo_type="dataset")
tsv = lambda s: pd.read_csv(hf_hub_download(REPO, f"transcript/ar/{s}.tsv", repo_type="dataset"), sep="\t", quoting=3)
def load_split(splits, tag):
    raw = f"{TMP}/{tag}"; os.makedirs(raw, exist_ok=True); dfs = []
    for s in splits:
        for f in [f for f in files if f.startswith(f"audio/ar/{s}/") and f.endswith(".tar")]:
            tarfile.open(hf_hub_download(REPO, f, repo_type="dataset")).extractall(raw, filter="data")
        dfs.append(tsv(s))
    mp3 = {os.path.basename(p): p for p in glob.glob(f"{raw}/**/*.mp3", recursive=True)}
    df = pd.concat(dfs).drop_duplicates("path"); df = df[df.path.isin(mp3)].reset_index(drop=True)
    df["mp3"] = df.path.map(mp3.get); return df
test_df = load_split(["test"], "cvtest")
HELD = set(test_df.client_id) | set(tsv("dev").client_id)
train_df = load_split(["train", "other"], "cvtrain"); train_df = train_df[~train_df.client_id.isin(HELD)]
print(f"train clips {len(train_df)} ({train_df.client_id.nunique()} spk) | test clips {len(test_df)} ({test_df.client_id.nunique()} spk)")
FRAUD = ["هلا يمه، جوالي خربان وأكلمك من رقم صاحبي، تكفين حولي لي ألف ريال ضروري.",
         "السلام عليكم، معك موظف من البنك، نبي نحدث بياناتك قبل ما يتوقف الحساب.",
         "وصلك كود على جوالك قبل شوي، عطني إياه بسرعة عشان أكمل الطلب.",
         "أبوي تعبان في المستشفى ويبون دفعة مقدمة، حول على هذا الحساب الله يخليك.",
         "أنا مديرك، أبيك تحول المبلغ للمورد الحين وأنا أرسل لك الأوراق بعدين."]
SENT = [s for s in pd.concat([train_df.sentence, test_df.sentence]).dropna().unique() if 25 <= len(str(s)) <= 140]

def make_ref(rows, out, sr=24000):
    """مقاطع Common Voice قصيرة (3-5 ث) → نلصق كم مقطع لنفس الشخص لين يصير المرجع 6-10 ثواني"""
    parts, dur = [], 0.0
    for i, r in enumerate(rows.itertuples()):
        w = f"{TMP}/part{i}.wav"; sh(f'ffmpeg -nostdin -v error -y -i "{r.mp3}" -ac 1 -ar {sr} "{w}"')
        if not os.path.exists(w): continue
        x, _ = sf.read(w, dtype="float32")
        if len(x) < sr: continue
        parts += [x, np.zeros(int(0.25 * sr), np.float32)]; dur += len(x) / sr
        if dur >= 8: break
    if dur < 5: return None
    sf.write(out, np.concatenate(parts[:-1]), sr); return out

def generate(df, n, tag, seed, minutes):
    out = f"{OUT}/cb_{tag}"; os.makedirs(out, exist_ok=True); rng = random.Random(seed)
    spk = list(df.client_id.unique()); rng.shuffle(spk); meta, t_end, i = [], time.time() + minutes * 60, 0
    while len(meta) < n and time.time() < t_end and i < len(spk) * 4:
        s = spk[i % len(spk)]; i += 1                                 # كل محاولة متحدث جديد (ما يعلق على متحدث واحد)
        rows = df[df.client_id == s].sample(frac=1, random_state=i).head(12)
        ref = make_ref(rows, f"{TMP}/ref.wav")
        if not ref: continue
        text = rng.choice(FRAUD) if rng.random() < 0.2 else rng.choice(SENT)
        try:
            try: wav = model.generate(text, language_id="ar", audio_prompt_path=ref)
            except TypeError: model.prepare_conditionals(ref); wav = model.generate(text, language_id="ar")
            k = len(meta); ta.save(f"{out}/clone_{k:04d}.wav", wav.cpu(), model.sr)
            shutil.copy(ref, f"{out}/real_{k:04d}.wav")                     # الصوت الحقيقي المرجع (نفس الجوال)
            meta.append({"id": k, "clone": f"clone_{k:04d}.wav", "real": f"real_{k:04d}.wav", "speaker": s[:12], "text": text})
            if len(meta) in (1, 10) or len(meta) % 50 == 0:
                print(f"  [{tag}] {len(meta)} clones ({(time.time() - T0) / 60:.0f} min)", flush=True)
                pd.DataFrame(meta).to_csv(f"{out}/meta.csv", index=False)
        except Exception as e: print("skip", repr(e)[:150])
    pd.DataFrame(meta).to_csv(f"{out}/meta.csv", index=False)
    print(f"[{tag}] {len(meta)} clones from {len(set(m['speaker'] for m in meta))} speakers ({(time.time() - T0) / 60:.0f} min)")

generate(test_df, 200, "test", 1, 45)                          # الاختبار أول (متحدثين ما يدخلون التدريب)
generate(train_df, 1500, "train", 0, LIMIT_MIN - (time.time() - T0) / 60)
print(f"\n✅ done in {(time.time() - T0) / 60:.0f} min — cb_train / cb_test في Output")
