# ===== SAQR — توليد أصوات مستنسخة بموديل حديث (Chatterbox Multilingual 2025، مجاني MIT، يدعم العربي) =====
# الهدف: اختبار الموديل على استنساخ "نص ← صوت شخص" (نفس أسلوب ElevenLabs) بأصوات متحدثين سعوديين ما تدرب عليهم
import os, sys, glob, zipfile, subprocess, importlib, random, time
T0 = time.time(); INP, TMP, OUT = "/kaggle/input", "/kaggle/temp", "/kaggle/working"
CL = f"{OUT}/chatterbox_clones"; os.makedirs(CL, exist_ok=True); os.makedirs(TMP, exist_ok=True)
sh = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True)

# ---------------------------------------------------------------- 1) التركيب (بدون ما نكسر مكتبات Kaggle قدر الإمكان)
print(sh("pip install -q --no-deps chatterbox-tts").stderr[-300:])
print(sh("pip install -q transformers==4.46.3 diffusers==0.29.0").stderr[-300:])
PIPNAME = {"perth": "resemble-perth", "s3tokenizer": "s3tokenizer", "pkuseg": "spacy-pkuseg", "spacy_pkuseg": "spacy-pkuseg"}
for _ in range(15):                                   # يركب أي مكتبة ناقصة وحدة وحدة
    try:
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS; break
    except ModuleNotFoundError as e:
        name = e.name.split(".")[0]; print("installing", name); sh(f"pip install -q {PIPNAME.get(name, name)}")
        importlib.invalidate_caches()
import torch, torchaudio as ta, numpy as np, pandas as pd, soundfile as sf
model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
print(f"model loaded ({(time.time() - T0) / 60:.0f} min), sr={model.sr}")

# ---------------------------------------------------------------- 2) أصوات مرجعية: متحدثين سعوديين من الاختبار الشامل (ما تدرب عليهم الموديل)
z = glob.glob(f"{INP}/**/battery.zip", recursive=True)
if z and not os.path.exists(f"{TMP}/battery"): zipfile.ZipFile(z[0]).extractall(f"{TMP}/battery")
BAT = next(os.path.dirname(p) for p in glob.glob(f"{INP}/**/manifest.csv", recursive=True) + glob.glob(f"{TMP}/**/manifest.csv", recursive=True))
man = pd.read_csv(f"{BAT}/manifest.csv"); man["path"] = man.path.map(lambda p: str(p).replace("/content/drive/MyDrive/saqr_battery", BAT))
refs = man[(man.cond == "clean") & (man.group == "SADA-real")].copy()
refs["dur"] = [sf.info(p).duration if os.path.exists(p) else 0 for p in refs.path]
refs = refs[refs.dur >= 4].sample(frac=1, random_state=5).head(30).reset_index(drop=True)
print("reference speakers:", len(refs))

TEXTS = [
    "هلا يمه، جوالي خربان وأكلمك من رقم صاحبي، تكفين حولي لي ألف ريال ضروري.",
    "السلام عليكم، معك موظف من البنك، نبي نحدث بياناتك قبل ما يتوقف الحساب.",
    "يا أخوي أنا في موقف صعب، لا تقول لأحد، أرسل لي المبلغ اليوم وأرجعه لك بكرة.",
    "وصلك كود على جوالك قبل شوي، عطني إياه بسرعة عشان أكمل الطلب.",
    "أبوي تعبان في المستشفى ويبون دفعة مقدمة، حول على هذا الحساب الله يخليك.",
    "معك مندوب التوصيل، الشحنة واقفة بالجمارك ولازم تدفع رسوم بسيطة اليوم.",
    "مبروك، فزت بجائزة المسابقة، بس نحتاج رقم الهوية عشان نرسلها لك.",
    "أنا مديرك، أبيك تحول المبلغ للمورد الحين وأنا أرسل لك الأوراق بعدين.",
    "لا تخاف، هذي مكالمة رسمية، بس أكد لي اسمك الكامل وتاريخ ميلادك.",
    "حبيبتي أنا برا وما معي شحن، أرسلي لي رقم التحقق اللي جاك الحين.",
]

# ---------------------------------------------------------------- 3) التوليد: كل صوت مرجعي يقول 3 جمل
rng, meta = random.Random(1), []
for i, r in refs.iterrows():
    for t in rng.sample(TEXTS, 3):
        out = f"{CL}/cb_{len(meta):03d}.wav"
        try:
            try: wav = model.generate(t, language_id="ar", audio_prompt_path=r.path)
            except TypeError:
                model.prepare_conditionals(r.path); wav = model.generate(t, language_id="ar")
            ta.save(out, wav.cpu(), model.sr)
            meta.append({"file": os.path.basename(out), "ref": r.path, "ref_id": r.id, "text": t})
        except Exception as e: print("skip", i, repr(e)[:200])
    if i % 5 == 0: print(f"  ref {i + 1}/{len(refs)} → {len(meta)} clones ({(time.time() - T0) / 60:.0f} min)")
pd.DataFrame(meta).to_csv(f"{CL}/meta.csv", index=False)
print(f"\n✅ {len(meta)} clones in {(time.time() - T0) / 60:.0f} min — اسمعي كم واحد من Output/chatterbox_clones")
