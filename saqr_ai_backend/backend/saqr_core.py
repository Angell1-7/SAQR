"""
SAQR — core analysis logic (model loading + audio analysis).

The model folder must contain:
    - the Hugging Face model files (config.json, model.safetensors, preprocessor_config.json)
    - thresholds.json   {"T_LOW": ..., "T_HIGH": ...} and, for v2, the analysis settings:
                        "score": "mean_logit", "tta_shifts": [0, 52], "overlap": false,
                        "max_speech_sec": 60, "cap_fake_on_call": false
    - calibration.json  (v2) score -> displayed fake probability table, measured on validation
    - temperature.json  (v1.1) {"T": ...}, used only when calibration.json is missing
To switch to a new model version, point SAQR_MODEL_DIR to the new folder. Nothing else changes.
"""
import json
import os
import subprocess

import numpy as np

SR = 16000            # sample rate the model expects
WIN = 96000           # 6-second windows (same as training/evaluation)
MIN_WIN = 16000       # ignore windows shorter than 1 s
TOP_DB = 30           # silence removal (same as evaluation)
MAX_SECONDS = 300     # analyse at most the first 5 minutes
CALL_CUTOFF_HZ = 4200 # phone audio is sampled at 8 kHz, so it has (almost) nothing above 4 kHz
CALL_HF_RATIO = 0.002 # energy share above the cutoff below this -> call-quality audio
MIN_SPEECH_SEC = 1.0  # less speech than this -> no_speech error
SILENCE_PEAK = 1e-3   # peak below -60 dBFS -> the file is silent
SILENCE_RMS = 3e-4    # "speech" quieter than about -70 dBFS -> not real speech
SHORT_SPEECH_SEC = 3.0

VERDICT_AR = {"real": "حقيقي", "suspicious": "مشكوك", "fake": "مزيف"}
WARN_CALL = "جودة الصوت منخفضة (نطاق ترددي ضيق مثل المكالمات)، وقد تكون دقة الكشف أقل"
WARN_SHORT = "الكلام في المقطع قصير، ودقة الكشف أقل"
WARN_TRUNC = "المقطع طويل، وتم تحليل أول 5 دقائق فقط"
WARN_SPEECH_CAP = "الكلام في المقطع طويل، وتم تحليل أول {} ثانية من الكلام"


class AnalysisError(Exception):
    def __init__(self, code, message_ar, http_status=400):
        super().__init__(code)
        self.code, self.message_ar, self.http_status = code, message_ar, http_status


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p):
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    return np.log(p / (1 - p))


def native_sample_rate(path):
    cmd = ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=sample_rate",
           "-of", "csv=p=0", path]
    try:
        return int(subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip())
    except Exception:
        return None


def decode_audio(path):
    """Decode any audio/video file to 16 kHz mono float32.

    ffmpeg only decodes (at the file's own sample rate); the resampling to 16 kHz is done with
    librosa (soxr_hq), exactly like librosa.load() used in training and evaluation. Resampling
    with ffmpeg instead changes the model's scores noticeably on some files.
    """
    import librosa
    sr_in = native_sample_rate(path)
    if not sr_in:
        raise AnalysisError("invalid_file", "تعذر قراءة الملف، تأكد إنه ملف صوت أو فيديو سليم")
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-vn", "-ac", "1",
           "-t", str(MAX_SECONDS + 1), "-f", "f32le", "pipe:1"]
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=120, check=True).stdout
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        raise AnalysisError("invalid_file", "تعذر قراءة الملف، تأكد إنه ملف صوت أو فيديو سليم")
    audio = np.frombuffer(out, dtype=np.float32).copy()
    if audio.size == 0:
        raise AnalysisError("invalid_file", "الملف لا يحتوي على صوت")
    if sr_in != SR:
        audio = librosa.resample(audio, orig_sr=sr_in, target_sr=SR, res_type="soxr_hq")
    return audio.astype(np.float32)


def media_duration(path):
    """Full duration in seconds (to report it even if we only analyse the first 5 minutes)."""
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path]
    try:
        return float(subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip())
    except Exception:
        return None


def speech_only(audio):
    import librosa
    iv = librosa.effects.split(audio, top_db=TOP_DB)
    return np.concatenate([audio[s:e] for s, e in iv]) if len(iv) else audio[:0]


def hf_ratio(audio):
    X = np.abs(np.fft.rfft(audio.astype(np.float32))) ** 2
    f = np.fft.rfftfreq(len(audio), 1 / SR)
    return float(X[f > CALL_CUTOFF_HZ].sum() / (X.sum() + 1e-12))


TTA_SHIFTS = [0, 52, 104, 156]   # each window is scored 4 times with tiny shifts (0-10 ms) and averaged


def window_starts(n, overlap=True):
    """Start positions of full 6-s windows (50% overlap, or none); the last one is aligned to the end of the speech,
    so no window is shorter than 6 s (short tail windows gave unstable scores)."""
    if n <= WIN:
        return [0]
    starts = list(range(0, n - WIN + 1, WIN // 2 if overlap else WIN))
    if starts[-1] != n - WIN:
        starts.append(n - WIN)
    return starts


def make_windows(speech, shift=0, overlap=True):
    out = []
    for s in window_starts(len(speech), overlap):
        x = speech[s + shift:s + shift + min(WIN, len(speech))]
        need = min(WIN, len(speech)) - len(x)
        out.append(np.pad(x, (0, need)) if need > 0 else x)
    return out


def decide(p_raw, t_low, t_high, is_call, cap_fake_on_call=True):
    verdict = "fake" if p_raw > t_high else ("real" if p_raw < t_low else "suspicious")
    if is_call and cap_fake_on_call and verdict == "fake":
        verdict = "suspicious"   # call audio: a definite "fake" is not reliable enough
    return verdict


def _smooth_table(cal):
    """The validation table is a staircase (many scores map to the same %). Keep one point per step,
    placed at the step's middle, so the displayed % changes smoothly with the score (still monotonic)."""
    x, y = np.asarray(cal["x"], float), np.asarray(cal["y"], float)
    xs, ys, i = [x[0]], [y[0]], 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and y[j + 1] == y[i]:
            j += 1
        xs.append((x[i] + x[j]) / 2); ys.append(y[i]); i = j + 1
    xs.append(x[-1]); ys.append(y[-1])
    return {"x": xs, "y": ys}


class SaqrDetector:
    def __init__(self, model_dir, device=None, cap_fake_on_call=True):
        import torch
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForAudioClassification.from_pretrained(model_dir).to(self.device).eval()
        self.fe = AutoFeatureExtractor.from_pretrained(model_dir)
        self.fake_id = int(self.model.config.label2id["fake"])
        self.real_id = 1 - self.fake_id

        th = json.load(open(os.path.join(model_dir, "thresholds.json")))
        self.t_low, self.t_high = float(th["T_LOW"]), float(th["T_HIGH"])
        # v1.1: score = mean probability of windows. v2: score = mean logit (no saturation), set in thresholds.json
        self.score_type = th.get("score", "mean_prob")
        self.shifts = th.get("tta_shifts", TTA_SHIFTS)
        self.overlap = th.get("overlap", True)
        self.max_speech_sec = th.get("max_speech_sec")          # None = no limit (only MAX_SECONDS of audio)
        self.cap_fake_on_call = th.get("cap_fake_on_call", cap_fake_on_call)
        cal = os.path.join(model_dir, "calibration.json")         # score -> probability table (from validation)
        self.cal = _smooth_table(json.load(open(cal))) if os.path.exists(cal) else None
        tp = os.path.join(model_dir, "temperature.json")
        self.T = float(json.load(open(tp))["T"]) if os.path.exists(tp) else 1.0
        self.version = os.path.basename(os.path.normpath(model_dir))

    def window_logits(self, windows):
        zs = []                                   # all windows of one file have the same length -> batch them
        with self.torch.no_grad():
            for i in range(0, len(windows), 8):
                iv = self.fe(windows[i:i + 8], sampling_rate=SR, return_tensors="pt").input_values.to(self.device)
                lg = self.model(iv).logits.float().cpu().numpy()
                zs.extend((lg[:, self.fake_id] - lg[:, self.real_id]).tolist())
        return np.array(zs)

    def analyze(self, path):
        audio = decode_audio(path)
        full_dur = media_duration(path) or len(audio) / SR
        truncated = full_dur > MAX_SECONDS + 0.5
        audio = audio[:MAX_SECONDS * SR]

        if np.max(np.abs(audio)) < SILENCE_PEAK:            # whole file is (near) digital silence
            raise AnalysisError("no_speech", "لم يتم العثور على كلام كافٍ في المقطع")
        speech = speech_only(audio)
        if len(speech) < MIN_SPEECH_SEC * SR or np.sqrt(np.mean(speech ** 2)) < SILENCE_RMS:
            raise AnalysisError("no_speech", "لم يتم العثور على كلام كافٍ في المقطع")

        is_call = hf_ratio(speech) < CALL_HF_RATIO
        capped = self.max_speech_sec and len(speech) > self.max_speech_sec * SR
        if capped:
            speech = speech[:int(self.max_speech_sec * SR)]
        windows = [w for s in self.shifts for w in make_windows(speech, s, self.overlap)]
        z = self.window_logits(windows)
        if self.score_type == "mean_logit":
            score = float(np.mean(z))                                    # mean logit over windows x shifts
        else:
            score = float(np.mean(_sigmoid(z)))                          # v1.1: mean probability
        verdict = decide(score, self.t_low, self.t_high, is_call, self.cap_fake_on_call)
        if self.cal:                                                     # probability measured on validation
            fake_prob = float(np.interp(score, self.cal["x"], self.cal["y"]))
        elif self.score_type == "mean_logit":
            fake_prob = float(_sigmoid(score / self.T))
        else:
            fake_prob = float(_sigmoid(_logit(score) / self.T))

        warnings = []
        if is_call:
            warnings.append(WARN_CALL)
        if len(speech) < SHORT_SPEECH_SEC * SR:
            warnings.append(WARN_SHORT)
        if truncated:
            warnings.append(WARN_TRUNC)
        if capped:
            warnings.append(WARN_SPEECH_CAP.format(int(self.max_speech_sec)))

        return {
            "status": "success",
            "verdict": verdict,
            "verdict_ar": VERDICT_AR[verdict],
            "fake_probability": round(fake_prob * 100, 1),
            "audio_quality": "call" if is_call else "normal",
            "warning": "، ".join(warnings) if warnings else None,
            "duration_sec": round(full_dur, 1),
            "model_version": self.version,
        }
