# ===== SAQR v2 — تجهيز الموديل النهائي (xlsr + الحدود + جدول النسبة) — CPU فقط =====
import glob, os, shutil, json, pickle
import numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression

V = pickle.load(open(glob.glob("/kaggle/input/**/xlsr_window_scores.pkl", recursive=True)[0], "rb"))["val"]
XDIR = os.path.dirname(glob.glob("/kaggle/input/**/saqr_v2_xlsr/config.json", recursive=True)[0])
OUT = "/kaggle/working/saqr_v2_final"
shutil.copytree(XDIR, OUT, dirs_exist_ok=True, ignore=shutil.ignore_patterns("*.csv"))

R = V[V.label == 0].mean_z.values
T_LOW, T_HIGH = float(np.quantile(R, 0.90)), float(np.quantile(R, 0.99))   # real90: حقيقي يتنبه ≤10%، يُتهم ≤1%
json.dump({"T_LOW": T_LOW, "T_HIGH": T_HIGH, "score": "mean_logit", "tta_shifts": [0, 52], "overlap": False,
           "max_speech_sec": 60, "cap_fake_on_call": False, "policy": "real90 on validation"},
          open(f"{OUT}/thresholds.json", "w"), indent=2)

# النسبة اللي تطلع للمستخدم: كم مرة كان المقطع مزيف فعلًا عند هالدرجة (من بيانات التحقق)
iso = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99).fit(V.mean_z, V.label)
x = np.unique(np.r_[np.quantile(V.mean_z, np.linspace(0, 1, 101)), T_LOW, T_HIGH])
json.dump({"x": x.round(4).tolist(), "y": iso.predict(x).round(4).tolist(),
           "note": "P(fake | score) on validation, isotonic"}, open(f"{OUT}/calibration.json", "w"))

print(f"T_LOW={T_LOW:.4f}  T_HIGH={T_HIGH:.4f}  | val: {len(V)} clips, {int((V.label == 1).sum())} fake")
print("displayed fake % at:", {f"{s:.2f}": f"{iso.predict([s])[0] * 100:.0f}%" for s in
      [-5, 0, 3, 5, T_LOW - 0.3, T_LOW, (T_LOW + T_HIGH) / 2, T_HIGH, T_HIGH + 0.3]})
print("verdict check on validation:  real→REAL", f"{np.mean(R < T_LOW) * 100:.1f}%",
      "| fake flagged", f"{np.mean(V[V.label == 1].mean_z >= T_LOW) * 100:.1f}%")
print("\nfiles:", sorted(os.listdir(OUT)))
print(f"size: {sum(os.path.getsize(f'{OUT}/{f}') for f in os.listdir(OUT)) / 1e9:.2f} GB")
print("✅ saqr_v2_final جاهز في Output")
