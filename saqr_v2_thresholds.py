# ===== SAQR v2_xlsr — الحدود النهائية + تجربة تحليل أسرع (CPU فقط، بدون GPU) =====
import glob, pickle, json
import numpy as np, pandas as pd
from sklearn.metrics import roc_curve

D = pickle.load(open(glob.glob("/kaggle/input/**/xlsr_window_scores.pkl", recursive=True)[0], "rb"))
V, B, WS = D["val"], D["battery"], D["windows"]
ok = np.array([w is not None for w in WS]); B, WS = B[ok].reset_index(drop=True), [w for w in WS if w is not None]

# ---------------------------------------------------------------- 1) قيم الحدود من بيانات التحقق
print("thresholds from validation (mean_z = متوسط الـ logit):")
TH = {}
for a in ["mean_z", "mean_p"]:
    R, F = V[V.label == 0][a].values, V[V.label == 1][a].values; th = float(np.quantile(R, 0.99))
    for pol, tl in [("old", min(np.quantile(F, 0.05), th)), ("real95", np.quantile(R, 0.95)), ("real90", np.quantile(R, 0.90))]:
        TH[(a, pol)] = (float(tl), th); print(f"  {a:7} {pol:7} T_LOW={tl:.4f}  T_HIGH={th:.4f}")

# ---------------------------------------------------------------- 2) نسخ أسرع من التحليل (نفس النتائج المحفوظة، بدون إعادة تشغيل)
def eer(y, s):
    fpr, tpr, _ = roc_curve(y, s); return float(fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]) * 100
def variant(W, shifts, overlap):
    W = W[:shifts]
    if not overlap and W.shape[1] > 2:                 # بدون تداخل: نافذة ونافذة + الأخيرة
        W = W[:, sorted(set(list(range(0, W.shape[1] - 1, 2)) + [W.shape[1] - 1]))]
    return float(W.mean())
core_m = ((B.src == "battery") & (B.cond != "partial")).values
tl, th = TH[("mean_z", "real90")]
v = lambda s: np.where(s > th, 2, np.where(s < tl, 0, 1))
full = np.array([variant(W, 4, True) for W in WS]); vfull = v(full)
rows = []
for shifts in [4, 2, 1]:
    for overlap in [True, False]:
        s = np.array([variant(W, shifts, overlap) for W in WS]); y = B.label.values
        n30 = (9 if overlap else 5) * shifts
        rows.append({"TTA shifts": shifts, "overlap": "50%" if overlap else "none",
                     "EER": round(eer(y[core_m], s[core_m]), 2),
                     "same verdict as full %": round(np.mean(v(s) == vfull) * 100, 1),
                     "real flagged %": round(np.mean(s[core_m & (y == 0)] >= tl) * 100, 1),
                     "fake flagged %": round(np.mean(s[core_m & (y == 1)] >= tl) * 100, 1),
                     "clones flagged %": round(np.mean(s[(B.src == "clone").values] >= tl) * 100, 1),
                     "real calls": "".join("RSF"[k] for k in v(s[(B.src == "extra").values])),
                     "CPU 30s clip (s)": round(n30 * 1.49)})
pd.set_option("display.width", 250)
print("\n================ cheaper inference (mean_z, real90 thresholds) ================")
print(pd.DataFrame(rows).to_string(index=False))
lens = [W.shape[1] for W in WS]; print(f"\nbattery clips: windows per clip median {np.median(lens):.0f}, max {max(lens)}")
