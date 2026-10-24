# -*- coding: utf-8 -*-
"""P1-29 方向三（上界来源）：为何止步 2*sqrt(2)（判据 P1 至 P5）

定位：v0 与二-2 路径甲把 2*sqrt2 在模型内落地为判据级事实（存在性、上界、迁移轮同数），
尚未回答"为什么是 2*sqrt2 而不是 4"。本脚本给出模型内的结构回答并逐条核对：锁定对的
相关结构由单位相量 <e^{i Δα}> 生成，相关矩阵的奇异值恰为 |<e^{i Δα}>| <= 1，Horodecki
形式给出全空间上界 S_max = 2*sqrt(1 + |<e^{i Δα}>|^2) <= 2*sqrt2；4 界需要奇异值 sqrt2
（"全对/全错"的 PR 图案，相关块 [[1,1],[1,-1]]，奇异值 (2,0)），单相位锁的单位相量给不出。

通道与相关矩阵：
  每侧两条通道：层位 Z（锁定，相关 1）与表针象限 (X, Y)（一个复数的实虚部）。
  指针象限相关块 B = [[c, s], [s, -c]]，其中 c = <cos Δα>、s = <sin Δα>，由相量
  z = <e^{i Δα}> = c + i s 生成；锐锁定给 |z| = 1（单位圆上），弥散给 |z| < 1。
  相关矩阵 T = diag(1, B)，三个奇异值 (1, |z|, |z|)。

判据：
  P1 相量结构：锐 Δα 给 B^2 = I（1e-15）、|z| = 1（1e-15）；wrapped 族给
      (c, s) = (e^{-v/2}, 0)（解析对采样，5 sigma）
  P2 上界：三奇异值 (1, |z|, |z|)（1e-15）；Horodecki S_max = 2*sqrt(1+|z|^2)（1e-12）；
      锐 Δα 给 2*sqrt2 = 2.828427124746（与 v0 与迁移轮读数一致）
  P3 4 界不可达：PR 相关块 [[1,1],[1,-1]] 奇异值 (sqrt2, sqrt2) 给 S = 4（1e-12），对应
      (c, s) = (1, 1) 的模 sqrt2 超出单位相量上限 1；模型族 |z| <= 1（随机分布族抽样核对，5 sigma）；
      缺口 4 - 2*sqrt2 = 1.171573
  P4 象限对齐与弥散损失：锐 Δα 的子午面最大 = 2*sqrt(1+cos^2 Δα)（1e-6）小于全空间
      2*sqrt2（差 = 象限对齐损失）；wrapped 族下两者合流为 2*sqrt(1+V^2)（1e-6）
  P5 体系窗口接口：窗内（gamma = 0.0139，kappa = 0.125）|z| = 0.993836 给
      S_max = 2.819724（与路径甲 J5 表一致，1e-12）

纪律：全确定性（种子 20260914）；判据先行，实测 vs 期望；图存 images/（绝对路径）。
边界：不声称公理级结论（信息因果性等原理级支持见方向三推导稿）；Y 象限不作相位语言合成，
      本脚本用相关矩阵结构处理全空间口径；不改动任何基准文档。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

FIG_PATH = r"D:\software\TianXingFC\images\tsirelson_bound_origin.png"
SEED = 20260914
TS = 2.0 * np.sqrt(2.0)
CL = 2.0
NS = 4.0                      # 非信号界（PR 箱）
GAP = NS - TS                 # 奇异性缺口 4 - 2*sqrt2
KAPPA = 0.125
GAMMA_WIN = 0.0139

results = []

def gate(name, ok, detail):
    results.append((name, ok, detail))
    print("[%s] %s: %s" % ("PASS" if ok else "FAIL", name, detail))

def B_block(c, s):
    """指针象限相关块：由相量 (c, s) 生成。"""
    return np.array([[c, s], [s, -c]], dtype=float)

def T_matrix(c, s):
    """相关矩阵：层位对层位 1，指针块 B，交叉 0。"""
    T = np.zeros((3, 3))
    T[0, 0] = 1.0
    T[1:, 1:] = B_block(c, s)
    return T

def singular(T):
    return np.linalg.svd(T, compute_uv=False)

def horodecki_S(T):
    sv = singular(T)
    return 2.0 * np.sqrt(sv[0] ** 2 + sv[1] ** 2)

def smax_meridian(V, n=721):
    """子午面（Z-X 平面）全角最大：闭式 2*sqrt(1+V^2)。"""
    b = np.linspace(-np.pi, np.pi, n)
    B0, B1 = np.meshgrid(b, b, indexing="ij")
    t0 = np.sqrt((np.cos(B0) + np.cos(B1)) ** 2 + V ** 2 * (np.sin(B0) + np.sin(B1)) ** 2)
    t1 = np.sqrt((np.cos(B0) - np.cos(B1)) ** 2 + V ** 2 * (np.sin(B0) - np.sin(B1)) ** 2)
    return float(np.max(t0 + t1))

def V_lock(gamma, kappa=KAPPA):
    return float(np.exp(-0.5 * gamma ** 2 / kappa ** 2))

print("=" * 78)
print("P1-29 方向三：为何止步 2*sqrt2（单位相量结构与奇异性缺口，判据先行）")
print("=" * 78)

rng = np.random.default_rng(SEED)

# ---------- P1 相量结构 ----------
sharps = (0.0, 0.5, np.pi / 2.0, np.pi, 2.37)
dev_sq, dev_mod = 0.0, 0.0
for g in sharps:
    B = B_block(np.cos(g), np.sin(g))
    dev_sq = max(dev_sq, float(np.abs(B @ B - np.eye(2)).max()))
    dev_mod = max(dev_mod, abs(np.hypot(np.cos(g), np.sin(g)) - 1.0))
n1 = 2_000_000
dev_ws = 0.0
for v in (0.0, 1.0, 2.5):
    da = np.mod(np.sqrt(v) * rng.standard_normal(n1), 2.0 * np.pi) if v > 0 else np.zeros(n1)
    c_hat, s_hat = float(np.mean(np.cos(da))), float(np.mean(np.sin(da)))
    V1 = float(np.exp(-0.5 * v))
    if v == 0.0:
        dev_ws = max(dev_ws, 0.0 if (abs(c_hat - 1.0) < 1e-15 and abs(s_hat) < 1e-15) else 1e9)
    else:
        sig = np.sqrt((1.0 - V1 ** 2) / n1)
        dev_ws = max(dev_ws, abs(c_hat - V1) / sig, abs(s_hat) / sig)
ok = dev_sq < 1e-15 and dev_mod < 1e-15 and dev_ws < 5.0
gate("P1 相量结构", ok,
     "锐锁定：B^2 对 I 偏差 %.1e、|z| 对 1 偏差 %.1e（%d 档）；wrapped 族 (c, s) 对 (e^{-v/2}, 0) 最大偏差 %.2f sigma"
     % (dev_sq, dev_mod, len(sharps), dev_ws))

# ---------- P2 上界 ----------
dev_sv, dev_h, S_sharp = 0.0, 0.0, 0.0
for g in sharps:
    c, s = float(np.cos(g)), float(np.sin(g))
    sv = singular(T_matrix(c, s))
    dev_sv = max(dev_sv, float(np.abs(sv - np.array([1.0, 1.0, 1.0])).max()))
    dev_h = max(dev_h, abs(horodecki_S(T_matrix(c, s)) - 2.0 * np.sqrt(1.0 + c ** 2 + s ** 2)))
    S_sharp = max(S_sharp, horodecki_S(T_matrix(c, s)))
ok = dev_sv < 1e-15 and dev_h < 1e-12 and abs(S_sharp - TS) < 1e-12
gate("P2 上界（Horodecki 形式）", ok,
     "三奇异值对 (1, |z|, |z|) 偏差 %.1e；S_max = 2*sqrt(1+|z|^2) 偏差 %.1e；锐锁定 S_max = %.12f（对 2*sqrt2 差 %.1e）"
     % (dev_sv, dev_h, S_sharp, abs(S_sharp - TS)))

# ---------- P3 4 界不可达 ----------
T_pr = np.zeros((3, 3))
T_pr[0, 0] = 1.0
T_pr[1:, 1:] = np.array([[1.0, 1.0], [1.0, -1.0]])
sv_pr = singular(T_pr)
S_pr = horodecki_S(T_pr)
dev_z = 0.0
for _ in range(200):
    v = float(rng.uniform(0.0, 4.0))
    m = float(rng.uniform(0.0, 2.0 * np.pi))
    da = np.mod(m + (np.sqrt(v) * rng.standard_normal(200_000) if v > 0 else 0.0), 2.0 * np.pi)
    zz = abs(np.mean(np.exp(1j * da)))
    dev_z = max(dev_z, max(0.0, zz - (1.0 + 5.0 / np.sqrt(200_000))))
mod_pr = float(np.hypot(1.0, 1.0))
ok = (abs(sv_pr[0] - np.sqrt(2.0)) < 1e-12 and abs(sv_pr[1] - np.sqrt(2.0)) < 1e-12
      and abs(S_pr - NS) < 1e-12 and dev_z < 1e-12 and abs(GAP - (4.0 - TS)) < 1e-12)
gate("P3 4 界不可达", ok,
     "PR 块奇异值 (%.6f, %.6f) = (sqrt2, sqrt2) 给 S = %.9f（恰 4）；该图案对应 (c, s) = (1, 1)，模 %.6f 超出单位相量上限 1；"
     "模型族 |z| <= 1（200 组随机分布，采样超出 1 + 5/sqrt(N) 的量 %.1e）；缺口 4 - 2*sqrt2 = %.6f"
     % (sv_pr[0], sv_pr[1], S_pr, mod_pr, dev_z, GAP))

# ---------- P4 象限对齐与弥散损失 ----------
dev_m1, dev_m2 = 0.0, 0.0
for g in (0.0, 0.5, 1.2, 2.5):
    c = float(np.cos(g))
    dev_m1 = max(dev_m1, abs(smax_meridian(c, 361) - 2.0 * np.sqrt(1.0 + c ** 2)))
c01 = float(np.cos(1.2))
S_mer_12 = smax_meridian(c01, 361)
S_full_12 = horodecki_S(T_matrix(c01, np.sin(1.2)))
for v in (0.0, 1.0, 2.5):
    V1 = float(np.exp(-0.5 * v))
    dev_m2 = max(dev_m2, abs(smax_meridian(V1, 361) - horodecki_S(T_matrix(V1, 0.0))))
ok = dev_m1 < 1e-6 and dev_m2 < 1e-6 and S_mer_12 < S_full_12 - 0.1 and abs(S_full_12 - TS) < 1e-12
gate("P4 象限对齐与弥散损失", ok,
     "锐 Δα 子午面最大对 2*sqrt(1+cos^2 Δα) 偏差 %.1e；Δα = 1.2 时子午面 %.6f < 全空间 %.6f（2*sqrt2，象限对齐损失 %.3f）；"
     "wrapped 族两口径合流偏差 %.1e" % (dev_m1, S_mer_12, S_full_12, S_full_12 - S_mer_12, dev_m2))

# ---------- P5 体系窗口接口 ----------
V_win = V_lock(GAMMA_WIN)
S_win = horodecki_S(T_matrix(V_win, 0.0))
S_win_ref = 2.0 * np.sqrt(1.0 + V_win ** 2)
ok = abs(S_win - 2.81972) < 5e-5 and abs(S_win - S_win_ref) < 1e-12
gate("P5 体系窗口接口", ok,
     "窗内（gamma = %.4f，kappa = %.3f）|z| = %.6f 给 S_max = %.6f（路径甲 J5 记录 2.819724，差 %.1e）"
     % (GAMMA_WIN, KAPPA, V_win, S_win, abs(S_win - 2.81972)))

# ---------- 图 ----------
fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))

ax = axes[0]
labels = ["模型（锐锁定）", "模型（弥散 v=1）", "PR 图案"]
sv_model_sharp = sorted(singular(T_matrix(1.0, 0.0)), reverse=True)
sv_model_v1 = sorted(singular(T_matrix(float(np.exp(-0.5)), 0.0)), reverse=True)
sv_pr_sorted = sorted(sv_pr, reverse=True)
arr = np.array([sv_model_sharp, sv_model_v1, sv_pr_sorted])
xs = np.arange(3)
for j, col in enumerate(("#2b6cb0", "#2f855a", "#c53030")):
    ax.bar(xs + (j - 1) * 0.26, arr[:, j], width=0.24, color=col, label="s%d" % (j + 1))
ax.axhline(TS / 2.0, color="0.5", lw=0.9, ls=":")
ax.text(2.35, TS / 2.0 + 0.03, "sqrt2（4 界所需）", fontsize=8)
ax.set_xticks(xs)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("相关矩阵奇异值")
ax.set_title("P2/P3 奇异值结构：(1,1,1) 对 (sqrt2,sqrt2,0)")

ax = axes[1]
zz = np.linspace(0.0, 1.05, 401)
ax.plot(zz, 2.0 * np.sqrt(1.0 + zz ** 2), lw=1.7, label="S_max = 2*sqrt(1+|z|^2)")
ax.axhline(TS, color="0.6", lw=0.9, ls=":", label="2*sqrt2")
ax.axhline(CL, color="k", lw=1.0, ls="--", label="经典界 2")
ax.axhline(NS, color="#c53030", lw=1.0, ls="-.", label="非信号界 4")
pts = [(1.0, TS), (float(np.exp(-0.5)), horodecki_S(T_matrix(float(np.exp(-0.5)), 0.0))), (V_win, S_win)]
for (x, y), lab in zip(pts, ("锐锁定", "弥散 v=1", "窗内 0.0139")):
    ax.plot([x], [y], "o", ms=6)
    ax.annotate(lab, (x, y), textcoords="offset points", xytext=(-38, 8), fontsize=8)
ax.set_xlabel("锁定相量模 |z| = |<e^{i Δα}>|")
ax.set_ylabel("S_max（全空间）")
ax.set_title("P2 上界曲线：缺口 4 - 2*sqrt2 = %.3f" % GAP)
ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig(FIG_PATH, dpi=140)
plt.close(fig)

# ---------- 汇总 ----------
print("-" * 78)
n_ok = sum(1 for _, ok, _ in results if ok)
print("汇总：%d/%d 通过（种子 %d）" % (n_ok, len(results), SEED))
print("图：%s" % FIG_PATH)
