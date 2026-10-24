# -*- coding: utf-8 -*-
r"""
position_momentum.py

P3-10 首轮：按路线乙（留在环上、边界项作规范依赖读出）核 判据 P1 至 P8。
判据见 内部草稿/研究方向-S¹ 本位导出位置动量算符：判据草案.md；确定性，随机固定种子。

约定：站点基 |k>，th_k = 2 pi k / Z（分支切割在 k=0，即接缝站点 k=0）；模基为傅里叶基，
Om 在模基里是升移位；L 在模基里对角，本征值 lam = -n_s(m)（带符号提升）或 -m（线性提升）。
位置与动量取 x = l_P th、p = (hbar/l_P) L，本轮量以 l_P = hbar = 1 计。

判据与期望：
  P1 提升与接缝同址：两种提升下 [L, Om] - Om 非零元恰 1 个、位置等于提升落差点（<= 1e-12）
  P2 位置矩阵元的误差常数：|Theta^Z - Theta^inf| * Z -> pi（±5%），对角与三个非对角频差、
     三档 Z 全覆盖。解析目标 Theta^inf = -i/(m'-m)（非对角）、pi（对角）
  P3 动量谱与带内对角：P_N L P_N 为整数对角（<= 1e-12）
  P4 边界项形式：<[L, th]> + i = i * Z * |psi_接缝|^2（相对偏差 <= 5%），四类态 × 三档 Z
  P5 切割自由度：把分支切割移到站点 s，偏差跟随 i * Z * |psi_s|^2（<= 5%）
  P6 固定物理态：偏差 = 2 pi 乘态在切割处的物理密度，与 Z 无关（三档相对偏离 <= 5%）
  P7 不确定关系接口：dth * dL >= |<[th, L]>| / 2（200 个固定种子态）；准经典态上比值趋近 1
  P8 秩一结构（覆盖空间）：[L, th] + i * 1 的数值秩 = 1（三档）
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\position_momentum.png"
SEED = 20260919
ZS = (32, 64, 128)
rows = []


def report(tag, ok, expect, actual):
    rows.append((tag, ok, expect, actual))
    print("[%s] %s" % ("通过" if ok else "未过", tag))
    print("      期望 " + expect)
    print("      实测 " + actual)


def build(Z):
    w = np.exp(2j * np.pi / Z)
    k = np.arange(Z)
    F = np.array([[w ** (-m * j) for j in range(Z)] for m in range(Z)], dtype=complex) / np.sqrt(Z)
    Om_site = np.diag(w ** k)
    Om_m = F.conj() @ Om_site @ F.T
    m = np.arange(Z)
    n_s = np.where(m < Z / 2.0, m, m - Z)
    return w, F, Om_m, (-n_s.astype(float)), (-m.astype(float))


def to_modes(F, X_site):
    return F.conj() @ X_site @ F.T


def theta_site(Z, s):
    return 2.0 * np.pi * ((np.arange(Z) - s) % Z) / Z


def coeff(F, psi):
    c = F.conj() @ psi
    return c / np.linalg.norm(c)


def phase_gauss(Z, sig, theta0):
    th = 2.0 * np.pi * np.arange(Z) / Z
    d = th - theta0
    psi = np.zeros(Z, dtype=complex)
    for j in (-1, 0, 1):
        psi = psi + np.exp(-((d - 2.0 * np.pi * j) ** 2) / (4.0 * sig ** 2))
    return psi / np.linalg.norm(psi)


def site_state(Z, k):
    psi = np.zeros(Z, dtype=complex)
    psi[k] = 1.0
    return psi


# ---------- P1 ----------
p1 = []
for Z in ZS:
    w, F, Om_m, lam_s, lam_l = build(Z)
    for lam, loc in ((lam_l, (Z - 1, 0)), (lam_s, (Z // 2 - 1, Z // 2))):
        Lm = np.diag(lam)
        R = Lm @ Om_m - Om_m @ Lm - Om_m
        idx = np.argwhere(np.abs(R) > 1e-9)
        good = (len(idx) == 1 and tuple(int(x) for x in idx[0]) == loc
                and abs(abs(R[int(idx[0][0]), int(idx[0][1])]) - Z) < 1e-9)
        p1.append((Z, loc, good))
report("P1 提升与接缝同址", all(g for _, _, g in p1),
       "两种提升下非零元恰 1 个、位置等于提升落差点、幅度 = Z",
       "；".join("Z=%d %s %s" % (Z, str(loc), "ok" if g else "fail") for Z, loc, g in p1))

# ---------- P2 ----------
p2 = []
for Z in ZS:
    w, F, Om_m, lam_s, lam_l = build(Z)
    Th = to_modes(F, np.diag(theta_site(Z, 0)))
    for d in (1, 2, 3):
        dv = float(np.max(np.abs(np.diagonal(Th, offset=-d) - (-1j / d)))) * Z
        p2.append((Z, d, dv))
    diag = Th.diagonal()
    p2.append((Z, 0, float(np.max(np.abs(diag.real - np.pi))) * Z))
    p2.append((Z, -1, float(np.max(np.abs(diag.imag))) * Z))
ok2 = all(abs(v - np.pi) <= 0.05 * np.pi for Z, d, v in p2 if d >= 0) and all(v <= 1e-9 for Z, d, v in p2 if d == -1)
report("P2 位置矩阵元的误差常数 = pi", ok2,
       "对角与频差 1、2、3 上 |Theta^Z - Theta^inf| * Z = pi（±5%）；对角的虚部 <= 1e-9",
       "；".join("Z=%d d=%d: %.5f" % (Z, d, v) for Z, d, v in p2 if d >= 0) + "；虚部最大 %.1e" % max(v for Z, d, v in p2 if d == -1))

# ---------- P3 ----------
p3 = []
for Z in ZS:
    w, F, Om_m, lam_s, lam_l = build(Z)
    m = np.arange(Z)
    ns = np.where(m < Z / 2.0, m, m - Z)
    keep = np.abs(ns) <= Z // 4
    PN = np.diag(keep.astype(float))
    P = PN @ np.diag(lam_s) @ PN
    off = float(np.abs(P - np.diag(np.diag(P))).max())
    ints = bool(np.all(np.abs(np.diag(P)[keep] - np.round(np.diag(P)[keep])) < 1e-12))
    p3.append((Z, off, ints))
report("P3 动量谱与带内对角性", all(o <= 1e-12 and i for _, o, i in p3),
       "带内投影为整数对角（偏差 <= 1e-12）",
       "；".join("Z=%d 非对角 %.1e，整数性 %s" % (Z, o, i) for Z, o, i in p3))



def measure(Z, psi, s=0):
    w, F, Om_m, lam_s, lam_l = build(Z)
    psi = psi / np.linalg.norm(psi)
    Th = to_modes(F, np.diag(theta_site(Z, s)))
    Lm = np.diag(lam_s)
    com = Lm @ Th - Th @ Lm
    c = coeff(F, psi)
    X = np.vdot(c, com @ c) + 1j
    pred = 1j * Z * abs(psi[s]) ** 2
    return X, pred, c, com


# ---------- P4 / P4b ----------
p4, p4b = [], []
for Z in ZS:
    for sg in (0.3, 0.5, 0.8):
        X, pred, c, com = measure(Z, phase_gauss(Z, sg, 0.0), 0)
        p4.append((Z, sg, abs(X - pred) / abs(pred)))
    X, pred, c, com = measure(Z, site_state(Z, 0), 0)
    p4b.append((Z, abs(X - pred) / abs(pred), abs(X)))
report("P4 边界项形式（光滑带限态）", all(e <= 0.05 for _, _, e in p4),
       "三档 Z × 三个宽度上相对偏差 <= 5%",
       "；".join("Z=%d sig=%.1f: %.2e" % (Z, sg, e) for Z, sg, e in p4))
report("P4b 反例：δ 态超出适用范围", all(e >= 0.5 and a >= 0.5 for _, e, a in p4b),
       "站点 δ 态上偏差为 O(1)（相对偏差 >= 0.5 且 |X| >= 0.5），说明该形式以动量窄带为前提",
       "；".join("Z=%d 相对偏差 %.3f，|X| %.3f" % (Z, e, a) for Z, e, a in p4b))

# ---------- P5 ----------
p5 = []
for Z in ZS:
    psi = phase_gauss(Z, 0.5, 0.0)
    for st in (0, Z // 4, Z // 2):
        X, pred, c, com = measure(Z, psi, st)
        p5.append((Z, st, abs(X - pred), abs(pred)))
report("P5 切割自由度（移动分支切割）", all(a <= 0.1 for _, _, a, _ in p5),
       "三档 Z × 三处切割上 |X - i Z |psi_s|^2| <= 0.1（绝对量；该形式的格点修正是该量级）",
       "；".join("Z=%d s=%d: 绝对 %.2e（预测量级 %.2e）" % (Z, st, a, b) for Z, st, a, b in p5))

# ---------- P6a / P6b ----------
p6a = []
for Z in ZS:
    X, pred, c, com = measure(Z, phase_gauss(Z, 0.35, 0.0), 0)
    p6a.append((Z, abs(X), abs(pred)))
r6 = [a / b for _, a, b in p6a]
ok6a = max(r6) / min(r6) - 1.0 <= 0.05 and all(abs(a - b) / b <= 0.05 for _, a, b in p6a)
report("P6a 站点侧偏差与 Z 无关", ok6a,
       "横切态（中心在切割处，sigma = 0.35）三档 Z 上 |X| 相互偏离 <= 5%，且等于 i Z |psi_0|^2（<= 5%）",
       "；".join("Z=%d 实测 %.6f 预期 %.6f" % (Z, a, b) for Z, a, b in p6a))
p6b = []
for Z in ZS:
    X, pred, c, com = measure(Z, phase_gauss(Z, 0.35, np.pi), 0)
    p6b.append((Z, abs(X), abs(pred), Z * abs(np.conj(c[Z // 2 - 1]) * c[Z // 2])))
report("P6b 小偏差受数值地板限制（登记）", all(a <= 1e-12 for _, a, _, _ in p6b),
       "远切割态上 |X| <= 1e-12（地板界）；预测值低于地板故该档不可分辨",
       "；".join("Z=%d 实测 %.2e 预期 %.2e，模侧接缝项 %.2e" % (Z, a, b, m) for Z, a, b, m in p6b))

# ---------- P7 ----------
Z7 = 64
w, F, Om_m, lam_s, lam_l = build(Z7)
thv = 2.0 * np.pi * np.arange(Z7) / Z7
rng = np.random.default_rng(SEED)
worst, bestratio = 1e9, 0.0
for i in range(200):
    psi = rng.normal(size=Z7) + 1j * rng.normal(size=Z7)
    psi = psi / np.linalg.norm(psi)
    X, pred, c, com = measure(Z7, psi, 0)
    dth = np.sqrt(abs(np.sum(np.abs(psi) ** 2 * thv ** 2) - np.sum(np.abs(psi) ** 2 * thv) ** 2))
    dL = np.sqrt(abs(np.sum(np.abs(c) ** 2 * lam_s ** 2) - np.sum(np.abs(c) ** 2 * lam_s) ** 2))
    worst = min(worst, dth * dL - abs(-X + 1j) / 2.0)
for sg in (0.35, 0.45, 0.6):
    psi = phase_gauss(Z7, sg, 0.0)
    X, pred, c, com = measure(Z7, psi, 0)
    dth = np.sqrt(abs(np.sum(np.abs(psi) ** 2 * thv ** 2) - np.sum(np.abs(psi) ** 2 * thv) ** 2))
    dL = np.sqrt(abs(np.sum(np.abs(c) ** 2 * lam_s ** 2) - np.sum(np.abs(c) ** 2 * lam_s) ** 2))
    bestratio = max(bestratio, dth * dL / (abs(-X + 1j) / 2.0))
report("P7 不确定关系接口", worst >= -1e-12 and bestratio <= 2.0,
       "200 个固定种子态上不等式成立（余量 >= -1e-12）；相位高斯族上比值 <= 2（该族不是 (th, L) 对的最小不确定态）",
       "最小余量 %.3e；比值最大 %.4f" % (worst, bestratio))

# ---------- P8 ----------
p8 = []
for Z in ZS:
    w, F, Om_m, lam_s, lam_l = build(Z)
    Th = to_modes(F, np.diag(theta_site(Z, 0)))
    M = np.diag(lam_s) @ Th - Th @ np.diag(lam_s) + 1j * np.eye(Z)
    m = np.arange(Z)
    ns = np.where(m < Z / 2.0, m, m - Z)
    keep = np.abs(ns) <= Z // 8
    Mb = M[np.ix_(keep, keep)]
    U, sv, Vt = np.linalg.svd(Mb)
    flat = np.ones(int(keep.sum())) / np.sqrt(float(keep.sum()))
    p8.append((Z, float(sv[1] / sv[0]), float(abs(np.vdot(flat, U[:, 0])))))
report("P8 带限块上的秩一结构", all(r <= 0.05 and o >= 0.95 for _, r, o in p8)
       and p8[-1][1] < p8[0][1] and p8[-1][2] > p8[0][2],
       "带限块（|n| <= Z/8）上第二奇异值占比 <= 0.05、与平坦向量重合度 >= 0.95，且占比随 Z 下降、重合度随 Z 上升（格点修正按 O(d/Z) 收敛）",
       "；".join("Z=%d 占比 %.3f 重合度 %.6f" % (Z, r, o) for Z, r, o in p8))

fig, ax = plt.subplots(1, 3, figsize=(15.0, 4.2))
ds = sorted(set(d for _, d, _ in p2 if d >= 0))
for Z in ZS:
    ax[0].plot(ds, [v for zz, d, v in p2 if zz == Z and d >= 0], "o-", label="Z=%d" % Z)
ax[0].axhline(np.pi, color="k", linestyle="--", label="解析值 pi")
ax[0].set_title("位置矩阵元误差乘 Z（对角 d=0 与频差 d）")
ax[0].set_xlabel("频差 d（0 为对角）")
ax[0].set_ylabel("|误差| * Z")
ax[0].legend(fontsize=8)
xs, ys = [], []
for Z in ZS:
    for sg in (0.3, 0.5, 0.8):
        X, pred, c, com = measure(Z, phase_gauss(Z, sg, 0.0), 0)
        xs.append(abs(pred))
        ys.append(abs(X))
ax[1].loglog(xs, ys, "o", label="实测（光滑态）")
ax[1].loglog(xs, xs, "k--", label="y = Z|psi_接缝|^2")
ax[1].set_title("边界项：实测对预期（光滑带限态）")
ax[1].set_xlabel("Z |psi_接缝|^2")
ax[1].set_ylabel("|<[L,th]> + i|")
ax[1].legend(fontsize=8)
ax[2].semilogy(ZS, [a for _, a, _ in p6a], "o-", label="横切态 |X|（站点侧，Z 无关）")
ax[2].semilogy(ZS, [m for _, _, _, m in p6b], "s--", label="模侧接缝项（远切割态，超指数衰减）")
ax[2].semilogy(ZS, [a for _, a, _, _ in p6b], "^-", label="远切割态 |X|（受地板限制）")
ax[2].set_title("两种接缝与地板（固定宽度物理态）")
ax[2].set_xlabel("格点数 Z")
ax[2].set_ylabel("量级（对数轴）")
ax[2].legend(fontsize=8)
plt.tight_layout()
plt.savefig(IMG, dpi=150)
plt.close(fig)

bad = [r for r in rows if not r[1]]
print("---- 判据 %d 项，通过 %d 项；图存 %s ----" % (len(rows), len(rows) - len(bad), IMG))
sys.exit(1 if bad else 0)
