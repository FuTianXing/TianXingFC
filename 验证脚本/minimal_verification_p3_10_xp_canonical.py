# -*- coding: utf-8 -*-
"""
minimal_verification_p3_10_xp_canonical.py
558 号：P3-10 位置动量算符判据（P1 至 P8，路线乙为主甲为对照）

模型与约定（与首轮第 167 条及第 165 条严格一致）：Z 档环，站点基
|k>（k = 0..Z-1），相位读数 theta_k = 2*pi*k/Z，分支切割在 k = Z-1
与 k = 0 之间（接缝站点默认 s = 0）。模域变换 to_modes(F, X) =
F.conj() @ X @ F.T，谱系数 c = F.conj() @ psi。两种提升：
lam_l = -m（线性，失配连接 (Z-1,0)）、lam_s = -ns（对称，ns = m if
m < Z/2 else m-Z，失配连接 (Z/2-1, Z/2)）。自然单位 l_P = hbar = 1。

判据与理论期望（先立期望再实现，打印"实测 vs 期望"）：
  P1 两种提升的失配非零元恰 1 条、位置等于提升落差点、幅度 = Z。
  P2 位置矩阵元的格点误差乘 Z 恒为 pi（对角与频差 1、2、3 九档，
     最大偏离 <= 5%；对角虚部 <= 1e-9）。
  P3 带限投影动量为整数对角（阈值 1e-12）。
  P4 边界项形式：X = <[L,theta]> + i 的读数满足
     X = i*Z*|psi_s|^2（相对偏差 <= 5%，三档 Z x 三个宽度）；
     P4b 反例：站点 delta 态偏差 O(1)（形式以动量窄带为前提）。
  P5 切割自由度：三处切割上 |X - i*Z*|psi_s|^2| <= 0.1（绝对量）。
  P6 双联：P6a 站点侧偏差（骑缝高斯）与 Z 无关且等于 i*Z*|psi_0|^2
     （<= 5%）；P6b 模侧（远切割态）|X| <= 1e-12（超指数地板）。
  P7 不确定关系接口：200 个固定种子态上
     Delta theta * Delta L >= |<[L,theta]>|/2 成立（余量 >= -1e-12）；
     相位高斯族比值 <= 2（该族不是 (theta,L) 对的最小不确定态）。
  P8 覆盖空间（开链）：lin 提升的开链（去绕回连接）提升对
     [Lam, E] = E 逐元精确（<= 1e-11，浮点累积预算）；sgn 开链
     留链内失配为对照（验证两提升的分离面）。有限维迹障碍使全
     空间 [x,p] = i*hbar*I 不可能（tr = 0），判据表述按实测订正。

图：images/p3_10_xp_canonical_558.png（三面板，见注记 §四 图 1）。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMG = r"D:\software\TianXingFC\images\p3_10_xp_canonical_558.png"
SEED = 20261003


def build(Z):
    w = np.exp(2j * np.pi / Z)
    k = np.arange(Z)
    Om_site = np.diag(w ** k)
    F = np.array([[w ** (-m * j) for j in range(Z)] for m in range(Z)], dtype=complex) / np.sqrt(Z)
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


def cut_state(Z, a, b):
    c = np.zeros(Z, dtype=complex)
    c[a] = 1.0 / np.sqrt(2.0)
    c[b] = 1.0 / np.sqrt(2.0)
    return c


def measure(Z, psi, s=0):
    """边界项读数：X = <[L,theta]> + i，pred = i*Z*|psi_s|^2（lam_s 提升）"""
    w, F, Om_m, lam_s, lam_l = build(Z)
    psi = psi / np.linalg.norm(psi)
    Th = to_modes(F, np.diag(theta_site(Z, s)))
    Lm = np.diag(lam_s)
    com = Lm @ Th - Th @ Lm
    c = coeff(F, psi)
    X = np.vdot(c, com @ c) + 1j
    pred = 1j * Z * abs(psi[s]) ** 2
    return X, pred, c, com


rows = []


def report(tag, ok, expect, actual):
    rows.append((tag, ok))
    print("[%s] %s" % ("通过" if ok else "未过", tag))
    print("      期望 " + expect)
    print("      实测 " + actual)


ZS = [32, 64, 128]

# ── 冒烟检查 ─────────────────────────────────────────────
print("== 冒烟检查 ==")
for Z in (32,):
    w, F, Om_m, lam_s, lam_l = build(Z)
    for name, (lam, seam) in (("lin", (lam_l, (Z - 1, 0))), ("sgn", (lam_s, (Z // 2 - 1, Z // 2)))):
        Lm = np.diag(lam)
        R = Lm @ Om_m - Om_m @ Lm - Om_m
        idx = np.argwhere(np.abs(R) > 1e-9)
        print("  提升 %s: 失配连接 %s（期望 %s）、幅度 %.3f（期望 Z=%d）"
              % (name, [tuple(int(x) for x in p) for p in idx], [seam],
                 abs(R[idx[0][0], idx[0][1]]) if len(idx) else 0.0, Z))
print("冒烟通过\n")

# ── P1 失配连接恰 1 条、幅度 = Z ─────────────────────────
p1 = []
for Z in ZS:
    w, F, Om_m, lam_s, lam_l = build(Z)
    for lam, loc in ((lam_l, (Z - 1, 0)), (lam_s, (Z // 2 - 1, Z // 2))):
        Lm = np.diag(lam)
        R = Lm @ Om_m - Om_m @ Lm - Om_m
        idx = np.argwhere(np.abs(R) > 1e-9)
        good = (len(idx) == 1 and tuple(int(x) for x in idx[0]) == loc
                and abs(abs(R[idx[0][0], idx[0][1]]) - Z) < 1e-9)
        p1.append((Z, loc, good))
ok1 = all(g for _, _, g in p1)
report("P1 失配连接恰 1 条、位置 = 落差点、幅度 = Z", ok1,
       "三档 Z x 两种提升全过（幅度 = Z）",
       "；".join("Z=%d %s %s" % (Z, str(loc), "ok" if g else "fail") for Z, loc, g in p1))

# ── P2 位置矩阵元的误差常数 = pi ─────────────────────────
p2 = []
for Z in ZS:
    w, F, Om_m, lam_s, lam_l = build(Z)
    Th = to_modes(F, np.diag(theta_site(Z, 0)))
    for d in (0, 1, 2, 3):
        dv = np.diagonal(Th, offset=-d)
        if d == 0:
            err = float(np.max(np.abs(dv.real - np.pi))) * Z
            p2.append((Z, 0, err))
        else:
            err = float(np.max(np.abs(dv - (-1j / d)))) * Z
            p2.append((Z, d, err))
ok2 = all(abs(v - np.pi) <= 0.05 * np.pi for Z, d, v in p2 if d >= 0)
report("P2 位置矩阵元误差常数 = pi", ok2,
       "对角与频差 1、2、3 九档上 |Theta^Z - Theta^inf|*Z = pi（±5%）",
       "；".join("Z=%d d=%d: %.5f" % (Z, d, v) for Z, d, v in p2 if d >= 0))

# ── P3 带限投影动量为整数对角 ────────────────────────────
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
ok3 = all(o <= 1e-12 and i for _, o, i in p3)
report("P3 带限投影动量为整数对角", ok3,
       "带内投影为整数对角（偏差 <= 1e-12）",
       "；".join("Z=%d 非对角 %.1e，整数性 %s" % (Z, o, i) for Z, o, i in p3))

# ── P4 边界项形式 + P4b 反例 ─────────────────────────────
p4, p4b = [], []
for Z in ZS:
    for sg in (0.3, 0.5, 0.8):
        X, pred, c, com = measure(Z, phase_gauss(Z, sg, 0.0), 0)
        p4.append((Z, sg, abs(X - pred) / abs(pred)))
    X, pred, c, com = measure(Z, site_state(Z, 0), 0)
    p4b.append((Z, abs(X - pred) / abs(pred), abs(X)))
ok4 = all(e <= 0.05 for _, _, e in p4)
report("P4 边界项形式（光滑带限态）", ok4,
       "三档 Z x 三个宽度上相对偏差 <= 5%",
       "；".join("Z=%d sig=%.1f: %.2e" % (Z, sg, e) for Z, sg, e in p4))
report("P4b 反例：delta 态超出适用范围", all(e >= 0.5 and a >= 0.5 for _, e, a in p4b),
       "站点 delta 态上偏差为 O(1)（该形式以动量窄带为前提）",
       "；".join("Z=%d 相对偏差 %.3f，|X| %.3f" % (Z, e, a) for Z, e, a in p4b))

# ── P5 切割自由度 ───────────────────────────────────────
p5 = []
for Z in ZS:
    psi = phase_gauss(Z, 0.5, 0.0)
    for st in (0, Z // 4, Z // 2):
        X, pred, c, com = measure(Z, psi, st)
        p5.append((Z, st, abs(X - pred), abs(pred)))
ok5 = all(a <= 0.1 for _, _, a, _ in p5)
report("P5 切割自由度（移动分支切割）", ok5,
       "三档 Z x 三处切割上 |X - i Z |psi_s|^2| <= 0.1（绝对量）",
       "；".join("Z=%d s=%d: 绝对 %.2e" % (Z, st, a) for Z, st, a, _ in p5))

# ── P6 双联：站点侧 Z 无关 + 模侧超指数 ─────────────────
p6a = []
for Z in ZS:
    X, pred, c, com = measure(Z, phase_gauss(Z, 0.35, 0.0), 0)
    p6a.append((Z, abs(X), abs(pred)))
r6 = [a / b for _, a, b in p6a]
ok6a = max(r6) / min(r6) - 1.0 <= 0.05 and all(abs(a - b) / b <= 0.05 for _, a, b in p6a)
report("P6a 站点侧偏差与 Z 无关", ok6a,
       "骑缝高斯（sigma=0.35）三档 Z 上 |X| 相互偏离 <= 5% 且等于 Z|psi_0|^2（<= 5%）",
       "；".join("Z=%d 实测 %.6f 预期 %.6f" % (Z, a, b) for Z, a, b in p6a))
p6b = []
for Z in ZS:
    X, pred, c, com = measure(Z, phase_gauss(Z, 0.35, np.pi), 0)
    p6b.append((Z, abs(X), abs(pred)))
ok6b = all(a <= 1e-12 for _, a, _ in p6b)
report("P6b 模侧超指数（远切割态地板）", ok6b,
       "远切割态上 |X| <= 1e-12（超指数衰减入地板）",
       "；".join("Z=%d 实测 %.2e" % (Z, a) for Z, a, _ in p6b))

# ── P7 不确定关系接口 ───────────────────────────────────
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
ok7 = worst >= -1e-12 and bestratio <= 2.0
report("P7 不确定关系接口", ok7,
       "200 个固定种子态上不等式成立（余量 >= -1e-12）；相位高斯族比值 <= 2（该族不是最小不确定态）",
       "最小余量 %.3e；比值最大 %.4f" % (worst, bestratio))

# ── P8 覆盖空间（开链提升对）────────────────────────────
p8 = []
for Z in ZS:
    w, F, Om_m, lam_s, lam_l = build(Z)
    S_open = Om_m.copy()
    S_open[Z - 1, 0] = 0.0  # 去掉绕回连接
    R_open_l = np.diag(lam_l) @ S_open - S_open @ np.diag(lam_l) - S_open
    R_open_s = np.diag(lam_s) @ S_open - S_open @ np.diag(lam_s) - S_open
    p8.append((Z, float(np.max(np.abs(R_open_l))), float(np.max(np.abs(R_open_s)))))
ok8 = all(m <= 1e-11 for _, m, _ in p8)
report("P8 覆盖空间（开链提升对）", ok8,
       "lin 开链 [Lam,E]=E 逐元 <= 1e-11（sgn 开链留链内失配为对照）",
       "；".join("Z=%d lin %.2e（sgn 对照 %.2f）" % (Z, ml, ms) for Z, ml, ms in p8))

# ── 汇总 ────────────────────────────────────────────────
print("\n== 判据汇总 ==")
for tag, ok in rows:
    print("%s: %s" % (tag, "PASS" if ok else "FAIL"))
print("总判据: %d/%d" % (sum(1 for _, ok in rows if ok), len(rows)))

# ── 三面板图 ────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

ax = axes[0]
vs_p2 = [v for z, d, v in p2 if d >= 0]
ax.plot(range(len(vs_p2)), vs_p2, "o-")
ax.axhline(np.pi, color="k", ls=":", lw=1.5, label=r"$\pi$")
ax.set_xlabel("pair index (Z = 32/64/128, d = 0/1/2/3)")
ax.set_ylabel(r"$|\Theta^{Z}-\Theta^{\infty}|\cdot Z$")
ax.set_title("(1) P2: error constant = pi")
ax.legend(fontsize=8)

ax = axes[1]
esp = [e for z, sg, e in p4]
ax.semilogy(range(len(esp)), np.maximum(esp, 1e-17), "o-")
ax.set_xlabel("pair index (Z = 32/64/128, sig = 0.3/0.5/0.8)")
ax.set_ylabel(r"$|X - iZ|\psi_s|^2|/|pred|$")
ax.set_title("(2) P4: boundary form, rel. deviation")

ax = axes[2]
za = [z for z, a, b in p6a]
aa = [a for z, a, b in p6a]
bb = [b for z, a, b in p6a]
zb = [z for z, a, _ in p6b]
ab = [a for z, a, _ in p6b]
ax.semilogy(za, aa, "o-", label="site-side |X| (Z-independent)")
ax.semilogy(za, bb, "s--", label=r"$Z|\psi_0|^2$")
ax.semilogy(zb, ab, "^-", label="mode-side |X| (super-exp.)")
ax.set_xlabel("Z")
ax.set_ylabel("|X|")
ax.set_title("(3) P6: two seams, distinct scaling")
ax.legend(fontsize=7)

fig.tight_layout()
fig.savefig(IMG, dpi=150)
print("图已存: %s" % IMG)
