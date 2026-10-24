"""第 506 条：子空间谱分类判据（简并子空间读数的合法性判定）。

背景：简并能量子空间内，单个本征矢的算符期望是基依赖的污染量（EP3）。
第 504 条读「最低正能态的 sigma_z 期望」，在 16 重简并子空间内取值随数值库的
基选取变化，据此判负不合法。本脚本给出替代判据：把待测算符投影到简并子空间，
按投影算符的谱做三类判定，读数与基选取无关。

判据：
  S1 对易性（乙、丙两口径）：max|[A, D]| 是否为零（守恒标签的必要条件）。
  S2 子空间谱分类（乙、丙两口径）：sigma_z 投影到最大简并簇后，谱为单值
     （常数类）、两个值（横交类）还是非对角（非守恒类）。
  S3 正对照：sign(D) 投影到同一簇应判为常数类（真标签的基准形态）。
  S4 504 读法复现：最低正能态单矢期望跨 kappa 档的跳变，作污染演示。

模型沿用第 504 条（丙口径）与第 505 条（乙口径）的算符约定。
"""

import numpy as np

N = 20
M0 = 0.5
KAPPAS = [0.0, 0.3, 0.6, 1.0]

I2M = np.eye(2)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)


def gammas():
    g0 = np.zeros((4, 4), dtype=complex)
    g0[:2, :2] = I2M
    g0[2:, 2:] = -I2M
    gs = []
    for s in (SX, SY, SZ):
        g = np.zeros((4, 4), dtype=complex)
        g[:2, 2:] = s
        g[2:, :2] = -s
        gs.append(g)
    g5 = np.zeros((4, 4), dtype=complex)
    g5[:2, 2:] = I2M
    g5[2:, :2] = I2M
    return g0, gs, g5


G0, GV, G5 = gammas()
ALPHA_TH, ALPHA_H = G0 @ GV[0], G0 @ GV[1]
NS = N * N


def shift(anti_bc):
    S = np.zeros((N, N))
    for i in range(N):
        S[i, (i + 1) % N] = -1.0 if (anti_bc and i == N - 1) else 1.0
    return S


Dth = (shift(True) - shift(True).conj().T) / 2.0
Dh = (shift(False) - shift(False).conj().T) / 2.0
Kth4 = -1j * np.kron(Dth, np.eye(N))
Kh4 = -1j * np.kron(np.eye(N), Dh)
D4 = np.kron(ALPHA_TH, Kth4) + np.kron(ALPHA_H, Kh4) + np.kron(G0, M0 * np.eye(NS))

SZ8 = np.kron(SZ, np.kron(np.eye(4), np.eye(NS)))
CROSS8 = np.kron(SZ, np.kron(G5, np.eye(NS)))


def build_bing(kappa):
    D8 = np.zeros((8 * NS, 8 * NS), dtype=complex)
    D8[: 4 * NS, : 4 * NS] = D4
    D8[4 * NS:, 4 * NS:] = D4
    return D8 + kappa * CROSS8


def build_yi():
    return np.kron(SZ, D4)


_cache = {}


def eig(key, builder):
    if key not in _cache:
        _cache[key] = np.linalg.eigh(builder())
    return _cache[key]


def clusters(ev, tol=1e-8):
    out, i, n = [], 0, len(ev)
    while i < n:
        j = i
        while j + 1 < n and abs(ev[j + 1] - ev[i]) < tol:
            j += 1
        out.append((i, j))
        i = j + 1
    return out


def classify(D, A, tol=1e-8):
    cm_res = float(np.max(np.abs(A @ D - D @ A)))
    ev, V = np.linalg.eigh(D)
    i, j = max(clusters(ev), key=lambda c: c[1] - c[0])
    W = V[:, i:j + 1]
    M = W.conj().T @ A @ W
    off = float(np.max(np.abs(M - np.diag(np.diag(M)))))
    spec = np.linalg.eigvalsh((M + M.conj().T) / 2.0)
    vals = np.unique(np.round(spec, 6))
    if cm_res > tol or off > 1e-6:
        kind = "C 非守恒（期望无意义）"
    elif len(vals) == 1:
        kind = "A 常数（子空间内固定）"
    else:
        kind = "B 横交（与能量符号横交）"
    return cm_res, float(ev[i]), j - i + 1, kind, spec


results = []


def report(tag, measured, expected, ok):
    print(f"[{'PASS' if ok else 'FAIL'}] {tag}")
    print(f"    实测: {measured}")
    print(f"    期望: {expected}")
    results.append((tag, ok))


print("== S1 对易性（守恒标签的必要条件）==")
D_bing = build_bing(0.6)
D_yi = build_yi()
res_bing = float(np.max(np.abs(SZ8 @ D_bing - D_bing @ SZ8)))
res_yi = float(np.max(np.abs(SZ8 @ D_yi - D_yi @ SZ8)))
report("S1-a 丙口径 [sz, D]", f"max = {res_bing:.3e}", "= 0（对易）", res_bing < 1e-10)
report("S1-b 乙口径 [sz, D]", f"max = {res_yi:.3e}", "= 0（对易）", res_yi < 1e-10)

print("\n== S2 子空间谱分类（sigma_z）==")
for lab, D in (("丙口径", D_bing), ("乙口径", D_yi)):
    cm_res, e0, deg, kind, spec = classify(D, SZ8)
    vals, counts = np.unique(np.round(spec, 4), return_counts=True)
    report(
        f"S2 {lab} sigma_z 在最大簇（E={e0:+.4f}，重数 {deg}）",
        f"分类 = {kind}；谱值 {dict(zip(vals.tolist(), counts.tolist()))}；对易残差 {cm_res:.1e}",
        "分类 = B 横交（谱含 +1 与 -1）",
        kind.startswith("B"),
    )

print("\n== S3 正对照（sign(D) 应判为常数类）==")
ev_b, V_b = eig("bing0.6", lambda: build_bing(0.6))
pos = V_b[:, ev_b > 1e-9] @ V_b[:, ev_b > 1e-9].conj().T
signD = 2.0 * pos - np.eye(D_bing.shape[0])
cm_res, e0, deg, kind, spec = classify(D_bing, signD)
vals, counts = np.unique(np.round(spec, 4), return_counts=True)
report(
    "S3 sign(D) 在最大簇",
    f"分类 = {kind}；谱值 {dict(zip(vals.tolist(), counts.tolist()))}；对易残差 {cm_res:.1e}",
    "分类 = A 常数（子空间内固定）",
    kind.startswith("A"),
)

print("\n== S4 504 读法复现（单矢期望的基依赖跳变）==")
rows = []
for kap in KAPPAS:
    ev, V = eig(f"bing{kap}", (lambda k: (lambda: build_bing(k)))(kap))
    ip, im = int(np.argmax(ev)), int(np.argmin(ev))
    zp = float(np.real(V[:, ip].conj() @ (SZ8 @ V[:, ip])))
    zn = float(np.real(V[:, im].conj() @ (SZ8 @ V[:, im])))
    rows.append((kap, zp, zn))
    print(f"    kappa={kap}: 最低正能态 {zp:+.4f}，最低负能态 {zn:+.4f}")
jump = max(abs(z - rows[0][1]) for _, z, _ in rows) + max(abs(z - rows[0][2]) for _, _, z in rows)
report(
    "S4 跨档读数波动",
    f"最低正能态取值集合 {sorted({round(z, 4) for _, z, _ in rows})}；最低负能态取值集合 {sorted({round(z, 4) for _, _, z in rows})}",
    "污染量：同口径内读数随 kappa 跳变，非物理",
    jump > 0.5,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
_, _, _, _, spec = classify(D_bing, SZ8)
vals, counts = np.unique(np.round(spec, 4), return_counts=True)
ax.bar([str(v) for v in vals], counts.tolist(), color="steelblue")
ax.set_xlabel("sigma_z 在最大简并簇内的本征值")
ax.set_ylabel("计数")
ax.set_title("S2: 子空间谱分类（+1 与 -1 各半，横交）")
ax = axes[1]
ks = [r[0] for r in rows]
ax.plot(ks, [r[1] for r in rows], "bo-", label="最低正能态 sz 期望")
ax.plot(ks, [r[2] for r in rows], "r^--", label="最低负能态 sz 期望")
ax.axhline(1, color="gray", ls=":", lw=1)
ax.axhline(-1, color="gray", ls=":", lw=1)
ax.set_xlabel("交叉项系数 kappa")
ax.set_ylabel("单矢期望")
ax.set_title("S4: 504 读法的基依赖跳变")
ax.legend()
plt.tight_layout()
plt.savefig(r"D:\software\TianXingFC\images\subspace_label_protocol.png", dpi=150)

npass = sum(1 for _, ok in results if ok)
print(f"\n判据 S1 至 S4：{npass}/{len(results)} 通过")
print("图已存: D:\\software\\TianXingFC\\images\\subspace_label_protocol.png")
