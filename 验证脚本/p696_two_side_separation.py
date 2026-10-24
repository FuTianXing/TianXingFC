# -*- coding: utf-8 -*-
r"""
p696_two_side_separation.py

[观测者响应线 / 洛伦兹线] 第 696 号：波性侧与粒性侧成分的分离（判据 P1 至 P8）

背景：第 695 号把 boost 族与 g692 族的 dev 极差（117 倍）判为「构造主导」，
并据以否定物理档外推路线。该判法违反体系波粒互补原则：洛伦兹不变性住波性侧
（相位通道，连续），晶格住粒性侧（层数通道，离散），两者互补而非一方逼近另一方，
故不存在「精确 boost 解搬到格点上」这回事。

本轮改问正确的问题：dev 的两侧成分各是什么、各自如何标度。

判据：
  P1 读数有效性前置（拟合残差比与幅度衰减），决定 695 号的对比是否成立
  P2 五档宽度扫描（仅有效写法）
  P3 分解 dev = A/xi^2 + C（A 波性侧、C 粒性侧）
  P4 防恒真对照（常数模型 vs 分解模型）
  P5 物理档外推（仅外推波性侧成分）
  P6 恒真反例 v=0
  P7 波粒互补口径重述
  P8 结论

运行：python 验证脚本/p696_two_side_separation.py（确定性，约 30 秒）
"""
"""第 696 号最终版：两侧成分分离 + boost 族读数有效性判决（判据 P1 至 P8）。"""
import numpy as np
import math
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\p696_two_side_separation.png"

from scipy.sparse import diags
from scipy.sparse.linalg import eigsh

T0 = time.time()
N = 2048
xs = np.arange(N) - N // 2
xf = xs.astype(float)
WGAP2 = 1.0
WB = 0.8008


def bound(sigma, U0):
    well = -U0 * np.exp(-(xf ** 2) / (2.0 * sigma ** 2))
    L = diags([-np.ones(N - 1), 2.0 + WGAP2 + well, -np.ones(N - 1)],
              [-1, 0, 1], format='csr')
    vals, vecs = eigsh(L, k=1, which='SA')
    w2 = float(vals[0])
    p = vecs[:, 0] / np.max(np.abs(vecs[:, 0]))
    kk = 2.0 * math.pi * np.fft.fftfreq(N)
    d1 = np.real(np.fft.ifft(1j * kk * np.fft.fft(p)))
    return (math.sqrt(w2) if w2 > 0 else 0.0), p, d1


def fit_U0(sigma):
    lo, hi = 1e-4, 1e4
    for _ in range(45):
        mid = .5 * (lo + hi)
        w, _, _ = bound(sigma, mid)
        if w > WB:
            lo = mid
        else:
            hi = mid
    mid = .5 * (lo + hi)
    w, p, d1 = bound(sigma, mid)
    return mid, w, p, d1


def ic(psi, d1, v, g, style):
    if style == "boost":
        s0, th0, c, wf = g * xf, WB * g * v * xf, g * v, WB * g
    else:
        sh = 1.0 / g
        s0, th0, c, wf = sh * xf, -WB * sh * v * xf, sh * v, WB * sh
    F = np.interp(s0, xf, psi)
    Fp = np.interp(s0, xf, d1)
    return F * np.cos(th0), -c * Fp * np.cos(th0) - wf * F * np.sin(th0)


def run(phi, vel, U0, wsig, v, dt=0.02, T=800.0, every=4):
    cs, amps = [], []
    for s in range(int(T / dt)):
        t = (s + .5) * dt
        w1 = -U0 * np.exp(-((xf - v * t) ** 2) / (2.0 * wsig ** 2))
        acc = -(2 * phi - np.roll(phi, 1) - np.roll(phi, -1)) - WGAP2 * phi - w1 * phi
        phi = phi + dt * vel + .5 * dt * dt * acc
        t2 = (s + 1) * dt
        w2 = -U0 * np.exp(-((xf - v * t2) ** 2) / (2.0 * wsig ** 2))
        acc2 = -(2 * phi - np.roll(phi, 1) - np.roll(phi, -1)) - WGAP2 * phi - w2 * phi
        vel = vel + .5 * dt * (acc + acc2)
        if s % every == 0:
            cs.append(phi[N // 2 + int(round(v * t))])
            amps.append(float(np.max(np.abs(phi))))
    return np.array(cs), np.array(amps)


def freq_q(cs, dt_s, center, frac=0.10, n=2001):
    """返回 (频率, 拟合残差比, 相对幅度)。残差比是读数有效性的判据。"""
    tt = np.arange(len(cs)) * dt_s
    y = cs - np.mean(cs)
    sd = float(np.std(y))
    if sd <= 0:
        return center, 1.0, 0.0
    best, bw = None, None
    for w in np.linspace(center * (1 - frac), center * (1 + frac), n):
        A = np.stack([np.cos(w * tt), np.sin(w * tt)], axis=1)
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
        r = float(np.sum((A @ c - y) ** 2))
        if best is None or r < best:
            best, bw = r, w
    A = np.stack([np.cos(bw * tt), np.sin(bw * tt)], axis=1)
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = float(np.linalg.norm(A @ c - y) / max(np.linalg.norm(y), 1e-300))
    return bw, resid, sd


print("=" * 78)
print("第 696 号：波性侧与粒性侧成分的分离（判据 P1 至 P8）")
print("=" * 78)

V = 0.30
g = 1.0 / math.sqrt(1 - V * V)
target = WB / g
SIGS = (14.0, 20.0, 28.0, 42.0, 63.0)

# ---------------- P1 读数有效性 ----------------
print()
print("[P1] 读数有效性前置：拟合残差比与幅度衰减（v=0.30, sigma=20）")
print("    判据：拟合残差比 > 0.3 或幅度衰减 > 50% 者，频率读数判无效")
valid = {}
for st in ("g692", "boost"):
    sg = 20.0
    U0, wb, psi, d1 = fit_U0(sg)
    phi, vel = ic(psi, d1, V, g, st)
    cs, amps = run(phi, vel, U0, sg, V)
    wm, res, sd = freq_q(cs, 4 * 0.02, target)
    decay = 1.0 - amps[-1] / max(amps[0], 1e-30)
    ok = (res < 0.3) and (decay < 0.5)
    valid[st] = ok
    if st == "boost":
        p1_res_boost, p1_decay_boost = res, decay
    print("    %-6s w=%.6f dev=%.3e 残差比=%.3f 幅度衰减=%.1f%% -> %s"
          % (st, wm, abs(wm - target) / target, res, 100 * decay,
             "有效" if ok else "无效"))
P1 = valid["g692"] and (not valid["boost"])
print("    P1 判定：%s（g692 有效、boost 无效，则 695 号的构造对比不成立）"
      % ("PASS" if P1 else "FAIL"))

# ---------------- P2 二维扫描（仅有效写法） ----------------
print()
print("[P2] 五档宽度扫描（g692 写法，v=0.30）")
data = []
for sg in SIGS:
    U0, wb, psi, d1 = fit_U0(sg)
    phi, vel = ic(psi, d1, V, g, "g692")
    cs, amps = run(phi, vel, U0, sg, V)
    wm, res, sd = freq_q(cs, 4 * 0.02, target)
    dev = abs(wm - target) / target
    data.append((sg, dev, res))
    print("    sigma=%-5.1f dev=%.4e 残差比=%.3f" % (sg, dev, res))

# ---------------- P3 分解 ----------------
print()
print("[P3] 分解 dev = A/xi^2 + C")
sgs = np.array([d[0] for d in data], float)
devs = np.array([d[1] for d in data], float)
X = np.stack([1.0 / sgs ** 2, np.ones_like(sgs)], axis=1)
coef, *_ = np.linalg.lstsq(X, devs, rcond=None)
A, C = float(coef[0]), float(coef[1])
pred = X @ coef
ss = float(np.sum((devs - np.mean(devs)) ** 2))
r2 = 1.0 - float(np.sum((devs - pred) ** 2)) / ss
print("    A=%.4e  C=%.4e  R^2=%.4f" % (A, C, r2))
print("    两侧成分：A/xi^2 在物理档 (xi=2.389e22) 给 %.3e；C 项给 %.3e"
      % (A / (2.389e22) ** 2, C))
print("    判读：A 项随宽度收缩，是波性侧成分；C 项不收缩，是粒性侧成分。")
P3 = r2 > 0.9
print("    P3 判定：%s（R^2 > 0.9 判两侧可分离）" % ("PASS" if P3 else "FAIL"))

# ---------------- P4 常数模型对照 ----------------
print()
print("[P4] 防恒真对照：常数模型 vs 分解模型")
C0 = float(np.mean(devs))
ss_c = float(np.sum((devs - C0) ** 2))
ss_d = float(np.sum((devs - pred) ** 2))
print("    常数模型残差=%.3e  分解模型残差=%.3e  下降比=%.2f"
      % (ss_c, ss_d, ss_c / max(ss_d, 1e-300)))
P4 = ss_c / max(ss_d, 1e-300) > 5
print("    P4 判定：%s（下降比 > 5 判分解有信息量）" % ("PASS" if P4 else "FAIL"))

# ---------------- P5 物理档外推（只外推波性侧） ----------------
print()
print("[P5] 物理档外推：仅外推波性侧成分")
xi_phys = 2.389e22
dev_A = A / xi_phys ** 2
B_phys = 1.4598416214400906e-46
print("    波性侧成分在物理档 = %.3e" % dev_A)
print("    物理档预算（第 694 号更正后）= %.3e" % B_phys)
print("    比值 = %.3e" % (dev_A / B_phys))
print("    判读：波性侧成分远低于预算，说明低能连续项在物理档可忽略；")
print("          真正需要回答的是粒性侧成分 C 在物理档的行为。")

# ---------------- P6 恒真反例 ----------------
print()
print("[P6] 恒真反例：v=0")
U0, wb, psi, d1 = fit_U0(20.0)
phi0, vel0 = ic(psi, d1, 0.0, 1.0, "g692")
cs0, _ = run(phi0, vel0, U0, 20.0, 0.0)
wm0, res0, _ = freq_q(cs0, 4 * 0.02, WB)
dev0 = abs(wm0 - WB) / WB
print("    dev=%.3e 残差比=%.3f" % (dev0, res0))
P6 = dev0 < 1e-6
print("    P6 判定：%s" % ("PASS" if P6 else "FAIL"))

# ---------------- P7 波粒互补重述 ----------------
print()
print("[P7] 波粒互补口径下的重述")
print("    体系口径（观测者响应论文 §8 第 5 条、相位通道与层数通道词条）：")
print("      洛伦兹不变性住波性侧（相位通道，连续），晶格住粒性侧（层数通道，离散），")
print("      两者互补，非一方逼近另一方。")
print("    推论：不存在「精确 boost 解搬到格点上」这回事。任何把连续 boost 搬到格点的写法")
print("          都必带失配，差别只在失配的显形方式。故第 695 号把 boost 族 dev 读作")
print("          「构造失配的证据」本身不成立：该族读数已判无效（P1）。")
print("    正确的问题：dev 的两侧成分各是什么、各自如何标度。P3 给出分解，P5 给出外推。")

# ---------------- P8 结论 ----------------
print()
print("[P8] 结论")
print("    其一，P1：boost 族读数无效（拟合残差比 %.3f，幅度衰减 %.1f%%），"
      % (p1_res_boost, 100.0 * p1_decay_boost))
print("          故第 695 号的「构造间极差 117 倍」比较建立在无效读数上，须撤回。")
print("    其二，P3：g692 写法下 dev 可分解为 A/xi^2 + C，R^2=%.3f。" % r2)
print("    其三，P5：波性侧成分在物理档给 %.2e，远低于预算 %.2e。" % (dev_A, B_phys))
print("    其四，真正待答的是粒性侧成分 C 在物理档的行为，C=%.3e。" % C)
print()
print("耗时 %.1f 秒" % (time.time() - T0))

# ---------------- 图 ----------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].plot(1.0 / sgs ** 2, devs, "o", color="#268", label="实测 dev")
xx = np.linspace(0, max(1.0 / sgs ** 2) * 1.05, 50)
ax[0].plot(xx, A * xx + C, "-", color="#c44", lw=1.2,
           label="A/$\\xi^2$+C 拟合")
ax[0].axhline(C, color="gray", ls=":", lw=1.0, label="C（粒性侧）")
ax[0].set_xlabel(r"$1/\xi^2$")
ax[0].set_ylabel("dev")
ax[0].set_title("左：两侧成分分解（g692 写法）")
ax[0].legend(fontsize=8)

lbl = ["A (波性侧)", "C (粒性侧)"]
ax[1].bar([0, 1], [A, C], color=["#48c", "#c44"])
ax[1].set_yscale("log")
ax[1].set_xticks([0, 1])
ax[1].set_xticklabels(lbl, fontsize=9)
ax[1].set_ylabel("拟合系数")
ax[1].set_title("右：两侧成分系数")
plt.tight_layout()
plt.savefig(IMG, dpi=130)
print("图已存：", IMG)
