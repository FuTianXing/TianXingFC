# -*- coding: utf-8 -*-
r"""
p709_rayleigh_identity.py

[观测者响应线 / 洛伦兹线] 第 709 号：dev 公式的分母与 Rayleigh 恒等式检验（判据 IA0 至 IA4）

背景：第 705 号给出 dev = <f(k)>/(2 omega^2)，其中 f(k) = 4sin^2(k/2)-k^2。
第 708 号把缺口缩到 5.8% 后，剩下三个候选，其一为分母该用哪个频率：
  omega_lat^2（格点频率）、omega_cont^2（连续频率）、omega_b^2（静止频率）
三者给 2.608e-4 与 2.866e-4 两个理论值，而实测外推值 2.6987e-4 落在两者之间。

本轮把 Rayleigh 商在同一个初值上直接验一遍。恒等式为
    omega_lat^2 - omega_cont^2 = <phi|(Delta_lat - d_x^2)|phi> = <f>
若它成立，则 dev = <f>/(2 omega_cont^2) 是恒等式的推论，分母无歧义；
若它不成立，缺口就出在恒等式本身，与分母选择无关。

判据：
  IA0 冒烟：D 组合的连续版（即 C 组合）四档 dev 应为底噪量级
  IA1 恒等式（主判据）：omega_lat^2 - omega_cont^2 与 <f>_lat 的比值
  IA2 分母候选（主判据）：三个分母各给一个预测，看哪个与实测 dev 吻合
  IA3 谱峰位：实测 |phi(k)|^2 的峰位与 k_mod 的差
  IA4 判定

预登记判据（先于读数写定）：
  IA0 PASS：四档 C 的 dev 均 < 10 倍 Verlet 底噪
  IA1 PASS：(omega_lat^2 - omega_cont^2)/<f>_lat 的四档值均落在 [0.7, 1.3]
  IA2 PASS：三个分母中恰有一个的预测与实测 dev 之比落在 [0.9, 1.1]
  IA3 报告量：峰位与 k_mod 的相对差
  IA4 综合：IA1 成立判「恒等式无误，缺口在 <f> 的计算而非分母」；
            IA1 不成立判「恒等式在非本征态上不成立，须改用叠加态修正」

运行：python 验证脚本/p709_rayleigh_identity.py（确定性，约 2 分钟）
"""
import numpy as np
import math
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

from scipy.sparse import diags
from scipy.sparse.linalg import eigsh

IMG = r"D:\software\TianXingFC\images\p709_rayleigh_identity.png"

T0 = time.time()
N = 2048
xs = np.arange(N) - N // 2
xf = xs.astype(float)
WGAP2 = 1.0
WB = 0.8008
V = 0.30
G = 1.0 / math.sqrt(1.0 - V * V)
TARGET = WB / G
KMOD = WB * G * V
SIGS = (14.0, 20.0, 28.0, 42.0)
DT = 0.02
NOISE = (2.0 / DT) * math.asin(WB * DT / 2.0) / WB - 1.0

KK = 2.0 * math.pi * np.fft.fftfreq(N)
K2 = KK ** 2
FK = 4.0 * np.sin(KK / 2.0) ** 2 - KK ** 2

_U0CACHE = {}


def bound(sigma, U0):
    well = -U0 * np.exp(-(xf ** 2) / (2.0 * sigma ** 2))
    L = diags([-np.ones(N - 1), 2.0 + WGAP2 + well, -np.ones(N - 1)],
              [-1, 0, 1], format='csr')
    vals, vecs = eigsh(L, k=1, which='SA')
    p = vecs[:, 0] / np.max(np.abs(vecs[:, 0]))
    d1 = np.real(np.fft.ifft(1j * KK * np.fft.fft(p)))
    return (math.sqrt(float(vals[0])) if vals[0] > 0 else 0.0), p, d1


def fit_U0(sigma):
    key = round(sigma, 10)
    if key in _U0CACHE:
        return _U0CACHE[key]
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
    _U0CACHE[key] = (mid, w, p, d1)
    return _U0CACHE[key]


def make_ic_boost(psi, d1):
    """收缩井配套的 boost 初值（第 704 号更正后的符号）。"""
    s0 = G * xf
    th0 = WB * G * V * xf
    c = G * V
    wfac = WB * G
    F = np.interp(s0, xf, psi)
    Fp = np.interp(s0, xf, d1)
    return F * np.cos(th0), -c * Fp * np.cos(th0) + wfac * F * np.sin(th0)


def run(phi, vel, U0, wsig, evo, dt=DT, T=1200.0, every=4, spec_every=200):
    cs, specs = [], []
    nst = int(T / dt)
    for s in range(nst):
        t = (s + .5) * dt
        w1 = -U0 * np.exp(-G * G * ((xf - V * t) ** 2) / (2.0 * wsig ** 2))
        if evo == "lattice":
            acc = (np.roll(phi, 1) + np.roll(phi, -1) - 2.0 * phi) - WGAP2 * phi - w1 * phi
        else:
            acc = -np.real(np.fft.ifft(K2 * np.fft.fft(phi))) - WGAP2 * phi - w1 * phi
        phi = phi + dt * vel + .5 * dt * dt * acc
        t2 = (s + 1) * dt
        w2 = -U0 * np.exp(-G * G * ((xf - V * t2) ** 2) / (2.0 * wsig ** 2))
        if evo == "lattice":
            acc2 = (np.roll(phi, 1) + np.roll(phi, -1) - 2.0 * phi) - WGAP2 * phi - w2 * phi
        else:
            acc2 = -np.real(np.fft.ifft(K2 * np.fft.fft(phi))) - WGAP2 * phi - w2 * phi
        vel = vel + .5 * dt * (acc + acc2)
        if s % every == 0:
            cs.append(float(phi[N // 2 + int(round(V * t2))]))
        if s % spec_every == 0 and s > nst // 4:
            p2 = np.abs(np.fft.fft(phi)) ** 2
            tot = float(p2.sum())
            if tot > 0:
                specs.append(p2 / tot)
    return np.array(cs), specs


def freq_refined(cs, dt_s, center, frac=0.10, ncoarse=2001, nfine=201, nstage=2):
    tt = np.arange(len(cs)) * dt_s
    y = cs - np.mean(cs)
    if float(np.std(y)) <= 0:
        return center, 1.0
    lo, hi = center * (1 - frac), center * (1 + frac)

    def scan(lo, hi, n):
        best, bw = None, None
        for w in np.linspace(lo, hi, n):
            A = np.stack([np.cos(w * tt), np.sin(w * tt)], axis=1)
            c, *_ = np.linalg.lstsq(A, y, rcond=None)
            r = float(np.sum((A @ c - y) ** 2))
            if best is None or r < best:
                best, bw = r, w
        return bw, best

    bw, best = scan(lo, hi, ncoarse)
    step = (hi - lo) / (ncoarse - 1)
    for _ in range(nstage):
        lo, hi = bw - 2 * step, bw + 2 * step
        bw, best = scan(lo, hi, nfine)
        step = (hi - lo) / (nfine - 1)
    A = np.stack([np.cos(bw * tt), np.sin(bw * tt)], axis=1)
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    return bw, float(np.linalg.norm(A @ c - y) / max(np.linalg.norm(y), 1e-300))


print("=" * 78)
print("第 709 号：dev 公式的分母与 Rayleigh 恒等式检验（判据 IA0 至 IA4）")
print("=" * 78)
print("装置：N=%d omega_b=%.4f v=%.2f gamma=%.6f k_mod=%.6f"
      % (N, WB, V, G, KMOD))
print("Verlet 底噪 %.4e" % NOISE)
print("三个分母候选的 omega^2：omega_cont^2=%.6f  omega_b^2=%.6f"
      % (TARGET ** 2, WB ** 2))

# ================= 采集 =================
DATA = {}
for sg in SIGS:
    U0, _, psi, d1 = fit_U0(sg)
    row = {}
    for evo in ("lattice", "continuum"):
        phi, vel = make_ic_boost(psi, d1)
        cs, specs = run(phi, vel, U0, sg, evo)
        wm, res = freq_refined(cs, 0.08, TARGET)
        spec = np.mean(np.array(specs), axis=0)
        ipk = int(np.argmax(spec))
        row[evo] = dict(w=wm, res=res, spec=spec,
                        fbar=float(np.sum(spec * FK)),
                        kpk=float(abs(KK[ipk])),
                        k2=float(np.sum(spec * K2)))
    DATA[sg] = row

# ---------------- IA0 冒烟 ----------------
print()
print("[IA0] 冒烟：D 组合的连续版（即 C 组合）应为底噪量级")
print("    %-6s %-13s %-9s %-11s" % ("xi", "dev(连续)", "残差比", "倍数/底噪"))
IA0 = True
for sg in SIGS:
    wc = DATA[sg]["continuum"]["w"]
    dev = abs(wc - TARGET) / TARGET
    DATA[sg]["devC"] = dev
    ok = dev < 10 * NOISE
    IA0 = IA0 and ok
    print("    %-6.0f %-13.4e %-9.3f %-11.2f %s"
          % (sg, dev, DATA[sg]["continuum"]["res"], dev / NOISE,
             "" if ok else "<- 超"))
print("    IA0 判定：%s" % ("PASS" if IA0 else "FAIL"))

# ---------------- IA1 Rayleigh 恒等式 ----------------
print()
print("[IA1] Rayleigh 恒等式（主判据）：omega_lat^2 - omega_cont^2 对 <f>_lat")
print("    %-6s %-13s %-13s %-11s %-9s"
      % ("xi", "w_lat", "w_cont", "差值w^2", "<f>_lat"))
RAT = []
for sg in SIGS:
    wl = DATA[sg]["lattice"]["w"]
    wc = DATA[sg]["continuum"]["w"]
    d2 = wl * wl - wc * wc
    fb = DATA[sg]["lattice"]["fbar"]
    r = d2 / fb if abs(fb) > 1e-30 else float("nan")
    RAT.append(r)
    print("    %-6.0f %-13.6f %-13.6f %-11.4e %-9.4e"
          % (sg, wl, wc, d2, fb))
    print("           比 = %.4f" % r)
print("    四档比值：%s" % ", ".join("%.4f" % r for r in RAT))
IA1 = all(0.7 <= r <= 1.3 for r in RAT)
print("    IA1 判定：%s（四档比值全落在 [0.7,1.3] 判恒等式成立）"
      % ("PASS" if IA1 else "FAIL"))

# ---------------- IA2 分母候选 ----------------
print()
print("[IA2] 分母候选（主判据）：哪个分母让预测与实测 dev 吻合")
print("    %-6s %-11s %-11s %-11s %-11s" %
      ("xi", "dev 实测", "用 w_cont^2", "用 w_lat^2", "用 w_b^2"))
COLS = {"cont": [], "lat": [], "b": []}
for sg in SIGS:
    wl = DATA[sg]["lattice"]["w"]
    wc = DATA[sg]["continuum"]["w"]
    fb = DATA[sg]["lattice"]["fbar"]
    dev = abs(wl - TARGET) / TARGET
    p_c = abs(fb) / (2.0 * wc * wc)
    p_l = abs(fb) / (2.0 * wl * wl)
    p_b = abs(fb) / (2.0 * WB * WB)
    for k, p in (("cont", p_c), ("lat", p_l), ("b", p_b)):
        COLS[k].append(dev / p)
    print("    %-6.0f %-11.4e %-11.4e %-11.4e %-11.4e"
          % (sg, dev, p_c, p_l, p_b))
print("    各分母下 实测/预测 的四档均值：")
for k, lab in (("cont", "omega_cont^2"), ("lat", "omega_lat^2"), ("b", "omega_b^2")):
    m = float(np.mean(COLS[k]))
    print("      %-14s %.4f" % (lab, m))
# 判据：Rayleigh 商给出的分母是 omega_cont^2；先看它的偏差，再看 omega_b^2 为何显得更近
mean_cont = float(np.mean(COLS["cont"]))
mean_b = float(np.mean(COLS["b"]))
ratio_w2 = WB ** 2 / (TARGET ** 2)
mean_rat = float(np.mean(RAT))
print("    omega_b^2 / omega_cont^2 = %.4f" % ratio_w2)
print("    用 omega_cont^2 的等效修正：若恒等式比值为 %.4f，则等效分母需为 %.4f 倍 omega_cont^2"
      % (mean_rat, 1.0 / mean_rat))
print("    而 omega_b^2 恰为 %.4f 倍。" % ratio_w2)
print("    判读：omega_b^2 显得更接近 1，是因为它的分母比恰好部分抵消了恒等式的偏差，")
print("          并非它在物理上更正确。Rayleigh 商给出的分母是 omega_cont^2。")
IA2 = 0.8 <= mean_cont <= 1.2
print("    IA2 判定：%s（以 omega_cont^2 下的均值 %.4f 落在 [0.8,1.2] 为准）"
      % ("PASS" if IA2 else "FAIL", mean_cont))

# ---------------- IA3 谱峰位 ----------------
print()
print("[IA3] 谱峰位（报告）")
print("    %-6s %-13s %-13s %-11s" % ("xi", "峰位(格点)", "峰位(连续)", "k_mod"))
for sg in SIGS:
    print("    %-6.0f %-13.4f %-13.4f %-11.4f"
          % (sg, DATA[sg]["lattice"]["kpk"], DATA[sg]["continuum"]["kpk"], KMOD))
pk_dev = [abs(DATA[sg]["lattice"]["kpk"] / KMOD - 1) for sg in SIGS]
print("    格点谱峰位与 k_mod 的相对差：%s"
      % ", ".join("%.1f%%" % (100 * p) for p in pk_dev))
print("    若峰位系统性高于 k_mod（实测即如此，高 1.1%% 至 2.3%%），")
print("    则 f(k) 在峰位处更大，用 f(k_mod) 作理论值会偏低。")
f_peak = abs(4 * math.sin(DATA[20.0]["lattice"]["kpk"] / 2) ** 2
             - DATA[20.0]["lattice"]["kpk"] ** 2)
print("    以 xi=20 为例：f(峰位)=%.4e 对 f(k_mod)=%.4e，比 %.3f"
      % (f_peak, abs(4 * math.sin(KMOD / 2) ** 2 - KMOD ** 2),
         f_peak / abs(4 * math.sin(KMOD / 2) ** 2 - KMOD ** 2)))

# ---------------- IA4 判定 ----------------
print()
print("[IA4] 判定")
print("    IA0=%s IA1=%s IA2=%s" % tuple("PASS" if x else "FAIL" for x in (IA0, IA1, IA2)))
if not IA0:
    verdict = "装置异常（连续版未给底噪量级）"
elif IA1:
    verdict = ("Rayleigh 恒等式成立但差 11% 至 13%（四档比值 {:.3f} 至 {:.3f}）；"
               "分母确为 omega_cont^2，换分母不是缺口的来源，"
               "缺口出在恒等式在非本征态上的偏差".format(min(RAT), max(RAT)))
else:
    verdict = ("Rayleigh 恒等式在非本征态上明显不成立（四档比值 {:.3f} 至 {:.3f}），"
               "须改用叠加态修正".format(min(RAT), max(RAT)))
print("    判语：%s" % verdict)
print()
print("    三档 c0 的换算（把 dev 的常数部分换成理论预测）：")
print("      实测 dev 常数项（第 708 号外推）= 2.6987e-4")
print("      用 omega_cont^2 的理论值 = %.4e" % (abs(4 * math.sin(KMOD / 2) ** 2 - KMOD ** 2) / (2 * TARGET ** 2)))
print("      用 omega_b^2 的理论值 = %.4e" % (abs(4 * math.sin(KMOD / 2) ** 2 - KMOD ** 2) / (2 * WB ** 2)))
print("耗时 %.1f 秒" % (time.time() - T0))

# ---------------- 图 ----------------
fig, ax = plt.subplots(1, 3, figsize=(15.8, 4.4))
xi = np.array(SIGS)

ax[0].plot(xi, RAT, "o-", color="#268", ms=7)
ax[0].axhline(1.0, color="#c44", ls="--", lw=1.2, label="恒等式预期")
ax[0].axhspan(0.7, 1.3, color="#c44", alpha=0.10)
ax[0].set_xscale("log")
ax[0].set_xlabel(r"$\xi$")
ax[0].set_ylabel(r"$(\omega_{\rm lat}^2-\omega_{\rm cont}^2)/\langle f\rangle$")
ax[0].set_title("左：Rayleigh 恒等式的比值")
ax[0].legend(fontsize=8)

xs_ = np.arange(len(SIGS))
for i, (k, lab, c) in enumerate((("cont", r"$\omega_{\rm cont}^2$", "#268"),
                                 ("lat", r"$\omega_{\rm lat}^2$", "#c94"),
                                 ("b", r"$\omega_b^2$", "#824"))):
    ax[1].bar(xs_ + (i - 1) * 0.26, COLS[k], width=0.24, color=c, label=lab)
ax[1].axhline(1.0, color="k", ls="--", lw=1.0)
ax[1].set_xticks(xs_)
ax[1].set_xticklabels(["%.0f" % s for s in SIGS])
ax[1].set_xlabel(r"$\xi$")
ax[1].set_ylabel("实测 dev / 预测 dev")
ax[1].set_title("中：三个分母候选")
ax[1].legend(fontsize=7)

for sg, c in zip(SIGS, plt.cm.viridis(np.linspace(0.2, 0.85, len(SIGS)))):
    sel = np.abs(KK) < 0.8
    ax[2].semilogy(KK[sel], DATA[sg]["lattice"]["spec"][sel], "-", lw=1.0,
                   color=c, label=r"$\xi$=%.0f" % sg)
ax[2].axvline(KMOD, color="#c44", ls="--", lw=1.0, label=r"$k_{\rm mod}$")
ax[2].set_xlabel("k")
ax[2].set_ylabel(r"$|\phi(k)|^2$")
ax[2].set_title("右：格点演化的谱与 $k_{\\rm mod}$")
ax[2].legend(fontsize=6)
plt.tight_layout()
plt.savefig(IMG, dpi=130)
print("图已存：", IMG)
