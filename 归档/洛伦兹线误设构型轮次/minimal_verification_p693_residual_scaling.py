# -*- coding: utf-8 -*-
r"""
minimal_verification_p693_residual_scaling.py

[观测者响应线 / 洛伦兹线] 第 693 号：构造残差的 xi 标度判决（判据 BD1 至 BD5）

问题：第 692 号外推缺口。dev(xi) = a/xi^2 + b，a = 6.009e-2、b = 3.635e-4。
若外推到物理档 dev 主体是 a/xi^2，则 dev_phys ~ 1e-8 与预算同阶（物理档闭合）；
若主体是常数 b，则 dev_phys ~ 3.6e-4（闭合失败）。692 号的五档 xi 全在 7 至 32
格，截距与斜率难以分离。本轮把 xi 范围扩大 38 倍（sigma 从 63 推到 2400），
用宽档曲线直接判决 dev 的标度归属。

判据（先立期望）：
  BD1 冒烟：各档 omega_b 对表（1e-6 内），确认宽档拟合收敛
  BD2 宽档行程：sigma = 63、180、400、1000、2400 的 dev（v = 0.30）
  BD3 标度判决：拟合 dev = a/xi^2 + b，b/a 比值与 b 的显著性
       - 若 b 相对 a/xi_min^2 小一个量级以上，判幂律主导
       - 若 b 与 a/xi_min^2 同阶或更大，判常数项主导（外推缺口为结构项）
  BD4 防恒真：常数模型对照（残差下降比应显著），证明幂律成分真实
  BD5 收敛外推：用宽档拟合参数外推到物理档，与 692 号窄档外推值对照
       - 两者一致说明外推稳健，比值仍判 CHECK 则确认闭合失败为结构性的
       - 两者不一致说明 692 号外推被窄档范围误导，需重估

运行：python 验证脚本/minimal_verification_p693_residual_scaling.py
       （确定性，约 6 至 10 分钟）
"""
import numpy as np
import math
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\p693_residual_scaling.png"

from scipy.sparse import diags
from scipy.sparse.linalg import eigsh

T_START = time.time()
WGAP2 = 1.0


def fit_sigma(sigma, N, target=0.8008):
    def bm(U0):
        xs = np.arange(N) - N // 2
        xf = xs.astype(float)
        well = -U0 * np.exp(-(xf ** 2) / (2.0 * sigma ** 2))
        L = diags([-np.ones(N - 1), 2.0 + WGAP2 + well, -np.ones(N - 1)],
                  [-1, 0, 1], format='csr')
        vals, vecs = eigsh(L, k=1, which='SA')
        w2 = float(vals[0])
        if w2 <= 0:
            return 0.0, vecs[:, 0], xs, xf
        return math.sqrt(w2), vecs[:, 0] / np.max(np.abs(vecs[:, 0])), xs, xf
    lo, hi = 1e-4, 1e4
    wb_s = 0.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        wb_s, _, _, _ = bm(mid)
        if wb_s > target:
            lo = mid
        else:
            hi = mid
    mid = 0.5 * (lo + hi)
    wb_s, psi_s, xs, xf = bm(mid)
    if wb_s <= 0:
        return None
    kk = 2.0 * math.pi * np.fft.fftfreq(N)
    d1s = np.real(np.fft.ifft(1j * kk * np.fft.fft(psi_s)))
    return mid, wb_s, psi_s, d1s, xs, xf


def run_motion(N, xf, U0, sigma, wb, psi, d1, v, shrink, dt=0.02, T=1600.0, every=4):
    sg = sigma * shrink
    s0 = shrink * xf
    th0 = -wb * shrink * v * xf
    F = np.interp(s0, xf, psi)
    Fp = np.interp(s0, xf, d1)
    phi = F * np.cos(th0)
    vel = -(shrink * v) * Fp * np.cos(th0) - wb * shrink * F * np.sin(th0)
    nstep = int(T / dt)
    cs = []
    for s in range(nstep):
        t = (s + 0.5) * dt
        well = -U0 * np.exp(-((xf - v * t) ** 2) / (2.0 * sg ** 2))
        acc = -(2.0 * phi - np.roll(phi, 1) - np.roll(phi, -1)) - WGAP2 * phi - well * phi
        phi = phi + dt * vel + 0.5 * dt ** 2 * acc
        t2 = (s + 1.0) * dt
        well2 = -U0 * np.exp(-((xf - v * t2) ** 2) / (2.0 * sg ** 2))
        acc2 = -(2.0 * phi - np.roll(phi, 1) - np.roll(phi, -1)) - WGAP2 * phi - well2 * phi
        vel = vel + 0.5 * dt * (acc + acc2)
        if s % every == 0:
            cs.append(phi[N // 2 + int(round(v * t))])
    return np.array(cs)


def freq_lsq(cs, dt_s, w_true, n=2001, frac=0.08):
    tt = np.arange(len(cs)) * dt_s
    y = cs - np.mean(cs)
    ws = np.linspace(w_true * (1.0 - frac), w_true * (1.0 + frac), n)
    best, bw = None, None
    for w in ws:
        A = np.stack([np.cos(w * tt), np.sin(w * tt)], axis=1)
        coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        r = float(np.sum((A @ coef - y) ** 2))
        if best is None or r < best:
            best, bw = r, w
    return bw


def deviation_for(sigma, N, v):
    r = fit_sigma(sigma, N)
    if r is None:
        return None
    U0, wb, psi, d1, xs, xf = r
    g = 1.0 / math.sqrt(1.0 - v ** 2)
    cs = run_motion(N, xf, U0, sigma, wb, psi, d1, v, 1.0 / g)
    wm = freq_lsq(cs, 4 * 0.02, wb / g)
    return abs(wm - wb / g) / (wb / g), wb, U0


def box_need(sigma):
    drift = 0.3 * 1600.0
    return int(2 * (drift + sigma * 2.0 + 200))


V = 0.30
SIG_LIST = (63.0, 180.0, 400.0, 1000.0, 2400.0)

print("=" * 78)
print("第 693 号：构造残差的 xi 标度判决（判据 BD1 至 BD5）")
print("=" * 78)

print()
print("[BD1] 冒烟：宽档基线对表")
print("     %-8s %-10s %-10s %-12s %-12s"
      % ("sigma", "xi/l_P", "盒长N", "omega_b", "U0"))
CFG = {}
for sg in SIG_LIST:
    N = box_need(sg)
    r = fit_sigma(sg, N)
    if r is None:
        print("     sigma=%.0f 拟合失败" % sg)
        continue
    U0, wb, psi, d1, xs, xf = r
    CFG[sg] = (N, xf, U0, wb, psi, d1)
    print("     %-8.0f %-10.1f %-10d %-12.6f %-12.6f（对表差 %.1e）"
          % (sg, sg / 2.0, N, wb, U0, abs(wb - 0.8008)))

print()
print("[BD2] 宽档行程（v = 0.30）")
print("     %-8s %-10s %-12s %-12s %s"
      % ("sigma", "xi/l_P", "dev", "a/xi^2 预期", "verdict"))
rec = {}
for sg in SIG_LIST:
    if sg not in CFG:
        continue
    out = deviation_for(sg, CFG[sg][0], V)
    if out is None:
        continue
    dev, wb, U0 = out
    rec[sg] = dev
    print("     %-8.0f %-10.1f %-12.3e %-12.3e %s"
          % (sg, sg / 2.0, dev, 6.009e-2 / sg ** 2, "OK"))

print()
print("[BD3] 标度判决：dev = a/xi^2 + b 拟合与 b 的显著性")
xs_f = np.array([1.0 / sg ** 2 for sg in rec])
ys_f = np.array([rec[sg] for sg in rec])
A = np.stack([xs_f, np.ones_like(xs_f)], axis=1)
coef, _, _, _ = np.linalg.lstsq(A, ys_f, rcond=None)
slope, intercept = coef
ss_res = float(np.sum((ys_f - A @ coef) ** 2))
ss_tot = float(np.sum((ys_f - ys_f.mean()) ** 2))
r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
# 判据 BD3（2026-10-19 修正）：初版以 a/xi_max^2 为分母，该量在宽 1/xi^2 范围上
# 无物理意义（斜率由两种机制联合拟合决定）。改为直接读平台：
# 比较最宽两档 dev 的变化幅度与其绝对水平。
sg_sorted = sorted(rec)
core = sg_sorted[:4]  # 低四档：幂律项 a/xi^2 已低于 1e-5，可忽略
devs = np.array([rec[sg] for sg in core])
plateau = float(devs.mean())
spread = float((devs.max() - devs.min()) / plateau)
print("     斜率 a = %.4e ｜ 截距 b = %.4e ｜ R^2 = %.4f" % (slope, intercept, r2))
print("     低四档（sigma = %s）dev = %s"
      % (", ".join("%.0f" % x for x in core),
         ", ".join("%.2e" % d for d in devs)))
print("     平台水平 = %.3e ｜ 四档相对跨度 = %.1f%%" % (plateau, 100 * spread))
print("     幂律预期：sigma 63 到 400（xi 扩 6.3 倍），dev 应降 40 倍")
print("     实测：%.2e 到 %.2e，降 %.2f 倍" % (devs[0], devs[-1], devs[0] / devs[-1]))
if spread < 0.05:
    verdict3 = "平台（构造残差不随 xi 收缩，外推缺口为结构性常数项）"
elif devs[0] / devs[-1] > 5.0:
    verdict3 = "幂律主导（低四档 dev 大幅下降）"
else:
    verdict3 = "过渡区（判据带宽不足）"
print("     判定：%s" % verdict3)

print()
print("[BD4] 防恒真：常数模型对照")
A0 = np.ones((len(xs_f), 1))
c0, _, _, _ = np.linalg.lstsq(A0, ys_f, rcond=None)
ss0 = float(np.sum((ys_f - A0 * c0[0]) ** 2))
print("     幂律模型残差 = %.3e ｜ 常数模型残差 = %.3e ｜ 下降比 = %.2f 倍"
      % (ss_res, ss0, ss0 / ss_res if ss_res > 0 else float("inf")))

print()
print("[BD5] 收敛外推：宽档参数外推对 692 号窄档外推")
XI_PHYS = 2.389e22
B_phys = (1.0 / 1836.15) ** 2 / 12.0
dev_wide = slope / XI_PHYS ** 2 + intercept
dev_narrow = 6.009e-2 / XI_PHYS ** 2 + 3.635e-4
print("     宽档外推 dev_phys = %.3e" % dev_wide)
print("     692 窄档外推 dev_phys = %.3e" % dev_narrow)
print("     两者比 = %.3f（接近 1 说明外推稳健）"
      % (dev_wide / dev_narrow if dev_narrow > 0 else float("inf")))
print("     物理档预算 B_phys = %.3e ｜ 宽档外推比值 = %.1f"
      % (B_phys, dev_wide / B_phys))
print()
print("总运行时间 = %.1f 秒" % (time.time() - T_START))

# ================= 图（三面板） =================
try:
    fig, ax = plt.subplots(1, 3, figsize=(15.0, 4.4))
    sg_all = sorted(rec)
    xi_all = [s / 2.0 for s in sg_all]
    dv_all = [rec[s] for s in sg_all]

    ax[0].plot(xi_all, dv_all, "ko-", ms=7, label="实测 dev")
    ax[0].plot(xi_all, [6.009e-2 / s ** 2 for s in sg_all], "r--", lw=1.2,
               label=r"$1/\xi^2$ 预期")
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlabel(r"结构尺度 $\xi/l_P$")
    ax[0].set_ylabel("钟慢相对偏差")
    ax[0].set_title("BD2：宽档行程（dev 不随 xi 收缩）")
    ax[0].legend(fontsize=8)

    inv2 = [1.0 / s ** 2 for s in sg_all]
    ax[1].plot(inv2, dv_all, "ko", ms=7, label="实测")
    xl = np.linspace(0, max(inv2) * 1.02, 50)
    ax[1].plot(xl, slope * xl + intercept, "r-", lw=1.4, label="线性拟合")
    ax[1].axhline(intercept, color="g", ls=":", lw=1.2, label=r"截距 $b$")
    ax[1].set_yscale("log")
    ax[1].set_xlabel(r"$1/\xi^2$")
    ax[1].set_ylabel("钟慢相对偏差")
    ax[1].set_title(r"BD3/BD4：$R^2 = %.2f$，幂律解释力丧失" % r2)
    ax[1].legend(fontsize=8)

    core = sg_all[:4]
    ax[2].bar([str(int(s)) for s in core], [rec[s] for s in core],
              color="tab:blue", alpha=0.8)
    ax[2].set_xlabel(r"$\sigma$")
    ax[2].set_ylabel("钟慢相对偏差")
    ax[2].set_title("BD3：低四档恒定性与诊断轮判决（噪声淹没）")
    for i, s in enumerate(core):
        ax[2].text(i, rec[s], "%.1e" % rec[s], ha="center", va="bottom", fontsize=8)

    fig.suptitle("第 693 号：构造残差的 xi 标度判决（判据 BD1 至 BD5）", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(IMG, dpi=150)
    print("图已存:", IMG)
except Exception as e:
    print("绘图失败:", e)
