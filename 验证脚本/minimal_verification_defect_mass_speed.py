# -*- coding: utf-8 -*-
r"""
minimal_verification_defect_mass_speed.py

缺陷质量与穿越速度的关系（P4-16 路线甲第九轮余项；第 224 条的续轮）。

背景：第 224 条给出无质量缺陷的群速度 v_g = cos(k/2) <= 1，取 v = 1 得联合条件
gamma <= sqrt(2) p 4 pi / L^2 = 0.01388，并登记余项「缺陷质量与速度的关系」。

本轮把体系自己的晶格色散（注释 03 / 第 196 条：omega^2 = 4 sum_a sin^2(k_a/2)）加上质量缺口：
    omega(k)^2 = m^2 + 4 sin^2(k/2)        （一维；单位 l_P = t_P = 1，c_s = 1）
即锁定晶格色散的质量化形式（m = 0 退化为注释 03 的无质量色散）。群速度
    v(k) = sin(k) / sqrt(m^2 + 4 sin^2(k/2))，
质量把速度上限从 1 压到 v_max(m) < 1，峰值动量移到 k* > 0。

判据与期望：
  Y1 格点本征值核对：有限环的键弹性加质量 Hessian 本征值 = m^2 + 4 sin^2(k/2)（<= 1e-12）
  Y2 无质量退化：m = 0 时群速度与 cos(k/2) 逐点相符（<= 1e-9），最大值 = 1（长波极限），
     即第 224 条的 v_g = cos(k/2) 是本轮 m = 0 的特例
  Y3 三维各向异性交叉核对：无质量三维群速首阶衰减系数沿 [100] = 1/8、沿 [111] = 1/24，
     与既有读数 0.124781、0.041642 一致（<= 1%），说明模型取的是体系自身色散
  Y4 质量压低上限：五档 m 的 v_max 严格递减且恒小于 1；闭式
     v_max^2 = mu (4 + mu - A) / (A (A + mu))，A = sqrt(mu^2 + 4 mu)，mu = m^2（<= 1e-9）
  Y4b 精确闭式：v_max = (sqrt(m^2 + 4) - m)/2，等价于 1/v_max - v_max = m（<= 1e-12）
  Y4c 峰值动量：解析式 x*^2 = m^2 (sqrt(1 + 4/m^2) - 1)、k* = 2 arcsin(x*/2)，
     与数值 argmax 一致（<= 1e-6）且随 m 单调增大（质量把峰值推离 k = 0 移向带内）
  Y5 小质量律：v_max = 1 - m/2 + m^2/8 + O(m^3)（m <= 0.05 档相对偏差 <= 5e-5）
  Y6 体系质量读数代入：m = 0.0936 与 0.0994（FK 链 K = 0.005 窗口协议实测）给
     v_max = 0.9543 与 0.9516；联合条件收紧到 gamma <= 0.01325 与 0.01322（比第 224 条紧约 5%）
  Y7 缺口只被收紧：使均匀化条件等于门槛带所需 v = 11.61，与上限之比由 11.61 升到 12.20；
     v_max(m) = 11.61 在 m >= 0 上无解（质量不能打开均匀化缺口）
  Y8 质量-速度对照：v_max = 0.9 / 0.75 / 0.5 / 0.25 所需质量，闭式逆式 m = (1 - v^2)/v 与数值反解一致
      （配套：v_max 精确闭式 v_max = (sqrt(m^2 + 4) - m)/2，等价于 1/v_max - v_max = m）
  Y9 登记项：质量的微观来源（绑定能与 K）与「重缺陷更慢」的观测量级，本轮不作判据
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\defect_mass_speed.png"
L, P_DEF, BAND = 8, 0.05, 1.29 * 0.125
rows = []


def report(tag, ok, expect, actual):
    rows.append((tag, ok, expect, actual))
    print("[%s] %s" % ("通过" if ok else "未过", tag))
    print("      期望 " + expect)
    print("      实测 " + actual)


def omega(k, m):
    return np.sqrt(m * m + 4.0 * np.sin(k / 2.0) ** 2)


def vgroup(k, m):
    return np.sin(k) / omega(k, m)


def vmax_closed(m):
    mu = m * m
    A = np.sqrt(mu * mu + 4.0 * mu)
    return np.sqrt(mu * (4.0 + mu - A) / (A * (A + mu))) if mu > 0 else 1.0


def vmax_exact(m):
    return (np.sqrt(m * m + 4.0) - m) / 2.0


def vmax_numeric(m):
    k = np.linspace(1e-9, np.pi, 400001)
    return float(np.max(vgroup(k, m)))


# Y1 格点本征值核对
def hessian_eigs(N, m):
    A = np.zeros((N, N))
    for n in range(N):
        A[n, n] = -1.0
        A[n, (n + 1) % N] = 1.0
    H = A.T @ A + (m * m) * np.eye(N)
    return np.sort(np.linalg.eigvalsh(H))


dev1 = 0.0
for N in (128, 256):
    for m in (0.0, 0.0994):
        ev = hessian_eigs(N, m)
        kk = 2.0 * np.pi * np.arange(N) / N
        ref = np.sort(m * m + 4.0 * np.sin(kk / 2.0) ** 2)
        dev1 = max(dev1, float(np.max(np.abs(ev - ref))))
report("Y1 格点本征值核对", dev1 <= 1e-12,
       "Hessian 本征值 = m^2 + 4 sin^2(k/2)（N = 128、256；m = 0、0.0994）",
       "最大偏差 %.2e" % dev1)

# Y2 无质量退化
k = np.linspace(1e-6, np.pi, 200001)
v0 = vgroup(k, 0.0)
dev2 = float(np.max(np.abs(v0 - np.abs(np.cos(k / 2.0)))))
report("Y2 无质量退化", dev2 <= 1e-9 and abs(vmax_numeric(0.0) - 1.0) <= 1e-6,
       "m = 0 时 v = cos(k/2)（<= 1e-9），max v = 1（长波极限），即第 224 条的读数",
       "最大偏差 %.2e；max v = %.6f（k -> 0）" % (dev2, vmax_numeric(0.0)))

# Y3 三维各向异性交叉核对
def v_3d(kvec):
    kv = np.asarray(kvec, dtype=float)
    w = np.sqrt(4.0 * np.sum(np.sin(kv / 2.0) ** 2) + 1e-300)
    return np.sqrt(np.sum(np.sin(kv) ** 2)) / w


def fit_coef(direction, kmax=0.05, n=400):
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    ks = np.linspace(1e-4, kmax, n)
    vs = np.array([v_3d(kk * d) for kk in ks])
    coef = float(np.polyfit(ks ** 2, 1.0 - vs, 1)[0])
    return coef


c100, c111 = fit_coef([1, 0, 0]), fit_coef([1, 1, 1])
dev3 = max(abs(c100 - 0.125) / 0.125, abs(c111 - 1.0 / 24.0) / (1.0 / 24.0))
ref100, ref111 = 0.124781, 0.041642
dev3b = max(abs(c100 - ref100) / ref100, abs(c111 - ref111) / ref111)
report("Y3 三维各向异性交叉核对", dev3 <= 1e-2 and dev3b <= 1e-2,
       "首阶系数 = 1/8 与 1/24，且与既有读数 0.124781、0.041642 一致（<= 1%）",
       "[100] %.6f；[111] %.6f；对 1/8 与 1/24 偏差 <= %.1e；对既有读数偏差 <= %.1e"
       % (c100, c111, dev3, dev3b))

# Y4 质量压低上限
ms = [0.0, 0.01, 0.05, 0.0994, 0.5]
vmaxs = [vmax_closed(m) for m in ms]
mono = all(vmaxs[i] > vmaxs[i + 1] for i in range(len(ms) - 1))
dev4 = max(abs(vmax_closed(m) - vmax_numeric(m)) for m in ms[1:])
report("Y4 质量压低上限", mono and vmaxs[-1] < 1.0 and dev4 <= 1e-9,
       "v_max 随 m 严格递减且恒小于 1；闭式与数值最大化一致（<= 1e-9）",
       "v_max = " + ", ".join("%.6f" % v for v in vmaxs) + "；闭式偏差 %.2e" % dev4)

# Y4c 峰值动量
def kstar_analytic(m):
    mu = m * m
    x2 = mu * (np.sqrt(1.0 + 4.0 / mu) - 1.0)
    return 2.0 * np.arcsin(np.sqrt(x2) / 2.0)


def kstar_numeric(m):
    kk = np.linspace(1e-9, np.pi, 400001)
    vv = vgroup(kk, m)
    i = int(np.argmax(vv))
    y0, y1, y2 = vv[i - 1], vv[i], vv[i + 1]
    dk = kk[1] - kk[0]
    denom = y0 - 2.0 * y1 + y2
    shift = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
    return float(kk[i] + shift * dk)


dev4c = 0.0
ks_list = []
for m in (0.01, 0.05, 0.0994, 0.5):
    ka, kn = kstar_analytic(m), kstar_numeric(m)
    ks_list.append((m, ka, kn))
    dev4c = max(dev4c, abs(ka - kn))
mono_c = all(ks_list[i][1] < ks_list[i + 1][1] for i in range(len(ks_list) - 1))
report("Y4c 峰值动量", dev4c <= 1e-8 and mono_c,
       "解析 k* 与数值 argmax（网格加抛物线细化）一致（<= 1e-8）且随 m 单调增大",
       "; ".join("m = %.4f: k* = %.6f（数值 %.6f）" % (m, ka, kn) for m, ka, kn in ks_list))

# Y4b 精确闭式与恒等式
def vmax_closed_np(m):
    mu = m * m
    A = np.sqrt(mu * mu + 4.0 * mu)
    return np.sqrt(mu * (4.0 + mu - A) / (A * (A + mu)))


mgrid = np.linspace(1e-6, 4.0, 4001)
dev4b1 = float(np.max(np.abs(vmax_exact(mgrid) - vmax_closed_np(mgrid))))
dev4b2 = float(np.max(np.abs(1.0 / vmax_exact(mgrid) - vmax_exact(mgrid) - mgrid)))
report("Y4b 精确闭式", dev4b1 <= 1e-12 and dev4b2 <= 1e-12,
       "v_max = (sqrt(m^2 + 4) - m)/2 与 mu 形式逐点相同，且满足 1/v_max - v_max = m（<= 1e-12）",
       "两式偏差 %.2e；恒等式偏差 %.2e" % (dev4b1, dev4b2))

# Y5 小质量律
mg = np.linspace(1e-4, 0.05, 21)
law = np.array([1.0 - m / 2.0 + m * m / 8.0 for m in mg])
exact = np.array([vmax_exact(m) for m in mg])
dev5 = float(np.max(np.abs(exact - law) / exact))
report("Y5 小质量律", dev5 <= 5e-5,
       "v_max = 1 - m/2 + m^2/8 + O(m^3)（m <= 0.05，相对偏差 <= 5e-5）",
       "最大相对偏差 %.2e" % dev5)

# Y6 体系质量读数代入
M_SYS = (0.0936, 0.0994)
gam224 = np.sqrt(2.0) * P_DEF * 1.0 * 4.0 * np.pi / (L ** 2)
out6 = []
ok6 = True
for m in M_SYS:
    vm = vmax_closed(m)
    Gam = P_DEF * vm
    gam = np.sqrt(2.0) * Gam * 4.0 * np.pi / (L ** 2)
    out6.append("m = %.4f: v_max = %.6f, Gamma_max = %.5f, gamma <= %.5f" % (m, vm, Gam, gam))
    ok6 = ok6 and vm < 1.0 and gam < gam224
report("Y6 体系质量读数代入", ok6,
       "两档质量给 gamma 上界紧于第 224 条的 %.5f（v = 1 情形）" % gam224,
       "; ".join(out6) + "；收紧 %.2f%% 与 %.2f%%"
       % (100.0 * (1 - np.sqrt(2.0) * P_DEF * vmax_closed(M_SYS[0]) * 4 * np.pi / (L ** 2) / gam224),
          100.0 * (1 - np.sqrt(2.0) * P_DEF * vmax_closed(M_SYS[1]) * 4 * np.pi / (L ** 2) / gam224)))

# Y7 缺口只被收紧
v_needed = BAND / gam224
v_ceiling = vmax_closed(M_SYS[1])
ratio_before, ratio_after = v_needed / 1.0, v_needed / v_ceiling


def solve_m(target):
    lo, hi = 1e-9, 5.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if vmax_closed(mid) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


no_sol = vmax_closed(1e-9) < v_needed
report("Y7 缺口只被收紧", no_sol and ratio_after > ratio_before,
       "所需 v = 11.61 与上限之比由 11.61 升到 12.20；v_max(m) = 11.61 在 m >= 0 无解",
       "所需 v = %.2f；上限 %.6f；比值 %.2f -> %.2f；解存在 = %s"
       % (v_needed, v_ceiling, ratio_before, ratio_after, (not no_sol)))

# Y8 质量-速度对照
targets = [0.90, 0.75, 0.50, 0.25]
tab = [(t, solve_m(t)) for t in targets]
dev8 = max(abs(m - (1.0 - t * t) / t) for t, m in tab)
report("Y8 质量-速度对照", all(m > 0 for _, m in tab) and tab[0][1] < tab[-1][1] and dev8 <= 1e-12,
       "v_max = 0.90 / 0.75 / 0.50 / 0.25 对应质量单调递增，且与逆式 m = (1 - v^2)/v 一致（<= 1e-12）",
       "; ".join("v_max = %.2f 需 m = %.4f（逆式 %.4f）" % (t, m, (1.0 - t * t) / t) for t, m in tab))

print("[登记] Y9：质量微观来源（绑定能、K 依赖）与「重缺陷更慢」的观测量级不作判据；")
print("       另登记一项口径订正：第 224 条与 P4-16 报告把本余项记作「归 P1-32」，而清单里 P1-32 为")
print("       已关闭的像素偏转条目，运动机制另有条目（有质量粒子运动机制），归属待作者裁定。")

fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.2))
kk = np.linspace(0, np.pi, 601)
for m, lst in zip((0.0, 0.0994, 0.5), ("-", "--", ":")):
    ax[0].plot(kk, omega(kk, m), lst, label="m = %.4f" % m)
ax[0].set_xlabel("k（单位 1/l_P）")
ax[0].set_ylabel("omega(k)")
ax[0].set_title("质量化晶格色散：质量抬带边")
ax[0].legend(fontsize=8)
for m, lst in zip((0.0, 0.05, 0.0994, 0.25, 0.5), ("-", "--", "-.", ":", "--")):
    ax[1].plot(kk, vgroup(np.maximum(kk, 1e-9), m), lst, label="m = %.4f" % m)
ax[1].axhline(1.0, ls=":", c="tab:red", lw=1, label="因果上限 1")
ax[1].set_xlabel("k（单位 1/l_P）")
ax[1].set_ylabel("v(k)")
ax[1].set_title("群速度：质量压低上限、峰值移向带内")
ax[1].legend(fontsize=8)
mplot = np.linspace(1e-6, 1.6, 601)
ax[2].plot(mplot, [vmax_closed(m) for m in mplot], "-", label="v_max(m) 闭式")
ax[2].axhline(1.0, ls=":", c="tab:red", lw=1, label="因果上限 1")
for m in M_SYS:
    ax[2].plot([m], [vmax_closed(m)], "o", c="tab:green", ms=5)
    ax[2].annotate("m = %.4f" % m, (m, vmax_closed(m)), textcoords="offset points",
                   xytext=(6, -12), fontsize=8, color="tab:green")
ax[2].annotate("所需 v = %.2f 远超上限" % v_needed, (1.0, 0.62), fontsize=8, color="tab:purple")
ax[2].set_xlabel("缺陷质量 m（单位 E_P）")
ax[2].set_ylabel("v_max")
ax[2].set_title("质量-速度上限与体系两档实测质量")
ax[2].legend(fontsize=8)
plt.tight_layout()
plt.savefig(IMG, dpi=150)
plt.close(fig)

bad = [r for r in rows if not r[1]]
print("---- 判据 %d 项，通过 %d 项；图存 %s ----" % (len(rows), len(rows) - len(bad), IMG))
sys.exit(1 if bad else 0)
