# -*- coding: utf-8 -*-
"""P1-29 值侧开放问题二的紧化：信息因果性渐近族在模型内的阈值核对（判据 A1 至 A6）

定位：369 已核到互信息和式的单副本形式不紧（模型内交叉点落在 S 属于 (3.08, 3.12)）；
第三轮文献核对（注记-文献 9 核对（第三轮））核到原文的紧形式是渐近族：m = 1、Alice 数据
N = 2^n 比特、两侧共享 N - 1 个无信号盒、各向同性族处 P_K = (1 + E^n)/2、和式
I(n) = 2^n [1 - h((1 + E^n)/2)]，判定为"存在 n 使 I(n) > 1"。本脚本把该族搬到模型族上
（对应 E = eps、S = 4 eps）做阈值核对与两形式对照。

判据先行（A1 至 A6）：
  A1 族公式内核化：P(n, eps) = (1 + eps^n)/2 与 I(n, eps) = 2^n [1 - h(P)]；与 369 的成功
     概率读数 P(1, eps) = (1 + eps)/2 逐位一致。
  A2 下界核对：I(n, eps) >= (2 eps^2)^n / (2 ln 2)（文献式 (23) 的二次下界链），网格逐档成立。
  A3 渐近阈值（两方向）：边之上（eps > 1/sqrt2）存在 n 使 I(n) > 1（给见证 n）；边与边之下
     （eps <= 1/sqrt2）在 n <= 4000 上 sup_n I <= 1。n 截断到 60 造成的表观阈值偏移作 A3c 教训登记。
  A4 三个锚点：eps = 0.5（经典）不违反；eps = 1（PR）对一切 n >= 1 违反；eps = 1/sqrt2 恰不违反。
  A5 单副本对照：解 2 [1 - h((1 + eps)/2)] = 1 得 eps1，S1 = 4 eps1 须落在 369 实测交叉区间
     (3.08, 3.12) 内。
  A6 判定：和式入口的紧形式是渐近族（存在 n），阈值 2*sqrt2；单副本形式的松弛由其阈值 S1 定量解释。

严格程度：族公式为引用级（文献式 (22) 至 (24) 的内核化），阈值判定与两形式对照为判据级。
边界：不重做 368/369 的读数；不改动 369 的 J 口径（口径不同，分开登记）；不触碰基准文档。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

FIG = r"D:\software\TianXingFC\images\information_causality_asymptotic.png"
EPS_EDGE = 1.0 / np.sqrt(2.0)
NMAX = 60
NSUP = 4000
LO2 = np.log(2.0)

def h2(p):
    q = np.clip(np.asarray(p, dtype=float), 1e-300, 1.0 - 1e-300)
    return -(q * np.log2(q) + (1.0 - q) * np.log2(1.0 - q))

def omh_y(y):
    """1 - h((1+y)/2) 的稳定实现（直接吃 y）：小 |y| 走正值幂级数（系数 1、1/6、1/15、1/28、1/45），
    否则走 log1p 闭式 [(1+y) log1p(y) + (1-y) log1p(-y)] / (2 ln2)。
    注意：不得由 P = (1+y)/2 反解 y（y 很小时 P 的舍入会把 y 的相对精度毁掉）。"""
    y = np.clip(np.asarray(y, dtype=float), -1.0 + 1e-15, 1.0 - 1e-15)
    y2 = y * y
    ser = y2 * (1.0 + y2 / 6.0 + y2 * y2 / 15.0 + y2 ** 3 / 28.0 + y2 ** 4 / 45.0) / (2.0 * LO2)
    close = 0.5 * ((1.0 + y) * np.log1p(y) + (1.0 - y) * np.log1p(-y)) / LO2
    return np.where(np.abs(y) < 1e-2, ser, close)

def one_minus_h(p):
    """按 P 入口的包装（仅在 |y| 不小、反解不丢精度处使用）。"""
    return omh_y(2.0 * np.asarray(p, dtype=float) - 1.0)

def P_of(n, eps):
    return 0.5 * (1.0 + eps ** n)

def logI_of(n, eps):
    """ln I(n, eps)：对数域实现，避免 2^n 上溢与 y 下溢的双重陷阱。"""
    n = np.asarray(n, dtype=float)
    eps = np.asarray(eps, dtype=float)
    ly = n * np.log(np.maximum(eps, 1e-300))
    y = np.exp(ly)
    big = np.abs(y) >= 1e-2
    corr = np.log(1.0 + y * y / 6.0 + y ** 4 / 15.0)
    small_lp = 2.0 * ly - np.log(2.0 * LO2) + corr
    big_lp = np.log(np.maximum(omh_y(y), 1e-300))
    return n * np.log(2.0) + np.where(big, big_lp, small_lp)

def I_of(n, eps):
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        return np.exp(logI_of(n, eps))

def I_lower(n, eps):
    return (2.0 * eps ** 2) ** n / (2.0 * LO2)

def sup_I(eps, nmax=NMAX):
    ns = np.arange(1, nmax + 1)
    return float(np.max(I_of(ns, eps)))

def gate(name, ok, detail):
    print("[%s] %s：%s" % ("PASS" if ok else "FAIL", name, detail))
    assert ok, name

print("=" * 78)
print("P1-29 值侧开放问题二：信息因果性渐近族（和式紧化）阈值核对（判据先行）")
print("=" * 78)

# ---------- A1 族公式与 369 成功概率入口一致 ----------
ok = abs(P_of(1, EPS_EDGE) - (1.0 + EPS_EDGE) / 2.0) < 1e-15 and abs(P_of(1, 0.75) - 0.875) < 1e-15
gate("A1 族公式内核化", ok, "P(1, 1/sqrt2) = %.15f 与 (1+eps)/2 逐位一致；P(1, 0.75) = %.12f" %
     (P_of(1, EPS_EDGE), P_of(1, 0.75)))

# 两支一致：切换点 y = 1e-2 处幂级数支对闭式支；y = 0.5 处闭式支对直算 1-h(0.75)
y_sw = 1e-2
close_sw = 0.5 * ((1.0 + y_sw) * np.log1p(y_sw) + (1.0 - y_sw) * np.log1p(-y_sw)) / LO2
d_sw = abs(float(one_minus_h(0.5 * (1.0 + y_sw))) - float(close_sw))
d_mid = abs(float(one_minus_h(0.75)) - float(1.0 - h2(0.75)))
ok = d_sw < 1e-17 and d_mid < 1e-15
gate("A1b 两支一致", ok, "切换点差 %.1e（期望 <= 1e-17）；y = 0.5 处对 1-h(0.75) 差 %.1e（期望 <= 1e-15）" % (d_sw, d_mid))

# ---------- A2 下界核对 ----------
grid_eps = np.linspace(0.0, 1.0, 41)
grid_n = np.arange(1, 41)
worst_rel = 1e9
for e in grid_eps:
    dn = np.maximum(I_lower(grid_n, e), 1e-12)
    worst_rel = min(worst_rel, float(((I_of(grid_n, e) - I_lower(grid_n, e)) / dn).min()))
gate("A2 二次下界", worst_rel >= -1e-9, "网格（41 eps × 40 n）上 (I - 下界)/下界 的最小值 = %.2e（期望 >= 0，浮点余量 1e-9）" % worst_rel)

# ---------- A3 渐近阈值（两方向） ----------
delta = 1e-3
eps_hi = EPS_EDGE + delta
n_w = int(np.ceil(np.log(2.0 * LO2) / np.log(2.0 * eps_hi ** 2))) + 2
I_w = float(I_of(n_w, eps_hi))
gate("A3a 边之上有见证", I_w > 1.0,
     "eps = 1/sqrt2 + 1e-3 处取 n = %d 时 I = %.6f > 1（违规见证；见证式 n 约 ln(2 ln2)/ln(2 eps^2)）" % (n_w, I_w))

ns_big = np.arange(1, 4001)
rows = []
for e in (0.5, 0.6, 0.7, EPS_EDGE):
    vals = I_of(ns_big, e)
    inm = int(np.argmax(vals))
    rows.append((e, float(vals[inm]), int(ns_big[inm])))
ok = all(v <= 1.0 + 1e-12 for _, v, _ in rows)
gate("A3b 边与边之下无违反", ok, "；".join("eps=%.6f 处 sup_n I = %.6f（n = %d）" % t for t in rows))

lo, hi = 0.65, 0.75
for _ in range(80):
    mid = 0.5 * (lo + hi)
    if sup_I(mid) > 1.0:
        hi = mid
    else:
        lo = mid
eps_trunc = 0.5 * (lo + hi)
print("[INFO] A3c 截断说明：n 截到 %d 时的表观阈值 = %.6f（对 1/sqrt2 差 %.1e）；该偏移由 n 截断解释，"
      "所需 n 随 ln(2 ln2)/ln(2 eps^2) 发散。故判据必须写成存在 n，不得写成截断 sup。" %
      (NMAX, eps_trunc, abs(eps_trunc - EPS_EDGE)))

# ---------- A4 三个锚点 ----------
s_cls, s_edge, s_pr = sup_I(0.5), sup_I(EPS_EDGE), sup_I(1.0)
ok = (s_cls <= 1.0 + 1e-12) and (s_edge <= 1.0 + 1e-12) and (s_pr > 1.0) and abs(I_of(1, 1.0) - 2.0) < 1e-12
gate("A4 三个锚点", ok, "经典 eps=0.5 处 sup_n I = %.6f；模型边处 sup_n I = %.6f（<= 1，恰不违反）；PR 处 I(1) = %.6f > 1" %
     (s_cls, s_edge, float(I_of(1, 1.0))))

# ---------- A5 单副本对照 ----------
lo, hi = 0.70, 0.85
for _ in range(80):
    mid = 0.5 * (lo + hi)
    if float(I_of(1, mid)) > 1.0:
        hi = mid
    else:
        lo = mid
eps1 = 0.5 * (lo + hi)
S1 = 4.0 * eps1
ok = 3.08 < S1 < 3.12
gate("A5 单副本对照", ok, "单副本阈值 eps1 = %.9f、S1 = %.9f，落在 369 实测交叉区间 (3.08, 3.12) 内；与 2 sqrt2 = %.6f 的差 = %.6f" %
     (eps1, S1, 2.0 * np.sqrt(2.0), S1 - 2.0 * np.sqrt(2.0)))

# ---------- A6 判定摘要 ----------
print("[INFO] A6 判定：和式入口的紧形式是渐近族，阈值 S* = 2 sqrt2（逐位同模型边）；单副本阈值 S1 = %.6f，松弛量 %.6f" % (S1, S1 - 2.0 * np.sqrt(2.0)))
print("[INFO] 口径提示：本脚本的 I(n) 以成功概率为入口；369 的 J 以协议类最大化为入口（模型边 J = 1.000000）。两口径的阈值一致，数值不同，分开登记。")

# ---------- 出图 ----------
eps_demo = [0.70, EPS_EDGE, 0.71, 0.75, 1.0]
ns = np.arange(1, 41)
fig, ax = plt.subplots(1, 2, figsize=(11.6, 4.4))
for e in eps_demo:
    ax[0].plot(ns, I_of(ns, e), "-o", ms=3.2, lw=1.4, label="eps = %.6f" % e)
ax[0].axhline(1.0, color="#c53030", lw=1.1, ls="--", label="IC 阈值 1（m = 1）")
ax[0].set_yscale("log")
ax[0].set_xlabel("族参数 n（Alice 数据 2^n 比特）")
ax[0].set_ylabel("和式 I(n, eps)")
ax[0].set_title("渐近族的和式：阈值上下两支")
ax[0].legend(fontsize=7.4)
ax[0].grid(alpha=0.25)

eg = np.linspace(0.65, 0.82, 171)
sup_vals = np.array([sup_I(e, NSUP) for e in eg])
i_cross = int(np.argmax(sup_vals > 1.0))
e_cross = float(eg[i_cross])
print("[INFO] 右图核对（n <= %d）：sup 曲线穿越阈值 1 的位置 = %.9f（对 1/sqrt2 差 %.1e）" %
      (NSUP, e_cross, abs(e_cross - EPS_EDGE)))
for e_chk in (0.65, 0.70, EPS_EDGE, 0.75):
    sv = sup_I(e_chk, NSUP)
    print("[INFO] 图核对：eps = %.6f 处 sup_n I = %s（n 上限 %d）；I(n = 1) = %.6f" %
          (e_chk, ("%.6f" % sv) if sv < 1e6 else "大于 1e6", NSUP, float(I_of(1, e_chk))))
ax[1].plot(eg, sup_vals, "-", lw=1.8, color="#2b6cb0", label="sup_n I(n, eps)")
ax[1].axhline(1.0, color="#c53030", lw=1.1, ls="--", label="IC 阈值 1")
ax[1].axvline(EPS_EDGE, color="#2f855a", lw=1.2, ls=":", label="1/sqrt2 = %.6f" % EPS_EDGE)
ax[1].axvline(eps1, color="#b7791f", lw=1.2, ls="-.", label="单副本阈值 %.6f" % eps1)
ax[1].set_yscale("log")
ax[1].set_xlabel("eps（S = 4 eps）")
ax[1].set_ylabel("sup_n I(n, eps)")
ax[1].set_title("阈值判定：渐近族在模型边翻转，单副本偏右")
ax[1].legend(fontsize=7.4)
ax[1].grid(alpha=0.25)
fig.tight_layout()
fig.savefig(FIG, dpi=150)
print("[INFO] 图已存：%s" % FIG)
print("=" * 78)
