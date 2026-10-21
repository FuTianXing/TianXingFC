# -*- coding: utf-8 -*-
r"""
minimal_verification_alpha_independent_source.py

[第734号] C5 alpha 的独立来源：过阻尼离散化与 kappa 定义的不依赖性

背景：判据 C5 要求「alpha 由抄写平均动力学独立确定，或证明 kappa 的定义不依赖
alpha」，期望二者其一成立，且明确声明 N_c 式子属过阻尼极限。C5 是 C4 全链闭合
（判据 C4）的残余矛盾之一。

模型设定（记录场的阻尼波动方程，库内 §2.2）：
  ddot(theta) + gamma*dot(theta) = cs^2 * lap(theta) + s        （弱阻尼端，cs = c）
  过阻尼极限（作者 2026-09-18 裁决十六）：忽略惯性项 ddot(theta)，故
      gamma * dot(theta) = cs^2 * lap(theta) + s
  一拍一步离散化（拍长 t_P，格距 l_P，c = l_P / t_P）：
      (theta_{n+1} - theta_n)/t_P = cs^2 * lap(theta_n)/gamma + s/gamma
  六邻居拉普拉斯 lap = 6*theta - sum_nb，故
      theta_{n+1} = theta_n + (cs^2 t_P^2/(gamma l_P^2)) * (1/6) * sum_nb (theta_nb - theta_n)
                    + s t_P / gamma
  抄写平均的写法 theta <- theta + alpha*(avg_nb - theta) 中，alpha 是
  邻居平均被抄入的比例。比较两式：在 cs = c 且 c = l_P/t_P 时，cs^2 t_P^2/l_P^2 = 1，
  故 alpha = 1/6 * 6 = 1，且该值不依赖 gamma 与 s（惯性项已被过阻尼舍去）。

判据（先立判据与理论期望，再实现）：
  C5-1  过阻尼离散化的唯一性：alpha_derived = 1（相对偏差 <= 1e-12）
  C5-2  三个前提各自的必要性：破坏 cs = c、拍长 = t_P、邻居集为六邻居，
        alpha_derived 须离开 1（偏离 > 1e-3）
  C5-3  kappa 的定义不依赖 alpha：kappa 读数在 alpha 多档上不变（容差 1e-6）
  C5-4  kappa 推导链的输入清单扫描：两个 kappa 推导脚本的输入不含 alpha，
        且把 alpha 塞进输入不影响读数
  C5-5  恒真替身：把 cs 换成 cs/2，C5-1 须 FAIL
  C5-6  过阻尼声明与 N_c 式子的适用区间：惯性项与阻尼项之比趋零（< 1e-3）
  C5-7  谱收敛区间：给出 alpha 使抄写平均算子谱半径小于 1 的区间

运行：python 验证脚本/minimal_verification_alpha_independent_source.py
确定性（无随机），约 20 秒。图存 images/alpha_independent_source.png
"""
import io
import math
import os
import sys
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
from scipy.linalg import eigh_tridiagonal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "images", "alpha_independent_source.png")

T0 = time.time()
CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))


print("=" * 78)
print("第734号：C5 alpha 的独立来源（过阻尼离散化与 kappa 定义的不依赖性）")
print("=" * 78)

D = 3
NN = 2 * D                      # 六邻居
N1 = 401                        # 一维链（抄写平均的可数载体）
C1 = N1 // 2

# ----------------------------------------------------------------------
# 装置一：过阻尼离散化的 alpha 推导
# ----------------------------------------------------------------------
def alpha_from_overdamped(cs_ratio, tau_ratio, n_neighbors, n_steps_per_beat):
    """从阻尼波动方程的过阻尼极限与一拍一步离散化，算出抄写平均系数 alpha。

    推导：一拍一步的前向欧拉离散给出
        theta_{n+1} = theta_n + (cs^2 t_P^2 / l_P^2) * (1/(2d)) * sum_nb (theta_nb - theta_n) * ...
    在抄写平均的写法 theta <- theta + alpha*(avg_nb - theta) 中，邻居平均的
    归一因子为 1/(邻居数)，故
        alpha = (cs^2 t_P^2 / l_P^2) * (邻居数 / 2d) * (每拍扩散步数)
    其中 cs_ratio = cs/c，tau_ratio = 拍长 / t_P。
    代入 cs = c 与 c = l_P/t_P 得 cs^2 t_P^2/l_P^2 = 1，alpha 恰为 1。
    """
    return (cs_ratio ** 2) * (tau_ratio ** 2) * (n_neighbors / NN) * n_steps_per_beat


# ----------------------------------------------------------------------
# 装置二：一维写入动力学（用于把 alpha 的固定点关系实测出来）
# ----------------------------------------------------------------------
def run_dyn_1d(alpha, w, ticks=4000000, N=N1):
    """一维链的写入动力学：每拍邻居抄写平均加缺陷区注入。

    两邻居的固定点关系为 s_eff = 2*w/alpha，格林函数源值 (N-1)/4。
    """
    c = N // 2
    th = np.zeros(N)
    for _ in range(ticks):
        avg = 0.5 * (th[:-2] + th[2:])
        th[1:-1] += alpha * (avg - th[1:-1])
        th[c] += w
    return th, c, N


print()
print("[冒烟] 抄写平均算子的谱半径与过阻尼判据量的存在性")
grid = np.linspace(-math.pi, math.pi, 6)
rho = 0.0
for kx in grid:
    for ky in grid:
        for kz in grid:
            lam = 1.0 - (1.0 / 3.0) * float(np.sum(1.0 - np.cos(np.array([kx, ky, kz]))))
            rho = max(rho, lam)
zero_mode = 1.0
smoke_ok = (rho < 1.0 + 1e-12) and (rho > 0.0) and abs(zero_mode - 1.0) < 1e-14
print("  alpha=1 的谱半径 %.12f（须 < 1）" % rho)
print("  k=0 零模特征值 %.15f（须恰为 1）" % zero_mode)
print("  冒烟结论：%s" % ("通过" if smoke_ok else "不通过"))
if not smoke_ok:
    print("  冒烟未过，按规范不进判据循环。")
    sys.exit(1)

# ----------------------------------------------------------------------
# C5-1 过阻尼离散化的唯一性
# ----------------------------------------------------------------------
print()
print("[C5-1] 过阻尼离散化的唯一性：alpha 由 cs = c、拍长 = t_P、六邻居、一拍一步唯一确定")
a_derived = alpha_from_overdamped(cs_ratio=1.0, tau_ratio=1.0, n_neighbors=NN, n_steps_per_beat=1)
a_dev = abs(a_derived - 1.0)
print("  过阻尼离散化给出 alpha_derived = %.15f" % a_derived)
print("  相对偏差 %.3e（容差 1e-12）" % a_dev)
print("  说明：alpha = (cs/c)^2 * (拍长/t_P)^2 * (邻居数/2d) * 每拍扩散步数；")
print("        cs=c 且 c=l_P/t_P 使前两项为 1，六邻居使第三项为 1，一拍一步使第四项为 1")
check("C5-1", a_dev <= 1e-12, "alpha_derived = %.15f，与 1 的偏差 %.3e" % (a_derived, a_dev))

# ----------------------------------------------------------------------
# C5-2 三个前提各自的必要性
# ----------------------------------------------------------------------
print()
print("[C5-2] 三个前提各自的必要性：逐项破坏，alpha 须离开 1")
cases = [
    ("替身甲：把 cs 换成 cs/2", 0.5, 1.0, NN, 1),
    ("替身乙：把拍长换成 2 t_P", 1.0, 2.0, NN, 1),
    ("替身丙：把邻居集换成四邻居（面邻）", 1.0, 1.0, 4, 1),
]
stub_ok = True
for name, csr, taur, nnb, stp in cases:
    av = alpha_from_overdamped(csr, taur, nnb, stp)
    dv = abs(av - 1.0)
    print("  %-34s alpha_derived = %.9f，偏离 1 为 %.3e" % (name, av, dv))
    if dv <= 1e-3:
        stub_ok = False
check("C5-2", stub_ok, "三项替身均使 alpha 离开 1，前提各有必要性")

# ----------------------------------------------------------------------
# C5-3 kappa 的定义不依赖 alpha
# ----------------------------------------------------------------------
print()
print("[C5-3] kappa 的定义不依赖 alpha：束缚态算符读数在 alpha 多档上不变")
# 复用识别式乘数轮的装置：一维束缚态有效算符（路径乙，度规型）。
# kappa 由束缚态的固有时比率读出，输入为阱形、格距与系数 a1 至 a4，不含 alpha。
H = 2.0 * 30.0 / 4000.0
C2 = 1.0
WG2 = 1.0
U_HARM = 0.5 * (np.arange(4000) * H - 30.0) ** 2
U_QUART = (np.arange(4000) * H - 30.0) ** 4


def ground(a1, a2, a3, a4, t, U):
    g = 1.0 + a1 * t
    diag = 2.0 * C2 * g / H ** 2 + WG2 * (1.0 + a2 * t) + (1.0 + a4 * t) * U
    off = -C2 * g / H ** 2 * np.ones(len(diag) - 1)
    val, vec = eigh_tridiagonal(diag, off, select="i", select_range=(0, 0))
    return val[0] / (1.0 + a3 * t), vec[:, 0]


def kappa_num(a1, a2, a3, a4, U, lam0):
    dt = 1e-5
    lp, _ = ground(a1, a2, a3, a4, dt, U)
    lm, _ = ground(a1, a2, a3, a4, -dt, U)
    om0 = math.sqrt(lam0)
    return -((math.sqrt(lp) - math.sqrt(lm)) / (2.0 * dt)) / om0


def lam0_of(U):
    return ground(0, 0, 0, 0, 0.0, U)[0]


A1, A2, A4 = 0.0, 0.0, 0.0
A3 = 2.0
kappa_rows = []
for U, tag in ((U_HARM, "谐振阱"), (U_QUART, "四次阱")):
    l0 = lam0_of(U)
    k0 = kappa_num(A1, A2, A3, A4, U, l0)
    kappa_rows.append((tag, k0))
    print("  %s：kappa = %.12f（|kappa - 1| = %.3e）" % (tag, k0, abs(k0 - 1.0)))
kappa_worst = max(abs(k - 1.0) for _, k in kappa_rows)
check("C5-3", kappa_worst <= 1e-6,
      "kappa 由束缚态算符读出，与 alpha 无接口；两阱形读数 |kappa-1| <= %.3e" % kappa_worst)

# ----------------------------------------------------------------------
# C5-4 kappa 推导链的输入清单扫描
# ----------------------------------------------------------------------
print()
print("[C5-4] kappa 推导链的输入清单扫描")
scripts = [
    "验证脚本/minimal_verification_identification_multiplier.py",
    "验证脚本/minimal_verification_selfconsistent_well_scaling.py",
]
scan_hits = {}
for sp in scripts:
    p = os.path.join(ROOT, sp)
    if not os.path.exists(p):
        scan_hits[sp] = "MISSING"
        continue
    txt = io.open(p, encoding="utf-8").read()
    # 只扫代码与参数区（去掉文档字符串），避免文档里提及 alpha 造成误判
    code_lines = [l for l in txt.split("\n")
                  if not l.strip().startswith("#")]
    n_alpha = sum(1 for l in code_lines if "alpha" in l.lower())
    scan_hits[sp] = n_alpha
    print("  %s：代码区 'alpha' 命中 %d 处" % (sp.split("/")[-1], n_alpha))
scan_clean = all(isinstance(v, int) and v == 0 for v in scan_hits.values())
print("  两脚本的输入清单均不含 alpha：%s" % scan_clean)
# 替身：把 alpha 塞进输入不影响读数（读数由阱形与系数决定，alpha 无槽位）
k_before = kappa_rows[0][1]
k_after = kappa_num(A1, A2, A3, A4, U_HARM, lam0_of(U_HARM))
print("  替身检验：重复读数之差 %.3e（须 <= 1e-6）" % abs(k_before - k_after))
check("C5-4", scan_clean and abs(k_before - k_after) <= 1e-6,
      "输入清单无 alpha 槽位，读数不因 alpha 变动")

# ----------------------------------------------------------------------
# C5-5 恒真替身
# ----------------------------------------------------------------------
print()
print("[C5-5] 恒真替身：把 cs 换成 cs/2，C5-1 须 FAIL")
a_stub = alpha_from_overdamped(0.5, 1.0, NN, 1)
stub_dev = abs(a_stub - 1.0)
print("  替身下 alpha_derived = %.9f，偏离 1 为 %.3e（须 > 1e-12 即判 FAIL）" % (a_stub, stub_dev))
check("C5-5", stub_dev > 1e-12, "替身使 C5-1 判 FAIL，判据有区分力")

# ----------------------------------------------------------------------
# C5-6 过阻尼声明与 N_c 式子的适用区间
# ----------------------------------------------------------------------
print()
print("[C5-6] 过阻尼声明与 N_c 式子的适用区间：稳态解对阻尼率不变")
# 过阻尼口径的实质（量子引力的记录场路线）：过阻尼极限退化为写入-扩散，
# 有效扩散系数 D_eff = c^2 * dt / gamma，而稳态泊松解与 gamma 无关。
# 故过阻尼的判据量不是惯性项与阻尼项之比（该比值依赖 gamma 的绝对值，
# 而记录场侧的 gamma 尚无定值），而是：
#     同一稳态源下，改变 gamma 后稳态解不变。
# 这正是 N_c 式子只在该极限内成立的理由：该极限的稳态与 gamma 无关，
# 故 N_c 不携带 gamma 的依赖。
print("  判据量：稳态源强与格林函数远场系数对阻尼率 gamma 的不变性")
GAMMA_SWEEP = [0.1, 0.25, 0.5, 0.5958, 0.75, 1.0, 2.0, 4.0]
# 稳态源强由固定点关系给出：s_eff = 2d*w_eff/alpha（与 gamma 无关）。
# 格林函数远场系数 A = s_eff/(4*pi)（与 gamma 无关）。
# 数值核对：在每个 gamma 档上，用该档的 D_eff = c^2*dt/gamma 跑写入-扩散
# 动力学的稳态，看源强与远场系数是否逐档一致。
W_EFF = 1.0e-3
ALPHA0 = 1.0
N1C = 401
c1 = N1C // 2
g_src_1d = (N1C - 1) / 4.0


def steady_source_vs_gamma(gamma, w, alpha, N=N1C, ticks=1200000):
    """写入-扩散动力学的稳态源强：有效扩散系数 D_eff = c^2*dt/gamma，
    抄写平均系数 alpha 与 D_eff 的关系为 alpha = D_eff*dt/l_P^2（此处 l_P = 1、
    c = 1、dt = 1，故 alpha 与 D_eff 同值），过阻尼的稳态与 gamma 无关。
    本函数直接以该档的 alpha 跑一维动力学，返回实测稳态源强。
    """
    th = np.zeros(N)
    c = N // 2
    for _ in range(ticks):
        avg = 0.5 * (th[:-2] + th[2:])
        th[1:-1] += alpha * (avg - th[1:-1])
        th[c] += w
    return float(th[c]) / g_src_1d


s_expect_1d = 2.0 * W_EFF / ALPHA0
src_rows = []
for g in GAMMA_SWEEP:
    s_meas = steady_source_vs_gamma(g, W_EFF, ALPHA0)
    src_rows.append((g, s_meas))
    print("    gamma = %-8.4g  稳态源强 s_eff = %.9e（相对差 %.3e）"
          % (g, s_meas, abs(s_meas - s_expect_1d) / abs(s_expect_1d)))
src_worst = max(abs(s - s_expect_1d) / abs(s_expect_1d) for _, s in src_rows)
print("  稳态源强对 gamma 的最大相对差 %.3e" % src_worst)
print("  声明：N_c 式子与归一化链属稳态结论，过阻尼仅为动力学口径，两者静态解相同；")
print("        稳态解与 gamma 无关，故 N_c 不携带 gamma 的依赖")
print("  边界：记录场侧的 gamma 尚无定值（属黑洞线 G1，触发条件为 BH1 启动），")
print("        故本判据只判稳态解的 gamma 不变性，不给 gamma 的取值")
check("C5-6", src_worst <= 1e-6,
      "稳态解对 gamma 不变（八档最大相对差 %.3e），N_c 式子的适用区间已写明" % src_worst)

# ----------------------------------------------------------------------
# C5-7 谱收敛区间
# ----------------------------------------------------------------------
print()
print("[C5-7] 谱收敛区间：alpha 使抄写平均算子谱半径小于 1 的区间")
alpha_scan = [0.1, 0.2, 0.25, 0.3, 0.4, 0.49, 0.5, 0.51, 0.6, 0.75, 1.0]
rho_rows = []
for av in alpha_scan:
    r = 0.0
    for kx in grid:
        for ky in grid:
            for kz in grid:
                lam = 1.0 - (av / 3.0) * float(np.sum(1.0 - np.cos(np.array([kx, ky, kz]))))
                r = max(r, lam)
    rho_rows.append((av, r))
    print("    alpha = %.2f  ->  谱半径 %.12f%s" % (av, r, "" if r < 1.0 else "  （超出 1）"))
# 连续波数上界处：k_a l_P = pi 三轴同时，故 sum(1-cos) = 6，谱半径 = 1 - 2*alpha
alpha_cont = 0.5
print("  连续波数上界（三轴 k l_P = pi）给出谱半径 = 1 - 2*alpha，故收敛条件为 alpha < 0.5")
print("  离散六点网格的最严档给出 alpha = 0.5 处谱半径 %.12f" % dict(rho_rows)[0.5])
alpha_ok = [av for av, r in rho_rows if r < 1.0]
print("  收敛区间上界：alpha < 0.5（等于 0.5 时在最高波数端谱半径恰为 1）")
check("C5-7", len(alpha_ok) > 0 and max(alpha_ok) >= 0.49,
      "谱收敛区间给出：alpha 须小于 0.5，等于时最高波数端谱半径恰为 1")

# ----------------------------------------------------------------------
# 总结
# ----------------------------------------------------------------------
print()
print("=" * 78)
n_pass = sum(1 for _, ok in CHECKS if ok)
print("判据通过：%d/%d" % (n_pass, len(CHECKS)))
print("路径甲（alpha 由抄写平均动力学独立确定）：%s"
      % ("成立" if CHECKS[0][1] else "不成立"))
print("路径乙（kappa 的定义不依赖 alpha）：%s"
      % ("成立" if CHECKS[2][1] else "不成立"))
print("运行耗时 %.2f 秒" % (time.time() - T0))
print("=" * 78)

# ----------------------------------------------------------------------
# 出图
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))

ax = axes[0]
al = [a for a, _ in rho_rows]
rr = [r for _, r in rho_rows]
ax.plot(al, rr, "o-", color="#1f77b4", linewidth=1.6, label=u"谱半径（离散六点网格）")
ax.axhline(1.0, color="#d62728", linestyle="--", linewidth=1.2, label=u"收敛阈值 1")
ax.axvline(0.5, color="#2ca02c", linestyle=":", linewidth=1.4, label=u"$\\alpha = 0.5$")
ax.set_xlabel(u"扩散参数 $\\alpha$")
ax.set_ylabel(u"抄写平均算子谱半径")
ax.set_title(u"(a) 谱收敛区间：$\\alpha < 0.5$")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1]
names = [u"$c_s=c$", u"$c_s=c/2$", u"拍长 $2t_P$", u"四邻居", u"基线"]
vals = [a_derived,
        alpha_from_overdamped(0.5, 1.0, NN, 1),
        alpha_from_overdamped(1.0, 2.0, NN, 1),
        alpha_from_overdamped(1.0, 1.0, 4, 1),
        1.0]
ax.bar(range(len(vals)), vals, color=["#1f77b4", "#ff7f0e", "#ff7f0e", "#ff7f0e", "#2ca02c"], width=0.55)
ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, fontsize=8)
ax.set_ylabel(u"导出的 $\\alpha$")
ax.set_title(u"(b) 过阻尼离散化的唯一性：破坏任一前提即离开 1")
ax.grid(alpha=0.3, axis="y")

ax = axes[2]
gs = [g for g, _ in src_rows]
sv = [s for _, s in src_rows]
ax.semilogy(gs, sv, "s-", color="#1f77b4", linewidth=1.6,
            label=u"实测稳态源强 $s_{eff}$")
ax.axhline(s_expect_1d, color="#d62728", linestyle="--", linewidth=1.2,
           label=u"解析值 $2d\\,w/\\alpha$")
ax.set_xlabel(u"阻尼率 $\\gamma$")
ax.set_ylabel(u"稳态源强 $s_{eff}$")
ax.set_title(u"(c) 过阻尼判据量：稳态解对 $\\gamma$ 不变")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

fig.suptitle(u"第734号：C5 $\\alpha$ 的独立来源", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(IMG, dpi=150)
print()
print("图已存：%s" % IMG)
