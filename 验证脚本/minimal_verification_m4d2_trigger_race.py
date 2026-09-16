# -*- coding: utf-8 -*-
"""M4-D2：两路触发的竞争（破坏是否必待满偏的判别形态）。

判据（先写期望，再算；设计见《黑洞线攻坚计划：坍缩机制重定位与强场终态》§7.2）：
  D2-1 解析锚点：一维均匀与三维均匀的 int_0^1 gamma dh 对闭式一致（相对偏差 <= 1e-3，
       梯形误差含在内）；gamma(h) = sqrt(-mu_min(h))。
  D2-2 交叉点结构：数值求解耦合时钟 t_c 与饱和时钟 t_sat = 1/Gamma 的交点，与解析
       Gamma_c = I / ln(1/eps0) 一致（相对偏差 <= 1e-6），其中 I = int_0^1 gamma dh。
  D2-3 一维构型不敏感：一维均匀、块 w=3、块 w=7 的 Gamma_c 两两之差 < 1%。
  D2-4 三维退化：三维立方体 2^3 的 Gamma_c 低于三维均匀的十分之一；且 L=16 对 32 的
       比值落在 1/sqrt(8) 的 +/-20% 内（1/sqrt(V) 标度）。
  D2-5 噪声底依赖：Gamma_c 比值 r(1e-6)/r(1e-2) 等于 ln(100)/ln(1e6)（<= 1e-9 相对）。
  D2-6 报告：一维与三维漏斗的 Gamma_c；单点构型的无触发（gamma 恒为 0）；
       指数写入律变体（一维均匀）的交叉点。

模型：模型与 D1 同源（一维交错链 N=256；三维简单立方 L=16/32、交错背景、度数 6），
写入律取线性斜坡 h_n(t) = Gamma t h_n^(0)，噪声底 eps0 = 1e-3，饱和阈值 h = 1。
确定性：本征值扫描无随机结果（三维稀疏求解器固定按序生成的初矢）。
图存 D:/software/TianXingFC/images/m4d2_trigger_race.png。
"""

import sys
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

EPS0 = 1e-3
LAMBDA = np.log(1.0 / EPS0)
CHECKS = []
T0 = time.time()
RNG = np.random.RandomState(20260917)


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))


def low_1d(h):
    N = len(h)
    n = np.arange(N)
    H = np.zeros((N, N))
    H[n, n] = 2.0 + ((-1.0) ** n) * h
    H[n, (n + 1) % N] += 1.0
    H[n, (n - 1) % N] += 1.0
    return float(np.linalg.eigvalsh(H)[0])


def low_1d_amp(a):
    return low_1d(np.full(N1, a))


def build_3d(L, h):
    N = L ** 3
    idx = np.arange(N)
    i = idx // (L * L); j = (idx // L) % L; k = idx % L
    sgn = (-1.0) ** (i + j + k)
    nxp = i * L * L + j * L + (k + 1) % L
    nyp = i * L * L + ((j + 1) % L) * L + k
    nzp = ((i + 1) % L) * L * L + j * L + k
    rows = np.concatenate([idx, idx, idx, nxp, nyp, nzp])
    cols = np.concatenate([nxp, nyp, nzp, idx, idx, idx])
    return sp.coo_matrix((-np.ones(6 * N), (rows, cols)), shape=(N, N)).tocsr() + sp.diags(6.0 + sgn * h)


def low_3d(L, h):
    N = L ** 3
    v0 = RNG.standard_normal(N)
    v0 /= np.linalg.norm(v0)
    return float(spla.eigsh(build_3d(L, h), k=1, which="SA", v0=v0, tol=0, return_eigenvectors=False)[0])


F1 = 2.0 / 3.0 * np.sqrt(np.sqrt(5.0) + 2.0) * (np.sqrt(5.0) - 4.0) + 8.0 / 3.0

def F3_exact():
    def anti(u):
        return (2.0 / 3.0) * np.sqrt(u + 6.0) * (u - 12.0)
    return anti(np.sqrt(37.0)) - anti(6.0)


def gamma_curve(lowfun, prof, npts, hmax=1.0):
    hs = np.linspace(0.0, hmax, npts)
    g = np.array([np.sqrt(max(0.0, -lowfun(a * prof))) for a in hs])
    return hs, g


def integral(hs, g, upto=1.0):
    m = hs <= upto
    return float(np.trapezoid(g[m], hs[m]))


def hc_from_exponent(hs, cum, exponent):
    """返回 int_0^{h_c} gamma dh = exponent 的 h_c（cum 为累积积分，单调）。"""
    return float(np.interp(exponent, cum, hs))


print("==== M4-D2：两路触发的竞争 ====")
print("噪声底 eps0 = %.0e，ln(1/eps0) = %.6f" % (EPS0, LAMBDA))

# 构型准备
N1 = 256
d1 = np.minimum(np.arange(N1), N1 - np.arange(N1))
prof_1d = {
    "一维均匀": np.ones(N1),
    "一维块 w=3": np.where(d1 <= 1, 1.0, 0.0),
    "一维块 w=7": np.where(d1 <= 3, 1.0, 0.0),
    "一维单点": np.where(d1 == 0, 1.0, 0.0),
    "一维漏斗": 1.0 / (1.0 + d1),
}
res1 = {}
for name, prof in prof_1d.items():
    hs, g = gamma_curve(low_1d, prof, 601, hmax=1.5)
    cum = np.concatenate([[0.0], np.cumsum((g[1:] + g[:-1]) * np.diff(hs) / 2.0)])
    I = integral(hs, g, upto=1.0)
    res1[name] = dict(hs=hs, g=g, cum=cum, I=I, gmax=float(g.max()))

# D2-1（一维）
rel1 = abs(res1["一维均匀"]["I"] - F1) / F1
print("\n  一维均匀积分 %.6f 对闭式 %.6f（相对偏差 %.2e）" % (res1["一维均匀"]["I"], F1, rel1))

# 三维
L = 16
i16, j16, k16 = np.meshgrid(np.arange(L), np.arange(L), np.arange(L), indexing="ij")
flat16 = (i16.ravel(), j16.ravel(), k16.ravel())
base_uniform3 = np.ones(L ** 3)
base_cube3 = ((flat16[0] < 2) & (flat16[1] < 2) & (flat16[2] < 2)).astype(float)
Ls = 32
idx32 = np.arange(Ls ** 3)
i32 = idx32 // (Ls * Ls); j32 = (idx32 // Ls) % Ls; k32 = idx32 % Ls
di = np.minimum(i32, Ls - i32); dj = np.minimum(j32, Ls - j32); dk = np.minimum(k32, Ls - k32)
base_funnel32 = 1.0 / (1.0 + np.sqrt(di * di + dj * dj + dk * dk))
base_cube32 = ((i32 < 2) & (j32 < 2) & (k32 < 2)).astype(float)

hs3, g_u3 = gamma_curve(lambda a: low_3d(16, a), base_uniform3, 61, hmax=1.5)
cum_u3 = np.concatenate([[0.0], np.cumsum((g_u3[1:] + g_u3[:-1]) * np.diff(hs3) / 2.0)])
I_u3 = integral(hs3, g_u3, upto=1.0)

hs3c, g_c3 = gamma_curve(lambda a: low_3d(16, a), base_cube3, 61, hmax=1.5)
cum_c3 = np.concatenate([[0.0], np.cumsum((g_c3[1:] + g_c3[:-1]) * np.diff(hs3c) / 2.0)])
I_c3 = integral(hs3c, g_c3, upto=1.0)

hs3c32, g_c3_32 = gamma_curve(lambda a: low_3d(32, a), base_cube32, 61, hmax=1.5)
I_c3_32 = integral(hs3c32, g_c3_32, upto=1.0)

hs3f, g_f32 = gamma_curve(lambda a: low_3d(32, a), base_funnel32, 61, hmax=1.5)
I_f32 = integral(hs3f, g_f32, upto=1.0)

F3 = F3_exact()
rel3 = abs(I_u3 - F3) / F3
check("D2-1", rel1 <= 1e-3 and rel3 <= 1e-3,
      "一维均匀 %.6f 对闭式 %.6f（%.2e）；三维均匀 %.6f 对闭式 %.6f（%.2e）"
      % (res1["一维均匀"]["I"], F1, rel1, I_u3, F3, rel3))

# D2-2 交叉点
from scipy.optimize import brentq


def crossover(hs, cum, I):
    """t_c(Gamma) = h_c/Gamma，t_sat = 1/Gamma；交点即 h_c = 1，等价 Gamma_c = I/LAMBDA。"""
    gc = I / LAMBDA
    f = lambda g: hc_from_exponent(hs, cum, g * LAMBDA) - 1.0
    lo, hi = gc * (1.0 - 1e-3), gc * (1.0 + 1e-3)
    assert f(lo) < -1e-9 and f(hi) > 1e-9, (f(lo), f(hi))
    g_num = float(brentq(f, lo, hi, xtol=1e-16, rtol=1e-15))
    ok = True
    for g in np.logspace(-4, 1, 401):
        hc = hc_from_exponent(hs, cum, g * LAMBDA)
        if hc >= hs[-1] or hc <= hs[0]:
            continue
        if np.sign(hc - 1.0) != np.sign(g - gc):
            ok = False
    return gc, g_num, ok


gc1, gn1, ok1 = crossover(res1["一维均匀"]["hs"], res1["一维均匀"]["cum"], res1["一维均匀"]["I"])
gc3, gn3, ok3 = crossover(hs3, cum_u3, I_u3)
rel = max(abs(gn1 - gc1) / gc1, abs(gn3 - gc3) / gc3)
check("D2-2", (rel <= 1e-6) and ok1 and ok3,
      "一维 Gamma_c 解析 %.6e 对数值 %.6e；三维 %.6e 对 %.6e（最大相对差 %.1e；两路先后的符号一致 %s/%s）"
      % (gc1, gn1, gc3, gn3, rel, ok1, ok3))

# D2-3 一维构型不敏感
gcs = {k: v["I"] / LAMBDA for k, v in res1.items()}
spread = max(abs(gcs["一维均匀"] - gcs["一维块 w=3"]), abs(gcs["一维均匀"] - gcs["一维块 w=7"])) / gcs["一维均匀"]
check("D2-3", spread < 0.01, "Gamma_c 均匀 %.6e、块 w=3 %.6e、块 w=7 %.6e（最大相对差 %.4f%%）"
      % (gcs["一维均匀"], gcs["一维块 w=3"], gcs["一维块 w=7"], 100 * spread))

# D2-4 三维退化
gc_c3 = I_c3 / LAMBDA
gc_c3_32 = I_c3_32 / LAMBDA
ratio_scale = (I_c3_32 / I_c3)
target = 1.0 / np.sqrt(8.0)
ok4 = (gc_c3 < 0.1 * gc3) and (abs(ratio_scale / target - 1.0) <= 0.2)
check("D2-4", ok4, "三维立方体 Gamma_c %.3e（均匀 %.3e，比值 %.3f）；L=16 对 32 的积分比 %.3f（目标 1/sqrt(8) = %.3f）"
      % (gc_c3, gc3, gc_c3 / gc3, ratio_scale, target))

# D2-5 噪声底依赖
r_a = 1.0 / np.log(1.0 / 1e-2)
r_b = 1.0 / np.log(1.0 / 1e-6)
rel5 = abs(r_b / r_a - np.log(100.0) / np.log(1e6)) / (np.log(100.0) / np.log(1e6))
check("D2-5", rel5 <= 1e-9, "比值 r(1e-6)/r(1e-2) = %.6f 对 ln(100)/ln(1e6) = %.6f"
      % (r_b / r_a, np.log(100.0) / np.log(1e6)))

# D2-6 报告
print("\n  [D2-6] 报告")
print("       单点构型：gamma 最大值 %.3e（无触发，期望 0）" % res1["一维单点"]["gmax"])
check("D2-6a", res1["一维单点"]["gmax"] <= 1e-12, "单点构型 gamma 恒为 0")
print("       一维漏斗：I = %.6f，Gamma_c = %.3e" % (res1["一维漏斗"]["I"], gcs["一维漏斗"]))
print("       三维漏斗（L=32）：I = %.6e，Gamma_c = %.3e" % (I_f32, I_f32 / LAMBDA))
print("       三维立方体（L=32）：I = %.6e，Gamma_c = %.3e" % (I_c3_32, gc_c3_32))
h0_exp = 1e-6
xs = np.logspace(np.log10(h0_exp), 0, 121)
g_exp = np.array([np.sqrt(max(0.0, -low_1d_amp(a))) / a for a in xs])
I_exp = float(np.trapezoid(g_exp, np.log(xs)))
print("       指数写入律（一维均匀，h0=1e-6）：int gamma dh/h = %.6f，Gamma_c = %.6f" % (I_exp, I_exp / LAMBDA))

# 图
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
fig, axs = plt.subplots(2, 2, figsize=(11.2, 8.0))

ax = axs[0, 0]
for name in ["一维均匀", "一维块 w=3", "一维漏斗"]:
    ax.semilogy(res1[name]["hs"][1:], res1[name]["g"][1:], label=name)
ax.set_xlabel("记录幅度 h（峰值）")
ax.set_ylabel("生长率 gamma = sqrt(-mu_min)")
ax.set_title("一维构型的生长率曲线", fontsize=9)
ax.legend(fontsize=8)

ax = axs[0, 1]
ax.semilogy(hs3[1:], g_u3[1:], label="三维均匀（L=16）")
ax.semilogy(hs3c[1:], g_c3[1:], label="三维立方体 2^3（L=16）")
ax.semilogy(hs3f[1:], g_f32[1:], label="三维漏斗（L=32）")
ax.set_xlabel("记录幅度 h（峰值）")
ax.set_ylabel("生长率 gamma")
ax.set_title("三维构型的生长率曲线", fontsize=9)
ax.legend(fontsize=8)

ax = axs[1, 0]
gs = np.logspace(-5, np.log10(2.5 * gc3), 200)
for name, hgrid, cum, I, color in [("三维均匀", hs3, cum_u3, I_u3, "tab:blue"),
                                    ("三维立方体 2^3", hs3c, cum_c3, I_c3, "tab:red")]:
    hs_sol = np.array([hc_from_exponent(hgrid, cum, g * LAMBDA) for g in gs])
    ax.loglog(gs, hs_sol / gs, "-", color=color, label="t_coupl：%s" % name)
    ax.loglog(gs, 1.0 / gs, "--", color=color, alpha=0.6)
    gc = I / LAMBDA
    ax.axvline(gc, color=color, lw=0.8, ls=":")
ax.text(2e-4, 3e2, "虚线为 t_sat = 1/Gamma；点线为交叉点", fontsize=8)
ax.set_xlabel("写入速率 Gamma")
ax.set_ylabel("触发时刻")
ax.set_title("两路时钟的竞争（三维）", fontsize=9)
ax.legend(fontsize=8)

ax = axs[1, 1]
labels = ["1D 均匀", "1D 块 w=3", "1D 漏斗", "3D 均匀", "3D 立方体 L=16", "3D 立方体 L=32", "3D 漏斗 L=32"]
vals = [gcs["一维均匀"], gcs["一维块 w=3"], gcs["一维漏斗"], gc3, gc_c3, gc_c3_32, I_f32 / LAMBDA]
ax.bar(labels, vals, color=["tab:blue", "tab:blue", "tab:blue", "tab:orange", "tab:red", "tab:red", "tab:brown"])
ax.set_yscale("log")
ax.tick_params(axis="x", labelrotation=30, labelsize=8)
ax.set_ylabel("交叉写入速率 Gamma_c")
ax.set_title("交叉点总览（eps0 = 1e-3）", fontsize=9)

fig.tight_layout()
fig.savefig(r"D:/software/TianXingFC/images/m4d2_trigger_race.png", dpi=150)
print("\n  图已保存：images/m4d2_trigger_race.png")

n_pass = sum(1 for _, ok in CHECKS if ok)
print("\n==== 判据汇总：%d / %d 通过（运行 %.0f 秒）====" % (n_pass, len(CHECKS), time.time() - T0))
sys.exit(0 if n_pass == len(CHECKS) else 1)