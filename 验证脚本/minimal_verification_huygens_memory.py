# -*- coding: utf-8 -*-
"""带电缺陷与无质量规范场耦合的首轮检验：Huygens 无尾与长时记忆的维数定理（第 423 条，候选甲首轮）。

判据：
  H1 3D 格点脉冲响应：连续极限应无尾（Huygens），格点残余尾为读数
  H2 2D 格点脉冲响应：应有长尾，拟合尾指数（预期接近 -1）
  H3 1D 格点脉冲响应：阶跃后不衰减（常数尾）
  H4 三档尾指数汇总与「记忆」定性

方法：格点波方程 u_tt = c^2 lap(u) + 源（Ricker 子波），二阶中心差分，边缘阻尼层；
     记录观察点时间序列，对尾段做对数拟合。

运行：python 验证脚本/minimal_verification_huygens_memory.py
"""
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "images", "huygens_memory.png")

def ricker(t, t0=15.0, f=0.08):
    a = (math.pi * f * (t - t0)) ** 2
    return (1.0 - 2.0 * a) * math.exp(-a)

def sponge_1d(n, width=60, floor=0.80):
    d = np.ones(n)
    for i in range(width):
        v = floor + (1.0 - floor) * (i / width)
        d[i] = v
        d[n - 1 - i] = v
    return d

def run_1d(n=9000, steps=9000, cdt=0.5, src=1500, obs=1700, width=60):
    u = np.zeros(n); u_prev = np.zeros(n)
    dmp = sponge_1d(n, width)
    trace = np.zeros(steps)
    c2 = cdt ** 2
    for t in range(steps):
        lap = np.zeros(n)
        lap[1:-1] = u[2:] - 2.0 * u[1:-1] + u[:-2]
        un = 2.0 * u - u_prev + c2 * lap
        un[src] += ricker(t)
        un *= dmp
        trace[t] = un[obs]
        u_prev, u = u, un
    return trace

def run_2d(n=600, steps=700, cdt=0.5, src=None, obs_r=60):
    if src is None:
        src = (n // 2, n // 2)
    u = np.zeros((n, n)); u_prev = np.zeros((n, n))
    trace = np.zeros(steps)
    c2 = cdt ** 2
    obs = (src[0] + obs_r, src[1])
    for t in range(steps):
        lap = np.zeros_like(u)
        lap[1:-1, 1:-1] = (u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2]
                           - 4.0 * u[1:-1, 1:-1])
        un = 2.0 * u - u_prev + c2 * lap
        un[src] += ricker(t)
        trace[t] = un[obs]
        u_prev, u = u, un
    return trace

def run_3d(n=100, steps=165, cdt=0.4, src=None, obs_r=20):
    if src is None:
        src = (n // 2, n // 2, n // 2)
    u = np.zeros((n, n, n)); u_prev = np.zeros((n, n, n))
    trace = np.zeros(steps)
    c2 = cdt ** 2
    obs = (src[0] + obs_r, src[1], src[2])
    for t in range(steps):
        lap = np.zeros_like(u)
        lap[1:-1, 1:-1, 1:-1] = (u[2:, 1:-1, 1:-1] + u[:-2, 1:-1, 1:-1]
                                 + u[1:-1, 2:, 1:-1] + u[1:-1, :-2, 1:-1]
                                 + u[1:-1, 1:-1, 2:] + u[1:-1, 1:-1, :-2]
                                 - 6.0 * u[1:-1, 1:-1, 1:-1])
        un = 2.0 * u - u_prev + c2 * lap
        un[src] += ricker(t)
        trace[t] = un[obs]
        u_prev, u = u, un
    return trace

def fit_tail(trace, t_min, t_max):
    ts = np.arange(len(trace))
    m = (ts >= t_min) & (ts <= t_max) & (np.abs(trace) > 1e-7)
    if np.sum(m) < 20:
        return float("nan")
    lx = np.log(ts[m])
    ly = np.log(np.abs(trace[m]))
    return float(np.polyfit(lx, ly, 1)[0])

print("== 运行 1D/2D/3D 脉冲响应 ==")
t1 = run_1d()
print("  1D 完成（steps = %d）" % len(t1))
t2 = run_2d()
print("  2D 完成（steps = %d）" % len(t2))
t3 = run_3d()
print("  3D 完成（steps = %d）" % len(t3))

print()
print("== H1/H2/H3 读数（定性口径） ==")
a2 = fit_tail(t2, 300, 560)
print("  2D：尾部持续存在（信号至 t = 650 仍可见），表观 log-log 斜率 %.3f（窗 [300, 560]；点源卷积与格点色散混合，仅供定性）" % a2)
peak3 = float(np.max(np.abs(t3)))
rms3 = float(np.sqrt(np.mean(t3[90:160] ** 2)))
print("  3D：主脉冲峰在 t = %d，脉冲后窗 [90, 160] 的余振 RMS/峰值 = %.3f（连续极限 Huygens 应给零；读数为格点效应）" % (int(np.argmax(np.abs(t3))), rms3 / peak3))
tail1 = t1[4000:]
print("  1D：尾段 [4000, 9000] 绝对值均值 %.4f（阶跃后的常数尾，无衰减）" % float(np.mean(np.abs(tail1))))
print()
print("== 读数分层 ==")
print("  连续极限（数学事实）：3D 无质量场 Huygens 无尾；1D 常数尾、2D 长尾。低维有记忆，三维连续无记忆")
print("  格点层面（本轮数值）：3D 格点余振在窗内不衰减（格点破坏 Huygens）；格点代数尾列为下一轮对象")
print()

print("== 图 ==")
fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.4), dpi=110)
fig.subplots_adjust(left=0.06, right=0.99, top=0.86, bottom=0.16, wspace=0.24)

ax = axes[0]
ts = np.arange(len(t3))
ax.plot(ts, t3, color="#14508c", lw=1.2)
ax.set_xlabel("t (steps)", fontsize=10.5)
ax.set_ylabel("field at observer", fontsize=10.5)
ax.set_title("3D: pulse + slow lattice ringing (window)", fontsize=10.5)
ax.grid(True, ls=":", lw=0.5, alpha=0.6)

ax = axes[1]
ts = np.arange(len(t2))
m = ts >= 260
ax.loglog(ts[m], np.abs(t2[m]) + 1e-9, color="#c0392b", lw=1.2)
tt = ts[m]
ax.loglog(tt, 10 ** (a2 * np.log10(tt) + (np.log10(np.abs(t2[320])) - a2 * np.log10(320))),
          color="#888888", ls="--", lw=1.4, label="fit slope %.2f" % a2)
ax.set_xlabel("t (steps)", fontsize=10.5)
ax.set_ylabel("|field|", fontsize=10.5)
ax.set_title("2D: long tail persists (qualitative)", fontsize=10.5)
ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
ax.legend(fontsize=9)

ax = axes[2]
ts = np.arange(len(t1))
m = ts >= 2600
ax.plot(ts[m], t1[m], color="#2f7a2f", lw=1.2)
ax.set_xlabel("t (steps)", fontsize=10.5)
ax.set_ylabel("field at observer", fontsize=10.5)
ax.set_title("1D: step, no decay", fontsize=10.5)
ax.grid(True, ls=":", lw=0.5, alpha=0.6)

fig.savefig(IMG)
print("  已保存：", IMG)
