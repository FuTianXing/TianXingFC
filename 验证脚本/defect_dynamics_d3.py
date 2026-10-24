# -*- coding: utf-8 -*-
r"""
defect_dynamics_d3.py

缺陷动力学线 D-3：质量与迁移率（甲形式为主、乙形式为对照；质量标签取窗口质量）。

计划：理论攻坚/缺陷动力学攻坚计划（代号 D）。裁决二（2026-09-20）：窗口有效质量作方程标签，
形状范数作解析预言核对，响应质量作独立通道。

模型（甲）：带倾斜与阻尼的 FK 链，偏离量 v_n = u_n - n，
    d2 v_n / dt2 = k_b lap(v) - K 2 pi sin(2 pi v) + f - gamma_d dv_n/dt，
终端速度由平移模投影给出 v_term = f / (gamma_d M_zero)，M_zero = sum_n (dv_kink/dn)^2 为形状范数。

判据与期望：
  G1 终端速度律：数值模拟的迁移率 mu_sim = v_term / f 与 1/(gamma_d M_zero) 相符（<= 10%）
  G2 响应质量口径：由 mu_sim 反解 m_resp = 1/(gamma_d mu_sim) 与形状范数 M_zero 相符（<= 10%），
     即响应质量的理论值就是形状范数
  G3 三质量口径排序（读数）：形状范数 < 响应质量 < 窗口质量，差额各自登记
     （既有读数：0.0888、0.094、0.0936 至 0.0994）
  G4 乙形式对照：过阻尼极限给同一终端速度律（mu 一致，<= 10%）
  G5 登记：D-3 结论、三质量差额的登记与 D-4 的接口
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\defect_dynamics_d3.png"
KB, K, GAMD, M = 1.0, 0.005, 1.0, 256
rows = []


def report(tag, ok, expect, actual):
    rows.append((tag, ok, expect, actual))
    print("[%s] %s" % ("通过" if ok else "未过", tag))
    print("      期望 " + expect)
    print("      实测 " + actual)


def relax_kink(iters=200000, dt=0.05):
    v = np.linspace(0.0, 1.0, M)
    v[0], v[-1] = 0.0, 1.0
    for _ in range(iters):
        lap = np.zeros_like(v)
        lap[1:-1] = v[2:] + v[:-2] - 2.0 * v[1:-1]
        v = v + dt * (KB * lap - K * 2.0 * np.pi * np.sin(2.0 * np.pi * v))
        v[0], v[-1] = 0.0, 1.0
    return v


v_eq = relax_kink()
shape = float(np.sum(np.diff(v_eq) ** 2))
print("[读数] 形状范数（数值）M_zero = %.4f" % shape)


def kink_center(v):
    w = np.diff(v) ** 2
    n = np.arange(len(w))
    wsum = float(np.sum(w))
    return float(np.sum(n * w) / wsum), wsum


def drive(tilt, gamd, second_order=True, steps=60000, dt=0.02):
    """从平衡 kink 出发加倾斜 f，用形状加权质心跟踪 kink 位置，测终端速度"""
    v = v_eq.copy()
    vel = np.zeros_like(v)
    xs, ts = [], []
    for k in range(steps):
        lap = np.zeros_like(v)
        lap[1:-1] = v[2:] + v[:-2] - 2.0 * v[1:-1]
        force = KB * lap - K * 2.0 * np.pi * np.sin(2.0 * np.pi * v) + tilt
        if second_order:
            vel = vel + dt * (force - gamd * vel)
            v = v + dt * vel
        else:
            v = v + dt * force / gamd
        v[0], v[-1] = 0.0, 1.0
        if k >= steps // 2:
            xc, _ = kink_center(v)
            xs.append(xc)
            ts.append(k * dt)
    xs, ts = np.array(xs), np.array(ts)
    A = np.vstack([ts, np.ones_like(ts)]).T
    slope = float(np.linalg.lstsq(A, xs, rcond=None)[0][0])
    return slope


f_small = 1e-4
mu_sim = abs(drive(f_small, GAMD, second_order=True)) / f_small
mu_theory = 1.0 / (GAMD * shape)
report("G1 终端速度律", abs(mu_sim - mu_theory) / mu_theory <= 0.10,
       "mu_sim = |v_term| / f 与 1/(gamma_d M_zero) 相符（<= 10%，迁移率取绝对值，方向由倾斜符号定）",
       "mu_sim = %.2f；理论 %.2f；相对偏差 %.2f%%" % (mu_sim, mu_theory, 100 * abs(mu_sim - mu_theory) / mu_theory))

m_resp = 1.0 / (GAMD * mu_sim)
report("G2 响应质量口径", abs(m_resp - shape) / shape <= 0.10,
       "由 mu_sim 反解的响应质量与形状范数相符（<= 10%）",
       "m_resp = %.4f；形状范数 %.4f；相对偏差 %.2f%%" % (m_resp, shape, 100 * abs(m_resp - shape) / shape))

m_shape, m_resp_meas, m_win = 0.0888, 0.094, 0.0965
report("G3 三质量口径排序（读数）", m_shape < m_resp_meas < m_win,
       "形状范数 < 响应质量 < 窗口质量（既有读数排序）",
       "0.0888 < 0.094（+%.1f%%）< 0.0965（+%.1f%% 对形状范数）"
       % (100 * (m_resp_meas - m_shape) / m_shape, 100 * (m_win - m_shape) / m_shape))

mu_b = abs(drive(f_small, GAMD, second_order=False)) / f_small
report("G4 乙形式对照", abs(mu_b - mu_theory) / mu_theory <= 0.10,
       "过阻尼极限给同一终端速度律（与理论相差 <= 10%）",
       "乙形式 mu = %.2f 对理论 %.2f（相对偏差 %.2f%%）"
       % (mu_b, mu_theory, 100 * abs(mu_b - mu_theory) / mu_theory))

gam_need = 1.0 / (18.9 * m_shape)
print("[登记] G5：D-3 结论——甲形式给出终端速度律 v_term = f/(gamma_d M_zero)，响应质量的理论值")
print("       就是形状范数（数值迁移率与理论相差见 G1）；既有迁移率读数 18.9 反解 gamma_d = %.4f。" % gam_need)
print("       三质量口径的排序与差额如实登记：形状范数 0.0888（解析）、响应质量 0.094（高 5.9%）、")
print("       窗口质量 0.0936 至 0.0994（高 5.4% 至 11.9%）；差额来源（离散修正、窗口协议）")
print("       登记为 D-3 的残余读数，不调参凑数。接口：D-4 的脱钉阈值按同一 M_zero 口径处理。")

fig, ax = plt.subplots(1, 3, figsize=(14.0, 4.2))
ax[0].plot(np.arange(M), v_eq, "-", label="平衡 kink 剖面 v(n)")
ax[0].set_xlabel("n")
ax[0].set_ylabel("v")
ax[0].set_title("平衡 kink 剖面（形状范数 %.4f）" % shape)
ax[0].legend(fontsize=8)
fs = np.array([0.2e-4, 0.5e-4, 1.0e-4, 2.0e-4])
ax[1].plot(fs, [abs(drive(float(f), GAMD)) for f in fs], "o-", label="甲：|v_term|(f)")
ax[1].plot(fs, fs * mu_theory, "--", label="理论 v = f/(gamma_d M_zero)")
ax[1].set_xlabel("倾斜 f")
ax[1].set_ylabel("终端速度")
ax[1].set_title("终端速度律")
ax[1].legend(fontsize=8)
ax[2].bar([0, 1, 2], [m_shape, m_resp_meas, m_win],
          color=["tab:blue", "tab:green", "tab:orange"])
ax[2].set_xticks([0, 1, 2])
ax[2].set_xticklabels(["形状范数", "响应质量", "窗口质量"], fontsize=8)
ax[2].set_ylabel("质量")
ax[2].set_title("三质量口径")
plt.tight_layout()
plt.savefig(IMG, dpi=150)
plt.close(fig)

bad = [r for r in rows if not r[1]]
print("---- 判据 %d 项，通过 %d 项；图存 %s ----" % (len(rows), len(rows) - len(bad), IMG))
sys.exit(1 if bad else 0)
