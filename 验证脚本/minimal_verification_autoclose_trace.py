# -*- coding: utf-8 -*-
"""BH3-5 留痕判据一维先导：自动闭合几何与结构留痕。

判据（先立判据，再实现；均可证伪）：
  J1a 坐标差收缩：退出点两侧邻居的坐标差由 2 变 1。
  J1b 格距与计数：全链相邻坐标差恒为 1；格点总数减一；总跨度减一；
       重标中任何格点坐标变动不超过一个格距。
  J2a 结构留痕非零：源图案固定、只变格点几何（链 101 与 100）时，
       记录场最大相对差 > 1e-4。
  J2b 项可分离：结构项与质量项之比 < 1（结构留痕不主导）。
  J3 线性分解：Δ_true 与 Δ_struct + Δ_mass 的残差 < 0.1。

模型：一维链、格距 1、离散泊松 (Φ_{i+1} − 2Φ_i + Φ_{i−1}) = −s_i、Dirichlet 0 边界；
源：固定图案（10 个质量单元、位置 46..55、各 s = 1）。单位化：格距、源强、4πG 均为 1。
确定性（无随机数）。图存 D:/software/TianXingFC/images/autoclose_trace.png。
"""

import sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))


def solve_poisson(n, src):
    A = np.zeros((n, n))
    b = -np.asarray(src, dtype=float)
    for i in range(n):
        A[i, i] = -2.0
        if i > 0:
            A[i, i - 1] = 1.0
        if i < n - 1:
            A[i, i + 1] = 1.0
    A[0, :] = 0.0
    A[0, 0] = 1.0
    b[0] = 0.0
    A[-1, :] = 0.0
    A[-1, -1] = 1.0
    b[-1] = 0.0
    return np.linalg.solve(A, b)


print("==== BH3-5 留痕判据一维先导 ====")

# ---------- J1 自动闭合几何 ----------
N0 = 101
i_A = 50
pos_before = {k: k for k in range(N0)}
gap_before = pos_before[51] - pos_before[49]
alive = [k for k in range(N0) if k != i_A]
pos_after = {k: (k if k < i_A else k - 1) for k in alive}
gap_after = pos_after[51] - pos_after[49]
moves = [abs(pos_after[k] - k) for k in alive]
span_after = max(pos_after.values()) - min(pos_after.values())
check("J1a", gap_before == 2 and gap_after == 1,
      "邻居坐标差 %d -> %d（退出点 50）" % (gap_before, gap_after))
check("J1b", len(alive) == N0 - 1 and span_after == (N0 - 1) - 1 and max(moves) <= 1,
      "格点总数 %d -> %d；跨度 %d -> %d；最大坐标变动 %d 格距"
      % (N0, len(alive), N0 - 1, span_after, max(moves)))

# ---------- J2/J3 场计算 ----------
def src_profile(n, keep):
    s = np.zeros(n)
    for i in keep:
        if 0 <= i < n:
            s[i] = 1.0
    return s


base_src = list(range(46, 56))
drop = 50

phi0 = solve_poisson(101, src_profile(101, base_src))
phi_struct = solve_poisson(100, src_profile(100, base_src))
phi_mass = solve_poisson(101, src_profile(101, [i for i in base_src if i != drop]))
phi_true = solve_poisson(100, src_profile(100, [i for i in base_src if i != drop]))

m = min(101, 100)
d_struct = phi_struct[:m] - phi0[:m]
d_mass = phi_mass[:m] - phi0[:m]
d_true = phi_true[:m] - phi0[:m]

amp = float(np.max(np.abs(phi0[:m])))
r_struct = float(np.max(np.abs(d_struct)) / amp)
r_mass = float(np.max(np.abs(d_mass)) / amp)
ratio = r_struct / r_mass
resid = float(np.max(np.abs(d_true - (d_struct + d_mass))) / np.max(np.abs(d_true)))

check("J2a", r_struct > 1e-4, "结构留痕 max|Δ|/幅度 = %.3e" % r_struct)
check("J2b", 0.0 < ratio < 1.0, "结构/质量比 = %.4f" % ratio)
check("J3", resid < 0.1, "线性分解残差 = %.4f" % resid)
print("  幅度明细：基准 %.4f；结构 %.5f；质量 %.5f；联合 %.5f"
      % (amp, np.max(np.abs(d_struct)), np.max(np.abs(d_mass)), np.max(np.abs(d_true))))

# ---------- 图 ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
fig, axs = plt.subplots(1, 2, figsize=(11.2, 4.4))

ax = axs[0]
ax.plot(range(101), phi0, label="基准：101 格、全源")
ax.plot(range(100), phi_struct, "--", label="仅几何变化：100 格、全源")
ax.plot(range(100), phi_true, "-.", label="联合：100 格、去中心源")
ax.set_xlabel("格点坐标（格距为单位）")
ax.set_ylabel("记录场 Φ（单位化）")
ax.set_title("记录场分布对照", fontsize=9)
ax.legend(fontsize=8)

ax = axs[1]
ax.plot(range(m), d_struct, label="结构项 Δ_struct")
ax.plot(range(m), d_mass, label="质量项 Δ_mass")
ax.plot(range(m), d_true, "--", label="联合 Δ_true")
ax.set_xlabel("格点坐标（格距为单位）")
ax.set_ylabel("差值")
ax.set_title("留痕分解", fontsize=9)
ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig("D:/software/TianXingFC/images/autoclose_trace.png", dpi=150)
print("图已保存：images/autoclose_trace.png")

n_fail = sum(1 for _, ok in CHECKS if not ok)
print("")
print("==== 判据汇总：%d 项，%d PASS，%d FAIL ====" % (len(CHECKS), len(CHECKS) - n_fail, n_fail))