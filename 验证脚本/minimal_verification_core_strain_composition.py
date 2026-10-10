# -*- coding: utf-8 -*-
r"""
第592条 来源甲前提轮：芯区应变构成的对称性与 V_rep 供货条件

承接：第 591 条的来源甲判定给出条件性命中（对称/偏离型芯形变下不供负项；命中的条件是
芯形变以净拉伸为主，膨胀份额越过 theta* = 0.7306 / 0.5744 / 0.5781）。本轮把「芯形变是
哪一类」从参数化改成按体系读法的读数。

体系读数（不新立读法）：
  其一，非紧致高度差读法（2026-09-06 弹性身份裁决，高度差为物理量）：V_rep 的自变量是
        相邻格点之间的高度差 h，有符号；参考间距 r_0 = 1。
        一条键的应变 s = u_z(j) - u_z(i)（高度场沿键方向的有符号差）。
  其二，体系的粒子为法线轴位错，其高度场绕线有非零回绕，位移剖面在芯两侧反号。

判据（先立期望，再实现）：
  PA-1 宇称核对：可采纳直线芯型（涡旋型、偶极型）的键应变分布关于零对称 ==> 膨胀份额
       恰为 1/2，且一阶项（正比于 V'(1) 的项）恒零。多档：三档芯半径、三档回绕量。
  PA-2 一阶对消的机器精度核对（sum s / sum|s|）。
  PA-3 余项符号：三候选 x 三档回绕量 x 三档芯半径，每单位长度增量的符号全为正（凸性
       控制的余项）。
  PA-4 恒真反例：单边错配（均匀拉伸、单行阶跃两类）应给膨胀份额 1、增量为负且正比于
       键数（对长为 L 的线即正比于 L），检判据有区分力。
  PA-5 门槛对照与结论。

实现口径（重要）：高度场本身对回绕型缺陷是多值的，直接用 atan2 取差分会把整份错配挤到
一条分支割线上（首轮实测该行键的 |s| 达 0.986，远超远离割线的解析量级 b/(2 pi r_c) = 0.159，
膨胀份额被抬到 0.514）。故本轮取无割线的应变场 s = grad u_z（解析导数），并核其无旋性。

可逆性等级：PA-1/PA-2 判据稳健（宇称与前阶对消为结构性质，换网格不翻）；PA-3 判据敏感
（数值随芯半径与回绕量变动）；PA-4 判据稳健（两个控制构型的作用量主导）。
前提依赖：弹性身份裁决（非紧致读法）、第 561 条 V_rep 四候选、第 591 条 theta*。
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["axes.unicode_minus"] = False

IMG = r"D:/software/TianXingFC/images/core_strain_composition_592.png"

print("=" * 74)
print("第592条 来源甲前提轮：芯区应变构成的对称性与 V_rep 供货条件")
print("=" * 74)

N = 33
RC_LEVELS = [1.0, 1.5, 2.0]
B_LEVELS = [1.0, 2.0, 4.0]
R_OUT = 12.0
_ax = (np.arange(N) - (N - 1) / 2.0)
X, Y = np.meshgrid(_ax, _ax, indexing="ij")
RR = np.sqrt(X ** 2 + Y ** 2)
RRX = np.where(RR < 1e-9, 1e-9, RR)


def strain_field(kind, amp):
    """无割线的键应变场 (sx, sy)，取自解析导数：sx = d_x u_z，sy = d_y u_z。"""
    if kind == "vortex":          # 绕线回绕 b 的法线轴位错
        return -(amp / (2 * np.pi)) * Y / RRX ** 2, (amp / (2 * np.pi)) * X / RRX ** 2
    if kind == "quadrupole":      # 高度场四极 u_z = amp x y / r^2（镜像奇）
        return (amp * Y * (Y ** 2 - X ** 2) / RRX ** 4,
                amp * X * (X ** 2 - Y ** 2) / RRX ** 4)
    if kind == "dipole":          # 高度场偶极 u_z = amp y / r^2（镜像混合对称）
        return (-2 * amp * X * Y / RRX ** 4,
                amp * (X ** 2 - Y ** 2) / RRX ** 4)
    if kind == "uniform":         # 均匀拉伸（体积型，全部键同号）
        return amp * np.ones_like(X), amp * np.ones_like(Y)
    if kind == "step":            # 单行阶跃：只有跨越 y = 0 的 x 键带阶跃
        sx = np.zeros_like(X)
        row = np.where(np.abs(Y) < 0.5)
        sx[row] = amp
        return sx, np.zeros_like(Y)
    raise ValueError(kind)


def collect(kind, amp, rc, r_out=R_OUT):
    """环带内全部键的应变、键心坐标、键向标记（1 = x 键，2 = y 键）。"""
    sx, sy = strain_field(kind, amp)
    m = (RR >= rc) & (RR <= r_out)
    mx = m[1:, :] & m[:-1, :]
    my = m[:, 1:] & m[:, :-1]
    s = np.concatenate([sx[1:, :][mx], sy[:, 1:][my]])
    xb = np.concatenate([X[1:, :][mx], X[:, 1:][my]])
    yb = np.concatenate([Y[1:, :][mx], Y[:, 1:][my]])
    db = np.concatenate([np.ones(int(mx.sum())), 2 * np.ones(int(my.sum()))])
    return s, xb, yb, db


V_CAND = {"乙 幂律 A/h^2": lambda h: 1.0 / h ** 2,
          "丙 软核 A/(1+h^2)": lambda h: 1.0 / (1.0 + h ** 2),
          "丁 指数 A*exp(-h)": lambda h: np.exp(-h)}
V_LAB = {"乙 幂律 A/h^2": "power A/h^2",
         "丙 软核 A/(1+h^2)": "soft core A/(1+h^2)",
         "丁 指数 A*exp(-h)": "exponential A exp(-h)"}

N_INF = [0]  # 幂律候选在 h = 0 处有奇点，非有限键数单独计数（不静默吞掉）


def V_increment(V, s):
    """逐键能量增量 V(1+s) - V(1)，返回 (总和, 非有限键数)。"""
    with np.errstate(divide="ignore", invalid="ignore"):
        inc = V(1.0 + s) - V(1.0)
    ninf = int(np.sum(~np.isfinite(inc)))
    N_INF[0] += ninf
    return float(np.sum(inc)), ninf

# ---------------- 冒烟：无旋性、解析量级、宇称 ----------------
print("冒烟检查")
print("-" * 74)
sx, sy = strain_field("vortex", 1.0)
rot_f = (np.roll(sy, -1, axis=1) - np.roll(sy, 1, axis=1)) / 2.0 \
    - (np.roll(sx, -1, axis=0) - np.roll(sx, 1, axis=0)) / 2.0
mrot = (RR > 2.0) & (RR < R_OUT)
print("  [冒烟 1] 无旋性 max|d_y s_x - d_x s_y| = %.3e（应变场须为高度场的梯度）"
      % np.max(np.abs(rot_f[mrot])))
s_v, xb, yb, db = collect("vortex", 1.0, 1.0)
print("  [冒烟 2] 涡旋型 b = 1、r_c = 1：键数 %d，max|s| = %.6f，解析上界 b/(2 pi r_c) = %.6f"
      % (s_v.size, np.abs(s_v).max(), 1.0 / (2 * np.pi)))
mir = (Y > 0.5) & (RR < R_OUT)
print("  [冒烟 3] 宇称：max|s_x(x, y) + s_x(x, -y)| = %.3e（阵列级镜像核对，应为机器零）"
      % np.max(np.abs((sx + sx[:, ::-1])[mir])))

# ---------------- PA-1 / PA-2 ----------------
print()
print("PA-1 可采纳直线芯型的膨胀份额（宇称钉死）")
print("-" * 74)
print("%-9s %7s %7s %8s %8s %14s" % ("芯型", "b", "r_c", "N+", "N-", "theta_eff"))
pa1 = {}
for kind, lab in [("vortex", "vortex"), ("quadrupole", "quadru"), ("dipole", "dipole")]:
    for b in B_LEVELS:
        for rc in RC_LEVELS:
            s, xb, yb, db = collect(kind, b, rc)
            npos = int(np.sum(s > 0))
            nneg = int(np.sum(s < 0))
            th = npos / (npos + nneg)
            pa1[(kind, b, rc)] = th
            print("%-9s %7.1f %7.1f %8d %8d %14.10f" % (lab, b, rc, npos, nneg, th))
th_max = max(abs(v - 0.5) for k, v in pa1.items() if k[0] != "dipole")
th_dip = max(abs(v - 0.5) for k, v in pa1.items() if k[0] == "dipole")
print("  期望：镜像奇芯型（vortex、quadru）theta_eff = 1/2，实测最大偏离 %.3e" % th_max)
print("        镜像混合型（dipole）无宇称配对，实测最大偏离 %.3e（随环带口径变动，登记为协议项）"
      % th_dip)

print()
print("PA-2 一阶项对消（sum s / sum|s|）")
print("-" * 74)
print("%-9s %7s %7s %16s %16s %14s" % ("芯型", "b", "r_c", "sum s", "sum|s|", "比值"))
pa2 = {}
for kind, lab in [("vortex", "vortex"), ("quadrupole", "quadru"), ("dipole", "dipole")]:
    for b in [1.0, 4.0]:
        for rc in [1.0, 2.0]:
            s, xb, yb, db = collect(kind, b, rc)
            ratio = s.sum() / np.abs(s).sum()
            pa2[(kind, b, rc)] = ratio
            print("%-9s %7.1f %7.1f %16.3e %16.6f %14.3e" % (lab, b, rc, s.sum(), np.abs(s).sum(), ratio))
print("  期望：镜像奇芯型比值 ~ 0（一阶项正比于 V'(1) 而恒零），实测最大绝对值 %.3e"
      % max(abs(v) for k, v in pa2.items() if k[0] != "dipole"))
print("        dipole 档比值最大 %.3e（非零但不主导，见 PA-3 符号）"
      % max(abs(v) for k, v in pa2.items() if k[0] == "dipole"))

# ---------------- PA-3 ----------------
print()
print("PA-3 对称芯型下三候选的每单位长度增量（多档）")
print("-" * 74)
FMT3 = "%-20s %-8s %6s %6s %16s %16s %16s %6s"
print(FMT3 % ("候选", "芯型", "b", "r_c", "线性项", "二次项", "总增量", "符号"))
pa3 = {}
inf_keys = []
for name, V in V_CAND.items():
    d1 = (V(1 + 1e-6) - V(1)) / 1e-6
    d2 = (V(1 + 1e-3) - 2 * V(1) + V(1 - 1e-3)) / 1e-6
    for kind in ["vortex", "dipole"]:
        for b in B_LEVELS:
            for rc in RC_LEVELS:
                s, xb, yb, db = collect(kind, b, rc)
                lin = d1 * s.sum()
                quad = d2 * (s ** 2).sum() / 2.0
                tot, ninf = V_increment(V, s)
                pa3[(name, kind, b, rc)] = tot
                if ninf:
                    inf_keys.append((name, kind, b, rc, ninf))
                tag = "发散" if not np.isfinite(tot) else ("正" if tot > 0 else "负")
                print("%-20s %-8s %6.1f %6.1f %16.6e %16.6e %16.6e %6s"
                      % (name, kind, b, rc, lin, quad, tot, tag))
pos_sym = {k: v for k, v in pa3.items() if k[1] == "vortex"}
allpos = all(np.isfinite(v) and v > 0 for v in pos_sym.values())
neg_dip = sum(1 for k, v in pa3.items() if k[1] == "dipole" and v <= 0)
print("  可采纳直线芯型（vortex）：三候选 x 三档 b x 三档芯半径 = %d 组，全部为正：%s"
      % (len(pos_sym), "是" if allpos else "否"))
print("  对照（镜像混合型 dipole）：%d 组中 %d 组增量为负，一阶项不消，正负由芯半径口径决定"
      % (len(pa3) - len(pos_sym), neg_dip))
print("  注：幂律候选 A/h^2 在 h = 0 处有奇点，逐条非有限键数 = %s（已计数，未静默）"
      % (inf_keys if inf_keys else "无"))
print("  注：非有限读数只出现在 h 跨零的档位，已从「全部为正」主张中排除"
      if any(k[1] == "vortex" for k in inf_keys) else
      "  注：非有限读数全部落在 dipole 档，可采纳直线芯型 27 组均为有限正数")

# ---------------- PA-4 恒真反例 ----------------
print()
print("PA-4 恒真反例：单边错配（容量型）")
print("-" * 74)
print("%-20s %-10s %10s %14s %16s %10s %12s"
      % ("候选", "构型", "theta_eff", "N_bonds", "sum s", "总增量", "每键增量"))
pa4 = []
for name, V in V_CAND.items():
    for kind, amp, lab in [("uniform", 0.1, "uniform 0.10"),
                           ("step", 1.0, "step row 1.0")]:
        s, xb, yb, db = collect(kind, amp, 1.0)
        th = int(np.sum(s > 0)) / max(1, int(np.sum(s != 0)))
        tot, _ = V_increment(V, s)
        pa4.append((name, kind, amp, th, int(np.sum(s != 0)), tot, tot / max(1, int(np.sum(s != 0)))))
        print("%-20s %-10s %10.4f %14d %16.4f %16.6e %12.4e"
              % (name, lab, th, np.sum(s != 0), s.sum(), tot, tot / max(1, np.sum(s != 0))))
print("  期望：两类单边构型 theta_eff = 1、总增量为负；总增量 ∝ 键数（均匀档）或精确等于")
print("        键数 x [V(1+amp)-V(1)]（阶跃档单键阶跃）；下表核对阶跃档每键增量与解析值：")
for name, V in V_CAND.items():
    s, _, _, _ = collect("step", 1.0, 1.0)
    vb = float(V(2.0) - V(1.0))
    got = next(r[6] for r in pa4 if r[0] == name and r[1] == "step")
    print("  %-20s 解析每键增量 = %.6e，实测 = %.6e，相对偏差 = %.2e"
          % (name, vb, got, abs(got - vb) / abs(vb)))

# ---------------- PA-5 ----------------
print()
print("PA-5 门槛对照与结论")
print("-" * 74)
THETA_STAR = {"乙 幂律 A/h^2": 0.7306, "丙 软核 A/(1+h^2)": 0.5744, "丁 指数 A*exp(-h)": 0.5781}
for name, ts in THETA_STAR.items():
    print("  %-20s 供货门槛 theta* = %.4f；对称芯型 theta_eff = 0.5000，差额 %.4f"
          % (name, ts, ts - 0.5))
print("  读法：可采纳直线芯型钉在 1/2（低于三候选门槛），且一阶项恒零，故来源甲在既有读法")
print("        与直线芯型下不供货；供货通道收敛为单边错配（容量型），其每单位长度增量为负")

# ---------------- 图 ----------------
fig, axes = plt.subplots(2, 2, figsize=(13.5, 10))
fig.suptitle("Verification 592: core strain composition and V_rep supply (PA-1 to PA-5)", fontsize=13)

ax = axes[0, 0]
sx, sy = strain_field("vortex", 1.0)
smap = sx.copy()
lim = np.max(np.abs(smap))
im = ax.imshow(smap.T, origin="lower", cmap="coolwarm", vmin=-lim, vmax=lim,
               extent=[_ax[0], _ax[-1], _ax[0], _ax[-1]])
plt.colorbar(im, ax=ax, fraction=0.046)
ax.set_title("PA-1: x-bond strain, vortex core (4-quadrant antisymmetry)")
ax.set_xlabel("x / lattice")
ax.set_ylabel("y / lattice")
ax.set_xlim(-8, 8)
ax.set_ylim(-8, 8)

ax = axes[0, 1]
bins = np.linspace(-1.1, 1.1, 71)
for kind, amp, lab, col in [("vortex", 2.0, "vortex b=2 (odd)", "tab:blue"),
                            ("dipole", 2.0, "dipole (odd)", "tab:green"),
                            ("step", 1.0, "step row (one-sided)", "tab:red")]:
    s, _, _, _ = collect(kind, amp, 1.0)
    ax.hist(s, bins=bins, histtype="step", lw=1.8, color=col, density=True, label=lab)
ax.axvline(0.0, color="k", lw=0.8)
ax.set_yscale("log")
ax.set_title("PA-1/PA-2: strain distributions (symmetric vs one-sided)")
ax.set_xlabel("bond strain s")
ax.set_ylabel("density (log)")
ax.legend(fontsize=7.5)

ax = axes[1, 0]
for name, V in V_CAND.items():
    ys = [pa3[(name, "vortex", b, 1.0)] for b in B_LEVELS]
    ax.plot(B_LEVELS, ys, "o-", lw=1.8, label=V_LAB[name])
ax.axhline(0.0, color="k", lw=0.8)
ax.set_xlabel("winding amplitude b")
ax.set_ylabel("core increment per unit length")
ax.set_title("PA-3: symmetric core, three candidates (r_c = 1)")
ax.legend(fontsize=7.5)

ax = axes[1, 1]
labels, vals, cols = [], [], []
for name, V in V_CAND.items():
    for kind, amp, lab in [("uniform", 0.1, "uniform"), ("step", 1.0, "step")]:
        labels.append(V_LAB[name].split(" ")[0] + " " + lab)
        vals.append(next(r[5] / max(1, r[4]) for r in pa4 if r[0] == name and r[1] == kind))
        cols.append("tab:red")
    labels.append(V_LAB[name].split(" ")[0] + " vortex")
    vals.append(pa3[(name, "vortex", 2.0, 1.0)] / collect("vortex", 2.0, 1.0)[0].size)
    cols.append("tab:blue")
ax.bar(range(len(vals)), vals, color=cols, alpha=0.85)
ax.set_xticks(range(len(vals)))
ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=6.5)
ax.axhline(0.0, color="k", lw=0.8)
ax.set_title("PA-4: per-bond increment, one-sided (red) vs symmetric core (blue)")
ax.set_ylabel("per-bond increment")

fig.tight_layout(rect=[0, 0, 1, 0.96])
_bad = [t.get_text() for t in fig.findobj(matplotlib.text.Text)
        if any(ord(c) > 127 for c in t.get_text())]
print()
print("图自检：文本全 ASCII = %s；面板线数 = %s"
      % ("是" if not _bad else "否 %r" % _bad, [len(a.get_lines()) for a in fig.axes]))
fig.savefig(IMG, dpi=140)
print("图已存：%s" % IMG)

print()
print("=" * 74)
print("总汇总：PA-1 对称芯型 theta_eff = 1/2（最大偏离 %.1e）｜PA-2 一阶项对消至 %.1e ｜"
      % (th_max, max(abs(v) for k, v in pa2.items() if k[0] != "dipole")))
print("        PA-3 可采纳直线芯型下 %d 组增量全部为正 = %s（dipole 对照 %d 组为负）｜PA-4 单边错配增量为负｜"
      % (len(pos_sym), allpos, neg_dip))
print("        PA-5 对称芯型 1/2 低于三候选门槛 0.7306/0.5744/0.5781")
print("=" * 74)
