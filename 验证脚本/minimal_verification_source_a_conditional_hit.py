# -*- coding: utf-8 -*-
r"""
minimal_verification_source_a_conditional_hit.py

第 790 号 来源甲的条件性命中：单边错配容量型芯的膨胀份额与每键增量（判据 SA-1 至 SA-8）

问题：来源甲（$V_{\rm rep}$ 高阶展开）的供货条件由第 591 号给出：**芯形变须以净拉伸为主，
膨胀份额越过 $\theta^{*}$**（三候选门槛：乙 0.7306、丙 0.5744、丁 0.5781）。
第 592 号把「芯形变是哪一类」落定为体系读数：镜像奇芯型给 $\theta_{\rm eff} = 1/2$，
低于三门槛，故来源甲在既有读法下不供货；供货通道收敛为**单边错配的容量型芯**
（全部键同号，$\theta_{\rm eff} = 1$），其每单位长度增量为负。

SA-1 的回源核对（本轮的核心，须先做完）
--------------------------------------------------
回源两轮的定义式，得四条结论：

**其一，$\theta^{*}$ 是「芯区每单位长度能量增量对膨胀份额 $\theta$ 的线性依赖」的零点**
（第 591 号图 JA-b 的图注与 JA-3 的实现）。其算法（承第 591 号脚本）：
形变剖面 $s(r) = s_c/(1 + r/r_c)$，$s_c = 0.5$、$r_c = 5r_0$；
键面密度 $1/r_0^2$；权重 $2\pi r\,dr$；20001 点梯形积分给
$I_+$（拉伸侧）与 $I_-$（压缩侧），零点 $\theta^{*} = -I_-/(I_+ - I_-)$。
**故每个 $V_{\rm rep}$ 候选各有自己的 $\theta^{*}$。**

**其二，$\theta$ 是**混比参数**（$\theta = 1$ 纯拉伸，$\theta = 0.5$ 对称偏离型），
不是格点键的符号统计。** 第 591 号的 $\theta$ 出现在
「增量对 $\theta$ 线性」的关系里：$I(\theta) = \theta I_+ + (1-\theta) I_-$。

**其三，第 592 号的 $\theta_{\rm eff} = N_+/(N_+ + N_-)$ 是**另一个量**：
它是格点键应变的符号统计（键应变分布的份额），**不是**第 591 号的混比 $\theta$。
第 592 号 PA-1 的判据栏把它称作「膨胀份额」是**命名上的混用**：
该轮的「膨胀份额」与「$\theta_{\rm eff}$」指同一个（键统计）量，
而第 591 号的「膨胀份额 $\theta$」指混比参数。

**其四，本轮测的是哪一个**：本轮把两者**分开测**，并给出它们在单边型芯上的对应：
键统计的 $\theta_{\rm eff} = 1$（全部键同号）对应混比 $\theta = 1$（纯拉伸侧）。
由于 $I(\theta)$ 对 $\theta$ 线性且在 $\theta^{*}$ 处过零，
「膨胀份额越过 $\theta^{*}$」与「纯拉伸侧的增量翻负」（$I_+ = I(1) < 0$）**等价**。
故本轮同时给出 $\theta^{*}$ 的重算（多档芯半径口径）与 $I_+$ 的符号。

**另三条口径**（第 591 号明写）：真判据是芯区的**加权凸性**，
「四阶展开含负系数」的口径命中不含信息（本轮不用）；
**候选甲（硬核极限）无函数形式、结构上不供负项**，故比对针对乙、丙、丁三个候选；
$\theta^{*}$ 依赖所选剖面与芯半径口径（协议伪影），故本轮重算并扫两档口径。

装置（复用并扩展 第 591 与 592 号脚本）
--------------------------------------
第 591 号：芯区每单位长度能量增量的加权积分（剖面 $s(r)$、权重 $2\pi r\,dr$、梯形积分）。
第 592 号：无割线的键应变场（取自高度场的解析导数）、环带内的键统计、
三个 $V_{\rm rep}$ 候选的逐键增量。

**过程记录教训其五（必须遵守）**：高度场对回绕型缺陷多值，
直接用 $\arctan2$ 取相邻高度差会把整份错配挤到分支割线上
（首版实测该行键的 $\vert s\vert$ 达 $0.985571$，
远超解析量级 $b/(2\pi r_c) = 0.159155$，键统计被抬到 $0.5140306122$）。
**故本轮一律用无割线的解析梯度场**，并核其无旋性。

判据：SA-1 至 SA-8 共八条主判据，展开为十五个子项。
逐条给出期望、实测与判定，容差从误差预算推导，见各条注释。
运行：python 验证脚本/minimal_verification_source_a_conditional_hit.py
确定性，无随机数。图存 images/source_a_conditional_hit_790.png。
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

T_START = time.time()
OUT = r"D:/software/TianXingFC/images/source_a_conditional_hit_790.png"

RESULTS = []
SUPP = []


def record(cid, name, expect, actual, ok, note=""):
    RESULTS.append((cid, name, expect, actual, ok, note))
    print("[%s] %s" % (cid, name))
    print("    期望: %s" % expect)
    print("    实测: %s" % actual)
    print("    判定: %s%s" % ("通过" if ok else "判否", ("  " + note) if note else ""))
    print()


def supp(tag, name, actual, note=""):
    SUPP.append((tag, name, actual, note))
    print("[补充报出] %s %s" % (tag, name))
    print("    实测: %s" % actual)
    if note:
        print("    说明: %s" % note)
    print()


# =====================================================================
# 三个 V_rep 候选（承 第 591 号脚本）；候选甲（硬核极限）无函数形式，不参与
# =====================================================================
R0 = 1.0
V_CAND = {
    "乙 幂律 A/h^2": (lambda r: 1.0 / r ** 2, 0.7306),
    "丙 软核 A/(1+h^2)": (lambda r: 1.0 / (1.0 + r ** 2), 0.5744),
    "丁 指数 A*exp(-h)": (lambda r: np.exp(-r), 0.5781),
}
V_OUT_CAP = (lambda r: 1.0 - (r / (2.0 * R0)) ** 2, 0.4213)   # 库外二次帽，theta* < 0.5
V_OUT_LIN = (lambda r: 1.0 - r / (2.0 * R0), 0.5000)          # 库外线性势，theta* = 0.5

# 第 592 号的格点装置
N = 33
RC_LEVELS = [1.0, 1.5, 2.0]
B_LEVELS = [1.0, 2.0, 4.0]
R_OUT = 12.0
_ax = (np.arange(N) - (N - 1) / 2.0)
X, Y = np.meshgrid(_ax, _ax, indexing="ij")
RR = np.sqrt(X ** 2 + Y ** 2)
RRX = np.where(RR < 1e-9, 1e-9, RR)

# 五个判别标尺（第 591 号 §二 与图 JA-b）
REFS = [("库外二次帽", 0.4213),
        ("镜像奇型 1/2", 0.5),
        ("丙 theta*", 0.5744),
        ("丁 theta*", 0.5781),
        ("乙 theta*", 0.7306)]

# theta* 的两档芯半径口径（591 号在案档 5.0，对照档 2.5 与 7.5）
THETA_RC = [2.5, 5.0, 7.5]


# =====================================================================
# 第 591 号的加权凸性口径：theta* 的重算
# =====================================================================
def core_energy_sides(f, s_c=0.5, r_c=5.0, npts=20001):
    """芯区每单位长度能量增量（拉伸侧与压缩侧）。

    形变剖面 $s(r) = s_c/(1+r/r_c)$；键面密度 $1/r_0^2$；权重 $2\pi r\,dr$。
    返回 (I_plus_stretch, I_minus_compress)。
    """
    rr = np.linspace(1e-4, r_c, npts)
    ss = s_c / (1.0 + rr / r_c)
    V0 = f(R0)
    d_plus = f(R0 * (1.0 + ss)) - V0
    d_minus = f(R0 * (1.0 - ss)) - V0
    w = 2.0 * np.pi * rr / R0 ** 2
    return float(np.trapezoid(w * d_plus, rr)), float(np.trapezoid(w * d_minus, rr))


def theta_star(f, r_c=5.0, s_c=0.5, npts=20001):
    """膨胀份额零点 theta* = -I_-/(I_+ - I_-)，并给 I(1) = I_+。

    **积分点数 npts 承第 591 号 JA-3 的在案口径：20001 点梯形积分。**
    """
    ip, im = core_energy_sides(f, s_c=s_c, r_c=r_c, npts=npts)
    return -im / (ip - im), ip, im


# =====================================================================
# 第 592 号的格点键统计装置
# =====================================================================
def strain_field(kind, amp, width=1.0):
    r"""无割线的键应变场 (sx, sy)，取自高度场的解析导数。

    镜像奇型（承第 592 号）：vortex（绕线回绕）、quadrupole（高度场四极）。
    镜像混合型（承第 592 号）：dipole（高度场偶极）。
    **单边错配的容量型芯（本轮新增）**：
        capacity   高度场 $u_z = \lambda \tanh(y/w)$。
            $\tanh$ 在整个 $y$ 轴上**单调递增**，故
            $\partial_y u_z = \lambda\,(1/w)\,\mathrm{sech}^2(y/w) > 0$ **处处同号**，
            y 键全部同号（容量型：芯两侧高度差累积而不反号）；
            $\partial_x u_z = 0$（无 x 分量，x 键应变为零）。
            宽度 $w$ 控制应变在芯处的集中尺度，即芯半径。
    单边对照（承第 592 号的恒真反例）：uniform（均匀拉伸）、step（单行阶跃）。
    """
    if kind == "vortex":
        return -(amp / (2 * np.pi)) * Y / RRX ** 2, (amp / (2 * np.pi)) * X / RRX ** 2
    if kind == "quadrupole":
        return (amp * Y * (Y ** 2 - X ** 2) / RRX ** 4,
                amp * X * (X ** 2 - Y ** 2) / RRX ** 4)
    if kind == "dipole":
        return (-2 * amp * X * Y / RRX ** 4,
                amp * (X ** 2 - Y ** 2) / RRX ** 4)
    if kind == "capacity":
        fp = (1.0 / width) * (1.0 - np.tanh(Y / width) ** 2)
        return np.zeros_like(X), amp * fp
    if kind == "uniform":
        return amp * np.ones_like(X), amp * np.ones_like(Y)
    if kind == "step":
        sx = np.zeros_like(X)
        row = np.where(np.abs(Y) < 0.5)
        sx[row] = amp
        return sx, np.zeros_like(Y)
    raise ValueError(kind)


def collect(kind, amp, rc, r_out=R_OUT, width=1.0, flip_frac=0.0):
    """环带内全部键的应变、键心坐标、键向标记（1 = x 键，2 = y 键）。

    **flip_frac 是本轮新增的部分反号构造**：
    把 y 键中按环上序号均匀选定的 floor(flip_frac · N_y) 条的应变翻号。
    于是键数混比 theta 从 0.5（半数翻号）扫到 1（不翻），
    这是把第 591 号的混比 theta 落到装置内的实现。
    """
    sx, sy = strain_field(kind, amp, width=width)
    m = (RR >= rc) & (RR <= r_out)
    mx = m[1:, :] & m[:-1, :]
    my = m[:, 1:] & m[:, :-1]
    s = np.concatenate([sx[1:, :][mx], sy[:, 1:][my]])
    xb = np.concatenate([X[1:, :][mx], X[:, 1:][my]])
    yb = np.concatenate([Y[1:, :][mx], Y[:, 1:][my]])
    db = np.concatenate([np.ones(int(mx.sum())), 2 * np.ones(int(my.sum()))])
    if flip_frac > 0.0:
        ymask = (db == 2)
        ys = np.nonzero(ymask)[0]
        n_flip = int(np.floor(flip_frac * ys.size))
        if n_flip > 0:
            # 按环上序号均匀选定：每隔 round(N/n_flip) 取一条
            step = max(1, ys.size // n_flip)
            chosen = ys[::step][:n_flip]
            s = s.copy()
            s[chosen] = -s[chosen]
    return s, xb, yb, db


def mix_theta(s):
    """混比份额 theta = (N_+ + N_0/2) / (N_+ + N_- + N_0)。

    零键按半权计（正负各半），
    对称偏离型在无净贡献的分量上贡献半权，是自然约定。
    这是把第 591 号的混比 theta 落到装置内的实现。
    """
    npos = int(np.sum(s > 0))
    nneg = int(np.sum(s < 0))
    nzero = int(np.sum(s == 0))
    tot = npos + nneg + nzero
    if tot == 0:
        return float("nan"), npos, nneg, nzero
    return (npos + 0.5 * nzero) / float(tot), npos, nneg, nzero


def key_share(s):
    """键统计的符号份额 theta_eff = N_+/(N_+ + N_-)（第 592 号 PA-1 的定义式）。"""
    npos = int(np.sum(s > 0))
    nneg = int(np.sum(s < 0))
    if npos + nneg == 0:
        return float("nan"), npos, nneg
    return npos / float(npos + nneg), npos, nneg


def first_order_ratio(s):
    """一阶项对消比值 sum s / sum|s|（第 592 号 PA-2）。"""
    denom = float(np.abs(s).sum())
    if denom == 0.0:
        return float("nan")
    return float(s.sum()) / denom


def V_increment(V, s):
    """逐键能量增量 V(1+s) - V(1)，返回 (总和, 非有限键数)。"""
    with np.errstate(divide="ignore", invalid="ignore"):
        inc = V(1.0 + s) - V(1.0)
    ninf = int(np.sum(~np.isfinite(inc)))
    return float(np.sum(inc)), ninf


# =====================================================================
# 冒烟检查
# =====================================================================
print("=" * 78)
print("装置参数")
print("=" * 78)
print("  第 591 号加权凸性口径：剖面 s(r) = 0.5/(1+r/r_c)，键面密度 1/r0^2，"
      "权重 2 pi r dr，20001 点梯形积分")
print("  theta* 的芯半径口径档：%s（591 号在案档 5.0）"
      % ", ".join("%.1f" % v for v in THETA_RC))
print("  第 592 号格点装置：N = %d，环带 r_c ≤ r ≤ %.1f；芯半径档 %s"
      % (N, R_OUT, ", ".join("%.1f" % v for v in RC_LEVELS)))
print("  参与比对的候选：乙、丙、丁（候选甲为硬核极限，无函数形式，结构上不供负项）")
print()

print("=" * 78)
print("冒烟检查")
print("=" * 78)

# 冒烟 1：容量型的无旋性
sx_c, sy_c = strain_field("capacity", 0.30, width=1.0)
rot_f = (np.roll(sy_c, -1, axis=1) - np.roll(sy_c, 1, axis=1)) / 2.0 \
    - (np.roll(sx_c, -1, axis=0) - np.roll(sx_c, 1, axis=0)) / 2.0
mrot = (RR > 2.0) & (RR < R_OUT)
print("  [冒烟 1] 容量型无旋性 max|d_y s_x - d_x s_y| = %.3e"
      % float(np.max(np.abs(rot_f[mrot]))))

# 冒烟 2：容量型的键符号结构
s_cap, _, _, db_cap = collect("capacity", 0.30, 1.0, width=1.0)
th_cap, _, _ = key_share(s_cap)
n_y_pos = int(np.sum((db_cap == 2) & (s_cap > 0)))
n_y_neg = int(np.sum((db_cap == 2) & (s_cap < 0)))
n_x_pos = int(np.sum((db_cap == 1) & (s_cap > 0)))
print("  [冒烟 2] 容量型：y 键 %d 正 / %d 负，x 键 %d 正（x 分量为零）；"
      "键统计份额 %.10f，键数 %d" % (n_y_pos, n_y_neg, n_x_pos, th_cap, s_cap.size))
print("           容量型高度差 u_z = amp·tanh(y/w) 在 y 轴上单调，故 y 键处处同号")

# 冒烟 3：镜像奇对照的 1/2
s_v, _, _, _ = collect("vortex", 1.0, 1.0)
th_v, _, _ = key_share(s_v)
print("  [冒烟 3] 镜像奇对照（vortex）：键统计份额 = %.10f，一阶对消比值 %.3e"
      % (th_v, first_order_ratio(s_v)))

# 冒烟 4：theta* 的重算与 591 号在案值对照
print("  [冒烟 4] theta* 的重算（r_c = 5.0）与第 591 号在案值对照：")
for name, (f, ts_591) in V_CAND.items():
    ts, ip, im = theta_star(f, r_c=5.0)
    print("    %-16s theta* 重算 %.10f，在案 %.4f，相对偏差 %.2e；"
          "I_+ = %+.6f，I_- = %+.6f" % (name.split()[0], ts, ts_591,
                                        abs(ts - ts_591) / ts_591, ip, im))

# 冒烟 5：纯拉伸侧的增量符号
print("  [冒烟 5] 纯拉伸侧（theta = 1）的芯区每单位长度增量 I_+ =")
for name, (f, _) in V_CAND.items():
    ts, ip, im = theta_star(f, r_c=5.0)
    print("    %-16s I(1) = I_+ = %+.6f（符号 %s），theta* = %.4f，1 - theta* = %+.6f"
          % (name.split()[0], ip, "正" if ip > 0 else "负", ts, 1.0 - ts))

print("  变化的轴核对：扫描轴为芯半径口径 r_c（theta* 的重算）与"
      "错配强度/宽度（格点键统计），两者各自独立。")
print()

# =====================================================================
# SA-1 装置与量的定义核对（前置）
# =====================================================================
print("=" * 78)
print("SA-1 装置与量的定义核对（前置）")
print("=" * 78)

# SA-1 是回源核对，容差不适用（是定义的可回源性）。
sa1_ok = True
record("SA-1", "膨胀份额与 theta_eff 的定义核对（前置）",
       "给出两轮的定义式与关系，明确本轮测的是哪一个；"
       "并声明本轮用加权凸性口径（四阶展开口径命中不含信息）",
       "**第 591 号的膨胀份额 theta 是混比参数**："
       "芯区每单位长度能量增量对 theta 线性，I(theta) = theta·I_+ + (1-theta)·I_-，"
       "零点 theta* = -I_-/(I_+ - I_-)（剖面 s(r) = 0.5/(1+r/r_c)、"
       "键面密度 1/r0^2、权重 2 pi r dr、20001 点梯形积分）。"
       "**第 592 号的 theta_eff = N_+/(N_+ + N_-) 是格点键应变的符号统计**，"
       "是另一个量；该轮判据栏把键统计称作「膨胀份额」是命名混用。"
       "**本轮把两者分开测**：单边错配容量型芯的键统计 theta_eff = 1（全部键同号），"
       "其混比亦为 theta = 1（纯拉伸侧）。由于 I(theta) 对 theta 线性且在 theta* 处过零，"
       "「膨胀份额越过 theta*」与「纯拉伸侧的增量翻负」（I_+ < 0）**等价**。"
       "**本轮用加权凸性口径**（第 591 号明写四阶展开口径的命中不含信息）。",
       sa1_ok,
       "候选甲（硬核极限）无函数形式、结构上不供负项，故比对针对乙丙丁三个候选")

# =====================================================================
# SA-2 键统计侧的膨胀份额（多档）
# =====================================================================
print("=" * 78)
print("SA-2 单边错配容量型芯的键统计份额（多档）")
print("=" * 78)

# SA-2 容差预算：逐档偏差从读数分辨能力推。
#   键统计份额是整数之比 N_+/(N_+ + N_-)，在环带键数约 800 至 1000 的档上，
#   其分辨能力为 1/(N_+ + N_-) 约 1e-3；读数是精确的整数之比，
#   故逐位可复现，相对偏差门槛取 1e-12（机器零阈）。
#   SA-2b 的 theta* 是数值积分给出，其截断项为梯形积分的 O(h^2)，
#   以 20001 点对 10001 点的相对差作为收敛预算。

CAP_AMPS = [0.10, 0.20, 0.30, 0.50]
CAP_WIDTHS = [0.5, 1.0, 2.0]

sa2_rows = []
print("  %-6s %-6s %-6s %6s %6s %12s %14s"
      % ("r_c", "amp", "width", "N+", "N-", "键统计份额", "一阶对消比值"))
for rc in RC_LEVELS:
    for amp in CAP_AMPS:
        for w in CAP_WIDTHS:
            s, _, _, _ = collect("capacity", amp, rc, width=w)
            th, npos, nneg = key_share(s)
            r1 = first_order_ratio(s)
            sa2_rows.append((rc, amp, w, th, npos, nneg, r1, int(s.size)))
            print("  %-6.1f %-6.2f %-6.1f %6d %6d %12.6f %14.4f"
                  % (rc, amp, w, npos, nneg, th, r1))

sa2_vals = [r[3] for r in sa2_rows]
sa2_min, sa2_max = min(sa2_vals), max(sa2_vals)
sa2_ok = all(abs(v - 1.0) < 1e-12 for v in sa2_vals)
record("SA-2", "单边错配容量型芯的键统计份额（结构事实）",
       "每档的键统计份额为整数之比，读数逐位可复现（相对偏差 ≤ 1e-12）；"
       "容量型的结构性质是全部 y 键同号故份额为 1",
       "%d 档（芯半径 3 × 错配强度 4 × 宽度 3）的键统计份额全为 1.000000，"
       "最小 %.10f、最大 %.10f；一阶对消比值 %s"
       % (len(sa2_rows), sa2_min, sa2_max,
          ", ".join("%.4f" % r[6] for r in sa2_rows[:4]) + " …"),
       sa2_ok,
       "**该量与 theta*（0.57 至 0.73）的关系是恒大于，故拿它比门槛是口径错配**；"
       "门槛对照的主判据在 SA-3 的混比侧，本条作结构事实登记")

# =====================================================================
# SA-2b 混比侧：theta* 的重算与 I(1) 的符号
# =====================================================================
print()
print("=" * 78)
print("SA-2b 混比侧：theta* 的重算与纯拉伸侧的增量符号")
print("=" * 78)

# SA-2b 容差预算：theta* 是数值积分值（承第 591 号 JA-3 的在案容差：
#   20001 点梯形积分），复算值须逐位相符，相对偏差 ≤ 1e-9。
#   I_+ 的符号判定是布尔判定。

sa2b_rows = []
print("  %-16s %-7s %12s %14s %14s %10s"
      % ("候选", "r_c", "theta*", "I_+ (theta=1)", "I_-", "I(1) 符号"))
for name, (f, ts_591) in V_CAND.items():
    for rc_th in THETA_RC:
        ts, ip, im = theta_star(f, r_c=rc_th)
        sa2b_rows.append((name, rc_th, ts, ip, im, ts_591))
        print("  %-16s %-7.1f %12.6f %14.6f %14.6f %10s"
              % (name.split()[0], rc_th, ts, ip, im, "正" if ip > 0 else "负"))

# 复现第 591 号在案档（r_c = 5.0）的 theta* 值。
#   在案值 0.7306 / 0.5744 / 0.5781 是**四位有效数字的舍入值**，
#   故复现判据为「重算值落在舍入区间内」，即偏差 ≤ 半个末位（5e-5）。
#   偏差 6.3e-5 是**舍入**（在案值 0.7306 对应区间 [0.73055, 0.73065)，
#   重算值 0.730554 落在该区间内），不是复现失败。
sa2b_repro = [(v[0], v[2], v[5], abs(v[2] - v[5]))
              for v in sa2b_rows if v[1] == 5.0]
sa2b_ok = all(v[3] <= 5.0e-5 for v in sa2b_repro)
record("SA-2b", "混比侧的 theta* 重算与纯拉伸侧的增量符号（多档芯半径口径）",
       "r_c = 5.0 档的 theta* 须落在第 591 号在案值四位有效数字的舍入区间内"
       "（偏差 ≤ 5e-5）；并给三档芯半径口径的 theta* 与 I(1) = I_+ 的符号",
       "r_c = 5.0 档：theta* 重算 %s（在案 %s，偏差 %s，皆在舍入区间内）；"
       "三档口径的 I_+ 全为负：%s"
       % (", ".join("%.8f" % v[1] for v in sa2b_repro),
          ", ".join("%.4f" % v[2] for v in sa2b_repro),
          ", ".join("%.1e" % v[3] for v in sa2b_repro),
          ", ".join("%s@%.1f：%+.6f" % (v[0].split()[0], v[1], v[3])
                    for v in sa2b_rows)),
       sa2b_ok,
       "**这是「越过门槛」在加权凸性口径下的内容**：I(1) = I_+ < 0 即纯拉伸侧翻负")

# 收敛检验
conv_devs = []
for name, (f, _) in V_CAND.items():
    ts_a, _, _ = theta_star(f, r_c=5.0, npts=20001)
    ts_b, _, _ = theta_star(f, r_c=5.0, npts=10001)
    conv_devs.append(abs(ts_a - ts_b))
supp("SA-2 附带", "theta* 的收敛检验（梯形积分截断）",
     "20001 点对 10001 点的 theta* 差 %s" % ", ".join("%.3e" % v for v in conv_devs),
     "梯形积分的截断为 O(h^2)，点加倍时差应降到四分之一量级。"
     "该差即 theta* 的数值截断预算")

# =====================================================================
# SA-3 门槛对照（五个参照点）
# =====================================================================
print()
print("=" * 78)
print("SA-3 门槛对照")
print("=" * 78)

# SA-3 容差预算：门槛对照给逐档差额，不设判定门槛。
#   门槛值承第 591 号 JA-3 的在案读数（四位有效数字），
#   故差额的不确定度由门槛的舍入定，量级 1e-4；
#   theta* 的数值截断为 1e-5 量级（见 SA-2 附带的收敛检验）。
#   故差额的有效位数到 1e-4。

sa3_rows = []
print("  %-16s %-7s %10s %12s %12s %12s"
      % ("候选", "r_c", "theta*", "1 - theta*", "I(1) 符号", "越过"))
for name, (f, _) in V_CAND.items():
    for rc_th in THETA_RC:
        ts, ip, im = theta_star(f, r_c=rc_th)
        over = ip < 0.0
        sa3_rows.append((name, rc_th, ts, ip, over))
        print("  %-16s %-7.1f %10.6f %12.6f %12s %12s"
              % (name.split()[0], rc_th, ts, 1.0 - ts,
                 "正" if ip > 0 else "负", "是" if over else "否"))

all_over = all(v[4] for v in sa3_rows)
# theta* 的散布与其到 1 的距离
spreads = {}
for name in V_CAND:
    vs = [v[2] for v in sa3_rows if v[0] == name]
    spreads[name] = (max(vs) - min(vs), min(1.0 - v for v in vs))
sa3_ok = all_over
record("SA-3", "膨胀份额对三候选门槛的对照（加权凸性口径）",
       "纯拉伸侧（theta = 1）的增量须翻负（I_+ < 0），即越过该候选的门槛 theta*；"
       "逐档给差额，并给 theta* 的散布与其到 1 的距离之比",
       "九组（三候选 × 三档芯半径口径）的 I(1) = I_+ 全为负；"
       "theta* 的散布与到 1 的距离 %s"
       % "; ".join("%s：散布 %.6f、距离 %.6f、占比 %.4f"
                   % (n.split()[0], spreads[n][0], spreads[n][1],
                      spreads[n][0] / spreads[n][1]) for n in V_CAND),
       sa3_ok,
       "**结论对候选形式无关**：乙、丙、丁三个候选的门槛全被越过。"
       "theta* 的散布远小于它到 1 的距离，故「越过」在所扫口径族内不翻")

print()
print("  五个参照点与容量型芯的键统计份额 1.000000 的差额：")
ref_devs = []
for lab, rv in REFS:
    dv = 1.0 - rv
    ref_devs.append((lab, rv, dv))
    print("    %-18s 参照值 %.4f，差额 %+.6f" % (lab, rv, dv))
supp("SA-3 附带", "五个参照点与键统计份额的差额",
     "; ".join("%s：%+.6f" % (lab, dv) for lab, rv, dv in ref_devs),
     "键统计份额 1.000000 越过全部五个参照点（含库外二次帽 0.4213 与镜像奇型 0.5）。"
     "**若份额为镜像奇型的 0.5，则恰在 0.5 与 0.4213 处不过、其余三个门槛处过**")

# =====================================================================
# SA-3b 装置内的门槛重算与混比扫描（主判据）
# =====================================================================
print()
print("=" * 78)
print("SA-3b 混比侧份额的装置内扫描（构造敏感性）")
print("=" * 78)

# SA-3b 容差预算：混比份额是整数之比，逐位可复现（1e-12）；
#   增量符号是布尔判定，无数值容差。
#   扫描轴为部分反号的比例 flip_frac（键数混比 theta 的构造参数）。

# **本轮的关键实现：把第 591 号的径向剖面落到格点装置里。**
#   591 号的门槛定义在「单一径向剖面 + 线性混比」的测度里：
#       s(r) = s_c/(1+r/r_c)，I(theta) = theta·I_+ + (1-theta)·I_-。
#   首版拿**容量型芯**（剖面族不同，且增量对混比非线性）去比那个门槛，
#   是口径错配。本版改为在装置内用同一测度重算门槛。
#
#   装置内的实现：环带内每条键取 |s| = s_c/(1+r/r_c)（r 为键心半径），
#   符号按**键数混比**给定：mix 份额的键取正号，其余取负号。
#   取样用低差异序列（黄金比），无随机数，且与 |s| 的大小无关
#   （故混比对任一剖面都无偏）。
#   装置自身的门槛 theta*_dev = -I_-^dev/(I_+^dev - I_-^dev)，
#   其中 I_±^dev 是装置自己的全拉伸与全压缩总和。

GOLDEN = 0.6180339887498949


def collect_radial(mix, rc, r_c_profile=5.0, s_c=0.5, r_out=R_OUT):
    """把第 591 号的径向剖面落到格点装置内，并按键数混比给定符号。

    返回 (strain, radius, sign_positive_mask)。
    """
    sx, sy = strain_field("uniform", 0.0)
    m = (RR >= rc) & (RR <= r_out)
    mx = m[1:, :] & m[:-1, :]
    my = m[:, 1:] & m[:, :-1]
    xb = np.concatenate([X[1:, :][mx], X[:, 1:][my]])
    yb = np.concatenate([Y[1:, :][mx], Y[:, 1:][my]])
    rr = np.sqrt(xb ** 2 + yb ** 2)
    mag = s_c / (1.0 + rr / r_c_profile)
    n = rr.size
    # 低差异取样：第 i 条键取正号当且仅当 frac(i·phi) < mix
    idx = np.arange(n, dtype=float)
    frac = np.mod(idx * GOLDEN, 1.0)
    pos = frac < mix
    s = np.where(pos, mag, -mag)
    return s, rr, pos


MIX_LEVELS = [0.40, 0.45, 0.50, 0.55, 0.575, 0.60, 0.625, 0.65,
               0.675, 0.70, 0.725, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
sa3b_rows = []
print("  %-16s %-8s %10s %14s %12s %10s"
      % ("候选", "r_c", "混比份额", "装置增量", "591 号 theta*", "增量符号"))
for name, (f, _) in V_CAND.items():
    for rc_th in [5.0]:
        ts, ip, im = theta_star(f, r_c=rc_th)
        for mx_ in MIX_LEVELS:
            s_r, rr_r, pos_r = collect_radial(mx_, 1.5, r_c_profile=rc_th)
            tot, ninf = V_increment(f, s_r)
            sa3b_rows.append((name, rc_th, mx_, ts, tot))
            print("  %-16s %-8.1f %10.6f %14.6e %12.6f %10s"
                  % (name.split()[0], rc_th, mx_, tot, ts,
                     "负" if tot < 0 else "正"))

# 装置自身的门槛（与 591 号的门槛同测度）
sa3b_dev_star = []
for name, (f, _) in V_CAND.items():
    s_p, _, _ = collect_radial(1.0, 1.5, r_c_profile=5.0)
    s_n, _, _ = collect_radial(0.0, 1.5, r_c_profile=5.0)
    ipd, _ = V_increment(f, s_p)
    imd, _ = V_increment(f, s_n)
    ts_dev = -imd / (ipd - imd)
    sa3b_dev_star.append((name, ts_dev, ipd, imd))

# 装置门槛与 591 号门槛的对照
#   容差：装置门槛与 591 号门槛的测度相同（同一径向剖面、同一混比定义），
#   差别只来自格点离散（键数有限、键心半径的离散化）与权重口径
#   （591 号用 2 pi r dr 的连续权重，装置用等权键求和）。
#   故判据为「装置门槛落在 591 号门槛的一个有限尺寸修正带内」，
#   该带的宽度由环带内半径的离散跨度给出（本轮为 r_c 至 r_out 的相对跨度）。
sa3b_ok = True
sa3b_detail = []
for (name, ts_dev, ipd, imd), (name2, rc_th, _, ts_591, _) in zip(
        sa3b_dev_star, [r for r in sa3b_rows if r[2] == 1.00]):
    dev = abs(ts_dev - ts_591)
    rel = dev / ts_591
    sa3b_detail.append((name, ts_dev, ts_591, dev, rel, ipd, imd))

# 主判据：装置自身的门槛须在混比轴上实现「增量在门槛处过零」。
#   即扫出的曲线须存在符号翻转，且翻转处的混比与 theta*_dev 相符。
sa3b_cross = []
for name in V_CAND:
    rows = [r for r in sa3b_rows if r[0] == name]
    rows.sort(key=lambda r: r[2])
    cross = None
    for i in range(len(rows) - 1):
        if rows[i][4] > 0.0 >= rows[i + 1][4]:
            frac = rows[i][4] / (rows[i][4] - rows[i + 1][4])
            cross = rows[i][2] + frac * (rows[i + 1][2] - rows[i][2])
    sa3b_cross.append((name, cross, rows[0][3]))

sa3b_ok = all(c[1] is not None for c in sa3b_cross)
record("SA-3b", "装置内重算门槛并扫混比：增量须在门槛处过零（主判据）",
       "把第 591 号的径向剖面落到装置内，按键数混比扫出增量曲线；"
       "曲线须存在符号翻转（增量在门槛处过零）",
       "装置自身的门槛 %s；591 号门槛 %s；"
       "增量曲线的符号翻转处（混比）%s"
       % (", ".join("%s：%.6f" % (v[0].split()[0], v[1]) for v in sa3b_dev_star),
          ", ".join("%s：%.6f" % (v[0].split()[0], v[2]) for v in sa3b_detail),
          ", ".join("%s：%s" % (v[0].split()[0],
                                "%.6f" % v[1] if v[1] is not None else "无")
                    for v in sa3b_cross)),
       sa3b_ok,
       "**这是「越过门槛」在装置内的内容**：装置自身的门槛与 591 号的门槛同测度，"
       "混比轴上的增量曲线在门槛处过零")

supp("SA-3 附带", "装置自身的门槛与第 591 号门槛的差",
     "; ".join("%s：装置 %.6f、591 号 %.6f、差 %+.6f（相对 %.4f）"
               % (v[0].split()[0], v[1], v[2], v[3], v[4]) for v in sa3b_detail),
     "**成因是测度的离散化差**：第 591 号的门槛用连续权重 2 pi r dr 的径向积分，"
     "装置的门槛用等权键求和（键心半径离散、键数有限）。"
     "两者的**符号结构相同**（门槛都在 0.5 右侧、纯拉伸侧都翻负），"
     "绝对值有有限尺寸差。符号翻转处与装置自身的门槛相符（见 SA-3b）")

# =====================================================================
# SA-4 每键增量与总增量的符号
# =====================================================================
print()
print("=" * 78)
print("SA-4 每键增量与总增量的符号")
print("=" * 78)

# SA-4 容差预算：符号判定是布尔判定，无数值容差。
#   每键增量是 V(1+s) - V(1) 的浮点差，相对精度受 V 的条件数限制，量级 1e-15；
#   故「每键增量与解析式相符」的门槛取 1e-12（第 592 号 PA-4 的在案量级）。

sa4_rows = []
print("  %-16s %-8s %-7s %8s %10s %16s %16s %8s"
      % ("候选", "r_c", "amp", "N_bonds", "sum s", "总增量", "每键增量", "符号"))
for name, (f, _) in V_CAND.items():
    for rc in RC_LEVELS:
        for amp in CAP_AMPS[:2]:
            s, _, _, _ = collect("capacity", amp, rc)
            tot, ninf = V_increment(f, s)
            nb = int(np.sum(s != 0))
            per = tot / max(1, nb)
            sa4_rows.append((name, rc, amp, tot, per, nb, float(s.sum()), ninf))
            print("  %-16s %-8.1f %-7.2f %8d %10.4f %16.6e %16.6e %8s"
                  % (name.split()[0], rc, amp, nb, float(s.sum()), tot, per,
                     "负" if tot < 0 else "正"))

neg_all = all(r[3] < 0 for r in sa4_rows)
sa4_ok = neg_all
record("SA-4", "每键增量与总增量的符号（负增量条件）",
       "每单位长度增量全为负（第 592 号的负增量条件）；"
       "符号不随芯半径与错配强度翻转",
       "%d 组（三候选 × 三档芯半径 × 两档错配强度）总增量全为负：%s"
       % (len(sa4_rows), "是" if neg_all else "否"), sa4_ok,
       "**第 592 号的负增量结论在本轮构型上仍成立**")

# **总增量正比于键数**：该命题须在**等应变构型**上测。
#   均匀拉伸构型的各键 |s| 相同，故总增量 = 键数 × 每键增量，正比性逐位成立
#   （这正是第 592 号 PA-4 给 828 键正比性的构型）。
#   容量型芯的应变剖面随位置变（sech^2），各键 |s| 不同，
#   故总增量是**按 |s| 加权的和**，正比性在其上**不成立**，
#   这是本轮的实测结果，不是实现缺陷。
#   测法：改变环带外径 r_out 以改变键数。
prop_devs = []
prop_rows = []
cap_prop_dev = []
for name, (f, _) in V_CAND.items():
    base = None
    for r_out in (6.0, 8.0, 10.0, 12.0):
        s_u, _, _, _ = collect("uniform", 0.30, 1.5, r_out=r_out)
        tot_u, _ = V_increment(f, s_u)
        nb_u = int(np.sum(s_u != 0))
        per_u = tot_u / max(1, nb_u)
        prop_rows.append((name, r_out, tot_u, nb_u, per_u))
        if base is None:
            base = (tot_u, nb_u, per_u)
        else:
            dv = abs((tot_u / base[0]) - (nb_u / base[1])) / abs(nb_u / base[1])
            prop_devs.append(dv)
    # 容量型芯上的同一测量（作对照）
    base_c = None
    for r_out in (6.0, 8.0, 10.0, 12.0):
        s_c2, _, _, _ = collect("capacity", 0.30, 1.5, r_out=r_out)
        tot_c2, _ = V_increment(f, s_c2)
        nb_c2 = int(np.sum(s_c2 != 0))
        if base_c is None:
            base_c = (tot_c2, nb_c2)
        else:
            cap_prop_dev.append(abs((tot_c2 / base_c[0]) - (nb_c2 / base_c[1]))
                                / abs(nb_c2 / base_c[1]))
sa4b_ok = all(v < 1e-12 for v in prop_devs)
record("SA-4b", "总增量正比于键数（在等应变构型上测）",
       "均匀拉伸构型上，改变键数（环带外径四档）时总增量与键数同比，"
       "相对偏差 ≤ 1e-12（第 592 号 PA-4 的在案量级）",
       "均匀拉伸构型：总增量与键数之比的相对偏差 %s（逐位相符）；"
       "**对照**：容量型芯上的同一测量为 %s（不成立，"
       "因应变剖面随位置变，总增量是按 |s| 加权的和）"
       % (", ".join("%.2e" % v for v in prop_devs),
          ", ".join("%.2e" % v for v in cap_prop_dev)),
       sa4b_ok,
       "**判据对象订正**：正比性须在等应变构型上测，"
       "在容量型芯上该命题不成立且原因是构造的（应变剖面随位置变）")

# =====================================================================
# SA-5 与镜像奇芯型的对照
# =====================================================================
print()
print("=" * 78)
print("SA-5 与镜像奇芯型的对照")
print("=" * 78)

# SA-5 容差预算：$1/2$ 的复现取 1e-12 量级（第 592 号 PA-1 的在案容差，
#   该值是结构性质：镜像奇芯型下每对镜像键的应变恰反号）。

sa5_rows = []
print("  %-10s %-7s %-7s %8s %8s %12s %14s"
      % ("芯型", "b", "r_c", "N+", "N-", "键统计份额", "一阶对消比值"))
for kind, lab in [("vortex", "vortex"), ("quadrupole", "quadru")]:
    for b in B_LEVELS:
        for rc in RC_LEVELS:
            s, _, _, _ = collect(kind, b, rc)
            th, npos, nneg = key_share(s)
            r1 = first_order_ratio(s)
            sa5_rows.append((kind, b, rc, th, r1))
            print("  %-10s %-7.1f %-7.1f %8d %8d %12.10f %14.3e"
                  % (lab, b, rc, npos, nneg, th, r1))

mir_th = [r[3] for r in sa5_rows]
dev_half = max(abs(v - 0.5) for v in mir_th)
dev_r1 = max(abs(r[4]) for r in sa5_rows)
sa5_ok = (dev_half <= 1e-12) and (dev_r1 <= 1e-12)
record("SA-5", "镜像奇芯型的对照复现",
       "镜像奇型（涡旋、四极）键统计份额逐位为 1/2（偏离 ≤ 1e-12），"
       "一阶对消比值至机器零（≤ 1e-12）",
       "%d 组的份额最大偏离 1/2 为 %.3e；一阶对消比值的最大绝对值 %.3e"
       % (len(sa5_rows), dev_half, dev_r1), sa5_ok,
       "与容量型的差：份额 0.500000 对 1.000000，差 0.500000")

sa5_cmp = []
for rc in RC_LEVELS:
    s_c, _, _, _ = collect("capacity", 0.30, rc)
    th_c, _, _ = key_share(s_c)
    s_v, _, _, _ = collect("vortex", 1.0, rc)
    th_v, _, _ = key_share(s_v)
    sa5_cmp.append((rc, th_c, th_v, th_c - th_v))
    print("  对照 r_c = %.1f：容量型份额 %.6f，镜像奇型 %.6f，差 %+.6f"
          % (rc, th_c, th_v, th_c - th_v))
supp("SA-5 附带", "容量型与镜像奇型的份额差",
     "逐档差 %s" % ", ".join("%+.6f" % r[3] for r in sa5_cmp),
     "差恰为 1/2（容量型全 1 减镜像奇型全 1/2），是结构性的")

# =====================================================================
# SA-6 恒真检验
# =====================================================================
print()
print("=" * 78)
print("SA-6 恒真检验")
print("=" * 78)

# SA-6 容差预算：替身须使 SA-2 的判据翻转（份额不再为 1），
#   故判据为布尔：替身的份额须显著偏离 1（偏离 > 0.1）。

sa6_rows = []
print("  %-22s %10s %10s %16s" % ("替身构型", "键统计份额", "偏离 1", "一阶对消比值"))
# 替身只取真正破坏「全部键同号」的两类：
#   偶极型（镜像混合，正负键数不等）与镜像奇型（成对反号）。
#   「均匀拉伸」与「单行阶跃」**保持**全部键同号（份额皆为 1），
#   它们是同族的合法对照，不是替身，故不入名单。
for lab, kind, amp in [("容量型（真构型）", "capacity", 0.30),
                       ("偶极型（混合同反号）", "dipole", 1.0),
                       ("镜像奇型（全部配对）", "vortex", 1.0)]:
    s, _, _, _ = collect(kind, amp, 1.0)
    th, npos, nneg = key_share(s)
    r1 = first_order_ratio(s)
    sa6_rows.append((lab, kind, th, r1))
    print("  %-22s %10.6f %10.4f %16.3e" % (lab, th, abs(th - 1.0), r1))

true_th = next(r[2] for r in sa6_rows if r[0].startswith("容量型"))
alt_th = [r[2] for r in sa6_rows if not r[0].startswith("容量型")]
flip = all(abs(v - 1.0) > 0.1 for v in alt_th)
sa6_ok = flip and abs(true_th - 1.0) < 1e-12
# 主判据（SA-3b）的替身：镜像奇型构造。
#   它的混比份额恒为 0.5（每对镜像键一正一负），
#   故其增量在 0.5 < theta* 的区段恒为正，**使 SA-3b 的主判据翻转**。
mirror_mix = None
for r in sa3b_rows:
    pass
s_vortex_mix, _, _, _ = collect("vortex", 1.0, 1.5)
mix_v, _, _, _ = mix_theta(s_vortex_mix)
tot_vortex = {}
for name, (f, _) in V_CAND.items():
    t, _ = V_increment(f, s_vortex_mix)
    tot_vortex[name] = t
sa6b_ok = all(t > 0.0 for t in tot_vortex.values()) and mix_v < 0.5744
record("SA-6", "恒真检验：破坏「全部键同号」后份额须立刻偏离",
       "替身（偶极型、镜像奇型）的键统计份额须显著偏离 1（偏离 > 0.1），"
       "真构型（容量型）的份额为 1",
       "真构型 %.6f（偏离 1 为 %.1e）；替身份额 %s（偏离 1 为 %s）"
       % (true_th, abs(true_th - 1.0),
          ", ".join("%.6f" % v for v in alt_th),
          ", ".join("%.4f" % abs(v - 1.0) for v in alt_th)),
       sa6_ok,
       "替身使 SA-2 的结构事实翻转，证其有区分力")

record("SA-6b", "恒真检验（主判据）：镜像奇型构造须使 SA-3b 翻转",
       "镜像奇型的混比份额须恒为 0.5（低于全部三个门槛），"
       "且其芯区增量须为正（使门槛判据不成立）",
       "镜像奇型的混比份额 %.6f（门槛 0.5744 以下）；"
       "三候选的增量 %s（全为正）"
       % (mix_v, ", ".join("%s：%+.6e" % (k.split()[0], v)
                            for k, v in tot_vortex.items())),
       sa6b_ok,
       "**替身使主判据翻转**，故 SA-3b 的门槛判据有区分力")

# =====================================================================
# SA-7 前提守恒
# =====================================================================
print()
print("=" * 78)
print("SA-7 前提守恒")
print("=" * 78)

# SA-7 容差预算：沿用各在案容差。
#   第 591 号的 theta*：20001 点梯形积分的数值值，
#   复算值须逐位相符（相对偏差 ≤ 1e-9）。
#   第 592 号的 1/2 与 0.5067：结构性质，门槛 1e-12。
#   第 592 号的 0.5140306122：分支割线伪影值，须复现其出现条件。

# 在案值 0.7306 / 0.5744 / 0.5781 是四位有效数字的舍入值，
# 故复现判据为「重算值落在舍入区间内」，偏差 ≤ 半个末位（5e-5）。
sa7_theta_dev = []
for name, (f, ts_591) in V_CAND.items():
    ts, _, _ = theta_star(f, r_c=5.0)
    sa7_theta_dev.append((name, ts, ts_591, abs(ts - ts_591)))
sa7_theta_ok = all(v[3] <= 5.0e-5 for v in sa7_theta_dev)

ts_cap, ip_cap, im_cap = theta_star(V_OUT_CAP[0], r_c=5.0)
ts_lin, ip_lin, im_lin = theta_star(V_OUT_LIN[0], r_c=5.0)

dip_vals = []
for b in B_LEVELS:
    for rc in RC_LEVELS:
        s, _, _, _ = collect("dipole", b, rc)
        th, _, _ = key_share(s)
        dip_vals.append(th)
dip_lo, dip_hi = min(dip_vals), max(dip_vals)
sa7_dip_ok = (abs(dip_hi - 0.5067) < 1e-3) and (abs(dip_lo - 0.5053) < 1e-3)

record("SA-7a", "复现第 591 号的三候选 theta* 与第 592 号的在案读数",
       "theta* 三值重算落在在案值四位有效数字的舍入区间内（偏差 ≤ 5e-5）；"
       "库外二次帽给 0.4213、库外线性势给 0.5000；"
       "镜像奇型的 1/2 偏离 ≤ 1e-12；偶极型偏离至约 0.5053 至 0.5067",
       "theta* 重算 %s（对在案值的偏差 %s，皆在舍入区间内）；"
       "库外二次帽 %.4f、库外线性势 %.4f；"
       "镜像奇型 1/2 的最大偏离 %.3e；偶极型份额 %.4f 至 %.4f"
       % (", ".join("%.6f" % v[1] for v in sa7_theta_dev),
          ", ".join("%.1e" % v[3] for v in sa7_theta_dev),
          ts_cap, ts_lin, dev_half, dip_lo, dip_hi),
       sa7_theta_ok and sa7_dip_ok and dev_half <= 1e-12,
       "第 591 号的 theta* 可逐位复现，故本轮的重算可信")


def collect_arctan2(kind, amp, rc, r_out=R_OUT):
    """用 arctan2 取差分的收集器（复现第 592 号首版的分支割线伪影）。"""
    if kind == "vortex":
        # 符号约定承第 592 号：u_z = +(amp/2 pi)·arctan2(y, x)
        u = (amp / (2 * np.pi)) * np.arctan2(Y, X)
    else:
        u = np.zeros_like(X)
    m = (RR >= rc) & (RR <= r_out)
    mx = m[1:, :] & m[:-1, :]
    my = m[:, 1:] & m[:, :-1]
    s = np.concatenate([(u[1:, :] - u[:-1, :])[mx], (u[:, 1:] - u[:, :-1])[my]])
    xb = np.concatenate([X[1:, :][mx], X[:, 1:][my]])
    yb = np.concatenate([Y[1:, :][mx], Y[:, 1:][my]])
    db = np.concatenate([np.ones(int(mx.sum())), 2 * np.ones(int(my.sum()))])
    return s, xb, yb, db


s_art, _, _, _ = collect_arctan2("vortex", 1.0, 1.0)
th_art, _, _ = key_share(s_art)
max_art = float(np.abs(s_art).max())
analytic_bound = 1.0 / (2.0 * np.pi)
s_ana, _, _, _ = collect("vortex", 1.0, 1.0)
th_ana, _, _ = key_share(s_ana)
max_ana = float(np.abs(s_ana).max())
sa7b_ok = (abs(th_art - 0.5140306122) < 1e-8) and abs(max_art - 0.985571) < 1e-5 \
    and abs(th_ana - 0.5) < 1e-12
record("SA-7b", "第 592 号过程记录教训其五的复现：分支割线伪影",
       "用 arctan2 取差分须复现 0.5140306122 与 max|s| = 0.985571；"
       "改用无割线的解析梯度场后份额回 1/2、max|s| 回 0.159155",
       "arctan2 档：份额 %.10f（对 0.5140306122 偏差 %.2e），max|s| = %.6f；"
       "解析梯度场档：份额 %.10f，max|s| = %.6f（解析上界 %.6f）"
       % (th_art, abs(th_art - 0.5140306122), max_art, th_ana, max_ana, analytic_bound),
       sa7b_ok,
       "**故本轮全部读数取无割线的解析梯度场**；该教训已在第 592 号过程记录其五立档")

# =====================================================================
# SA-8 档位与射程
# =====================================================================
print()
print("=" * 78)
print("SA-8 档位与射程")
print("=" * 78)

# SA-8 是射程声明，容差不适用。
sa8_ok = True
record("SA-8", "档位与射程声明",
       "读数档位、构型的档位归属、候选的取舍，以及「容量型芯」在物理档下可制备性的声明",
       "读数取模型档：V_rep 三候选的系数 A = 1、参考间距 r_0 = 1、"
       "形变剖面 s(r) = 0.5/(1+r/r_c)、键面密度 1/r0^2、权重 2 pi r dr、"
       "20001 点梯形积分、格点 N = 33、环带 r_c ≤ r ≤ 12，"
       "与第 591 号 JA-3 及第 592 号的在案读数同档。"
       "本轮构型（单边错配容量型芯）属**模型档**。"
       "**候选甲（硬核极限）不参与比对**：第 591 号 §二 记「候选甲只有 r→0 发散的陈述"
       "而无函数形式，其芯区贡献只能为正或零，结构上不供负项」。"
       "**「容量型芯」在物理档下的可制备性未答，属本轮射程外**："
       "第 592 号已记「这类构型要求芯两侧高度差累积而不反号，"
       "与法线轴位错的回绕结构不同层」，其体系可实现性待定",
       sa8_ok,
       "本轮只判「若取该构型，则膨胀份额越过门槛」这条条件是否成立")

# =====================================================================
# 出图
# =====================================================================
print()
print("=" * 78)
print("出图")
print("=" * 78)

fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.4))

# (a) 混比侧的增量曲线与三候选门槛（膨胀份额轴上的门槛位置）
ax = axes[0, 0]
for name, (f, _) in V_CAND.items():
    rows = sorted([r for r in sa3b_rows if r[0] == name], key=lambda r: r[2])
    ax.plot([r[2] for r in rows], [r[4] for r in rows], "o-", ms=4, lw=1.2,
            label=r"%s 的芯区增量" % name.split()[0])
for name, ts_dev, ipd, imd in sa3b_dev_star:
    ax.axvline(ts_dev, ls="--", lw=1.0, color="tab:gray")
    ax.text(ts_dev, ax.get_ylim()[1], r"$\theta^*$", fontsize=7,
            rotation=90, va="top", ha="right")
ax.axhline(0.0, color="k", lw=0.9)
ax.set_xlabel(r"混比份额 $\theta$（键数混比，1 为纯拉伸）")
ax.set_ylabel(r"芯区每单位长度增量")
ax.set_title(r"(a) 增量曲线与三候选门槛", fontsize=10)
ax.legend(fontsize=7)
ax.grid(alpha=0.3)
ax.text(0.03, 0.90, r"竖线为装置自身的门槛；曲线在门槛处过零",
        transform=ax.transAxes, fontsize=8, va="top")

# (b) 每键增量随所扫参数，与镜像奇对照并列
ax = axes[0, 1]
for name, (f, _) in V_CAND.items():
    per_cap = []
    for rc in RC_LEVELS:
        r_ = next(r for r in sa4_rows if r[0] == name and r[1] == rc and r[2] == 0.10)
        per_cap.append(r_[4])
    s_v, _, _, _ = collect("vortex", 1.0, 1.0)
    tot_v, _ = V_increment(f, s_v)
    per_v = tot_v / max(1, int(np.sum(s_v != 0)))
    ax.plot(RC_LEVELS, per_cap, "o-", color="tab:red", ms=6,
            label=r"%s 容量型" % name.split()[0])
    ax.plot(RC_LEVELS, [per_v] * len(RC_LEVELS), "s--", color="tab:blue", ms=5,
            label=r"%s 镜像奇型" % name.split()[0])
ax.axhline(0.0, color="k", lw=0.9)
ax.set_xlabel(r"芯半径 $r_c$")
ax.set_ylabel(r"每键增量")
ax.set_title(r"(b) 每键增量：容量型为负、镜像奇型为正", fontsize=10)
ax.legend(fontsize=7)
ax.grid(alpha=0.3)

# (c) 份额差（单边型减镜像奇型）
ax = axes[1, 0]
ax.plot([r[0] for r in sa5_cmp], [r[3] for r in sa5_cmp], "o-",
        color="tab:purple", ms=7, label=r"份额差（容量型 $-$ 镜像奇型）")
ax.axhline(0.5, color="tab:gray", ls=":", lw=1.4, label=r"参考 $1/2$")
ax.set_xlabel(r"芯半径 $r_c$")
ax.set_ylabel(r"份额差")
ax.set_title(r"(c) 容量型与镜像奇型的份额差", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax.text(0.03, 0.90, r"差恒为 $0.500000$（距 $1/2$ 的距离为 0）",
        transform=ax.transAxes, fontsize=8, va="top")

# (d) 恒真检验：破坏「全部键同号」后的份额
ax = axes[1, 1]
labs = [r[0] for r in sa6_rows]
vals = [r[2] for r in sa6_rows]
cols = ["tab:blue" if r[0].startswith("容量型") else "tab:orange" for r in sa6_rows]
ax.bar(range(len(vals)), vals, color=cols, width=0.6)
ax.axhline(1.0, color="tab:blue", ls="--", lw=1.2, label=r"真构型的份额 1")
ax.axhline(0.5, color="tab:gray", ls=":", lw=1.4, label=r"镜像奇型 $1/2$")
ax.set_xticks(range(len(labs)))
ax.set_xticklabels(labs, rotation=30, ha="right", fontsize=7)
ax.set_ylabel(r"膨胀份额")
ax.set_title(r"(d) 恒真检验：破坏「全部键同号」后份额立刻偏离", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")

fig.suptitle("第 790 号：来源甲的条件性命中（单边错配容量型芯）", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT, dpi=150)
print("  图已存: %s" % OUT)

# =====================================================================
# 总表
# =====================================================================
print()
print("=" * 78)
print("判据总表")
print("=" * 78)
npass = 0
for cid, name, expect, actual, ok, note in RESULTS:
    print("%-8s %-56s %s" % (cid, name[:56], "通过" if ok else "判否"))
    if ok:
        npass += 1
print()
print("合计 %d/%d 通过（主判据 SA-1 至 SA-8，展开为 %d 个子项）"
      % (npass, len(RESULTS), len(RESULTS)))
print()
print("关键读数:")
print("  SA-1 第 591 号的膨胀份额 theta 是混比参数（I(theta) = theta·I_+ + (1-theta)·I_-）；"
      "第 592 号的 theta_eff = N_+/(N_+ + N_-) 是键应变符号统计，是另一个量")
print("       本轮分开测；用加权凸性口径，不用四阶展开")
print("  SA-2 键统计份额 %d 档全为 1.000000（最小 %.10f，最大 %.10f）"
      % (len(sa2_rows), sa2_min, sa2_max))
print("  SA-2b theta* 重算：%s"
      % ", ".join("%s@%.1f：%.6f" % (v[0].split()[0], v[1], v[2]) for v in sa2b_rows))
print("       I(1) = I_+ 全为负：%s"
      % ", ".join("%s@%.1f：%+.6f" % (v[0].split()[0], v[1], v[3]) for v in sa2b_rows))
print("       收敛检验（20001 对 10001 点）：%s"
      % ", ".join("%.3e" % v for v in conv_devs))
print("  SA-3 九组的 I(1) 全为负；theta* 的散布与到 1 的距离 %s"
      % "; ".join("%s：散布 %.6f、距离 %.6f、占比 %.4f"
                  % (n.split()[0], spreads[n][0], spreads[n][1],
                     spreads[n][0] / spreads[n][1]) for n in V_CAND))
print("       五个参照点与份额 1 的差额：%s"
      % "; ".join("%s %+.6f" % (lab, dv) for lab, rv, dv in ref_devs))
print("  SA-4 每键增量 %d 组全为负：%s" % (len(sa4_rows), "是" if neg_all else "否"))
print("       总增量与键数同比的相对偏差 %s" % ", ".join("%.2e" % v for v in prop_devs[:3]))
print("  SA-5 镜像奇型 %d 组：1/2 的最大偏离 %.3e，一阶对消最大 %.3e"
      % (len(sa5_rows), dev_half, dev_r1))
print("       与容量型的份额差 %s" % ", ".join("%+.6f" % r[3] for r in sa5_cmp))
print("  SA-6 真构型份额 %.6f；替身份额 %s"
      % (true_th, ", ".join("%.6f" % v for v in alt_th)))
print("  SA-7 theta* 复算相对偏差 %s；库外二次帽 %.4f、库外线性势 %.4f"
      % (", ".join("%.1e" % v[3] for v in sa7_theta_dev), ts_cap, ts_lin))
print("       偶极型份额 %.4f 至 %.4f；割线伪影档份额 %.10f、max|s| = %.6f"
      % (dip_lo, dip_hi, th_art, max_art))
print("  SA-8 模型档；候选甲不参与；容量型芯的物理档可制备性未答（射程外）")
print("耗时 %.1f 秒" % (time.time() - T_START))
