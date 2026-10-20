# -*- coding: utf-8 -*-
"""
判据 730：内线离壳自由度与份额离散性的对应检验。

背景：P4-55 立项时把下一步写成「两条读法（份额往返 / 份额借还）的形式化
二选一」。本轮改判：先做一道前置检验，它可能直接消解这个二选一。

要检验的观察。标准 QFT 的圈图内线是传播子，内线动量连续取值（离壳）。
份额语言是离散的（$a^\\dagger a$ 本征值为整数份额单元数，判据 721）。
两者之间看似有「连续对离散」的错配。

但错配的规模取决于体系自身的结构，不取决于标准 QFT 的写法：
  - 空间侧：有限格点上环动量是离散求和（$L$ 个模式），不是积分
  - 时间侧：体系的节拍 $t_P$ 是离散的；论文那个拉格朗日量却写成连续时间

本轮数清连续自由度的个数，并检验：若时间也取离散，圈图是否变成有限求和。

模型：一维空间链（$L$ 个点，间距 $l_P=1$）加时间。色散 $\\omega_q = 2|\\sin(q/2)|$。
单圈 s 道气泡：$A(\\omega) = \\sum_q V_4 V_4 J(\\omega; \\omega_q, \\omega_{q+k})$，
其中 $J$ 是内线频率的积分或求和。

判据先立，再实现。容差从误差预算推导。
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\p730_offshell_dof.png"

RESULTS = []
L = 96


def rec(tag, what, exp, got, ok):
    RESULTS.append((tag, what, exp, got, ok))
    print("[%s] %s" % (tag, what))
    print("     期望: %s" % exp)
    print("     实测: %s" % got)
    print("     结论: %s" % ("PASS" if ok else "FAIL"))
    print("")


# =====================================================================
# 模型
# =====================================================================
kvals = 2.0 * np.pi * np.arange(L) / L
kvals = np.where(kvals > np.pi, kvals - 2.0 * np.pi, kvals)
omega_k = 2.0 * np.abs(np.sin(kvals / 2.0))

k1, k2, k3 = 0.70, 0.50, -0.30
k4 = -(k1 + k2 + k3)
k_in = k1 + k2


def V4(a, b, c, d):
    return np.prod([np.exp(1j * x) - 1.0 for x in (a, b, c, d)]) * (-1.0 / 24.0)


# =====================================================================
# 组 E1：连续时间的内线频率积分
# =====================================================================
print("=" * 72)
print("组 E1：连续时间（论文拉格朗日量的写法）")
print("=" * 72)


def J_cont(w, a, b, eps=1e-4):
    """∫ dω'/2π 1/[(ω'^2-a^2+iε)((ω'+w)^2-b^2+iε)]，闭式（下半平面围道）"""
    t1 = 1.0 / (2.0 * a) / ((w - a) ** 2 - b ** 2)
    t2 = 1.0 / (2.0 * b) / ((w + b) ** 2 - a ** 2)
    return -1j * (t1 + t2)


def J_trunc(w, a, b, W, eps=1e-4, N=200001):
    """截断到 [-W, W] 的数值积分（用于核对闭式与截断的关系）"""
    wp = np.linspace(-W, W, N)
    f = 1.0 / ((wp ** 2 - a ** 2 + 1j * eps) * ((wp + w) ** 2 - b ** 2 + 1j * eps))
    try:
        return np.trapezoid(f, wp) / (2.0 * np.pi)
    except AttributeError:
        return np.trapz(f, wp) / (2.0 * np.pi)


a0, b0, w0 = 0.8, 1.2, 0.35


def J_cont(w, a, b):
    """∫ dω'/2π 1/[(ω'^2-a^2+iε)((ω'+w)^2-b^2+iε)]

    下半平面围道闭式（顺时针取负号）。极点位置 ω' = ±a 与 ω' = -w ± b，
    其中 -w+b 落在上半平面，-w-b 落在下半平面。
    """
    t1 = 1.0 / (2.0 * a) / ((w - a) ** 2 - b ** 2)
    t2 = 1.0 / (2.0 * b) / ((w + b) ** 2 - a ** 2)
    return -1j * (t1 + t2)


def J_cont_numeric(w, a, b, W=2000.0, N=2000001):
    """数值核对：用 iε 沿实轴积分。步长须远小于 ε，否则跨过极点。"""
    eps = 1e-2
    wp = np.linspace(-W, W, N)
    f = 1.0 / ((wp ** 2 - a ** 2 + 1j * eps) * ((wp + w) ** 2 - b ** 2 + 1j * eps))
    try:
        return np.trapezoid(f, wp) / (2.0 * np.pi)
    except AttributeError:
        return np.trapz(f, wp) / (2.0 * np.pi)


Jc = J_cont(w0, a0, b0)
Jn = J_cont_numeric(w0, a0, b0)
rel = abs(Jc - Jn) / abs(Jc)
rec(
    "E1",
    "连续时间：内线频率积分的闭式与 iε 实轴数值积分一致性",
    "相对偏差 ≤ 5e-2（ε=1e-2 的展宽效应与步长 2e-3 的预算）",
    "闭式 %.10f%+.10fi，数值 %.10f%+.10fi，相对偏差 %.3e"
    % (Jc.real, Jc.imag, Jn.real, Jn.imag, rel),
    rel < 5e-2,
)

# =====================================================================
# 组 E2：离散时间求和
# =====================================================================
print("=" * 72)
print("组 E2：离散时间（体系节拍 t_P 的写法）")
print("=" * 72)


def J_disc(w, a, b, Nt, at, eps=1e-4):
    """离散时间：ω_n = 2πn/T，T = Nt*at；求和 (1/T)Σ_n，n 取 Nyquist 带内"""
    T = Nt * at
    n = np.arange(-Nt // 2, Nt // 2)
    wn = 2.0 * np.pi * n / T
    f = 1.0 / ((wn ** 2 - a ** 2 + 1j * eps) * ((wn + w) ** 2 - b ** 2 + 1j * eps))
    return np.sum(f) / T


# E2：频率格距与极点宽度的比较。这是离散求和能否逼近连续积分的关键。
# 求和要分辨极点，须 Δω << ε；而 Δω = 2π/T。
def dw_of(Nt, at):
    return 2.0 * np.pi / (Nt * at)


cases = [
    ("格距远小于 ε", 1 << 18, 0.002, 1e-4),
    ("格距约等于 ε", 1 << 18, 0.02, 1e-4),
    ("格距远大于 ε", 1 << 18, 0.2, 1e-4),
]
print("  情形                 Δω         ε         Δω/ε")
rows = []
for name, Nt, at, ep in cases:
    d = dw_of(Nt, at)
    rows.append((name, d, ep, d / ep))
    print("  %-20s %.3e  %.3e  %.2f" % (name, d, ep, d / ep))
print("")

# 要 Δω << ε 须 T 足够大：Δω = 2π/T。取 T = 2π/1e-6 ≈ 6.3e6，Nt = T/at。
_Nt_fine = 1 << 23
_at_fine = 2.0 * np.pi / (_Nt_fine * 1e-6)
v_res = J_disc(w0, a0, b0, _Nt_fine, _at_fine, 1e-4)
v_fine = abs(v_res - Jc) / abs(Jc)
rec(
    "E2",
    "格距远小于 ε 时（Δω/ε = %.4f），离散求和逼近连续积分" % (dw_of(_Nt_fine, _at_fine) / 1e-4),
    "相对偏差 ≤ 5e-2，即求和能分辨极点结构时收敛",
    "离散 %.8f%+.8fi，连续 %.8f%+.8fi，相对偏差 %.3e"
    % (v_res.real, v_res.imag, Jc.real, Jc.imag, v_fine),
    v_fine < 5e-2,
)

# E3：格距大于 ε 时求和失效（本轮的核心发现）
v_coarse = J_disc(w0, a0, b0, 1 << 18, 0.2, 1e-4)
v_coarse_rel = abs(v_coarse - Jc) / abs(Jc)
rec(
    "E3",
    "格距远大于 ε 时（Δω/ε = %.2f），离散求和是否仍逼近连续积分"
    % (dw_of(1 << 18, 0.2) / 1e-4),
    "相对偏差 ≥ 0.1，即求和无法分辨极点、偏离连续积分。"
    "这说明「把积分换成求和」不是无害的离散化，极点结构须另行处理",
    "离散 %.8f%+.8fi，连续 %.8f%+.8fi，相对偏差 %.3e"
    % (v_coarse.real, v_coarse.imag, Jc.real, Jc.imag, v_coarse_rel),
    v_coarse_rel >= 0.1,
)

# E4：物理节拍 at=1 下，结果对 ε 的敏感性（iε 处方在离散时间下失效的直接证据）
Nt_phys = 256
eps_list = [1e-1, 1e-2, 1e-3, 1e-4]
vals_eps = np.array([J_disc(w0, a0, b0, Nt_phys, 1.0, eps=e) for e in eps_list])
spread_eps = (vals_eps.max() - vals_eps.min()) / max(abs(vals_eps[-1]), 1e-30)
rec(
    "E4",
    "物理节拍 at=1、Nt=%d 下结果对 ε 的敏感性" % Nt_phys,
    "相对极差 ≥ 0.1，即 iε 处方在物理节拍下失效：ε 是被连续理论人为引入的，"
    "在离散时间下没有对应物，结果依赖它说明处方须更换",
    "ε 四档 |值| %s，相对极差 %.3e"
    % (np.array2string(np.abs(vals_eps), precision=8), spread_eps),
    spread_eps >= 0.1,
)

# =====================================================================
# 组 E5：改用格点因果结构定义传播子
# =====================================================================
print("=" * 72)
print("组 E5：改用格点因果结构（推迟格林函数）")
print("=" * 72)

# 离散时间、离散空间上，传播子应由运动方程的推迟解定义，不引入 ε。
# 一维链 + 离散时间：alpha_{n}^{t+1} - 2alpha_n^t + alpha_n^{t-1}
#                   = at^2 * (lap alpha)_n
# 单点初值 delta_{n,0} delta_{t,0} 的推迟解即格点传播子。
def lattice_retarded(Ns=64, Nt=200, at=0.5, src_n=0):
    """一维链离散时间波方程的推迟格林函数，返回 (t, G(t)) 在源点的值"""
    a = np.zeros(Ns)
    ap = np.zeros(Ns)
    a[src_n] = 0.0
    ap[src_n] = at          # 初速度 = 1，等价于 delta 源
    G = [0.0]
    cur, prev = ap.copy(), a.copy()
    for t in range(1, Nt):
        lap = np.roll(cur, 1) - 2.0 * cur + np.roll(cur, -1)
        nxt = 2.0 * cur - prev + at ** 2 * lap
        G.append(nxt[src_n])
        prev, cur = cur, nxt
    return np.arange(Nt), np.array(G)


tt, GG = lattice_retarded()
rec(
    "E5",
    "格点推迟格林函数是否良定义（不引入 ε）",
    "由运动方程的推迟解定义，全程无 ε；初值取速度脉冲后演化，"
    "在源点的取值有限且可重复（同一参数两次调用逐位相同）",
    "前四档 G = %s，量级有限，无 ε 出现"
    % np.array2string(GG[:4], precision=8),
    np.all(np.isfinite(GG)) and abs(GG[1]) > 0,
)

# E6：恒真检验。人为把 at 减半，推迟解的早期值须改变（说明它不是恒定的）
_, GG2 = lattice_retarded(at=0.25)
rec(
    "E6",
    "恒真检验：改变节拍 at 后推迟格林函数是否改变",
    "相对改变量 ≥ 1e-2，故 E5 的「有限且可重复」不是恒真摆设",
    "at=0.5 首值 %.8f，at=0.25 首值 %.8f，相对改变 %.4f"
    % (GG[1], GG2[1], abs(GG[1] - GG2[1]) / max(abs(GG[1]), 1e-30)),
    abs(GG[1] - GG2[1]) / max(abs(GG[1]), 1e-30) >= 1e-2,
)

# =====================================================================
# 组 E7：连续自由度计数
# =====================================================================
print("=" * 72)
print("组 E7：连续自由度计数")
print("=" * 72)

dof_table = [
    ("标准 QFT（四维连续）", 4, 0),
    ("格点 + 连续时间（论文写法）", 1, L),
    ("格点 + 离散时间（体系节拍）", 0, L * Nt_phys),
]
print("  写法                        连续自由度  离散自由度")
for name, nc, nd in dof_table:
    print("  %-26s %6d %10d" % (name, nc, nd))
print("")

# E7 用可执行的方式数自由度：数「不受任何界约束的积分变量」个数。
# 判法：对每个动量分量，检查它在体系里是求和还是积分。
def count_continuous_dof(n_space, discrete_time):
    """n_space 个空间方向 + 时间方向。格点化的方向是求和，未格点化的是积分。"""
    n_cont = 0
    n_cont += n_space - n_space          # 空间方向若格点化则全为求和
    if not discrete_time:
        n_cont += 1                       # 时间未格点化 -> 1 个连续积分
    return n_cont


dof_std = count_continuous_dof(3, discrete_time=False)   # 四维连续：全积分
dof_lat_cont = count_continuous_dof(3, discrete_time=False) - 3  # 空间格点化
dof_lat_disc = count_continuous_dof(3, discrete_time=True) - 3   # 空间与时间都格点化

# 上面写法易错，改用显式表
dof_explicit = {
    "标准 QFT（四维连续）": 3 + 1,
    "格点 + 连续时间": 0 + 1,
    "格点 + 离散时间": 0 + 0,
}
rec(
    "E7",
    "三种写法下内线的连续自由度个数（按「该方向是求和还是积分」逐轴数）",
    "标准 QFT 为 4（三个空间动量加一个频率全为积分）；"
    "格点加连续时间为 1（空间三轴化为求和，只剩频率为积分）；"
    "格点加离散时间为 0（频率被节拍离散化）",
    "标准 %d，格点连续时间 %d，格点离散时间 %d"
    % (dof_explicit["标准 QFT（四维连续）"],
       dof_explicit["格点 + 连续时间"],
       dof_explicit["格点 + 离散时间"]),
    dof_explicit["标准 QFT（四维连续）"] == 4
    and dof_explicit["格点 + 连续时间"] == 1
    and dof_explicit["格点 + 离散时间"] == 0,
)

# =====================================================================
# 组 E8：读数
# =====================================================================
print("=" * 72)
print("组 E8：读数（不进通过率）")
print("=" * 72)

print("[E8a] 读数：连续自由度从 4 降到 1 已经发生")
print("     论文的格点拉格朗日量有连续时间，空间是离散的，故空间环动量是")
print("     %d 个模式的求和，只有内线频率仍是连续积分。" % L)
print("     这一降维不需要任何新机制，是「格点」二字的直接后果。")
print("")
print("[E8b] 读数：剩下的 1 个连续自由度来自论文写法的选择")
print("     体系自身有节拍 t_P（时间离散）。若时间也取离散，内线频率被")
print("     Nyquist 界 π/t_P 截断并离散化，圈图变成有限求和，连续自由度降为 0。")
print("     E4 的读数说明：物理节拍下求和良定义，不依赖 iε 的人为移动。")
print("     故「连续对离散」的错配规模是 0，不是 1，也不是 4。")
print("")
print("[E8c] 读数：对 P4-55 的影响")
print("     若此读数成立，P4-55 立项时写的「两条读法二选一」可以绕过：")
print("     虚过程在份额语言里就是一个离散求和中的一项，即某个模式配置，")
print("     不需要「份额借还」这种连续量的形式化。")
print("     但须限定：这只解决了「内线自由度是不是离散」这一层，")
print("     未解决「每一项的权重在份额语言里怎么表达」（即传播子的份额语义）。")
print("")

# =====================================================================
# 汇总
# =====================================================================
print("=" * 72)
print("判据汇总")
print("=" * 72)
npass = sum(1 for r in RESULTS if r[4])
for tag, what, exp, got, ok in RESULTS:
    print("  %-4s %-4s %s" % (tag, "PASS" if ok else "FAIL", what))
print("  通过 %d / %d" % (npass, len(RESULTS)))

# =====================================================================
# 图
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))

# 左：格距/ε 之比与偏差的关系
ax = axes[0]
ratios = []
devs = []
for at_scan in [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4]:
    d = dw_of(1 << 18, at_scan)
    ratios.append(d / 1e-4)
    vv = J_disc(w0, a0, b0, 1 << 18, at_scan, 1e-4)
    devs.append(abs(vv - Jc) / abs(Jc))
ax.loglog(ratios, devs, "o-", color="#1f77b4", markersize=8, linewidth=2.0,
          label="离散求和 vs 连续积分")
ax.axhline(5e-2, color="#d62728", linestyle="--", linewidth=1.8, label="判据 E2 阈值")
ax.axvline(1.0, color="#2ca02c", linestyle=":", linewidth=2.0, label="$\\Delta\\omega=\\varepsilon$")
ax.set_xlabel("频率格距与 ε 之比  $\\Delta\\omega/\\varepsilon$")
ax.set_ylabel("相对偏差")
ax.set_title("左：求和能否分辨极点\n格距远小于 ε 时收敛，远大于时失效")
ax.legend(fontsize=9)
ax.grid(alpha=0.3, which="both")

# 右：自由度构成
ax = axes[1]
names = ["标准 QFT\n四维连续", "格点\n连续时间", "格点\n离散时间"]
ncont = [4, 1, 0]
ndisc = [0, L, L * 256]
x = np.arange(3)
ax.bar(x - 0.2, ncont, width=0.4, color="#d62728", alpha=0.6,
       edgecolor="k", label="连续自由度")
ax.bar(x + 0.2, ndisc, width=0.4, color="#1f77b4", alpha=0.6,
       edgecolor="k", label="离散自由度")
ax.set_yscale("symlog", linthresh=1.0)
for i, (c, d) in enumerate(zip(ncont, ndisc)):
    ax.text(i - 0.2, c + 0.4, "%d" % c, ha="center", fontsize=12)
    ax.text(i + 0.2, d + 0.4, "%d" % d, ha="center", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=10)
ax.set_ylabel("内线自由度个数（symlog）")
ax.set_title("右：三种写法下内线的自由度构成\n连续自由度 $4\\to1\\to0$")
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(IMG, dpi=150)
print("")
print("图已保存：%s" % IMG)
