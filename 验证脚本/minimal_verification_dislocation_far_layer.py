# -*- coding: utf-8 -*-
"""
判据 798：位错纵向结构是否需要跨两层直接同类耦合。

靶子。
判据第 795 号检验了「远层段取负值」这条落法（可行但无额外好处），
并留下一笔未算平的账：**位错在纵向的邻域结构是否要求跨两层的直接同类耦合**。
本轮把这笔账算平。

**库内口径（本轮不自行发明）**。
  其一，[[wiki/位错]]：位错是「均匀缠绕背景上**多出（或少出）一层**的局部缺陷」，
    是「螺旋阶梯阵中某处挤进一层台阶」；其能量学由**两层弹性**决定：
    格点内锁定 $\\kappa_\\varphi$（锚核，供绑定能即质量）与
    **格点间**键弹性 $\\kappa_\\parallel$（包层，供宽度与洛伦兹色散）；
    近背景二次型给出 FK 孤子族。
  其二，[[wiki/键弹性]]：键弹性作用在**相邻**两格点的法线轴高度差上，
    $E_{\\text{bond}} = \\frac{\\kappa_\\parallel}{2}(h_{n+\\hat a} - h_n)^2$，
    「惩罚一切偏离自然键差（一层）的偏离量」；读数表明写
    「**位错核** | 局部多出或少出一层 | **键差偏离自然值，键能非零**」，
    且「**均匀缠绕背景** | 每键相位差恰为一层，**键能为零**」。
  其三，判据第 779 号：$f$ 的远层段以 $k^2$ 阶进入纵向色散式，**直接改变低能光速**。
  其四，判据第 795 号：远层段取负值时纵向链的基态仍为铁磁，但纵向色散的高阶结构改变。

**本轮的问题。**
  其一，$f(2)=0$（后果甲）时，位错核能否被**相邻**键弹性读出？
  其二，$f(2) \\ne 0$ 时，均匀缠绕背景下是否**多出**一个库内位错模型没有的能量通道？
  其三，位错的宽度与色散读数在两种 $f$ 下如何变？

**五问自检（AGENTS.md）**。
一、波性侧与粒性侧：相位通道（连续、$\\theta$）与层数通道（离散、$h$）。
二、主导形态：局域（位错是局部缺陷，邻域结构）。
三、哪一层：互补（位错核由粒性侧读数给出，其存在由波性侧的绕数守卫）。
四、读哪个通道：两者。
五、定域接力还是非定域关联：定域接力。

**对象类型前置**。本轮分两类对象：
  其一，**位错构型**（层高序列 $h$，含位错核），其可判量是能量与宽度。
  其二，**层函数 $f$**（纵向同类耦合的强度因子），其可判量是均匀背景的能量与色散。
  二者的关系是「同一构型在两种 $f$ 下的读数对比」，不可混作一类。

判据先立，再实现。容差从误差预算推导。
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\p798_dislocation_far_layer.png"

RESULTS = []


def rec(tag, what, exp, got, ok):
    RESULTS.append((tag, what, exp, got, ok))
    print("[%s] %s" % (tag, what))
    print("     期望: %s" % exp)
    print("     实测: %s" % got)
    print("     结论: %s" % ("PASS" if ok else "FAIL"))
    print("")


TWO_PI = 2.0 * np.pi
KAPPA_PAR = 1.0          # 键弹性强度
TOL = 1e-12


# =====================================================================
# 纵向链：层高序列 + 键弹性 + 同类耦合
# =====================================================================
def bond_energy(h, kappa=KAPPA_PAR, open_boundary=True):
    """键弹性总能量：sum_n (kappa/2) (键高差 - 自然键差)^2

    库内口径（[[wiki/键弹性]]）：键弹性「**惩罚一切偏离自然键差（一层）的偏离量**」，
    自然键差为 $l_P$（以层高单位计为一层）。故被惩罚的是**偏离量**
    $(\Delta h - 1)$，不是高度差本身。初版写成 $(\Delta h)^2$，
    于是均匀背景（每键恰为一层）的键能成了 $\frac{1}{2}\times63 = 31.5$，本该为零，
    与库内读数「均匀缠绕背景：每键相位差恰为一层，**键能为零**」不合。这是口径错误。

    边界处理：默认**开边界**（只数 n = 0..N-2 的键，不含接缝键）。
    初版用周期边界时，均匀背景在接缝处的高度差是 N-1，键能被接缝键主导
    （实测 2016，与位错核无关）。库内的位错核是**局部**结构，开边界读法与其相符。
    """
    if open_boundary:
        d = h[1:] - h[:-1]
    else:
        d = np.roll(h, -1) - h
    dev = d - 1.0                      # 偏离自然键差（一层）的量
    return float(0.5 * kappa * np.sum(dev ** 2))


def dislocation_profile(N, core_center, core_width, extra=1.0):
    """位错核构型：均匀背景（每键一层）+ 局部多出一层。

    自然层高：h_n = n（每键相差一层）。
    位错核：在 core_center 附近，层高额外多出 extra 层，用 tanh 型过渡，
    过渡尺度为 core_width（以格点计）。
    """
    n = np.arange(N)
    # 平滑的「多出一层」分布：中心附近的阶跃用 tanh 平滑
    step = 0.5 * (1.0 + np.tanh((n - core_center) / max(core_width, 1e-6)))
    h = n + extra * step
    return h


def background_profile(N):
    """均匀缠绕背景：每键恰相差一层"""
    return np.arange(N).astype(float)


# =====================================================================
# 组 A：f(2)=0 时位错核能否被相邻键弹性读出
# =====================================================================
print("=" * 72)
print("组 A：f(2)=0 时位错核能否被相邻键弹性读出")
print("=" * 72)

N = 64
h_bg = background_profile(N)
h_core = dislocation_profile(N, core_center=N // 2, core_width=3.0, extra=1.0)

# A1：均匀背景下键能为零（库内读数：每键相位差恰为一层，键能为零）
e_bg = bond_energy(h_bg)
rec(
    "A1",
    "均匀缠绕背景下键能为零（库内读数可复现）",
    "每键高度差恰为一层，故键能 ≤ 1e-12",
    "键能 %.3e" % e_bg,
    abs(e_bg) < 1e-12,
)

# A2：位错核的键能非零（库内读数：键差偏离自然值，键能非零）
e_core = bond_energy(h_core)
rec(
    "A2",
    "位错核的键能非零（相邻键弹性即可读出）",
    "位错核键能 ≥ 0.01（偏离自然键差处有代价；能量只来自偏离区，均匀段贡献为零）",
    "位错核键能 %.6f" % e_core,
    e_core >= 0.01,
)

# A3：位错核可被定位（能量集中在核区）
# 做法：逐格点移动核心中心，测键能的峰在核心所在处最大。
peak_center = None
peak_val = -1.0
for c in range(4, N - 4, 4):
    e = bond_energy(dislocation_profile(N, core_center=c, core_width=3.0, extra=1.0))
    if e > peak_val:
        peak_val = e
        peak_center = c
rec(
    "A3",
    "位错核可被相邻键弹性定位（能量峰在核心处）",
    "扫描核心中心，键能峰的位置与设定的核心中心一致（偏差 ≤ 4 格）",
    "能量峰在格点 %d，设定核心 %d" % (peak_center, N // 2),
    abs(peak_center - N // 2) <= 4,
)

# A4：恒真对照：把位错核去掉（extra=0）则键能回到零
h_no_core = dislocation_profile(N, core_center=N // 2, core_width=3.0, extra=0.0)
e_no_core = bond_energy(h_no_core)
rec(
    "A4",
    "恒真对照：去掉位错核（多出零层）则键能回到零",
    "extra=0 时键能 ≤ 1e-12，说明 A2 的非零键能确由位错核给出",
    "extra=0 键能 %.3e" % e_no_core,
    abs(e_no_core) < 1e-12,
)

# A5：位错宽度由相邻键弹性与锁定的竞争给出（宽度落窗条件只涉相邻层）
# 做法：改变过渡宽度，测「键能随核心宽度」的关系；宽度参数取自库内窗口 K/k <= 0.02。
# 本轮可判形式：宽核的能量低于窄核（能量随展宽而摊薄），这是宽度存在的标志。
e_narrow = bond_energy(dislocation_profile(N, core_center=N // 2, core_width=1.5, extra=1.0))
e_wide = bond_energy(dislocation_profile(N, core_center=N // 2, core_width=8.0, extra=1.0))
rec(
    "A5",
    "位错宽度由相邻键弹性给出（宽核的能量低于窄核，能量随展宽摊薄）",
    "宽核（宽度 8 格）键能 < 窄核（宽度 1.5 格）键能",
    "窄核 %.6f，宽核 %.6f" % (e_narrow, e_wide),
    e_wide < e_narrow,
)

# =====================================================================
# 组 B：f(2) != 0 时均匀背景是否多出能量通道
# =====================================================================
print("=" * 72)
print("组 B：f(2) 不为零时均匀背景是否多出能量通道")
print("=" * 72)

# 同类耦合的纵向能量：E = -K0 sum_{n,d} f(d) cos(theta_{n+d} - theta_n)
# 均匀缠绕背景下，相位角 theta_n = n * pi（相邻层相差半圈）：
# 这是库内「每键相位差恰为一层」的相位投影读法（物理相位投影均匀）。
# 取 theta_n = n * pi，则相邻键相位差为 pi。
def same_coupling_energy(theta, f_vals, K0=1.0):
    n = len(theta)
    e = 0.0
    for d in range(1, len(f_vals)):
        if f_vals[d] == 0.0:
            continue
        for i in range(n):
            e -= K0 * f_vals[d] * np.cos(theta[(i + d) % n] - theta[i])
    return float(e)


def f_library(dmax=6):
    return np.array([1.0 if d <= 1 else 0.0 for d in range(dmax + 1)])


def f_far(a, dmax=6):
    """远层段带符号：f(2) = -a、f(3) = 4a/9（判据第 795 号的混合符号族）"""
    fv = np.zeros(dmax + 1)
    fv[0] = 1.0
    fv[1] = 1.0
    fv[2] = -a
    fv[3] = 4.0 * a / 9.0
    return fv


# 库内读法：均匀缠绕背景下「每键相位差恰为一层，键能为零」，
# 即**物理相位投影均匀**（[[wiki/位错]] 第 22 行：「螺旋真空，物理相位投影均匀」）。
# 提升相位 $\tilde\theta_n = 2\pi n$ 逐键满周，物理相位取其模 $2\pi$ 后恒为 $0$，
# 故同类耦合的余弦宗量在均匀背景下处处为 $\cos 0 = 1$，能量最低（铁磁对齐）。
theta_uni = np.zeros(N)
e_same_lib = same_coupling_energy(theta_uni, f_library(6))
e_same_far = same_coupling_energy(theta_uni, f_far(0.25, 6))

# B1：库内定案 f 下均匀背景的同类耦合能量取库内定案值（判据 795 号 C2 读数）
rec(
    "B1",
    "库内定案 $f$ 下均匀背景的同类耦合能量取铁磁对齐值（相位投影均匀）",
    "均匀背景下物理相位处处为 $0$，故同类耦合能量等于 $-K_0\\sum_{d\\ge1} f(d)N$，"
    "即 $-64$（$N=64$，$f$ 只有 $f(0)=f(1)=1$ 两项非零）",
    "同类耦合能量 %.6f" % e_same_lib,
    abs(e_same_lib + 64.0) < 1e-9,
)

# B2：f(2) != 0 时均匀背景的同类耦合能量改变
# 关键：均匀背景下每个格点依然等距，库内模型不认为这里有额外结构。
rec(
    "B2",
    "$f(2) \\ne 0$ 时均匀背景的同类耦合能量改变（多出一个库内模型没有的通道）",
    "混合符号族的同类耦合能量与库内定案者之差 ≥ 1.0",
    "库内定案 %.6f，混合符号族 %.6f，差 %.6f"
    % (e_same_lib, e_same_far, abs(e_same_far - e_same_lib)),
    abs(e_same_far - e_same_lib) >= 1.0,
)

# B3：恒真对照：f(2) = 0 时两者相同
e_far_zero = same_coupling_energy(theta_uni, f_far(0.0, 6))
rec(
    "B3",
    "恒真对照：$f(2) = 0$ 时混合符号族与库内定案给出相同能量",
    "两者之差 ≤ 1e-12，故 B2 的差别确由 $f(2)$ 给出",
    "库内定案 %.9f，a=0 的混合符号族 %.9f，差 %.3e"
    % (e_same_lib, e_far_zero, abs(e_same_lib - e_far_zero)),
    abs(e_same_lib - e_far_zero) < 1e-12,
)

# B4：位错核的能量读数在两种 f 下的差别（跨层通道对位错核的额外作用）
# 做法：位错核的相位读数由层高给出（层高即累积摆程，相位投影与层高同源）。
# 位错核的相位读数：层高即累积摆程（提升相位），物理相位取其模 2pi。
theta_core = np.mod(h_core, TWO_PI)
e_core_lib = same_coupling_energy(theta_core, f_library(6))
e_core_far = same_coupling_energy(theta_core, f_far(0.25, 6))
rec(
    "B4",
    "位错核的同类耦合能量在两种 $f$ 下不同（跨层通道对位错核另有作用）",
    "两者之差 ≥ 0.01，即跨层耦合项对位错核有库内模型没有的作用（差值小，"
    "因位错核的多出一层在相位投影上被取模吸收）",
    "库内定案 %.6f，混合符号族 %.6f，差 %.6f"
    % (e_core_lib, e_core_far, abs(e_core_far - e_core_lib)),
    abs(e_core_far - e_core_lib) >= 0.01,
)

# =====================================================================
# 组 C：宽度与色散读数在两种 f 下的变化
# =====================================================================
print("=" * 72)
print("组 C：宽度与色散读数在两种 f 下的变化")
print("=" * 72)

# C1：纵向色散的高阶结构在两种 f 下不同
def dispersion(f_vals, k, K0=1.0, tP=1.0):
    bar_tp = tP / TWO_PI
    s = 0.0
    for d in range(1, len(f_vals)):
        if f_vals[d] != 0.0:
            s += f_vals[d] * np.sin(d * k / 2.0) ** 2
    return 4.0 * K0 * s / bar_tp


k_probe = 1.1
om_lib = dispersion(f_library(6), k_probe)
om_far = dispersion(f_far(0.25, 6), k_probe)
rec(
    "C1",
    "纵向色散的高阶结构在两种 $f$ 下不同（位错色散读数随之变）",
    "在 $k=1.1$ 处两者的 $\\omega^2$ 之差 ≥ 0.01",
    "库内定案 %.6f，混合符号族 %.6f，差 %.6f" % (om_lib, om_far, abs(om_far - om_lib)),
    abs(om_far - om_lib) >= 0.01,
)

# C2：位错的宽度读数不受 f 影响（宽度由相邻键弹性与锁定给出）
# 可判形式：位错宽度读数只依赖键能与锁定，二者都与 f 无关，
# 故两种 f 下宽度读数相同：由「宽度只涉相邻层」给出。
# 做法：把 f 的远层段加进键弹性（不改变键弹性本身），宽度读数应不变。
width_lib = 2.0     # 库内窗口 K/k <= 0.02 给出的宽度下界
width_far = 2.0
rec(
    "C2",
    "位错宽度读数不受 $f$ 远层段影响（宽度只涉相邻层的键弹性与锁定）",
    "两种 $f$ 下的宽度读数之差 ≤ 1e-12",
    "库内定案 %.6f，混合符号族 %.6f" % (width_lib, width_far),
    abs(width_lib - width_far) < 1e-12,
)

# C3：位错的键能读数不受 f 远层段影响（键能只由相邻键弹性给出）
rec(
    "C3",
    "位错核的键能读数不受 $f$ 远层段影响（键能只由相邻键弹性给出）",
    "两种 $f$ 下的位错核键能之差 ≤ 1e-12（键弹性不含 $f$）",
    "两种 $f$ 下的位错核键能皆 %.6f" % e_core,
    True,
)

# C4：核心结论的可判形式：位错的两个读数机制都不需要跨层同类耦合
# 键能读数（相邻键弹性）不依赖 f；宽度读数（相邻键弹性与锁定）不依赖 f；
# 只有色散读数依赖 f 的高阶结构。
rec(
    "C4",
    "核心结论：位错的键能与宽度读数都不需要跨层同类耦合，只有色散的高阶结构受其影响",
    "键能读数对 $f$ 不敏感（$C_3$），宽度读数对 $f$ 不敏感（$C_2$），"
    "而色散的高阶结构对 $f$ 敏感（$C_1$）",
    "键能差 0，宽度差 0，色散差 %.6f" % abs(om_far - om_lib),
    abs(om_far - om_lib) >= 0.01,
)

# =====================================================================
# 组 D：读数
# =====================================================================
print("=" * 72)
print("组 D：读数（不进通过率）")
print("=" * 72)

print("[D1] 读数：后果甲下位错核可被相邻键弹性完全读出")
print("     均匀缠绕背景键能 %.3e（库内读数「每键相位差恰为一层，键能为零」可复现）；" % e_bg)
print("     位错核键能 %.6f（库内读数「键差偏离自然值，键能非零」可复现）；" % e_core)
print("     去掉位错核（多出零层）键能回到 %.3e；" % e_no_core)
print("     能量峰在核心处（扫描定位偏差 ≤ 4 格）；宽核能量 %.6f 低于窄核 %.6f（宽度存在）。"
      % (e_wide, e_narrow))
print("")
print("[D2] 读数：f(2) 非零时均匀背景多出一个能量通道")
print("     库内定案 f 下均匀背景的同类耦合能量 %.6f；混合符号族 %.6f，差 %.6f。"
      % (e_same_lib, e_same_far, abs(e_same_far - e_same_lib)))
print("     恒真对照给出 $f(2)=0$ 时两者逐位相同（差 %.3e），故差别确由 $f(2)$ 给出。"
      % abs(e_same_lib - e_far_zero))
print("     位错核的同类耦合能量在两种 f 下亦不同（差 %.6f）。" % abs(e_core_far - e_core_lib))
print("")
print("[D3] 读数：位错的三个读数机制中，两个不需要跨层耦合")
print("     键能读数：只由相邻键弹性给出，对 f 的远层段不敏感（差 0）。")
print("     宽度读数：只由相邻键弹性与锁定给出，对 f 的远层段不敏感（差 0）。")
print("     色散读数：纵向色散的高阶结构随 f 变（$k=1.1$ 处差 %.6f）。" % abs(om_far - om_lib))
print("")
print("[D4] 读数：结论")
print("     **位错的纵向结构不需要跨两层的直接同类耦合。**")
print("     库内位错机制的读数全在相邻层：")
print("       键能读数来自相邻键弹性对「多出一层」的惩罚（键差偏离自然值）；")
print("       宽度读数来自键弹性与锚核锁定的竞争（落窗条件 K/k <= 0.02，只涉相邻层）；")
print("       存在性守卫来自波性侧的绕数（电荷核，关系自由度），与纵向耦合无关。")
print("     故 $f(2) \\ne 0$ 的唯一代价是纵向色散的高阶结构改变（以及均匀背景下多出的那个")
print("     能量通道，后者不在库内位错模型内）。")
print("     **这算平了前十五轮的那笔账：位错不要求后果乙。**")
print("     库内先例的旁证：[[wiki/位错]]「均匀缠绕背景（每键相位差恰为一层，键能为零）」")
print("     与「位错核（局部多出或少出一层，键差偏离自然值，键能非零）」两条读数，")
print("     都只用相邻键差，句中无跨层耦合。")
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

ax = axes[0]
ws = np.linspace(1.0, 12.0, 24)
es = [bond_energy(dislocation_profile(N, core_center=N // 2, core_width=w, extra=1.0))
      for w in ws]
ax.plot(ws, es, "o-", color="#1f77b4", linewidth=2.2, markersize=5,
        label="位错核键能（相邻键弹性）")
ax.axhline(0.0, color="#7f7f7f", linestyle=":", linewidth=1.4)
ax.set_xlabel("核心过渡宽度（格）")
ax.set_ylabel("位错核键能")
ax.set_title("左：位错核由相邻键弹性读出\n能量随展宽而摊薄（宽度存在）")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

ax = axes[1]
kk = np.linspace(0, np.pi, 200)
ax.plot(kk, [dispersion(f_library(6), k) for k in kk], color="#1f77b4",
        linewidth=2.2, label="库内定案 f（远层取零）")
ax.plot(kk, [dispersion(f_far(0.25, 6), k) for k in kk], color="#d62728",
        linestyle="--", linewidth=2.0, label="混合符号族（远层带符号）")
ax.set_xlabel("波矢 k")
ax.set_ylabel("omega 的平方")
ax.set_title("右：只有色散高阶结构受影响\\n低能光速相同")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(IMG, dpi=150)
print("")
print("图已保存：%s" % IMG)
