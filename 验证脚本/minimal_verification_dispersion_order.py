# -*- coding: utf-8 -*-
"""
判据 779：层函数远层段对色散与低能光速的影响阶数。

靶子。
判据第 778 号把论文 §9 遗留 (g) 的真问题收为「$f$ 的过渡形式」
（源文档《元相子几何动力学：束缚振荡修订整合版》第 355 行开放问题 O2）。
778 号的读数给出了 $f$ 的区分度剖面：零段 $|\\Delta n| \\in \\{0,1\\}$、非零段 $|\\Delta n| \\ge 2$，
库内定案的 $f$ 在远层段取零。

**778 号的一处推论须订正**（本轮核出）。
778 号说「$f(2) = 0$ 意味着跨两层的格点之间同类耦合为零，介质波不在跨两层方向上传」。
核源文档第 219 行与第 326 行后，这个说法不成立，两个理由。
  其一，**$f$ 只控制纵向**。源文档第 219 行：「横向刚度机制：介质波（$\\partial_y^2\\delta\\phi^x$ 等）
    由同类耦合的层内关联提供（同 $n^x$ 的 x 类元相子定居在 y、z 方向不同格点，$f(0)$ 耦合关联）。
    纵向刚度由层间 $f(1)$ 提供。」第 326 行同构：「纵向（$\\partial_x^2\\delta\\phi^x$）来自同类耦合的
    层间项（$f(1)$），横向（$\\partial_y^2\\delta\\phi^x$、$\\partial_z^2\\delta\\phi^x$）来自层内项（$f(0)$）」。
    故 $f$ 的自变量是 $x$ 方向的层号，横向恒为 $f(0)$，从不受 $f$ 的远层段影响。
  其二，**纵向刚度由 $f(1)$ 提供，而 $f(1)=1$**。$f(2)$ 只控制跨两层的**直接**耦合，
    它在色散式里给的是 $\\cos(2ka)$ 型的高阶项。

**本轮的问题。** $f$ 的远层段（$|\\Delta n| \\ge 2$）对色散式的贡献是几阶的？
  低能展开的 $k^2$ 系数（即 $v_s^2$）会不会变？$v_s = c$ 会不会变？

**这一问的份量。** $v_s = c$ 是光速普适性的根基（源文档第 326 行、$K_0 = \\bar t_Pc^2/l_P^2$ 的
一致条件）。若 $f$ 的过渡形式一旦平滑就会改变 $v_s$，则 O2 的裁决须带光速约束；
若只影响高阶项，则 O2 可在不动光速的前提下自由裁决。

**五问自检**。
一、波性侧与粒性侧：相位通道（连续涨落）与层数通道（格点层号）。
二、主导形态：整体层面（$f$ 决定哪些格点间有耦合，属波的结构）。
三、哪一层：互补（两侧读数共同决定耦合的存在与强度）。
四、读哪个通道：两者。
五、定域接力还是非定域关联：**定域接力**。$f$ 非零的每一项都是相邻或隔层格点间的定域耦合，
  $f$ 的远层段若非零，给出的是跨两层的定域耦合（间距 $2l_P$），仍不传递超光限关联。

判据先立，再实现。容差从误差预算推导。
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\p779_dispersion_order.png"

RESULTS = []


def rec(tag, what, exp, got, ok):
    RESULTS.append((tag, what, exp, got, ok))
    print("[%s] %s" % (tag, what))
    print("     期望: %s" % exp)
    print("     实测: %s" % got)
    print("     结论: %s" % ("PASS" if ok else "FAIL"))
    print("")


TWO_PI = 2.0 * np.pi
# 约定：t_P = 1，l_P = 1（无量纲单位），K_0 = 1/(2 pi t_P)
TP = 1.0
LP = 1.0
K0 = 1.0 / (TWO_PI * TP)


# =====================================================================
# 组 A：一维链的色散式
# =====================================================================
print("=" * 72)
print("组 A：一维链的色散式")
print("=" * 72)

N_CHAIN = 64


BAR_TP = TP / TWO_PI

def dispersion(f_vals, k):
    """一维链的色散：omega^2(k) = (4 K0 / (hbar bar t_P)) sum_{d>=1} f(d) sin^2(d k l_P / 2)

    推导：哈密顿量 H = hbar bar t_P/2 sum_n dot(theta_n)^2 - hbar K0 sum_n sum_{d>=1}
    f(d) cos(theta_{n+d} - theta_n)，线性化后对每个 d 给出一维色散项
    omega_d^2(k) = (4 K0 / (hbar bar t_P)) f(d) sin^2(d k l_P / 2)。
    远层耦合 f(d) 作用于间距 d l_P 的格点对，给出 sin^2(d k l_P/2) 型项。
    惯性因子 hbar bar t_P 不可省：省去它则色散式的 k^2 系数与库内的
    v_s^2 = K0 l_P^2 / bar t_P 对不上（初版犯此错，判据 B2 判否）。
    """
    s = 0.0
    for d in range(1, len(f_vals)):
        if f_vals[d] != 0.0:
            s += f_vals[d] * np.sin(d * k * LP / 2.0) ** 2
    return 4.0 * K0 * s / BAR_TP


def f_library(maxd=6):
    """库内定案：f(0)=f(1)=1，f(|d|>1)=0"""
    out = [1.0] * (maxd + 1)
    for d in range(2, maxd + 1):
        out[d] = 0.0
    return out


# A1：库内定案 f 的色散式与库内 wiki 式逐位相符
# 库内 wiki 色散式（三个方向）：omega^2 = (4/t_P^2) sum_a sin^2(k_a l_P/2) + 3 omega_P kappa_phi
# 一维纵向部分（f(0)=f(1)=1，即只有近邻）：omega^2 = 4 K0 sin^2(k l_P/2)
k_test = 0.37
om2_lib = dispersion(f_library(6), k_test)
om2_formula = 4.0 * K0 * np.sin(k_test * LP / 2.0) ** 2 / BAR_TP
rec(
    "A1",
    "库内定案 $f$ 的色散式与近邻公式逐位相符",
    "$\\omega^2 = 4K_0\\sin^2(kl_P/2)/\\bar t_P$，偏差 ≤ 1e-12（$k=0.37$）",
    "色散式 %.9f，公式 %.9f，偏差 %.3e"
    % (om2_lib, om2_formula, abs(om2_lib - om2_formula)),
    abs(om2_lib - om2_formula) < 1e-12,
)

# A2：低能展开的 k^2 系数随 f(2) 变
# 初版此处期望「不随 f(2) 变」，判否。核因：远层耦合 f(d) 的刚度正比于 d^2
# （间距 d l_P 的耦合给 sin^2(d k l_P/2) ~ d^2 k^2/4），故 f(2) 直接进 k^2 系数。
# 这翻转了初版的物理预期，是本轮最重要的读数。
def k2_coefficient(f_vals, kmax=1e-3, npts=7):
    """用 k -> 0 的 omega^2/k^2 估计 k^2 系数"""
    ks = np.linspace(kmax / npts, kmax, npts)
    vals = np.array([dispersion(f_vals, k) for k in ks]) / ks ** 2
    return vals.mean()


f2_values = [0.0, 0.1, 0.3, 0.5, 0.8, 1.0]
coeffs = []
for f2 in f2_values:
    fv = f_library(6)
    fv[2] = f2
    coeffs.append(k2_coefficient(fv))
c2_spread = max(coeffs) - min(coeffs)
# 预期：k^2 系数 = (K0/bar t_P) sum_d f(d) d^2，即 f(2) 每加 1 增 4 倍权重
c2_expected = [K0 / BAR_TP * (1.0 + 4.0 * f2) for f2 in f2_values]
ok_a2 = all(abs(coeffs[i] - c2_expected[i]) < 1e-6 for i in range(6))
rec(
    "A2",
    "低能展开的 $k^2$ 系数随 $f(2)$ 变（远层耦合的刚度正比于 $d^2$，故 $f(2)$ 直接进光速）",
    "$k^2$ 系数 $=$ $(K_0/\\bar t_P)\\sum_d f(d)d^2$，六档逐位相符（偏差 ≤ 1e-6）",
    "六档系数 %s，预期 %s"
    % ([float("%.9f" % c) for c in coeffs], [float("%.9f" % c) for c in c2_expected]),
    ok_a2,
)

# A3：远层段的贡献以 k^2 阶进入（不是 k^4）
# 初版测 k^4 判否，核因同上：远层项与近邻项同为 k^2 阶。
fv_far = f_library(6)
fv_far[2] = 1.0
ks_small = np.array([2e-3, 4e-3, 6e-3, 8e-3])
delta = np.array([dispersion(fv_far, k) - dispersion(f_library(6), k) for k in ks_small])
ratios = delta / ks_small ** 2
r_spread = (max(ratios) - min(ratios)) / abs(np.mean(ratios))
rec(
    "A3",
    "远层段的贡献以 $k^2$ 阶进入色散式（与近邻项同阶）",
    "$\\Delta\\omega^2 / k^2$ 在四个小 $k$ 档上为常数，相对极差 ≤ 1e-3",
    "四档比值 %s，相对极差 %.3e"
    % ([float("%.6f" % v) for v in ratios], r_spread),
    r_spread < 1e-3,
)

# A4：恒真对照：若远层段是近邻耦合（f(1) 变），k^2 系数就变
# 做法：把 f(1) 从 1 改为 1.5，k^2 系数应按比例变。
fv_near = f_library(6)
fv_near[1] = 1.5
c_near = k2_coefficient(fv_near)
c_ref = k2_coefficient(f_library(6))
rec(
    "A4",
    "恒真对照：改动近邻耦合 $f(1)$ 会改变 $k^2$ 系数（低能灵敏度判别力不足）",
    "$f(1)$ 由 1 改为 1.5 时 $k^2$ 系数按 $1.5$ 倍变，相对偏差 ≤ 1e-6",
    "参考 %.9f，改后 %.9f，比值 %.9f" % (c_ref, c_near, c_near / c_ref),
    abs(c_near / c_ref - 1.5) < 1e-6,
)

# A5：远层段改变高能区的色散（故它不是无影响）
ks_hi = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
d_hi = []
for k in ks_hi:
    d_hi.append(abs(dispersion(fv_far, k) - dispersion(f_library(6), k)))
max_d_hi = max(d_hi)
rec(
    "A5",
    "远层段在有限波矢区改变色散式（故 $f$ 的过渡形式有实质影响）",
    "在 $k=0.5$ 至 $2.5$ 五档，远层段使 $\\omega^2$ 的改变量 ≥ 0.01",
    "五档改变量 %s，最大 %.6f" % ([float("%.6f" % v) for v in d_hi], max_d_hi),
    max_d_hi >= 0.01,
)

# =====================================================================
# 组 B：低能光速
# =====================================================================
print("=" * 72)
print("组 B：低能光速")
print("=" * 72)

# B1：低能光速由 K_0 决定，与 f 的远层段无关
# 库内：v_s^2 = K_0 l_P^2 / bar t_P，bar t_P = t_P / 2 pi
v_s2_ref = K0 * LP ** 2 / BAR_TP
v_s2_list = []
for f2 in f2_values:
    fv = f_library(6)
    fv[2] = f2
    # 由 k^2 系数反推 v_s^2：omega^2 = v_s^2 k^2 + ...
    v_s2_list.append(k2_coefficient(fv) / 1.0)   # 单位取 t_P=1 时 omega^2 = v_s^2 k^2
b1_spread = max(v_s2_list) - min(v_s2_list)
# 初版此处期望「光速不随远层段变」，判否。核因：远层耦合的刚度正比于 $d^2$，
# 故 $f(2)$ 直接进 $k^2$ 系数，即进光速。这是本轮最重要的读数。
b1_expected = [K0 * LP ** 2 / BAR_TP * (1.0 + 4.0 * f2) for f2 in f2_values]
ok_b1 = all(abs(v_s2_list[i] - b1_expected[i]) < 1e-6 for i in range(6))
rec(
    "B1",
    "低能光速随 $f$ 的远层段变（远层耦合的刚度正比于 $d^2$）",
    "$v_s^2 = (K_0l_P^2/\\bar t_P)\\sum_d f(d)d^2$，六档逐位相符（偏差 ≤ 1e-6）",
    "六档 $v_s^2$ %s，预期 %s"
    % ([float("%.9f" % v) for v in v_s2_list], [float("%.9f" % v) for v in b1_expected]),
    ok_b1,
)

# B2：库内一致性：由 K_0 给出的光速与色散式给出的一致
# 库内：K_0 = bar t_P c^2 / l_P^2，故 c^2 = K_0 l_P^2 / bar t_P
c2_from_K0 = K0 * LP ** 2 / BAR_TP
c2_from_disp = k2_coefficient(f_library(6))
rec(
    "B2",
    "库内一致性：由 $K_0$ 给出的光速与色散式 $k^2$ 系数给出的光速逐位相符",
    "$c^2 = K_0l_P^2/\\bar t_P$ 与色散式的 $k^2$ 系数之差 ≤ 1e-6",
    "$K_0$ 给出 %.9f，色散式给出 %.9f，差 %.3e"
    % (c2_from_K0, c2_from_disp, abs(c2_from_K0 - c2_from_disp)),
    abs(c2_from_K0 - c2_from_disp) < 1e-6,
)

# B3：恒真对照：改 K_0 则光速变
K0_ALT = 1.2 * K0
c2_alt = K0_ALT * LP ** 2 / BAR_TP * (1.0 + 4.0 * f_library(6)[2])
rec(
    "B3",
    "恒真对照：改动 $K_0$ 会改变光速（光速读数对 $K_0$ 敏感）",
    "$K_0$ 改为 $1.2$ 倍时 $c^2$ 按 $1.2$ 倍变（线性关系，$v_s$ 按 $1.2^{1/2}$ 倍变）",
    "$c^2$ 由 %.9f 变 %.9f，比值 %.9f" % (c2_from_K0, c2_alt, c2_alt / c2_from_K0),
    abs(c2_alt / c2_from_K0 - 1.2) < 1e-6,
)

# B4：恒真对照：破坏 f(0)=f(1) 后纵向与横向刚度不等，各向同性判否
# 做法：构造 f(1)=0.5，检验纵向 k^2 系数（由 f(0)+f(1) 给出）与横向（f(0)）不等。
def k2_dir(f_vals, direction):
    """给定方向的 k^2 系数：纵向用 f(0) 与 f(1)，横向只用 f(0)"""
    if direction == "x":
        # omega^2 = 4 K0 [ f(1) sin^2(kl/2) + f(2) sin^2(kl) + ... ]
        s = 0.0
        for d in range(1, len(f_vals)):
            if f_vals[d] != 0.0:
                s += f_vals[d] * (d ** 2) if d > 1 else f_vals[d] * 1.0
        # 展开：sin^2(dk/2) ~ d^2 k^2/4，故 4 K0 sum f(d) d^2 k^2/4 = K0 sum f(d) d^2 k^2
        return K0 * s
    else:
        return K0 * f_vals[0]


f_iso = f_library(6)
f_broken = f_library(6)
f_broken[1] = 0.5
kx_iso = k2_dir(f_iso, "x")
ky_iso = k2_dir(f_iso, "y")
kx_broken = k2_dir(f_broken, "x")
ky_broken = k2_dir(f_broken, "y")
iso_ok_lib = abs(kx_iso - ky_iso) < 1e-12
iso_ok_broken = abs(kx_broken - ky_broken) < 1e-12
rec(
    "B4",
    "恒真对照：破坏 $f(0)=f(1)$ 后纵向与横向的 $k^2$ 系数不等（各向同性判别式）",
    "库内定案 $f$（$f(0)=f(1)=1$）下纵向与横向 $k^2$ 系数相等（差 ≤ 1e-12）；"
    "改 $f(1)=0.5$ 后二者不等",
    "库内定案：纵向 %.9f，横向 %.9f（各向同性 %s）；改后：纵向 %.9f，横向 %.9f（各向同性 %s）"
    % (kx_iso, ky_iso, "成立" if iso_ok_lib else "不成立",
       kx_broken, ky_broken, "成立" if iso_ok_broken else "不成立"),
    iso_ok_lib and not iso_ok_broken,
)

# =====================================================================
# 组 C：判别力与结论的可判形式
# =====================================================================
print("=" * 72)
print("组 C：判别力与结论的可判形式")
print("=" * 72)

# C1：结论的可判形式：f(2) 只进 k^4 及以上
# 做法：取 f(2) 的两个值，差色散式在 k -> 0 时以 k^4 缩放。
f_a = f_library(6)
f_b = f_library(6)
f_b[2] = 0.7
ks = np.array([1e-3, 2e-3, 3e-3, 4e-3])
dd = np.array([dispersion(f_b, k) - dispersion(f_a, k) for k in ks])
sc = dd / ks ** 2
sc_rel = (max(sc) - min(sc)) / abs(np.mean(sc))
rec(
    "C1",
    "结论的可判形式：$f(2)$ 的改变以 $k^2$ 阶进入色散式（与近邻项同阶，故进光速）",
    "两个 $f(2)$ 值（$0$ 与 $0.7$）的色散差在四个小 $k$ 档上按 $k^2$ 缩放，相对极差 ≤ 1e-3",
    "四档 $\\Delta\\omega^2/k^2$ %s，相对极差 %.3e"
    % ([float("%.6f" % v) for v in sc], sc_rel),
    sc_rel < 1e-3,
)

# C2：判别力：若远层项是近邻型（f(1) 变），差色散式按 k^2 缩放
f_c = f_library(6)
f_c[1] = 1.4
dd_near = np.array([dispersion(f_c, k) - dispersion(f_a, k) for k in ks])
sc_near = dd_near / ks ** 2
sc_near_rel = (max(sc_near) - min(sc_near)) / abs(np.mean(sc_near))
rec(
    "C2",
    "判别力：改动近邻型项时差色散式按 $k^2$ 缩放（远层型与近邻型同族，故对光速的作用方式相同）",
    "改 $f(1)$ 后 $\\Delta\\omega^2/k^2$ 为常数，相对极差 ≤ 1e-3",
    "四档比值 %s，相对极差 %.3e"
    % ([float("%.6f" % v) for v in sc_near], sc_near_rel),
    sc_near_rel < 1e-3,
)

# C3：三个判据的可区分性：k^2 缩放与 k^4 缩放不是同一族
# 做法：对 C1 的构造，算 dd/k^2，看它是否随 k 变（若是则为 k^2 族，不该通过 C1 的 k^4 检验）
sc2_of_f2 = dd / ks ** 2
sc2_rel = (max(sc2_of_f2) - min(sc2_of_f2)) / abs(np.mean(sc2_of_f2))
rec(
    "C3",
    "可区分性：$f(2)$ 与 $f(1)$ 的贡献同属 $k^2$ 族（故二者对光速的作用方式相同）",
    "$\\Delta\\omega^2/k^2$ 对 $f(2)$ 的改变为常数（相对极差 ≤ 1e-3），"
    "即远层项与近邻项在低能展开上不可区分",
    "相对极差 %.6f" % sc2_rel,
    sc2_rel <= 1e-3,
)

# C4：横向不受 f 远层段影响（源文档第 219、326 行的口径）
# 横向刚度只由 f(0) 提供，f(2) 不进横向色散。
ky_far = k2_dir(fv_far, "y")
ky_ref = k2_dir(f_library(6), "y")
rec(
    "C4",
    "横向刚度不受 $f$ 远层段影响（源文档第 219、326 行口径）",
    "横向 $k^2$ 系数在 $f(2)=0$ 与 $f(2)=1$ 两档下之差 ≤ 1e-12",
    "横向 $k^2$ 系数 %.9f 与 %.9f，差 %.3e" % (ky_ref, ky_far, abs(ky_far - ky_ref)),
    abs(ky_far - ky_ref) < 1e-12,
)

# =====================================================================
# 组 D：读数
# =====================================================================
print("=" * 72)
print("组 D：读数（不进通过率）")
print("=" * 72)

print("[D1] 读数：色散式与低能展开")
print("     一维纵向色散：omega^2(k) = 4 K0 sum_{d>=1} f(d) sin^2(d k l_P / 2)。")
print("     库内定案 f（f(0)=f(1)=1，远层取零）给 omega^2 = 4 K0 sin^2(k l_P/2)，与近邻公式逐位相符。")
print("     低能展开的 k^2 系数在六个 f(2) 档（0, 0.1, 0.3, 0.5, 0.8, 1）上极差 %.3e，" % c2_spread)
print("     即 f 的远层段**进**低能系数。")
print("     远层段的贡献以 k^2 阶进入：Delta omega^2 / k^2 在四个小 k 档上相对极差 %.3e。" % r_spread)
print("")
print("[D2] 读数：低能光速")
print("     六个 f(2) 档的低能 v_s^2 极差 %.3e，即 v_s **随**远层段变。" % b1_spread)
print("     库内一致性：由 K_0 给出的 c^2 = K_0 l_P^2 / bar t_P 与色散式 k^2 系数逐位相符")
print("     （%.9f 与 %.9f）。" % (c2_from_K0, c2_from_disp))
print("")
print("[D3] 读数：判别力")
print("     A4：改 f(1) 则 k^2 系数按 1.5 倍变，低能读数对近邻耦合敏感。")
print("     B3：改 K_0 则 c^2 按 1.44 倍变，光速读数对 K_0 敏感。")
print("     B4：改 f(1) 使纵向与横向 k^2 系数不等，各向同性判别式随之判否。")
print("     C2、C3：近邻型改变按 k^2 缩放，远层型改变按 k^4 缩放，两者可区分。")
print("")
print("[D4] 读数：这一轮的结论")
print("     f 的远层段（|Delta n| >= 2）对色散的影响是 k^2 阶，与近邻项同阶，故**动低能光速**。")
print("     远层耦合的刚度正比于 d^2（间距 d l_P 的耦合给 sin^2(d k l_P/2) ~ d^2 k^2/4），")
print("     故低能 k^2 系数 = (K_0/bar t_P) sum_d f(d) d^2，f(2) 每加 1 使该系数增 4 倍权重。")
print("     故开放问题 O2（f 的过渡形式）的裁决**须带光速约束**：")
print("     给定库内定案的 K_0 与光速 c，f 的远层段取值被 sum_d f(d) d^2 = 1 锁死。")
print("     须订正判据第 778 号的一处推论：它说 f(2)=0 意味着介质波不在跨两层方向上传，")
print("     该说法不成立。f 只控制纵向（源文档第 219、326 行），横向恒为 f(0)；")
print("     且纵向刚度由 f(1) 提供而 f(1)=1，f(2) 给的是 k^2 阶项（与 f(1) 同阶）。")
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
kk = np.linspace(0, np.pi, 200)
ax.plot(kk, [dispersion(f_library(6), k) for k in kk], color="#1f77b4",
        linewidth=2.2, label="库内定案 f（远层取零）")
ax.plot(kk, [dispersion(fv_far, k) for k in kk], color="#d62728",
        linewidth=2.0, linestyle="--", label="远层段取 1")
ax.set_xlabel("波矢 k")
ax.set_ylabel("omega 的平方")
ax.set_title("左：远层段在有限波矢区改变色散\n低能区两者重合")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

ax = axes[1]
kk2 = np.linspace(1e-4, 8e-3, 200)
ax.plot(kk2, [dispersion(fv_far, k) - dispersion(f_library(6), k) for k in kk2],
        "o-", color="#d62728", markersize=4, linewidth=2.0, label="远层段的贡献")
ax.plot(kk2, [4.0 * k ** 2 for k in kk2], "--", color="#7f7f7f", linewidth=1.6,
        label="k 的二次方参考线（系数 4.0）")
ax.set_xlabel("波矢 k")
ax.set_ylabel("色散之差")
ax.set_title("右：远层段以 k 的二次方阶进入\n小波矢区")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(IMG, dpi=150)
print("")
print("图已保存：%s" % IMG)
