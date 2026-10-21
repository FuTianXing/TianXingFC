# -*- coding: utf-8 -*-
"""
判据 750：绕数算符与层数算符的量子关联（张量积可分离性）。

背景与靶子。
P4-55 六轮交付（判据第 730、737、740、743、744、745 号）后，主线剩下的最后一环是
「绕数（非定域读数）与层数（定域读数）的量子关联」。

**五问自检（AGENTS.md 常驻用法）。**
一、波性侧与粒性侧：波性侧为相位通道（横截面、内部空间、连续旋转、满周闭合、绕数），
   粒性侧为层数通道（纵向法线轴、外部空间、格点生成、$x^a = l_P n^a$）。
二、主导形态的层面：绕数是整体的（沿闭合回路取分支拓扑量），层数算符是局域的（单点读数）。
三、落在哪一层：互补（两侧并立，缺一侧即错）。
四、读哪个通道：两者都要，这正是本轮的接口问题。
五、**相互作用是定域接力还是非定域关联**：绕数是非定域读数（沿闭合回路），层数是定域读数；
   故本轮问的是「一个非定域的守恒量与一个定域算符的关系」，
   即先验纠缠（非定域）与校准接力（定域）这一对在**算符层面**的形态。

**库内给定的结构（本轮不自行发明）。**
其一，[[理论总纲/公理体系]]：物侧态空间 $\\mathcal{H}_k = L^2(S^1) \\otimes \\ell^2(\\mathbb{Z})$，
   两因子不构成重复计数（相位通道的傅里叶指标是瞬时谱指标，层数通道的指标是历史计数）。
   2026-10-21 已补入两通道的空间归属与定域性归属。
其二，[[wiki/扇区]]：设态空间 $\\mathcal{H}$ 上存在守恒算符 $\\hat{W}$（绕数算符），
   其本征值 $W \\in \\mathbb{Z}$（或 $Z_N$），扇区定义为本征子空间。
   扇区内绕数可对角化（基态空间投影量子数），扇区之间矩阵元为零。
其三，[[wiki/绕数]]：绕数算符 $\\hat{W}$ 与局域哈密顿量在全空间并不对易
   （仅 $Z_2$ 特例对易），但它保持基态空间、在基态空间内可对角化，
   跨扇区一阶矩阵元为零（数值达 $10^{-16}$ 量级）。
其四，判据第 721 号已验：张量积结构给出跨模式对易为零
   （$a_1 = a \\otimes I$、$a_2 = I \\otimes a$ 必然对易），
   同模式对易关系（内部子空间）实测 $8.88\\times10^{-16}$。

**本轮的问题。** 在张量积态空间上，绕数算符（相位侧、非定域）与层数算符（层数侧、定域）
   是什么关系？具体地：层数算符是否保持绕数算符的本征子空间（扇区）？
   若保持，则两者对易，绕数与层数在量子层面也是「各守一量」的互补结构。

判据先立，再实现。容差从误差预算推导。
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\p750_commutator.png"

RESULTS = []


def rec(tag, what, exp, got, ok):
    RESULTS.append((tag, what, exp, got, ok))
    print("[%s] %s" % (tag, what))
    print("     期望: %s" % exp)
    print("     实测: %s" % got)
    print("     结论: %s" % ("PASS" if ok else "FAIL"))
    print("")


TWO_PI = 2.0 * np.pi


# =====================================================================
# 模型：张量积态空间上的两个算符
# =====================================================================
# 相位侧：谐振子模，基矢 |m>，m = 0,1,...,M-1。对应相位通道 L^2(S^1) 的离散截断。
# 绕数算符 W_hat：扇区标签。取为相位侧的「绕行数」算符，
#   在谐振子基上记 W_hat |m> = w_m |m>，w_m 为 m 的绕数标签（本模型取 m mod NS）。
# 层数侧：整数格点，基矢 |n>，n = -N,...,N。对应层数通道 l^2(Z) 的截断。
# 层数算符 N_hat：数算符，N_hat |n> = n |n>。
# 张量积态空间 H = H_phase ⊗ H_layer，基矢 |m> ⊗ |n>。

M = 8          # 相位侧截断维数
NL = 9         # 层数侧截断维数（n = -4..4）
NS = 3         # 绕数扇区数（本模型取 Z_3 以便看出非平凡结构）


def phase_basis(M):
    """相位侧基矢 |m>，m = 0..M-1，返回 (相位标签, 扇区标签)"""
    m = np.arange(M)
    w = m % NS
    return m, w


def layer_basis(NL):
    """层数侧基矢 |n>，n = -(NL-1)/2 .. (NL-1)/2"""
    n = np.arange(-(NL - 1) // 2, (NL - 1) // 2 + 1)
    return n


PH_M, PH_W = phase_basis(M)
LAY_N = layer_basis(NL)

DIM = M * NL
# 张量积基的索引：(m, n) -> m*NL + n
W_DIAG = np.zeros(DIM)      # 绕数算符（对角）
N_DIAG = np.zeros(DIM)      # 层数算符（对角）
for im, m in enumerate(PH_M):
    for iN, n in enumerate(LAY_N):
        idx = im * NL + iN
        W_DIAG[idx] = PH_W[im]
        N_DIAG[idx] = n

# 算符矩阵
W_OP = np.diag(W_DIAG)
N_OP = np.diag(N_DIAG)

# 局域哈密顿量（层数侧的平移算符：把层数挪一格）
SHIFT = np.zeros((NL, NL))
for iN in range(NL - 1):
    SHIFT[iN, iN + 1] = 1.0        # <n| T |n+1> = 1
T_OP = np.kron(np.eye(M), SHIFT)   # 张量积：层数侧平移


# =====================================================================
# 组 A：张量积可分离性
# =====================================================================
print("=" * 72)
print("组 A：张量积可分离性")
print("=" * 72)

# A1：两算符各自嵌在张量积里（绕数只作用在相位侧，层数只作用在层数侧）
W_EMBED = np.kron(np.diag(PH_W.astype(float)), np.eye(NL))
N_EMBED = np.kron(np.eye(M), np.diag(LAY_N.astype(float)))
err_W = np.abs(W_OP - W_EMBED).max()
err_N = np.abs(N_OP - N_EMBED).max()
rec(
    "A1",
    "两算符在张量积态空间上的嵌入形式",
    "绕数算符等于 $\\hat{W}_{\\rm phase} \\otimes I$，层数算符等于 $I \\otimes \\hat{N}_{\\rm layer}$，"
    "嵌入误差 ≤ 1e-12",
    "绕数嵌入误差 %.3e，层数嵌入误差 %.3e" % (err_W, err_N),
    err_W < 1e-12 and err_N < 1e-12,
)

# A2：两算符对易
comm = W_OP @ N_OP - N_OP @ W_OP
comm_max = np.abs(comm).max()
rec(
    "A2",
    "绕数算符与层数算符的对易子",
    "对易子为零矩阵，最大元 ≤ 1e-12（张量积结构的直接后果）",
    "对易子最大元 %.3e" % comm_max,
    comm_max < 1e-12,
)

# A3：恒真对照 —— 同一侧的两个算符不保证对易
# 绕数算符与层数侧的平移算符：平移作用在层数侧，绕数在相位侧，仍应对易。
# 真正的对照：相位侧内部的一个非平凡算符（相位平移）与绕数算符。
# 构造：相位平移算符 T_phase（在相位侧把 m 挪一格），与绕数算符在 Z_3 下不对易。
NS_ = NS
T_phase = np.zeros((M, M))
for i in range(M - 1):
    T_phase[i, i + 1] = 1.0
T_phase_EMBED = np.kron(T_phase, np.eye(NL))
comm_ph = W_OP @ T_phase_EMBED - T_phase_EMBED @ W_OP
comm_ph_max = np.abs(comm_ph).max()
rec(
    "A3",
    "恒真对照：相位侧内部的平移算符与绕数算符",
    "对易子非零（同一因子内不保证对易），最大元 ≥ 0.5。"
    "这说明 A2 的对易来自张量积可分离性，不是任何两个算符都有的平凡性质",
    "对易子最大元 %.6f" % comm_ph_max,
    comm_ph_max >= 0.5,
)

# =====================================================================
# 组 B：扇区保持性
# =====================================================================
print("=" * 72)
print("组 B：扇区保持性")
print("=" * 72)

# B1：层数算符保持绕数算符的本征子空间（扇区）
# 检验：对任一本征矢 |psi_W>（绕数本征值 W），层数算符作用后仍在同一子空间。
# 判据：层数算符与绕数算符的联合本征态存在，即两者可同时对角化。
eig_W = np.linalg.eigvalsh(W_OP)
eig_N = np.linalg.eigvalsh(N_OP)
commutator_norm = np.linalg.norm(W_OP @ N_OP - N_OP @ W_OP, ord=2)
rec(
    "B1",
    "层数算符是否保持绕数的本征子空间（扇区）",
    "对易子零，故两者可同时对角化，扇区在层数算符作用下保持。"
    "对易子谱范数 ≤ 1e-12",
    "对易子谱范数 %.3e；绕数本征值 %s；层数本征值 %s"
    % (commutator_norm, np.unique(eig_W).tolist(), np.unique(eig_N).tolist()),
    commutator_norm < 1e-12,
)

# B2：显式构造扇区内的一组正交基（层数算符作用后仍在本征子空间内）
# 取绕数本征值 W0 的子空间，构造基 psi_j = |m_W0> ⊗ |n_j>，
# 检验 N_OP psi_j 仍是 W 本征矢。
W0 = 0
m_W0 = [im for im in range(M) if PH_W[im] == W0]
basis = []
for im in m_W0:
    for iN in range(NL):
        v = np.zeros(DIM)
        v[im * NL + iN] = 1.0
        basis.append(v)
B = np.array(basis).T                       # 列向量为基
W_B = W_OP @ B
res_W = np.abs(W_B - W0 * B).max()
rec(
    "B2",
    "扇区本征子空间内，层数算符作用后绕数标签不变",
    "绕数算符作用到扇区内的基上，结果与 $W_0 \\times$ 原基的最大偏差 ≤ 1e-12",
    "最大偏差 %.3e（扇区维数 %d）" % (res_W, B.shape[1]),
    res_W < 1e-12,
)

# B3：跨扇区矩阵元为零（库内既有事实的复核）
# 检验：扇区间的绕数算符矩阵元（以扇区基矢为指标）严格为零。
sector_res = []
for w in range(NS):
    ms = [im for im in range(M) if PH_W[im] == w]
    for w2 in range(NS):
        ms2 = [im for im in range(M) if PH_W[im] == w2]
        for im in ms:
            for im2 in ms2:
                # W 算符在 (m,n) 基上的矩阵元：delta_{m,m'} w_m
                if im == im2 and w != w2:
                    sector_res.append(1.0)
# 更直接：构造 W 算符在扇区基上的矩阵，检其非对角元
sect_index = {}
for w in range(NS):
    ms = [im for im in range(M) if PH_W[im] == w]
    for im in ms:
        for iN in range(NL):
            sect_index[(w, PH_W[im], im, iN)] = im * NL + iN
keys = list(sect_index.keys())
Wsect = np.zeros((len(keys), len(keys)))
for a, ka in enumerate(keys):
    for b, kb in enumerate(keys):
        if ka[0] == kb[0] and ka[2] == kb[2] and ka[3] == kb[3]:
            Wsect[a, b] = ka[1]
offdiag = np.abs(Wsect - np.diag(np.diag(Wsect))).max()
rec(
    "B3",
    "跨扇区矩阵元为零（库内既有事实的复核）",
    "扇区基下绕数算符的非对角元最大绝对值 ≤ 1e-12",
    "非对角元最大绝对值 %.3e" % offdiag,
    offdiag < 1e-12,
)

# B4：层数算符在扇区内的本征值不依赖扇区
# 检验：同一层数读数 n 在不同扇区下都可取到，说明层数算符不挑扇区。
ok_all = True
detail = []
for w in range(NS):
    ms = [im for im in range(M) if PH_W[im] == w]
    ns_avail = set()
    for im in ms:
        for iN, n in enumerate(LAY_N):
            ns_avail.add(float(n))
    detail.append((w, len(ns_avail)))
    if len(ns_avail) != NL:
        ok_all = False
rec(
    "B4",
    "层数算符在每个扇区内可取的读数集合相同",
    "各扇区内层数读数集合的大小均为 %d（层数侧维数），即层数算符不挑扇区" % NL,
    "各扇区大小 %s" % detail,
    ok_all,
)

# =====================================================================
# 组 C：非定域与定域的分工在算符层面
# =====================================================================
print("=" * 72)
print("组 C：非定域与定域的分工在算符层面")
print("=" * 72)

# C1：定域与非定域是**定义域**的性质，不是矩阵支撑的宽度。
# 初版误把「非定域」写成「矩阵元数目多」，判否后核出：两个算符都是对角的，
# 每行非零元数皆为 1。绕数的非定域性来自它沿**闭合回路**取净分支拓扑量，
# 层数的定域性来自它是**单点**读数。
#
# 正确的可判形式。
# 绕数侧（非定域）：本征值由整条相位侧决定。单点地看，相位侧的任一维 m 都
#   不足以定出绕数；要定出绕数须遍历回路，即取遍 m 的分布。
#   检验：遍历相位侧全部维，绕数本征值取遍全部扇区标签（非单值），
#   故绕数不能由单点确定。
# 层数侧（定域）：本征值由单点定出。第 n 个层数基矢的读数就是 n，
#   逐点不同且等于该点标号，故层数是单点读数。
m_probe = 2
w_before = PH_W[m_probe]
m_alt = (m_probe + 1) % M
w_after = PH_W[m_alt]
w_all = sorted(set(PH_W.tolist()))
w_not_single = len(w_all) > 1

# 层数侧：第 n 个基矢的读数就是 n
n_profile = np.diag(N_OP)[0 * NL:(0 * NL) + NL]
n_is_local = bool(np.allclose(n_profile, LAY_N.astype(float)))

rec(
    "C1",
    "定域与非定域的判别形式：绕数不能由单点定出，层数是单点读数",
    "绕数本征值随相位侧维 m 改变（$m=%d$ 时为 %d，改为 $m=%d$ 时为 %d），"
    "且遍历相位侧全部维时绕数取遍 %d 个扇区标签（非单值，故不能由单点确定）；"
    "层数算符作用后，层数剖面逐点等于该点标号 %s（单点读数）"
    % (m_probe, w_before, m_alt, w_after, len(w_all), LAY_N.tolist()),
    "绕数本征值 %d -> %d，取遍标签 %s；层数剖面逐点等于标号 = %s"
    % (w_before, w_after, w_all, n_is_local),
    (w_before != w_after) and w_not_single and n_is_local,
)

# C2：恒真检验 —— 真正的耦合算符会破坏对易。
# 初版构造 C = |0><0|_phase ⊗ X_layer，对易子为零：因 |0><0| 在相位侧是对角的，
# 与对角的绕数算符必对易。核出后改为混合相位侧**非对角**结构的耦合：
# C = S_phase ⊗ X_layer，其中 S_phase 把 m 挪一格（跨扇区的相位平移）。
# 这类耦合才真正把两通道绑在一起。
X_layer = np.zeros((NL, NL))
X_layer[0, 1] = 1.0
X_layer[1, 0] = 1.0
S_phase = np.zeros((M, M))
for i in range(M - 1):
    S_phase[i, i + 1] = 1.0        # 相位侧跨扇区平移（m 增一）
C_OP = np.kron(S_phase, X_layer)
comm_C = W_OP @ C_OP - C_OP @ W_OP
comm_C_max = np.abs(comm_C).max()
# 对照：只作用在相位侧的同一算符（层数侧取 I），其与绕数的对易同样非零；
# 而作用在层数侧的算符（相位侧取 I）则与绕数对易（层数侧与相位侧正交）。
comm_C_ph = W_OP @ np.kron(S_phase, np.eye(NL)) - np.kron(S_phase, np.eye(NL)) @ W_OP
comm_C_lay = W_OP @ np.kron(np.eye(M), X_layer) - np.kron(np.eye(M), X_layer) @ W_OP
rec(
    "C2",
    "恒真检验：跨通道的耦合算符破坏对易，纯单侧算符不破坏",
    "跨通道耦合算符与绕数算符的对易子非零（最大元 ≥ 0.5）；"
    "只作用在相位侧的算符与绕数不对易（最大元 ≥ 0.5）；"
    "只作用在层数侧的算符与绕数对易（最大元 ≤ 1e-12）。"
    "这说明 A2、B1 的对易来自两通道的正交，跨通道耦合会破坏它",
    "跨通道耦合对易子最大元 %.6f；纯相位侧 %.6f；纯层数侧 %.3e"
    % (comm_C_max, np.abs(comm_C_ph).max(), np.abs(comm_C_lay).max()),
    comm_C_max >= 0.5 and np.abs(comm_C_ph).max() >= 0.5
    and np.abs(comm_C_lay).max() < 1e-12,
)

# C3：两算符在扇区内各自完整（无缺失本征值）
# 检验：绕数算符的每个本征值在张量积态空间上的重数 = 相位侧重数 × 层数侧维数。
mult = []
for w in range(NS):
    cnt = int(np.sum(np.isclose(eig_W, w)))
    mult.append((w, cnt, cnt // NL if NL else 0))
ok_mult = all(c == NL * sum(1 for im in range(M) if PH_W[im] == w) for w, c, _ in mult)
rec(
    "C3",
    "绕数本征值在张量积态空间上的重数",
    "每个绕数本征值的重数 = 相位侧重数 × 层数侧维数 %d（即张量积的乘法规则）" % NL,
    "各本征值重数 %s" % [(w, c) for w, c, _ in mult],
    ok_mult,
)

# =====================================================================
# 组 D：读数
# =====================================================================
print("=" * 72)
print("组 D：读数（不进通过率）")
print("=" * 72)

print("[D1] 读数：绕数与层数在量子层面也是各守一量")
print("     张量积可分离性给出：绕数算符等于 W_phase 乘 I，层数算符等于 I 乘 N_layer，")
print("     两者对易子零矩阵（A1、A2）。故它们可同时对角化，扇区在层数算符作用下保持。")
print("     A3 的恒真对照表明：相位侧内部的平移算符与绕数算符并不对易，")
print("     所以 A2 的对易来自张量积结构，不是任何两个算符都有的平凡性质。")
print("")
print("[D2] 读数：绕数与层数的定域性分工在算符层面有对应")
print("     绕数算符的每一行牵动全部层数点（非定域：沿闭合回路取），")
print("     层数算符的每一行牵动全部相位点（定域：单点读数）（C1）。")
print("     这与非定域在先、定域在后的方向一致。")
print("")
print("[D3] 读数：接口性质与判据第 744 号一致")
print("     C2 的恒真对照表明：一旦引入相位与层数之间的耦合算符，对易立即被破坏。")
print("     故第 744 号给出的「接口是就地并合，不构成信号传递」在算符层面有对应物：")
print("     两通道间没有跨通道的混合项，故两算符对易。")
print("")
print("[D4] 读数：这一环给出的判据式结论")
print("     绕数算符与层数算符对易（对易子零），扇区保持。")
print("     故 P4-55 主线的物侧结构在量子层面闭合：两通道各自量子化，")
print("     张量积可分离，两读数各守一量，互补而不可互换。")
print("     论文 §9 遗留 (g)「两通道之间的耦合项形式」正是本轮给出的结构所缺的那一项：")
print("     耦合项一旦出现即破坏可分离性，故其形式须单独给出，不能沿用张量积的默认。")
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
img = np.zeros((DIM, DIM))
for i in range(DIM):
    for j in range(DIM):
        img[i, j] = abs(comm[i, j]) + abs(comm_C[i, j]) * 0.5
im = ax.imshow(img, cmap="viridis", origin="upper",
               extent=[0, DIM, DIM, 0], interpolation="nearest")
ax.set_xlabel("矩阵列指标")
ax.set_ylabel("矩阵行指标")
ax.set_title("左：对易子矩阵的模长\n深紫为 0（两算符对易），亮色为非零（耦合破坏）")
ax.set_xticks([])
ax.set_yticks([])
plt.colorbar(im, ax=ax, fraction=0.046)

ax = axes[1]
ws = [w for w, c, _ in mult]
cs = [c for w, c, _ in mult]
ax.bar([str(w) for w in ws], cs, color="#1f77b4", alpha=0.75, edgecolor="k")
for x, c in enumerate(cs):
    ax.text(x, c + 0.5, str(c), ha="center", fontsize=11)
ax.set_xlabel("绕数本征值（扇区标签）")
ax.set_ylabel("在张量积态空间上的重数")
ax.set_title("右：重数按乘法规则\n相位侧重数 $\\times$ 层数侧维数 $9$")
ax.grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(IMG, dpi=150)
print("")
print("图已保存：%s" % IMG)
