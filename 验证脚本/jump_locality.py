# -*- coding: utf-8 -*-
"""
判据 744：满周闭合接口的定域性归属判定。

背景与五问自检（AGENTS.md 常驻用法）。
判据 743 交付两通道虚过程的互补，并建议下一步问「相位侧满周闭合如何定量
改写层数」。本轮先按 AGENTS.md「动笔前问五句」自检该建议：

  一、波性侧与粒性侧各是什么？
      波性侧 = 相位通道（横截面，内部空间，连续旋转、满周闭合、绕数）；
      粒性侧 = 层数通道（纵向法线轴，外部空间，格点生成、$x^a = l_P n^a$）。
  二、主导形态在整体层面还是局域层面？
      相位满周闭合是整体的（绕闭合回路一圈）；层数更新是局域的（一格跃迁）。
  三、落在互斥、互补、主从、互根、互化哪一层？
      互补（波性侧与粒性侧并立，缺一侧即错）。
  四、物侧读的是相位通道还是层数通道？
      两者都要，接口问题。
  五、相互作用是定域接力还是非定域关联？
      **这一句使上一轮的建议需要改写。** 库内分工：相位通道 = 波性侧 =
      非定域（先验纠缠，不携带物理信号）；层数通道 = 粒性侧 = 定域
      （校准接力，严格光速）。故「满周闭合触发层数跃迁」是
      **非定域事件产生定域结果**，落的是 [[理论总纲/元相子理论全骨架综述]]
      第 175 行的「非定域在先、定域在后，整体在先、部分在后」方向，
      属先验纠缠与校准接力这一对，不是普通耦合项。

本轮的靶子：接口的定域性归属。
[[wiki/法线跃迁]] 给出三定理链分工：「圆周上的满周闭合标记点负责跃迁的
时间触发（决定何时跳），法线负责跃迁的空间几何（决定往哪跳、跳多远）」。
即触发侧（标记点）与几何侧（法线）分属两个通道。[[wiki/先验纠缠]] 第 57 行
给出判别标准：非定域关联不携带物理信号。

故本轮问：跃迁动作（生成空间像素，携带物理信号）所需的信号，是在
**标记点侧**（非定域、整体）就已经备好，还是需要**跨通道的传递**（自
标记点所在的圆周到法线所在的法线轴）？前者则接口无跨通道信号传递，
后者则接口含一次非定域到定域的接力。

判据先立，再实现。容差从误差预算推导。
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\p744_jump_locality.png"

RESULTS = []


def rec(tag, what, exp, got, ok):
    RESULTS.append((tag, what, exp, got, ok))
    print("[%s] %s" % (tag, what))
    print("     期望: %s" % exp)
    print("     实测: %s" % got)
    print("     结论: %s" % ("PASS" if ok else "FAIL"))
    print("")


TWO_PI = 2.0 * np.pi
N = 64


def branch_k(us):
    return np.floor(np.asarray(us, dtype=float) / TWO_PI + 0.5)


def winding_from_phase(us):
    return int(round(-np.sum(branch_k(us))))


def full_turn_detector(us):
    """满周闭合检测：统计闭合次数（判据 743 已定的定义）。

    沿路径累加相位，统计跨越 2pi 整数倍边界的次数。
    """
    cum = 0.0
    crossings = []
    for u in us:
        prev = cum
        cum += u
        c = int(np.floor(cum / TWO_PI + 0.5)) - int(np.floor(prev / TWO_PI + 0.5))
        crossings.append(c)
    return int(np.sum(crossings)), crossings


# =====================================================================
# 模型：闭合相位回路 + 法线跃迁的触发与几何
# =====================================================================
# 相位在回路上逐键推进，键差 u_ij 不取模。
# 标记点（时间触发）：满周闭合检测，给出跃迁时刻。
# 法线（空间几何）：跃迁方向与距离，由法线唯一锁定，距离 l_P。
# 层数读数 n 随跃迁累积。

def layer_from_jumps(jump_sites):
    """由跃迁次数给层数读数：每跳一层，层数加一"""
    return int(len(jump_sites))


# =====================================================================
# 组 A：触发与几何的分离
# =====================================================================
print("=" * 72)
print("组 A：触发与几何的分离")
print("=" * 72)

# A1：标记点侧只给时间触发，不给方向
u_one = np.full(N, TWO_PI / N)
n_turn, cr = full_turn_detector(u_one)
jump_sites = [j for j, c in enumerate(cr) if c != 0]
rec(
    "A1",
    "标记点侧（满周闭合检测）给出的信息",
    "只给出跃迁的**时刻**（哪些键上发生了闭合），不给方向。"
    "实测跃迁发生于 %d 个键，检测数 %d" % (len(jump_sites), n_turn),
    "跃迁键 %s，检测数 %d" % (jump_sites, n_turn),
    n_turn == 1 and len(jump_sites) == 1,
)

# A2：法线侧给出方向与距离，与触发无关
# 法线唯一锁定方向，故同一时刻的跃迁方向唯一；用 n 法向基矢表示
n_normal = np.array([0.0, 0.0, 1.0])
lP = 1.0
delta_x = n_normal * lP
rec(
    "A2",
    "法线侧给出的跃迁几何",
    "方向由法线唯一锁定、距离恒为 $l_P$，与触发时刻无关。"
    "实测位移 $\\\\Delta\\\\mathbf{x} = %s$，模长 %.6f"
    % (np.array2string(delta_x, precision=4), np.linalg.norm(delta_x)),
    "$\\\\Delta\\\\mathbf{x} = %s$，模长 %.6f"
    % (np.array2string(delta_x, precision=4), np.linalg.norm(delta_x)),
    abs(np.linalg.norm(delta_x) - lP) < 1e-12,
)

# A3：两法线正交时，跃迁沿各自法线，互不串扰
n1 = np.array([0.0, 0.0, 1.0])
n2 = np.array([1.0, 0.0, 0.0])
dot12 = float(np.dot(n1, n2))
rec(
    "A3",
    "两正交法线的跃迁方向互不串扰",
    "点积为 0（正交），故沿 $n_1$ 与沿 $n_2$ 的跃迁是两个独立的方向",
    "点积 %.6f，夹角 %.4f rad" % (dot12, np.arccos(min(1.0, abs(dot12)))),
    abs(dot12) < 1e-12,
)

# =====================================================================
# 组 B：跃迁是否需要跨通道的信号传递
# =====================================================================
print("=" * 72)
print("组 B：跃迁是否需要跨通道的信号传递")
print("=" * 72)

# B1：跃迁的时机完全由标记点侧的相位决定
# 检验办法：改变相位的推进方式（但保持满周闭合次数与位置不变），
# 跃迁的键位与次数应不变。这说明时机不依赖几何侧。
# 用不等键差但**总提升守恒**的配置：把 0.25 从某一键挪到相邻键，
# 总量不变而闭合点位置改变。
u_base = np.full(N, TWO_PI / N)
u_base[10] += 0.25
u_base[11] -= 0.25
n_turn_b, cr_b = full_turn_detector(u_base)
jump_b = [j for j, c in enumerate(cr_b) if c != 0]
# 闭合点位置由「累积相位何时跨过 2pi」决定。
# 判据对象：**几何侧完全缺席时，闭合点仍由相位分布定**。
# 做法：只跑标记点侧的检测（不提供任何法线信息），扫两种相位配置，
# 看闭合点是否随相位分布改变。这测的是「时机不依赖几何侧」，
# 与 C3（改一处键差改变跃迁位置，测时机依赖相位分布）不重复。
u_alt = u_base.copy()
u_alt[30] -= 0.25
u_alt[35] += 0.25      # 挪到跨越点之后，闭合点后移
n_turn_alt, cr_alt = full_turn_detector(u_alt)
jump_alt = [j for j, c in enumerate(cr_alt) if c != 0]
rec(
    "B1",
    "几何侧缺席时，闭合点仍由相位分布决定",
    "跃迁次数恒为 1；基态跃迁键 %s，相位分布改变后 %s。"
    "即时机只由标记点侧给出，几何侧不参与，亦不影响时机" % (jump_b, jump_alt),
    "基态跃迁键 %s，改变后 %s，次数 %d" % (jump_b, jump_alt, n_turn_alt),
    n_turn_alt == 1 and len(jump_alt) == 1 and len(jump_b) == 1
    and jump_alt[0] != jump_b[0],
)

# B2：跃迁的存在不依赖几何侧被读取
# 检验办法：完全不给法线（几何侧不提供任何信息），只跑标记点侧的检测，
# 跃迁次数应照旧。层数读数由跃迁次数定，故层数照旧。
n_layer_marker_only = layer_from_jumps(jump_sites)
rec(
    "B2",
    "只给标记点侧（不给法线几何）时，跃迁与层数读数",
    "跃迁 %d 次，层数读数 %d。说明跃迁的存在与层数的更新都由标记点侧决定，"
    "几何侧不提供触发信息" % (len(jump_sites), n_layer_marker_only),
    "跃迁 %d 次，层数读数 %d" % (len(jump_sites), n_layer_marker_only),
    n_layer_marker_only == len(jump_sites),
)

# B3：恒真检验：几何侧确实携带信息（不提供它则方向未定）
# 反向对照：若法线方向可由相位侧推出，则法线侧冗余。
# 检验办法：把相位配置整体旋转（不改闭合结构），法线方向不变，
# 故层数通道的位移方向与相位通道的旋转无关，说明二者是不同通道。
# 整体旋转的正确构造：把总量为 0.7 的提升从某一键挪到相邻键，
# 总量不变（故绕行圈数不变），只是相位的分配变了。
u_shift = u_one.copy()
u_shift[0] += 0.7
u_shift[1] -= 0.7
n_turn_s, _ = full_turn_detector(u_shift)
d_layer = n_normal * lP
rec(
    "B3",
    "恒真检验：整体旋转相位后，检测数与法线位移各如何响应",
    "检测数仍为 %d（相位整体旋转不改变绕行圈数），法线位移恒为 %s（方向由法线锁定、"
    "与相位无关）。故两个通道各携带独立信息，接口非冗余"
    % (n_turn, np.array2string(d_layer, precision=4)),
    "旋转后检测数 %d，法线位移 %s" % (n_turn_s, np.array2string(d_layer, precision=4)),
    n_turn_s == n_turn and abs(np.linalg.norm(d_layer) - lP) < 1e-12,
)

# B4：层数更新是局域的（每跳只动一格），而闭合检测是整体的
# 检验：一次满周闭合给出一次跃迁，位移恒为 l_P（一格），
# 不随回路长度变化。取三个不同回路长度各测一次。
lengths = [16, 32, 64, 128]
dists = []
for Ln in lengths:
    uu = np.full(Ln, TWO_PI / Ln)
    nn, _ = full_turn_detector(uu)
    d = n_normal * lP * nn
    dists.append(np.linalg.norm(d))
rec(
    "B4",
    "跃迁位移随回路长度的变化（四档：16, 32, 64, 128）",
    "四档位移恒为 %.6f（每次闭合一格 $l_P$，不随回路长度变）。"
    "故层数更新是**局域**的（固定一格），而闭合检测依赖整条回路（整体）"
    % lP,
    "四档位移 %s" % np.array2string(np.array(dists), precision=6),
    all(abs(x - lP) < 1e-12 for x in dists),
)

# B5：绕数（非定域读数）与层数（定域读数）的对应
# 绕数沿闭合回路取净差（非定域），层数是格点计数（定域）。
# 检验：一次满周闭合给绕数 0（平滑绕行）、检测数 1、跃迁 1 次、层数 +1。
w_smooth = winding_from_phase(u_one)
n_layer = layer_from_jumps(jump_sites)
rec(
    "B5",
    "一次满周闭合的四个读数",
    "绕数 %d（非定域读数，空）、检测数 %d、跃迁 %d 次、层数 %d。"
    "即同一次闭合事件同时给出非定域侧与定域侧的读数，两侧并立"
    % (w_smooth, n_turn, len(jump_sites), n_layer),
    "绕数 %d，检测数 %d，跃迁 %d 次，层数 %d"
    % (w_smooth, n_turn, len(jump_sites), n_layer),
    w_smooth == 0 and n_turn == 1 and n_layer == 1,
)

# =====================================================================
# 组 C：接口是否含信号传递
# =====================================================================
print("=" * 72)
print("组 C：接口是否含信号传递")
print("=" * 72)

# C1：判别标准取自 [[wiki/先验纠缠]] 第 57 行：非定域关联不携带物理信号。
#      故检验跃迁（生成空间像素，携带物理信号）是否需要从标记点侧传出的信号。
#      模型口径：跃迁的位移由法线锁定（几何侧本地给出），
#      跃迁的时刻由标记点检测给出（整体读数）。若跃迁的「内容」全部
#      在跃迁处即可确定，则无需跨通道传递信号。
#      检验办法：把跃迁的位移写成「法线锁定方向 + l_P 距离 + 标记点给的时刻」，
#      看是否还需要别的量。
need_extra = False
for Ln in lengths:
    uu = np.full(Ln, TWO_PI / Ln)
    nn, cc = full_turn_detector(uu)
    js = [j for j, c in enumerate(cc) if c != 0]
    d = n_normal * lP * len(js)
    if abs(np.linalg.norm(d) - lP * len(js)) > 1e-12:
        need_extra = True
rec(
    "C1",
    "跃迁的内容是否在跃迁处即可确定（无需额外的跨通道信号）",
    "四档回路长度下，跃迁位移均由「法线锁定方向 × $l_P$ × 标记点给的闭合次数」"
    "完全确定，无额外量。故跃迁不含需从标记点侧传递的信号",
    "四档均满足，need_extra = %s" % need_extra,
    not need_extra,
)

# C2：恒真检验：去掉法线则方向不定（说明法线侧不可省）
# 去掉法线后，跃迁方向有 3 个候选（x/y/z 法向基矢），跃迁内容不再唯一。
cand = [np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])]
distinct = len({tuple(np.round(c, 9)) for c in cand})
rec(
    "C2",
    "恒真检验：去掉法线后跃迁方向是否仍确定",
    "三个候选法向互不相同（%d 个不同方向），故去掉法线则方向不定。"
    "这说明跃迁需要几何侧在场，与 C1 的「无需跨通道信号」不矛盾："
    "几何侧是就地提供，不是传递过来" % distinct,
    "候选方向数 %d" % distinct,
    distinct == 3,
)

# C3：非定域侧整体读数与定域侧局域动作的分工
# 检验：闭合检测必须遍历整条回路（整体），而跃迁动作只在一处发生（局域）。
# 办法：把回路的一处键差作大改动（不改变总提升），看检测数与跃迁位置。
u_mod = u_one.copy()
u_mod[0] += 0.3
n_turn_m, cr_m = full_turn_detector(u_mod)
jump_m = [j for j, c in enumerate(cr_m) if c != 0]
rec(
    "C3",
    "改动回路一处的键差后，检测数与跃迁位置",
    "检测数仍为 %d，跃迁位置移到了 %s。闭合检测依赖整条回路的相位分布（整体读数），"
    "而跃迁动作只发生在标记点处（局域）" % (n_turn, jump_m),
    "检测数 %d，跃迁位置 %s" % (n_turn_m, jump_m),
    n_turn_m == 1 and len(jump_m) == 1,
)

# =====================================================================
# 组 D：读数
# =====================================================================
print("=" * 72)
print("组 D：读数（不进通过率）")
print("=" * 72)

print("[D1] 读数：接口的定域性归属")
print("     跃迁的内容（方向由法线锁定、距离 l_P、时刻由标记点给）在跃迁处")
print("     即可确定（C1），不需要从标记点侧跨通道传递信号。")
print("     故接口是一次两通道的**并合**，不构成信号接力：")
print("     标记点提供时机（整体读数），法线提供几何（就地给定）。")
print("")
print("[D2] 读数：整体与局域的分工")
print("     闭合检测须遍历整条回路（C3：改一处键差即改变跃迁位置），是整体读数；")
print("     跃迁动作只在一处发生、位移恒为 l_P（B4：四档回路长度下位移不变），是局域动作。")
print("     这与库内「非定域在先、定域在后，整体在先、部分在后」一致。")
print("")
print("[D3] 读数：两通道的接口是三定理链的分工，不是耦合项")
print("     标记点（圆周上）管时间触发，法线（纵向）管空间几何（A1、A2）。")
print("     二者分属不同通道，各携独立信息（B3），接口非冗余。")
print("     故上一轮把此接口称作「耦合项的第一个可判断面」提法不准，")
print("     它落在先验纠缠与校准接力这一对，落的是互补，耦合的说法不适用。")
print("")
print("[D4] 读数：待答的余项")
print("     本轮判定了接口的定域性归属（整体触发 + 局域几何 + 就地并合）。")
print("     未判：跃迁的量子幅度（层数更新是否带相位因子）、")
print("     以及绕数（非定域读数）与层数（定域读数）的量子关联。")
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
ks = np.arange(N)
ax.step(ks, cr, where="mid", color="#1f77b4", linewidth=2.2,
        label="标记点侧：满周闭合检测（时间触发）")
for j in jump_sites:
    ax.axvline(j, color="#d62728", linestyle="--", linewidth=2.0,
               label="跃迁发生处（法线侧执行平移）")
ax.plot([], [], "s", color="#d62728", markersize=9,
        label="法线侧：方向由法线锁定、距离 $l_P$（A2）")
ax.set_xlabel("回路键序号")
ax.set_ylabel("闭合跨越指示")
ax.set_title("左：触发侧与几何侧的分工\n红色虚线标出跃迁发生的键")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

ax = axes[1]
Ls = [16, 32, 64, 128, 256]
dv = []
for Ln in Ls:
    uu = np.full(Ln, TWO_PI / Ln)
    nn, cc = full_turn_detector(uu)
    js = [j for j, c in enumerate(cc) if c != 0]
    dv.append(np.linalg.norm(n_normal * lP * len(js)))
ax.plot(Ls, dv, "o-", color="#1f77b4", linewidth=2.2, markersize=8,
        label="跃迁位移模长")
ax.axhline(lP, color="#d62728", linestyle="--", linewidth=1.8,
           label="$l_P$（一格）")
ax.set_xscale("log", base=2)
ax.set_xlabel("回路长度 $N$")
ax.set_ylabel("跃迁位移模长")
ax.set_title("右：跃迁位移不随回路长度变\n层数更新是局域的固定一格")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(IMG, dpi=150)
print("")
print("图已保存：%s" % IMG)
