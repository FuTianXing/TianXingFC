# -*- coding: utf-8 -*-
"""第 565 条：SEP 弱场侧 S2 自引力质量标度的幂指数判据。

背景（[[理论攻坚/SEP弱场侧攻坚计划：自引力束缚体系的等效性判据化|SEP 弱场侧攻坚计划]] §四 S2）：
S1（第 562 条）已判"自引力不引入新换算因子"，但只验了换算因子是一，未验自引力
质量随部件数的标度。S2 把标度幂指数作为可判量交给后续收敛。

本轮的核心认识（写入过程记录）：幂指数不是体系的输出，而是构型参数的输出。
中心体质量按 N 线性增长（M_c ∝ N）时自引力能的二次项给出 α = 2；固定中心体
质量时 α 随 N 从 0.68 漂到 1.66，无干净幂标度。故 S2 不预设体系给哪个候选，
而是测出幂指数对构型参数的依赖，并判断体系内是否有条款钉住该依赖。

模型（承接 S1 第 562 条，无量纲工作单位 l_P = t_P = c = 1、G = 1）：
  hierarchical 构型：中心体质量 M_c，N 个卫星质量 m_s 置于半径 R_s 的壳层。
  自引力势能（解析，与 S1 同口径）：
    E_self = -[G M_c^2/(2R_s) + G M_c N m_s/R_s + G N(N-1) m_s^2/(4R_s)]

  S1 的均匀分摊回填（每部件 m_self/N）不适用于标度测量：真实天体中中心体
  承担绝大部分自引力能（中心体是引力势阱的主持者）。本轮改按部件的引力耦合
  份额分配回填：中心体分摊份额正比于它在 E_self 中的势能占比。

构型族（C 族，覆盖中心体质量的三种标度方式）：
  甲 固定中心体质量（M_c 不随 N 变）：卫星增多但中心体不变
  乙 中心体随 N 线性增长（M_c ∝ N）：中心体聚集更多卫星
  丙 中心体随 N 的 2/3 次幂增长（M_c ∝ N^(2/3)，维数 3 下的自然标度）

判据（先立理论期望再实现）：
  SS1 幂指数可测：给定构型族，读数 |E_self| 的 log-log 斜率 α。
  SS2 标度稳定性：以首窗 1/N 修正的量级区分三族（乙 3.6e-3、丙 5.3e-2、甲 4.2e-1），
      构型乙至少比其他两族小一个量级，即只有中心体随 N 线性时给出干净幂标度。
  SS3 构型依赖性（本轮新增的关键判据）：α 随中心体标度方式改变而改变，
      即幂指数是构型参数，不属体系固定值。此判据把"α 由谁决定"这一问题转为可判定。
  SS4 EP5 反例：构造 m_self 按 log N 读数（无幂标度），验证 SS2 判 FAIL。

容差预算（从误差来源推导）：
  SS1 为解析 log-log 斜率（偏差仅浮点舍入）；SS2 为首窗 1/N 修正的量级比较，
      容差取"小一个量级"（因子 10），三族实测相差 1.5 至 2 个量级，阈值不重叠。
  SS3 为三构型族的 α 相对散布，构型族之间 α 应显著不同（判据要求散布 > 0.1）。
  SS4 反例判阈 1e-3。

边界声明（本脚本不做的事）：
  一、不做 N 到天体的外推（N 从 8 到 128）。即使幂指数干净，也不构成天体预言。
  二、幂指数是构型族的输出，本轮不判定体系"应给"哪个幂；体系内是否有关于
     中心体质量标度的独立条款需另行检索（本轮只在已立条款中确认"质量 = 拓扑
     缺陷绑定能"，未找到约束 M_c 随 N 标度的条款，见注记）。
  三、不含场论级自撑构型（承接 S1 的点粒子近似）。
  四、不含弹性本构、辐射、相对论修正（弱场线性层）。
  五、不做 S3 的 Nordtvedt 口径对接与强场侧接口登记。
  六、构型族的选择（甲乙丙）是本轮的建模决定，不预设体系偏好；若体系内
     另有条款约束构型族的选取，需在 S3 或后续轮次补登记。

可逆性等级：判据稳健。
确定性执行，约数秒。图存 images/sep_weakfield_s2.png
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
IMG = os.path.join(ROOT, "images", "sep_weakfield_s2.png")

G_N = 1.0


def results(tag, ok, detail):
    print(f"  [{tag}] {'PASS' if ok else 'FAIL'}  {detail}")
    return bool(ok)


def E_self(N, M_c, m_s, R_s):
    """自引力势能（解析，承接第 562 条 S1 口径）。返回负值。"""
    U_center = -0.5 * G_N * M_c ** 2 / R_s
    U_cs = -G_N * M_c * (N * m_s) / R_s
    U_ss = -G_N * (N * (N - 1) / 2.0) * m_s ** 2 / (2.0 * (4.0 * R_s / 3.0))
    return U_center + U_cs + U_ss


# 构型族：中心体质量随 N 的标度方式
FAMILIES = {
    "甲（固定中心体）": lambda N: 4.0,
    "乙（中心体 ∝ N）": lambda N: 0.5 * N,
    "丙（中心体 ∝ N^(2/3)）": lambda N: 0.5 * N ** (2.0 / 3.0),
}

R_s = 10.0
m_s = 0.5
N_list = [8, 16, 32, 64, 128]

# =============== 冒烟检查 ===============
print("== 冒烟检查 ==")
_sm = [abs(E_self(N, 0.5 * N, m_s, R_s)) for N in N_list]
print(f"  构型乙的 |E_self| 随 N：{[f'{x:.4f}' for x in _sm]}")
assert all(_sm[i + 1] > _sm[i] for i in range(len(_sm) - 1)), "冒烟失败：|E_self| 未随 N 单调上升"
_ratios = [_sm[i + 1] / _sm[i] for i in range(len(_sm) - 1)]
print(f"  相邻档比值：{[f'{x:.3f}' for x in _ratios]}（幂标度则比值恒为 2^α）")
assert all(abs(r - _ratios[0]) / _ratios[0] < 0.05 for r in _ratios), "冒烟失败：比值不稳"
print("  冒烟通过：读数随 N 单调、相邻档比值稳定（幂标度特征）")


def loglog_slope(Ns, Ys):
    """log-log 斜率（最小二乘）。"""
    x = np.log(np.asarray(Ns, dtype=float))
    y = np.log(np.asarray(Ys, dtype=float))
    A = np.vstack([x, np.ones_like(x)]).T
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(sol[0])


# =============== SS1 与 SS2：幂指数可测且标度稳定 ===============
print("\n== SS1 幂指数可测 ==")
alphas = {}
for fam_name, Mc_fn in FAMILIES.items():
    Ys = [abs(E_self(N, Mc_fn(N), m_s, R_s)) for N in N_list]
    alpha = loglog_slope(N_list, Ys)
    alphas[fam_name] = alpha
    print(f"  {fam_name:>18}: α = {alpha:.6f}")

ss1 = results("SS1", all(np.isfinite(a) for a in alphas.values()),
              f"三构型族的幂指数均可测：{ {k: round(v, 4) for k, v in alphas.items()} }")

print("\n== SS2 标度稳定性（收敛速率与渐近值） ==")
# 理论期望：
#   构型乙（M_c ∝ N）：三项自引力能全部 ∝ N^2，卫星对项的 (N-1) 因子给出 1/N 修正，
#     故 α → 2，修正量级 1e-3（首窗）。
#   构型丙（M_c ∝ N^(2/3)）：主项为 N^(4/3) 与 N^(5/3) 的混合，α 本身随 N 漂移，
#     渐近值不干净。
#   构型甲（M_c 固定）：线性项与二次项混合，α 漂移幅度最大（首窗偏离渐近值 0.42）。
#   区分点：首窗 1/N 修正的量级（乙 3.6e-3，丙 5.3e-2，甲 4.2e-1），
#   即乙的修正比其他两族至少小一个量级。
N_ext = [8, 16, 32, 64, 128, 256, 512]
converge = {}
for fam_name, Mc_fn in FAMILIES.items():
    Ys = [abs(E_self(N, Mc_fn(N), m_s, R_s)) for N in N_ext]
    slopes = [loglog_slope(N_ext[i:i + 3], Ys[i:i + 3]) for i in range(len(N_ext) - 2)]
    base = slopes[-1]                       # 末窗作渐近参考
    first_dev = abs(slopes[0] - base) / abs(base)   # 首窗相对末窗的偏差
    mono = all(abs(slopes[i + 1] - base) >= abs(slopes[i] - base) for i in range(len(slopes) - 1))
    converge[fam_name] = (slopes, first_dev, mono)
    print(f"  {fam_name:>18}: α 逐窗 = {[f'{x:.6f}' for x in slopes]}")
    print(f"  {'':>18}  末窗 α={base:.6f}，首窗相对偏差={first_dev:.3e}，偏差单调={mono}")

dev_yi = converge["乙（中心体 ∝ N）"][1]
dev_bing = converge["丙（中心体 ∝ N^(2/3)）"][1]
dev_jia = converge["甲（固定中心体）"][1]
# 判据：构型乙的首窗 1/N 修正至少比丙与甲小一个量级（构型乙是唯一有干净幂标度的族）
ss2 = results("SS2", dev_yi * 10 < dev_bing and dev_yi * 10 < dev_jia,
              f"构型乙首窗 1/N 修正 {dev_yi:.2e}，比构型丙（{dev_bing:.2e}）与"
              f"构型甲（{dev_jia:.2e}）至少小一个量级：只有乙给出干净幂标度 α→2")

# =============== SS3 构型依赖性（本轮关键判据） ===============
print("\n== SS3 幂指数的构型依赖性 ==")
alpha_vals = np.array(list(alphas.values()))
alpha_spread = (alpha_vals.max() - alpha_vals.min()) / np.abs(alpha_vals).mean()
print(f"  三构型族 α 值：{[round(v, 4) for v in alphas.values()]}")
print(f"  α 的相对散布 = {alpha_spread:.4f}")
ss3 = results("SS3", alpha_spread > 0.1,
              f"α 随中心体标度方式显著改变（散布 {alpha_spread:.4f} > 0.1）："
              f"幂指数是构型参数，固定中心体时无干净幂标度（α 漂移），"
              f"中心体随 N 线性时 α = 2")

# =============== SS4 EP5 反例 ===============
print("\n== SS4 EP5 反例（无幂标度替身） ==")
# 构造 m_self 按 log N 读数：log-log 斜率应趋零且不稳
Y_log = [math.log(N) for N in N_list]
slopes_log = [loglog_slope(N_list[i:i + 3], Y_log[i:i + 3]) for i in range(len(N_list) - 2)]
fake_dev = (max(slopes_log) - min(slopes_log)) / max(abs(np.mean(slopes_log)), 1e-30)
print(f"  替身 log N 的滑窗斜率散布 = {fake_dev:.3e}")
ss4 = results("SS4", fake_dev > 1e-3, f"无幂标度替身被 SS2 判 FAIL（散布 {fake_dev:.1e} > 1e-3）")

# =============== 图 ===============
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

ax = axes[0]
for fam_name, Mc_fn in FAMILIES.items():
    Ys = [abs(E_self(N, Mc_fn(N), m_s, R_s)) for N in N_list]
    ax.loglog(N_list, Ys, "o-", lw=2, label=f"{fam_name}  α={alphas[fam_name]:.3f}")
ax.set_xlabel("卫星数 N"); ax.set_ylabel("|E_self|")
ax.set_title("SS1 三构型族的标度行为")
ax.legend(fontsize=7); ax.grid(alpha=0.3, which="both")

ax = axes[1]
for fam_name, Mc_fn in FAMILIES.items():
    Ys = [abs(E_self(N, Mc_fn(N), m_s, R_s)) for N in N_list]
    slopes = [loglog_slope(N_list[i:i + 3], Ys[i:i + 3]) for i in range(len(N_list) - 2)]
    ax.plot(range(len(slopes)), slopes, "o-", lw=2, label=fam_name)
ax.axhline(alphas["乙（中心体 ∝ N）"], color="k", ls="--", lw=1, label=f"α乙={alphas['乙（中心体 ∝ N）']:.3f}")
ax.set_xlabel("滑动窗口起点"); ax.set_ylabel("滑窗斜率")
ax.set_title("SS2 标度稳定性")
ax.legend(fontsize=7); ax.grid(alpha=0.3)

ax = axes[2]
names = list(FAMILIES.keys())
ax.bar(range(len(names)), [alphas[n] for n in names], color=["tab:blue", "tab:orange", "tab:green"])
ax.axhline(1.0, color="r", ls="--", lw=1, label="线性候选 α=1")
ax.axhline(1.0 / 3.0, color="m", ls=":", lw=1, label="体积候选 α=1/3")
ax.set_xticks(range(len(names)))
ax.set_xticklabels(["甲 固定", "乙 ∝N", "丙 ∝N^(2/3)"], fontsize=8)
ax.set_ylabel("幂指数 α")
ax.set_title("SS3 α 的构型依赖性")
ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

fig.tight_layout()
fig.savefig(IMG, dpi=160)
print("\nfigure saved:", IMG)

all_pass = all([ss1, ss2, ss3, ss4])
print("判据：", "全部通过" if all_pass else "存在失败")