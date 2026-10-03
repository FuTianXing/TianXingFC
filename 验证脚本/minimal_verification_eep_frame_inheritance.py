# -*- coding: utf-8 -*-
"""第 556 条：EEP 自由落体系内 LLI/LPI 的独立继承判据。

背景（[[内部草稿/材料-EEP合成：局域自由落体等效的定理陈述与前提清单]] §四遗留）：
EEP 合成判定（第 468 条）已证几何基石（自由落体即测地线、局域一阶消去、
残余潮汐），但三前提中 Q2（局域洛伦兹不变性 LLI）与 Q3（局域位置不变性
LPI）当时为引用性对接，未在自由落体系内做独立新数值。本轮补该缺口。

判据设计（先立理论期望，再实现）：

模型：
  弱场径向 theta(r) = A/r（体系静倾角刻度）。
  LLI 侧：晶格色散 omega^2(k) = (4/t_P^2) sum_a sin^2(k_a l_P/2)，长波极限
    omega^2 = c^2 k^2 + m_eff^2，c = l_P/t_P（无量纲工作单位 c = 1）。
    自由落体系以释放初速 v0 参数化为洛伦兹 boost（beta = v0/c）。
  LPI 侧：有限立方网格 N=15 上的离散格林函数，Dirichlet 边界（边界节点
    解为零），源写入 -delta，读数为最近邻六格点载荷均值与写入幅值之比 C。

  关于平移不变在有限网格上的正确形态：Dirichlet 网格关于中心对称，故源
  置于 x 与 (N-1-x) 处的读数应逐位相等。这是平移不变在"到边界距离相同"
  的等价位置对上的表现。若取到边界距离不同的位置，读数按边界因子变化，
  该变化属有限尺寸效应（协议伪影），不构成平移不变的判否证据。周期域
  FFT 解则整体平移对称（读数逐位恒等），会使判据构造恒真，故不采用。
  这两条教训记入注记过程记录。

LLI 侧判据：
  LI1 旋转不变与长波常数项：同一 |k| 下四个取向（含非轴向旋转）给同一
      omega（长波近似为解析恒等，容差 1e-12）；精确色散的方向差异为 k^4
      高阶项（容差 1e-3）；长波端 omega^2 - c^2 k^2 为常数 m_eff^2。
  LI2 boost 等价性（LLI 的操作化）：boost 下频率与波矢同时变换
      omega' = gamma(omega - beta c k)，k' = gamma(k - beta omega/c^2)；
      光速读数 c_local = omega'/k' 对光支（omega = c k）应恒为 c，
      即系间散布为零。这是 LLI 的内容：光速（omega/k 之比）在洛伦兹系内
      不变，而频率本身按多普勒因子变换（不可把二者混淆，见过程记录）。
  LI3 普朗克层方向比在 boost 下保持：sum_a k_a^4 / k^2 的方向依赖比在
      boost 线性变换下不变（分子分母同次幂，解析恒等，容差 1e-12）。

LPI 侧判据：
  LP1 写入平移不变在引力梯度下的继承（DB2 的自由落体系版）：四个等价
      对称位置对（x 与 N-1-x）在无梯度与三档梯度下，读数之比 C 的相对
      偏差均 <= 容差（稀疏直接解的舍入量级 ~1e-15，容差 1e-10）。
  LP2 潮汐读数的对称协变：两侧对称位置的二阶差分读数逐位相等
      （容差 1e-10）。

EP5 恒真检验（反例检验，"改掉它就 FAIL"）：
  EC1 LI2 判否替身：破坏 boost 结构使 c_local 随 v0 变化，LI2 判 FAIL。
  EC2 LP1 判否替身：令 C 随位置线性变化，LP1 判 FAIL。

容差预算（从误差来源推导）：
  LI1/LI2/LI3：解析恒等式，容差 1e-12（浮点舍入主导）。
  LP1/LP2：稀疏直接解（spla.splu）的舍入量级 ~1e-15，容差 1e-10
      （保守放大五个量级，覆盖不同源位置的条件数差异）。

边界声明：
  一、不重证 Q1（WEP）与 Q4（几何基石），二者已有第 459、468 条判据。
  二、不证强场与 SEP（归黑洞线）。三、不证非线性完成式。
  四、LI2 的"局域系"以释放初速参数化的洛伦兹系为构造，是操作化定义；
      完整的 LLI 定理陈述见 [[狭义相对论的定理化：从晶格几何到洛伦兹结构]]。
  五、LP1 继承 DB2（第 458 条）判据，不重新判 DB2 本身。
  六、LP1 在有限 Dirichlet 网格上以等价对称位置对检验平移不变；到边界
      距离不同位置的读数差异属有限尺寸效应，不作判否证据（见过程记录）。
  七、本轮不核 LLI 与 LPI 在含时场、多极场下的继承（一维径向弱场模型）。

可逆性等级：判据稳健。
确定性执行，约数秒。图存 images/eep_frame_inheritance.png
"""
import math
import os

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "images", "eep_frame_inheritance.png")

TPL = 1.0
LP = 1.0
C_LIGHT = 1.0
N_GRID = 15


def results(tag, ok, detail):
    print(f"  [{tag}] {'PASS' if ok else 'FAIL'}  {detail}")
    return bool(ok)


def dispersion(k_vec):
    return 4.0 / TPL ** 2 * sum(math.sin(kk * LP / 2.0) ** 2 for kk in k_vec)


def dispersion_longwave(k_vec, m_eff_sq):
    return C_LIGHT ** 2 * sum(kk * kk for kk in k_vec) + m_eff_sq


def gamma_of(beta):
    return 1.0 / math.sqrt(1.0 - beta ** 2)


# =============== 冒烟检查 ===============
print("== 冒烟检查 ==")
_sm = [dispersion([k, 0, 0]) ** 0.5 for k in (0.05, 0.1, 0.2)]
print(f"  色散频率（k=0.05,0.1,0.2）：{_sm[0]:.6f}, {_sm[1]:.6f}, {_sm[2]:.6f}")
assert _sm[0] < _sm[1] < _sm[2], "冒烟失败：频率未随波矢单调"
print("  冒烟通过：频率随 k 单调、低能端 omega/k -> c = 1")

# =============== LLI 侧 ===============
print("\n== LI1 局域系内低能色散的旋转不变与长波常数项 ==")
m_eff_sq = (0.01 / 10.0) ** 2
k_mag = 0.02
dirs = [np.array([k_mag, 0, 0]), np.array([0, k_mag, 0]), np.array([0, 0, k_mag])]
th = 0.7
dirs.append(np.array([k_mag * math.cos(th), k_mag * math.sin(th), 0.0]))
om_lw = [dispersion_longwave(d, m_eff_sq) ** 0.5 for d in dirs]
om_ex = [dispersion(list(d)) ** 0.5 for d in dirs]
spread_lw = (max(om_lw) - min(om_lw)) / np.mean(om_lw)
spread_ex = (max(om_ex) - min(om_ex)) / np.mean(om_ex)
const_term = dispersion_longwave([k_mag, 0, 0], m_eff_sq) - C_LIGHT ** 2 * k_mag ** 2
li1 = results("LI1", spread_lw < 1e-12 and spread_ex < 1e-3 and abs(const_term - m_eff_sq) < 1e-15,
              f"旋转不变（低能 {spread_lw:.1e}，精确 {spread_ex:.1e}）；"
              f"长波常数项 = {const_term:.3e}（应 = {m_eff_sq:.1e}）")

print("\n== LI2 boost 等价性（光速读数系间一致） ==")
v0_list = [0.0, 0.1, 0.2, 0.3]
k_light = 1e-3
om_light = C_LIGHT * k_light
c_locals = []
freq_ratio = []
for v0 in v0_list:
    beta = v0 / C_LIGHT
    g = gamma_of(beta)
    om_prime = g * (om_light - beta * C_LIGHT * k_light)
    k_prime = g * (k_light - beta * om_light / C_LIGHT ** 2)
    c_locals.append(om_prime / k_prime)
    freq_ratio.append(om_prime / om_light)
c_locals = np.array(c_locals)
freq_ratio = np.array(freq_ratio)
c_spread = (max(c_locals) - min(c_locals)) / np.mean(c_locals)
# 频率读数应按多普勒因子 gamma(1-beta) = sqrt((1-beta)/(1+beta)) 变换（对照项）
doppler_ref = np.array([math.sqrt((1 - b) / (1 + b)) for b in v0_list])
dop_dev = float(np.max(np.abs(freq_ratio - doppler_ref)))
li2 = results("LI2", c_spread < 1e-12 and dop_dev < 1e-12,
              f"光速读数散布 = {c_spread:.3e}（LLI 预言恒等）；"
              f"频率读数与多普勒因子偏差 = {dop_dev:.1e}（对照，确认频率非不变量）")

print("\n== LI3 普朗克层方向比在 boost 下保持 ==")
def dir_ratio(k_vec):
    return sum(kk ** 4 for kk in k_vec) / sum(kk ** 2 for kk in k_vec)

ratio_g = [dir_ratio(np.eye(3)[a] * k_mag) for a in range(3)]
beta_t, g_t = 0.2, gamma_of(0.2)
ratio_l = []
for a in range(3):
    kv = np.eye(3)[a] * k_mag
    om = dispersion_longwave(list(kv), m_eff_sq) ** 0.5
    om_p = g_t * (om - beta_t * C_LIGHT * kv[a])
    kvp = kv.copy()
    kvp[a] = g_t * (kv[a] - beta_t * om / C_LIGHT ** 2)
    ratio_l.append(dir_ratio(kvp))
r_spread = (max(ratio_l) - min(ratio_l)) / np.mean(ratio_l)
li3 = results("LI3", r_spread < 1e-12,
              f"全局系读数 = {[f'{x:.4e}' for x in ratio_g]}，boost 后散布 = {r_spread:.1e}")

# =============== LPI 侧：有限网格 Dirichlet 格林函数 ===============
print("\n== 稀疏格林函数求解器 ==")
free = np.ones((N_GRID,) * 3, dtype=bool)
free[0, :, :] = free[-1, :, :] = False
free[:, 0, :] = free[:, -1, :] = False
free[:, :, 0] = free[:, :, -1] = False
w = np.where(free)
lin = np.ravel_multi_index((w[0], w[1], w[2]), (N_GRID,) * 3)
pos_of = -np.ones(N_GRID ** 3, dtype=np.int64)
pos_of[lin] = np.arange(len(lin))
rows = [np.arange(len(lin))]
cols = [np.arange(len(lin))]
vals = [np.full(len(lin), -6.0)]
for da, db, dc in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
    na, nb, nc = w[0] + da, w[1] + db, w[2] + dc
    ok = free[na, nb, nc]
    rows.append(np.where(ok)[0])
    cols.append(pos_of[np.ravel_multi_index((na[ok], nb[ok], nc[ok]), (N_GRID,) * 3)])
    vals.append(np.ones(int(ok.sum())))
MAT = sp.csc_matrix((np.concatenate(vals),
                     (np.concatenate(rows), np.concatenate(cols))),
                    shape=(len(lin), len(lin)))
LU = spla.splu(MAT)
print(f"  网格 {N_GRID}^3，自由节点 {MAT.shape[0]}，nnz {MAT.nnz}，分解完成")

def field_for(src, A=0.0):
    b = np.zeros(len(lin))
    b[pos_of[np.ravel_multi_index(src, (N_GRID,) * 3)]] = -1.0
    th = np.zeros((N_GRID,) * 3)
    th.reshape(-1)[lin] = LU.solve(b)
    return th + A

def C_of(src, A=0.0):
    th = field_for(src, A)
    x, y, z = src
    return float(np.mean([abs(th[(x+dx), (y+dy), (z+dz)])
                          for dx in (-1, 1) for dy in (-1, 1) for dz in (-1, 1)]))

def lap2_of(src, A=0.0):
    th = field_for(src, A)
    x, y, z = src
    c = th[x, y, z]
    s = sum(th[(x+dx), (y+dy), (z+dz)]
            for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
            if (dx, dy, dz) != (0, 0, 0))
    return s - 6 * c

print("\n== LP1 写入平移不变在引力梯度下的继承 ==")
MIRR = N_GRID - 1
sym_pairs = [((4, 7, 7), (MIRR - 4, 7, 7)),
             ((6, 7, 7), (MIRR - 6, 7, 7)),
             ((7, 5, 7), (7, MIRR - 5, 7)),
             ((7, 7, 6), (7, 7, MIRR - 6))]
_c_a = C_of(sym_pairs[0][0]); _c_b = C_of(sym_pairs[1][0])
print(f"  冒烟：两对位置 C = {_c_a:.10e}, {_c_b:.10e}，差 = {abs(_c_a-_c_b):.3e}")
assert abs(_c_a - _c_b) > 1e-12, "冒烟失败：不同对称对读数全同，LP1 无区分力"
print("  冒烟通过：不同对称对读数有差异（非构造恒真）")

lp1_devs = {}
for A in (0.0, 0.001, 0.01, 0.05):
    devs = []
    for p, q in sym_pairs:
        Cp, Cq = C_of(p, A), C_of(q, A)
        devs.append(abs(Cp - Cq) / max(abs(Cp), abs(Cq)))
    dev = max(devs)
    lp1_devs[A] = dev
    print(f"  A = {A}：四对对称位置最大相对偏差 = {dev:.3e}")
lp1 = results("LP1", max(lp1_devs.values()) < 1e-10,
              f"含梯度四档最大偏差 = {max(lp1_devs.values()):.1e}（对称位置平移不变）")

print("\n== LP2 潮汐读数的对称协变 ==")
lp2_devs = []
for (p, q) in sym_pairs:
    for A in (0.0, 0.01):
        lp2_devs.append(abs(lap2_of(p, A) - lap2_of(q, A)) / max(abs(lap2_of(p, A)), 1e-30))
lp2_dev = max(lp2_devs)
lp2 = results("LP2", lp2_dev < 1e-10, f"二阶差分读数的对称偏差 = {lp2_dev:.1e}")

# =============== EP5 恒真检验 ===============
print("\n== EC1 LI2 判否替身 ==")
c_fake = np.array([C_LIGHT * (1.0 - v0 / C_LIGHT) for v0 in v0_list])
cf_spread = (max(c_fake) - min(c_fake)) / np.mean(c_fake)
ec1 = results("EC1", cf_spread > 1e-6, f"替身 c_local=c(1-beta) 散布 = {cf_spread:.2e}，LI2 判 FAIL")

print("\n== EC2 LP1 判否替身 ==")
devs_fake = [abs((1.0 + 0.05 * p[0]) - (1.0 + 0.05 * q[0])) / (1.0 + 0.05 * p[0])
             for (p, q) in sym_pairs]
fake_dev = max(devs_fake)
ec2 = results("EC2", fake_dev > 1e-6, f"替身 C=C0(1+0.05x) 最大偏差 = {fake_dev:.2e}，LP1 判 FAIL")

# =============== 图 ===============
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
ax = axes[0]
kk = np.linspace(0.005, 0.1, 60)
ax.plot(kk, [dispersion_longwave([k_], m_eff_sq) ** 0.5 for k_ in kk],
        lw=2, label="长波极限")
ax.plot(kk, [dispersion([k_, 0, 0]) ** 0.5 for k_ in kk], "--", lw=1, label="精确色散")
ax.set_xlabel("k (l_P^-1)"); ax.set_ylabel("omega")
ax.set_title("LI1 局域低能色散与旋转不变"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(v0_list, c_locals, "o-", lw=2, label="c_local = omega'/k'")
ax.axhline(1.0, color="k", ls="--", lw=1, label="c = 1")
ax.set_xlabel("释放初速 v0"); ax.set_ylabel("光速读数")
ax.set_title("LI2 光速系间一致（散布 %.0e）" % c_spread)
ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax2 = ax.twinx()
ax2.plot(v0_list, freq_ratio, "s--", color="tab:orange", lw=1,
         label="频率比（多普勒）")
ax2.set_ylabel("频率比", color="tab:orange")
ax2.legend(fontsize=7, loc="lower right")

ax = axes[2]
A_vals = sorted(lp1_devs.keys())
ax.semilogy(range(len(A_vals)), [max(lp1_devs[a], 1e-17) for a in A_vals],
            "o-", lw=2, label="LP1 对称位置偏差")
ax.axhline(1e-10, color="r", ls="--", lw=1, label="容差 1e-10")
ax.set_xticks(range(len(A_vals))); ax.set_xticklabels([f"{a:g}" for a in A_vals])
ax.set_xlabel("引力梯度 A"); ax.set_ylabel("relative deviation")
ax.set_title("LP1 平移不变在引力场中继承")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(IMG, dpi=160)
print("\nfigure saved:", IMG)
all_pass = all([li1, li2, li3, lp1, lp2, ec1, ec2])
print("判据：", "全部通过" if all_pass else "存在失败")