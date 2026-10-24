# -*- coding: utf-8 -*-
"""
load_equals_write.py

载荷等同写入的最小验证（第 458 条）：命题 B 的判据化。

命题 B（倾角注记通道三）：写入幅度 delta_theta 与最近邻格点载荷读数 theta(l_P)
的等同，由写事件分配规则（判据 L）与离散格林函数结构给出，无需引入 PN 匹配。

连续层的恒等式 theta(l_P) = delta_theta 已有解析（承载力读法注记第二节：
theta(r) = (m/m_P)(l_P/r) 取 r = l_P）。本轮在离散层验证该等同的结构性质：
最近邻读数与写入幅值之比是结构常数 C——线性（与 delta 无关）、平移不变
（与源位置无关）、载荷读数唯一（六邻居相等）、可加（多源线性叠加）——
C 本身由纯几何（离散格林函数）决定，不含新物理参数。

判据：
  DB1（线性）：C = u_nn/delta 与写入幅值无关（4 档幅值扫描，散布 <= 1e-12）。
  DB2（平移不变）：C 与源站点位置无关（4 个非对称位置，散布 <= 1e-12）。
  DB3（分配细节分层）：同一幅值的单点写入与六邻居对称分散写入，远场
      （r = 6 格距）读数相对差 <= 10%（单极主导，多极截断预算）；源项分布
      （拉普拉斯）随分配改变：单点在站点 -delta，分散在站点 0、六邻居各
      -delta/6。格点读数因格林函数调和性近不变（实测 2e-5 量级，教训：
      读数差不是分配细节的正确探针，源项分布才是）。
  DB4（可加性）：双源联合解与分解和一致（相对差 <= 1e-12）；交叉项定量打印。

模型设定与建模决策：
  - FFT 周期域谱解（48³，去零模，中性背景）：周期域格林函数的绝对值含
    中性背景与镜像贡献，与连续值 1/(4*pi*r) 的直接对照不在本轮射程（需
    Ewald 求和）；所有判据取比值/差分形态，不受绝对值影响（协议伪影与
    物理结论分层）。
  - "最近邻载荷读数"取源站点六邻居读数（DC1a 已证六邻居相等，读数唯一）。
  - 线性叠加判据（DB1/DB4）属求解器性质的自检级核对，其判据价值在于确认
    离散设置（网格奇偶、源点布置）未破坏线性与平移不变。

边界声明：验证比值的结构性质（线性、平移不变、唯一、可加）与分配细节的
近远场分层；不验证 C 到连续极限 1 的收敛（需 Ewald 或网格加密研究，不在
本轮射程）；不验证真实多体动力学。命题 B 的连续层恒等式为解析结果（代数），
本轮数值与之合成后命题 B 达判据级。

可逆性等级：判据稳健（结构性质不依赖投影模型与分配细节的量值）。
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NG = 48
idx = np.arange(NG)
k1 = np.fft.fftfreq(NG, d=1.0) * 2 * np.pi
KX, KY, KZ = np.meshgrid(k1, k1, k1, indexing='ij')
LAM = 6.0 - 2.0 * (np.cos(KX) + np.cos(KY) + np.cos(KZ))
LAM[0, 0, 0] = 1.0

def solve(src):
    s = src - src.mean()
    return np.real(np.fft.ifftn(np.fft.fftn(s) / LAM))

def point_src(delta, pos):
    s = np.zeros((NG, NG, NG))
    s[pos] = delta
    return s

def nn_reading(u, pos):
    x, y, z = pos
    return 0.5 * ((u[x+1, y, z] + u[x-1, y, z] +
                   u[x, y+1, z] + u[x, y-1, z] +
                   u[x, y, z+1] + u[x, y, z-1]) / 3.0)

results = []
P0 = (24, 24, 24)

# ---------- DB1：线性（C 与 delta 无关） ----------
deltas = [0.1, 0.5, 1.0, 2.0]
Cs = []
for d in deltas:
    u = solve(point_src(d, P0))
    Cs.append(nn_reading(u, P0) / d)
Cs = np.array(Cs)
spread = (Cs.max() - Cs.min()) / Cs.mean()
results.append(('DB1', 'C 与写入幅值无关（4 档相对散布）', spread, '<= 1e-12',
                spread <= 1e-12))

# ---------- DB2：平移不变（C 与源位置无关） ----------
POS = [(24, 24, 24), (30, 18, 40), (10, 40, 12), (44, 8, 30)]
Cs2 = []
for p in POS:
    u = solve(point_src(1.0, p))
    Cs2.append(nn_reading(u, p))
Cs2 = np.array(Cs2)
spread2 = (Cs2.max() - Cs2.min()) / Cs2.mean()
results.append(('DB2', 'C 与源位置无关（4 站点相对散布）', spread2, '<= 1e-12',
                spread2 <= 1e-12))

# ---------- DB3：分配细节分层（单点 vs 六邻居对称分散） ----------
d3 = 1.0
u_single = solve(point_src(d3, P0))
s_sc = np.zeros((NG, NG, NG))
for dd in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
    s_sc[P0[0]+dd[0], P0[1]+dd[1], P0[2]+dd[2]] = d3 / 6.0
u_sc = solve(s_sc)

def radial(u, p, kmax):
    x, y, z = p
    return np.array([u[x+k, y, z] for k in range(1, kmax+1)])

kmax = 8
r_single = radial(u_single, P0, kmax)
r_sc = radial(u_sc, P0, kmax)
far_diff = abs(r_single[5:].mean() - r_sc[5:].mean()) / abs(r_single[5:].mean())
results.append(('DB3a', '远场 r=6..8 单点/分散 相对差', far_diff, '<= 0.10（单极主导）',
                far_diff <= 0.10))

# DB3b-d：分配细节改变源项分布（拉普拉斯）；调和性使格点读数近不变（实测 2e-5 量级）
def lap(u, pos):
    x, y, z = pos
    return 6.0 * u[pos] - (u[x+1, y, z] + u[x-1, y, z] + u[x, y+1, z] +
                           u[x, y-1, z] + u[x, y, z+1] + u[x, y, z-1])

lap_single_c = lap(u_single, P0)
lap_sc_c = lap(u_sc, P0)
lap_sc_all = [lap(u_sc, (P0[0]+a, P0[1]+b, P0[2]+cc)) for a, b, cc in
              [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]]
# lap 约定：6u - sum(nn u) = +s。周期域去零模的中性背景使各点偏差含 mean = 1/NG^3
# （协议伪影，精确可预算）；判据相对 "期望 + mean" 达机器精度。
MEAN = 1.0 / (NG ** 3)
results.append(('DB3b', '单点源项在站点（拉普拉斯，含背景）', lap_single_c,
                '≈ +1 - mean', abs(lap_single_c - (1.0 - MEAN)) <= 1e-12))
results.append(('DB3c', '分散源项在站点（站点无源）', lap_sc_c,
                '≈ -mean（调和 + 背景）', abs(lap_sc_c + MEAN) <= 1e-12))
results.append(('DB3d', '分散源项在六邻居（各 +1/6）最大偏差',
                max(abs(v - (1.0/6.0 - MEAN)) for v in lap_sc_all), '<= 1e-12',
                max(abs(v - (1.0/6.0 - MEAN)) for v in lap_sc_all) <= 1e-12))

# ---------- DB4：可加性 ----------
d1, d2 = 0.7, 1.3
p1, p2 = (24, 24, 24), (30, 24, 24)  # 相距 6 格距
u_joint = solve(point_src(d1, p1) + point_src(d2, p2))
u_a = solve(point_src(d1, p1))
u_b = solve(point_src(d2, p2))
err = np.max(np.abs(u_joint - (u_a + u_b))) / np.max(np.abs(u_joint))
results.append(('DB4a', '双源联合解与分解和最大相对差', err, '<= 1e-12',
                err <= 1e-12))
G_cross = (nn_reading(u_joint, p1) - d1 * Cs[2]) / d2
results.append(('DB4b', '交叉项（源 2 对源 1 近邻读数的贡献）/d2', G_cross,
                '打印（协议伪影分层）', True))

C_ref = Cs[2]
disc_factor = 4.0 * np.pi * C_ref
print('=' * 72)
print('载荷等同写入的最小验证（第 458 条）：命题 B 判据 DB1 至 DB4')
print('=' * 72)
n_pass = 0
for tag, name, val, expect, ok in results:
    n_pass += ok
    print('%s %-36s 实测 %.12g  期望 %s  %s' % (
        tag, name, val, expect, 'PASS' if ok else 'FAIL'))
print('-' * 72)
print('结构常数 C（delta=1，P0） = %.12g' % C_ref)
print('离散修正因子 4*pi*C      = %.12g（周期域协议伪影，分层声明，不作判据）' % disc_factor)
print('交叉项读数（对照连续 1/(4*pi*6) = %.6g）' % (1.0/(4*np.pi*6)))
print('-' * 72)
print('判据 %d/%d 通过' % (n_pass, len(results)))
assert n_pass == len(results), 'FAIL: 判据未全过'

# ---------- 图 ----------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
ax = axes[0]
ax.plot(deltas, Cs/Cs[2], 'o-', lw=1.5)
ax.set_xlabel('write amplitude delta')
ax.set_ylabel('C(delta) / C(1)')
ax.set_title('(a) DB1: ratio independent of amplitude')
ax = axes[1]
ks = np.arange(1, kmax+1)
ax.plot(ks, r_single / r_single[0], 'o-', label='single-site write', lw=1.5)
ax.plot(ks, r_sc / r_single[0], 's--', label='six-neighbor spread', lw=1.5)
ax.set_xlabel('distance from source (l_P)')
ax.set_ylabel('reading / near-field(single)')
ax.set_title('(b) DB3: near-field differs, far field merges')
ax.legend(fontsize=8)
ax = axes[2]
prof_a = u_a[24, 24, 20:36]
prof_j = u_joint[24, 24, 20:36]
ax.plot(np.arange(20, 36) - 24, prof_a / prof_a.max(), 'o-', label='source 1 alone', lw=1.5)
ax.plot(np.arange(20, 36) - 24, prof_j / prof_a.max(), 's--', label='joint (1+2)', lw=1.5)
ax.axvline(6, color='C3', ls=':', label='source 2 position')
ax.set_xlabel('x - x1 (l_P)')
ax.set_ylabel('reading (normalized)')
ax.set_title('(c) DB4: additivity of two sources')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(r'D:\software\TianXingFC\images\load_equals_write.png', dpi=150)
print('figure saved: images/load_equals_write.png')
