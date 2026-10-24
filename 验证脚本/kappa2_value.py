# -*- coding: utf-8 -*-
"""
判据第 597 条：kappa_2 量值反推（自旋锚定强度攻坚阶段二）
=========================================================
主读法（源读法，与第 449 条「该式读作束缚电流密度」判据语义一致）：
    J_sys = (kappa_2 / l_P^2) * curl(s_hat)          （体系静磁方程的源，[J]=1/L^3）
    J_sys = (e*mu0/hbar) * j_SI,  j_SI = curl(M)     （AB 相位桥，SI 对接）
    M = M_s * s_hat（饱和均匀磁化，M_s 均匀）=>  kappa_2_src = e*mu0*M_s*l_P^2/hbar
对照档（场读法，阶段一 T3 原式）：
    B_sys = (kappa_2/l_P) * curl(s_hat),  B_sys = (e/hbar)*B_SI,  B_SI = mu0*M_s
    alpha = |curl(s_hat)| = pi/delta（Bloch 180 度壁，壁厚 delta）
    =>  kappa_2_fld = e*mu0*M_s*l_P*delta/(hbar*pi)

判据：
    V1 量纲桥闭合（AB 桥与源桥）+ 恒真反例（漏 mu0 则不闭合）
    V2 源读法多档读数（材料 Fe/Co/Ni/FeCo），漂移 = M_s 漂移（语义红旗登记）
    V3 场读法对照档（3 材料 x 2 壁厚约定），读法分叉 = delta/(pi*l_P) 个数量级
    V4 与 kappa_1=1/(4*pi*alpha_fs) 对比 + 换算族带距离（判否倾向的量化）
    V5 l_P 幂敏感性扫描（判否对读法约定稳健性）

数据来源（标准磁学教科书/文献值）：
    饱和极化 mu0*Ms：Fe 2.15 T、Co 1.76 T、Ni 0.61 T、FeCo(permendur) 2.35 T
      （Crangle & Hallam 1971 绝对测量 Fe/Ni；Wikipedia "Saturation (magnetic)" 汇总）
    交换刚度 A 与各向异性 K1：Fe A=20 pJ/m K1=48 kJ/m^3；Co A=30 pJ/m K1=410 kJ/m^3；
      Ni A=8.6 pJ/m |K1|=4.5 kJ/m^3（Kuz'Min 2020 (HAL) 与标准文献汇总，量级档）
    壁厚约定：delta0=sqrt(A/|K1|)（特征壁宽）与 delta=pi*delta0（180 度 Bloch 壁全厚）两档扫描
可逆性等级：判据稳健（换 SI 版本或换材料数据档不翻；读法语义定案属阶段三）。
"""
import numpy as np
import sympy as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------- 物理常数（SI 2019） ----------
E_CH   = 1.602176634e-19        # C（精确）
HBAR   = 1.054571817e-34        # J s（精确）
MU0    = 1.25663706212e-6       # N/A^2（CODATA 2018）
LP     = 1.616255e-35           # m（CODATA 2018 普朗克长度）
ALPHA  = 1.0 / 137.035999084    # 精细结构常数
KAPPA1 = 1.0 / (4.0 * np.pi * ALPHA)   # = 1/e_nat^2，Heaviside-Lorentz 口径

# ---------- 材料数据（来源见 docstring） ----------
# (名称, mu0*Ms [T], A [J/m], |K1| [J/m^3])
MATS = [
    ('Fe',   2.15, 20.0e-12, 48.0e3),
    ('Co',   1.76, 30.0e-12, 410.0e3),
    ('Ni',   0.61, 8.6e-12,  4.5e3),
    ('FeCo', 2.35, None,     None),   # permendur，无壁厚参数，只进源读法档
]

results = {}
ok_all = True

def check(name, ok, detail):
    global ok_all
    results[name] = (ok, detail)
    if not ok:
        ok_all = False
    print('[%s] %s' % ('PASS' if ok else 'FAIL', name))
    for line in detail:
        print('    ' + line)

# ---------- V1：量纲桥闭合（sympy 符号量纲） ----------
m_, l_, t_, q_ = sp.symbols('m l t q', positive=True)
dim_B_si   = m_ / (q_ * t_)                 # tesla = kg/(A s^2), A=q/t
dim_e_hbar = q_ * t_ / (m_ * l_**2)         # [e/hbar] = C s /(kg m^2)
dim_mu0    = m_ * l_ / q_**2                # N/A^2 = kg m / C^2
dim_Ms     = q_ / (t_ * l_)                 # A/m
dim_j_si   = q_ / (t_ * l_**2)              # A/m^2
dim_LP2    = l_**2

bridge_B   = sp.simplify(dim_e_hbar * dim_B_si)            # 应为 1/l^2
bridge_J   = sp.simplify(dim_e_hbar * dim_mu0 * dim_j_si)  # 应为 1/l^3
bridge_J_no_mu0 = sp.simplify(dim_e_hbar * dim_j_si)       # 恒真反例：漏 mu0
kappa2_src_dim = sp.simplify(dim_e_hbar * dim_mu0 * dim_Ms * dim_LP2)  # 完整表达式 e*mu0*Ms*l_P^2/hbar 的量纲，应为 1

v1_ok = (sp.simplify(bridge_B - 1/l_**2) == 0
         and sp.simplify(bridge_J - 1/l_**3) == 0
         and sp.simplify(kappa2_src_dim - 1) == 0
         and sp.simplify(bridge_J_no_mu0 - 1/l_**3) != 0)
check('V1 量纲桥闭合（AB 桥 + 源桥）+ 恒真反例（漏 mu0）', v1_ok, [
    'B 桥 (e/hbar)*B_SI 实测 %s vs 期望 1/l**2' % bridge_B,
    '源桥 (e/hbar)*mu0*j_SI 实测 %s vs 期望 1/l**3' % bridge_J,
    'kappa_2_src = e*mu0*Ms*l_P^2/hbar 实测 %s vs 期望 1（无量纲）' % kappa2_src_dim,
    '恒真反例（漏 mu0）：实测 %s，判非 1/l**3（反例成立，判据有区分力）' % bridge_J_no_mu0,
])

# ---------- V2：源读法多档读数 ----------
print('\n--- V2 源读法（主读法）：kappa_2_src = e*mu0*Ms*l_P^2/hbar ---')
src_log = {}
ms_vals = {}
for name, b_sat, _, _ in MATS:
    Ms = b_sat / MU0                       # A/m
    ms_vals[name] = Ms
    k2 = E_CH * MU0 * Ms * LP**2 / HBAR
    src_log[name] = np.log10(k2)
    print('    %-4s mu0*Ms=%.2f T  Ms=%.3e A/m  kappa_2_src=%.3e (log10=%.2f)'
          % (name, b_sat, Ms, k2, src_log[name]))
ms_ratio = max(ms_vals.values()) / min(ms_vals.values())
k2_ratio = 10**(max(src_log.values()) - min(src_log.values()))
v2_ok = all(-57.0 <= v <= -52.0 for v in src_log.values()) \
        and abs(k2_ratio - ms_ratio) / ms_ratio < 1e-9
check('V2 源读法四档读数量级 + 漂移一致性', v2_ok, [
    'log10(kappa_2_src) 四档区间 [%.2f, %.2f] vs 期望 [-57, -52]' % (min(src_log.values()), max(src_log.values())),
    '材料漂移比 kappa_2 max/min=%.3f vs Ms 漂移比=%.3f（读数严格跟随 Ms）' % (k2_ratio, ms_ratio),
    '语义红旗（登记不判否）：kappa_2 若为体系常数不应随材料变；漂移=Ms 漂移说明源读法读的是材料磁化，不是体系常数',
])

# ---------- V3：场读法对照档 ----------
print('\n--- V3 场读法（对照档）：kappa_2_fld = e*mu0*Ms*l_P*delta/(hbar*pi) ---')
fld_log = {}
for name, b_sat, A, K1 in MATS:
    if A is None:
        continue
    Ms = b_sat / MU0
    delta0 = np.sqrt(A / K1)               # 特征壁宽
    for tag, delta in (('sqrt(A/K)', delta0), ('pi*sqrt(A/K)', np.pi * delta0)):
        k2 = E_CH * MU0 * Ms * LP * delta / (HBAR * np.pi)
        key = '%s/%s(delta=%.1fnm)' % (name, tag, delta * 1e9)
        fld_log[key] = np.log10(k2)
        print('    %-28s kappa_2_fld=%.3e (log10=%.2f)' % (key, k2, fld_log[key]))
fork = 10**(np.mean(list(fld_log.values())) - np.mean(list(src_log.values())))
v3_ok = all(-30.0 <= v <= -25.0 for v in fld_log.values())
check('V3 场读法六档读数量级', v3_ok, [
    'log10(kappa_2_fld) 六档区间 [%.2f, %.2f] vs 期望 [-30, -25]' % (min(fld_log.values()), max(fld_log.values())),
    '读法分叉均值 log10(kappa_2_fld/kappa_2_src) = %.2f，解析 log10 delta/(pi*l_P) = %.2f（纯 l_P 幂分叉，与材料无关）'
        % (np.log10(fork), np.log10(np.mean([np.sqrt(A/K1) for _,_,A,K1 in MATS if A]) / (np.pi * LP))),
])

# ---------- V4：与 kappa_1 对比 + 换算族带距离 ----------
print('\n--- V4 与 kappa_1=%.3f (log10=%.2f) 对比 ---' % (KAPPA1, np.log10(KAPPA1)))
FAMILY = []
for n in range(-6, 7):
    for base in (2*np.pi, np.pi, 4*np.pi, np.pi**2/4.0, 1.0/(2*np.pi)):
        FAMILY.append(np.log10(base) * n)   # 换算族常数的对数（幂展开）
FAMILY = np.array(sorted(set(FAMILY)))
def dist_to_family(logv):
    return float(np.min(np.abs(FAMILY - logv)))
d_src  = float(np.mean(list(src_log.values())) - np.log10(KAPPA1))
d_fld  = float(np.mean(list(fld_log.values())) - np.log10(KAPPA1))
f_src, f_fld = dist_to_family(d_src), dist_to_family(d_fld)
v4_ok = (abs(d_src) > 25.0 and abs(d_fld) > 25.0 and f_src > 20.0 and f_fld > 20.0)
check('V4 与 kappa_1 的分离度 + 换算族带距离（判否倾向量化）', v4_ok, [
    'log10(kappa_2_src/kappa_1) = %.2f（分离 %s 55 个数量级）' % (d_src, '负向' if d_src < 0 else '正向'),
    'log10(kappa_2_fld/kappa_1) = %.2f（分离约 28 个数量级）' % d_fld,
    '到换算族带（2pi^±6 等幂展开，%d 个对数格点）最近距离：源读法 %.2f、场读法 %.2f vs 期望均 > 20' % (len(FAMILY), f_src, f_fld),
    '读否倾向：两读法均远离 kappa_1 达 25 个数量级以上，任何换算族常数（O(1)-O(10^6)）无法跨越',
])

# ---------- V5：l_P 幂敏感性扫描（判否对读法约定的稳健性） ----------
print('\n--- V5 l_P 幂敏感性：kappa_2^(p) = e*mu0*Ms*l_P^(2+p)...（对源读式乘 l_P^p 扫描）---')
best = {}
for name, b_sat, _, _ in MATS:
    Ms = b_sat / MU0
    base = E_CH * MU0 * Ms / HBAR            # p=0：不含任何 l_P 幂的裸读数
    for p in range(-1, 4):
        logv = np.log10(base) + p * np.log10(LP)
        d = dist_to_family(logv - np.log10(KAPPA1))
        best[(name, p)] = d
min_dist = min(best.values())
argmin = min(best, key=best.get)
v5_ok = min_dist > 5.0
check('V5 l_P 幂 p in [-1,3] 扫描下换算族最小距离', v5_ok, [
    '全档最小换算族距离 = %.2f（命中档 %s）vs 期望 > 5' % (min_dist, argmin),
    '结论：判否对 l_P 幂约定稳健（p=-1..3 无一档能把 kappa_2 拉进换算族带 ±3 对数单位）',
    '若阶段三终判否，原因在物理语义（磁化强度与联络锚定强度不同物），换算约定无法救回',
])

# ---------- 汇总 ----------
print('\n===== 判据结果总表（第 597 条） =====')
n_pass = sum(1 for ok, _ in results.values() if ok)
for name, (ok, _) in results.items():
    print('%s  %s' % ('PASS' if ok else 'FAIL', name))
print('判据 %d/%d' % (n_pass, len(results)))

# ---------- 出图 ----------
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
ax = axes[0]
xs = range(len(src_log))
ax.scatter(list(xs), list(src_log.values()), s=70, color='C0', marker='o', label='source reading (main)')
xf = range(len(fld_log))
ax.scatter(list(xf), list(fld_log.values()), s=55, color='C1', marker='^', label='field reading (control)')
ax.axhline(np.log10(KAPPA1), color='r', lw=1.8, label=r'$\kappa_1=1/e^2\simeq10.9$')
ax.axhspan(np.log10(KAPPA1)-3, np.log10(KAPPA1)+3, color='r', alpha=0.12, label='conversion-family band ($\\pm$3 dec)')
ax.set_xticks(list(xs) + [])
ax.set_xticklabels(list(src_log.keys()), rotation=30, ha='right', fontsize=8)
ax.set_ylabel(r'$\log_{10}\kappa_2$')
ax.set_title('kappa2 readings by tier (judgment 597)')
ax.legend(fontsize=8, loc='lower left')
ax.grid(alpha=0.3)

ax = axes[1]
name0, b_sat, _, _ = MATS[0]
Ms = b_sat / MU0
ps = np.arange(-1, 4)
logs = [np.log10(E_CH * MU0 * Ms / HBAR) + p * np.log10(LP) for p in ps]
ax.plot(ps, logs, 'o-', color='C0', label='Fe tier')
ax.axhline(np.log10(KAPPA1), color='r', lw=1.8, label=r'$\kappa_1$')
ax.axhspan(np.log10(KAPPA1)-3, np.log10(KAPPA1)+3, color='r', alpha=0.12)
ax.set_xlabel(r'$l_P$ power $p$ (source formula $\times\, l_P^{p}$)')
ax.set_ylabel(r'$\log_{10}\kappa_2$')
ax.set_title('l_P power sensitivity: no tier hits the band')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('D:/software/TianXingFC/images/kappa2_value_tiers_597.png', dpi=160)
print('\n图已存 D:/software/TianXingFC/images/kappa2_value_tiers_597.png')
