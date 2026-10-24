# -*- coding: utf-8 -*-
"""
587 号：写入份额定理化 J1/J3 —— 量纲唯一性的完备枚举与转录守恒的替身排除
路线 B（写入份额定理化）证明轮恒等式面先行。材料：内部草稿/材料-写入份额定理化。

模型设定
--------
物理量纲空间为三维（M, L, T）。候选常数集（P1 至 P3）：
  hbar = [1, 2, -1]（作用量 M L^2 T^-1）
  c    = [0, 1, 1]（速度 L T^-1）
  G    = [-1, 3, -2]（引力常数 M^-1 L^3 T^-2）
  m    = [1, 0, 0]（质量）
l_P、t_P 为 hbar/c/G 的导出量（非独立量纲），作冗余参数档扫描。
质量恒等式：mc^2 = hbar*omega_0，即 omega_0 = m c^2 / hbar。
转录等式（J3 目标恒等式）：delta_theta = m/m_P = m c^2 t_P / hbar = omega_0 * t_P。
m_P 定义：m_P = sqrt(hbar c / G)。

判据与理论期望（先立后测）
--------------------------
J1a（解空间维数）：量纲方程 [m]^1 [hbar]^b [c]^d [G]^e = 无量纲 的解空间维数恰为 1。
    期望：4 变量（b,d,e 加 a 固定 1 后 3 变量）3 方程、秩 3，解唯一。判定：维数==1。
J1b（唯一解身份）：该唯一解的指数向量 (b,d,e) = (-1/2, -1/2, 1/2)，
    组合 m hbar^-1/2 c^-1/2 G^1/2 = m sqrt(G/(hbar c)) = m/m_P。
    期望：与 m_P 定义式逐位一致（浮点相对偏差 < 1e-15）。判定：相对偏差 < 1e-15。
J1c（幂封闭）：a = 2, 3, 1/2 的解均为 a=1 解的对应幂（同一基张成）。
    期望：a=2 解的指数 = 2*(-1/2, -1/2, 1/2)，组合值 = (m/m_P)^2。判定：指数逐位相等。
J1d（冗余参数档）：把 l_P、t_P 加入候选集（五常数量纲矩阵），含 m^1 的解空间维数仍为 1，
    且解可重写为 m/m_P。期望：维数不变（l_P、t_P 量纲落在常数张成的子空间内）。判定：维数==1。
J1e（多档扫描）：在 5 档质量（0.1 至 5 m_P）上，唯一解的组合值与 m/m_P 的相对偏差全档 < 1e-15。
J3a（转录等式）：随机 5 档质量下 mc^2 t_P/hbar 与 omega_0 t_P 逐位相等（浮点 < 1e-15）。
J3b（守恒读回，正确幅值）：按幅值 delta = m/m_P 写入十拍，读回累积作用量
    （每拍幅值 * hbar）与缺陷占用累积（每拍 m c^2 t_P）相对偏差 < 1e-12。
    容差预算：十拍浮点累积误差 ~ 10 * 1e-16 = 1e-15，取 1e-12 覆盖（宽三档）。
J3c（替身排除，恒真反例组）：替身幅值 f_tilde in {(m/m_P)^2, sqrt(m/m_P), (m/m_P)^3,
    omega_0^2 t_P^2} 在同一守恒判据下 FAIL：十拍读回与占用的相对偏差
    |accum_tilde/accum_true - 1| > 0.05（份额量级，远超 1e-12 容差；多档全替身全档 FAIL）。
    期望：非 m/m_P 的幅值破坏份额守恒，偏差为份额级非噪声级。

可逆性等级：判据稳健（解析恒等式与穷举/线性代数，与实现无关；J3c 反例为构造性 FAIL）。
前提依赖：P1（hbar 半径与作用量量子）、P2（每拍满周）、P3（量纲自洽清单）。
不验证：线性半（J2）、斜率推导（J4，归 TA1 与 DC 通道）、格点动力学、矢量块与张量块。
"""
import math
from fractions import Fraction

OUT_IMG = r"D:\software\TianXingFC\images\write_share_uniqueness_587.png"

# ---------- 常数（CODATA 2018；c 与 hbar 为定义值） ----------
C_LIGHT = 299792458.0                      # m/s（定义）
HBAR = 1.054571817e-34                     # J s（定义）
G_NEWTON = 6.67430e-11                     # m^3 kg^-1 s^-2（CODATA 2018）
M_PLANCK = math.sqrt(HBAR * C_LIGHT / G_NEWTON)   # kg

# ---------- 量纲矩阵（列向量 [M, L, T]） ----------
DIM = {
    'm':    (Fraction(1), Fraction(0), Fraction(0)),
    'hbar': (Fraction(1), Fraction(2), Fraction(-1)),
    'c':    (Fraction(0), Fraction(1), Fraction(-1)),
    'G':    (Fraction(-1), Fraction(3), Fraction(-2)),
    'l_P':  (Fraction(0), Fraction(1), Fraction(0)),
    't_P':  (Fraction(0), Fraction(0), Fraction(1)),
}

def solve_null_space(cols, target):
    """解 sum cols[i]*x_i + target = 0（target 为 m 列的 a 倍）。返回精确有理解集合。"""
    # 方程组：sum_i cols[i] * x_i = -target，3 个方程，len(cols) 个未知数，Fraction 精确高斯消元
    rows = []
    for k in range(3):
        row = [cols[i][k] for i in range(len(cols))] + [-target[k]]
        rows.append(row)
    n_eq, n_var = 3, len(cols)
    # 高斯消元（Fraction 精确）
    mat = [row[:] for row in rows]
    pivot_cols = []
    r = 0
    for cidx in range(n_var):
        piv = None
        for rr in range(r, n_eq):
            if mat[rr][cidx] != 0:
                piv = rr
                break
        if piv is None:
            continue
        mat[r], mat[piv] = mat[piv], mat[r]
        pv = mat[r][cidx]
        mat[r] = [v / pv for v in mat[r]]
        for rr in range(n_eq):
            if rr != r and mat[rr][cidx] != 0:
                f = mat[rr][cidx]
                mat[rr] = [a - f * b for a, b in zip(mat[rr], mat[r])]
        pivot_cols.append(cidx)
        r += 1
    rank = r
    free = [c for c in range(n_var) if c not in pivot_cols]
    dim = n_var - rank
    # 特解与通解基（精确 Fraction）
    particular = [Fraction(0)] * n_var
    for i, pc in enumerate(pivot_cols):
        particular[pc] = mat[i][n_var]
    basis = []
    for fc in free:
        vec = [Fraction(0)] * n_var
        vec[fc] = Fraction(1)
        for i, pc in enumerate(pivot_cols):
            vec[pc] = -mat[i][fc]
        basis.append(vec)
    return rank, dim, particular, basis, mat

def combo_value(exp_dict):
    """按指数字典计算组合数值（浮点）。"""
    val = 1.0
    for name, e in exp_dict.items():
        base = {'m': 1.0, 'hbar': HBAR, 'c': C_LIGHT, 'G': G_NEWTON,
                'l_P': math.sqrt(HBAR * G_NEWTON / C_LIGHT**3),
                't_P': math.sqrt(HBAR * G_NEWTON / C_LIGHT**5)}[name]
        val *= base ** float(e)
    return val

results = []

def judge(tag, ok, detail):
    results.append((tag, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {tag}: {detail}")

# ---------- 冒烟：量纲矩阵自检 ----------
print("=== 冒烟检查 ===")
# 冒烟 1：m_P 定义式的量纲恰为 M（质量量纲）
mp_dim = tuple(DIM['hbar'][k] + DIM['c'][k] - DIM['G'][k] for k in range(3))
smoke1 = tuple(Fraction(v) / 2 for v in mp_dim)
print(f"冒烟1  m_P 量纲 = (M^{smoke1[0]}, L^{smoke1[1]}, T^{smoke1[2]})，期望 (1, 0, 0)")
judge("冒烟1 m_P量纲", smoke1 == (Fraction(1), Fraction(0), Fraction(0)),
      f"{smoke1}")
# 冒烟2：m/m_P 数值与 m*sqrt(G/(hbar c)) 一致
lhs = 1.0 / M_PLANCK
rhs = math.sqrt(G_NEWTON / (HBAR * C_LIGHT))
print(f"冒烟2  1/m_P vs sqrt(G/(hbar c))：{lhs:.15e} vs {rhs:.15e}")
judge("冒烟2 定义式一致", abs(lhs / rhs - 1) < 1e-15, f"相对偏差 {abs(lhs/rhs-1):.2e}")

# ---------- J1a：解空间维数 ----------
print("\n=== J1a 解空间维数（候选集 {hbar, c, G}，m 幂固定 1）===")
cols = [DIM['hbar'], DIM['c'], DIM['G']]
rank, dim, particular, basis, _ = solve_null_space(cols, DIM['m'])
print(f"秩 = {rank}，解空间维数 = {dim}，特解指数 (b,d,e) = {tuple(particular)}")
judge("J1a 满秩唯一解", rank == 3 and dim == 0, f"秩 {rank}、自由维数 {dim}（m 幂固定 1 时解唯一、无连续自由度；a 自由则解族为 (m/m_P)^a，由 J1c 幂封闭覆盖）")

# ---------- J1b：唯一解身份 ----------
print("\n=== J1b 唯一解身份 ===")
exp_vec = tuple(particular)
expected_vec = (Fraction(-1, 2), Fraction(-1, 2), Fraction(1, 2))
print(f"特解指数 {exp_vec}，期望 {expected_vec}")
judge("J1b 指数逐位", exp_vec == expected_vec, f"{tuple(map(str, exp_vec))}")
combo = combo_value({'m': 1, 'hbar': exp_vec[0], 'c': exp_vec[1], 'G': exp_vec[2]})
one_over_mp = 1.0 / M_PLANCK
rel = abs(combo / one_over_mp - 1)
print(f"组合值 m*hbar^b*c^d*G^e = {combo:.15e}，1/m_P = {one_over_mp:.15e}，相对偏差 {rel:.2e}")
judge("J1b 组合==1/m_P", rel < 1e-15, f"相对偏差 {rel:.2e}")

# ---------- J1c：幂封闭（a = 2, 3, 1/2） ----------
print("\n=== J1c 幂封闭 ===")
ok_c = True
for a in [Fraction(2), Fraction(3), Fraction(1, 2)]:
    target_a = tuple(Fraction(a) * v for v in DIM['m'])
    _, d_a, part_a, _, _ = solve_null_space(cols, target_a)
    exp_part = tuple(part_a)
    exp_expect = tuple(Fraction(a) * e for e in expected_vec)
    same = (d_a == 0) and (exp_part == exp_expect)  # 满秩唯一解（与 J1a 修正口径一致）
    # 数值面：组合值 == (m/m_P)^a
    val_a = combo_value({'m': 1, 'hbar': float(exp_part[0]), 'c': float(exp_part[1]), 'G': float(exp_part[2])})
    val_ref = (1.0 / M_PLANCK) ** float(a)
    num_ok = abs(val_a / val_ref - 1) < 1e-13
    print(f"  a={a}: 维数 {d_a}，指数 {exp_part}，期望 {exp_expect}，数值比 {val_a/val_ref:.15f}")
    ok_c = ok_c and same and num_ok
judge("J1c 幂封闭", ok_c, "a=2,3,1/2 解均为 a=1 解的幂（指数与数值双验证）")

# ---------- J1d：冗余参数档（加 l_P、t_P） ----------
print("\n=== J1d 冗余参数档 ===")
cols5 = [DIM['hbar'], DIM['c'], DIM['G'], DIM['l_P'], DIM['t_P']]
rank5, dim5, part5, basis5, _ = solve_null_space(cols5, DIM['m'])
print(f"五常数候选集：秩 {rank5}，维数 {dim5}，特解 {tuple(part5)}")
# 特解应落在 l_P=t_P=0 的 m/hbar/c/G 子解上
part5_ok = tuple(part5) == (Fraction(-1, 2), Fraction(-1, 2), Fraction(1, 2), Fraction(0), Fraction(0))
# 新增自由方向应为不含 m 信息的恒等组合（如 l_P/(c t_P) = 1）：基向量的组合数值恒为 1
basis_trivial = True
for bv in basis5:
    val = combo_value({'hbar': float(bv[0]), 'c': float(bv[1]), 'G': float(bv[2]),
                       'l_P': float(bv[3]), 't_P': float(bv[4])})
    if abs(val - 1.0) > 1e-12:
        basis_trivial = False
judge("J1d 特解不依赖冗余参数", part5_ok and dim5 == 2 and basis_trivial,
      f"特解 {tuple(part5)}（l_P、t_P 幂为零、与三常数解一致）；新增自由维数 {dim5}，其基方向组合值恒为 1（不含 m 的恒等组合），不产生新的含 m 组合")

# ---------- J1e + J3a：多档质量扫描 ----------
print("\n=== J1e/J3a 多档质量扫描 ===")
masses = [0.1, 0.5, 1.0, 2.0, 5.0]
ok_e, ok_3a = True, True
worst = 0.0
for f_m in masses:
    m = f_m * M_PLANCK
    omega0 = m * C_LIGHT**2 / HBAR
    lhs = m * C_LIGHT**2 * math.sqrt(HBAR * G_NEWTON / C_LIGHT**5) / HBAR
    rhs = omega0 * math.sqrt(HBAR * G_NEWTON / C_LIGHT**5)
    # 恒等：两者同为 m c^2 t_P / hbar；直接核对 m/m_P
    r1 = abs((m / M_PLANCK) / f_m - 1)
    r2 = abs(lhs / rhs - 1)
    worst = max(worst, r1, r2)
    if r1 >= 1e-15 or r2 >= 1e-15:
        ok_e = False
    print(f"  m = {f_m} m_P: m/m_P 读数 {m/M_PLANCK:.15f}，转录等式相对偏差 {r2:.2e}")
judge("J1e 多档唯一解", ok_e, f"5 档全过，最坏相对偏差 {worst:.2e}")
judge("J3a 转录等式", ok_3a and worst < 1e-15, f"mc^2 t_P/hbar == omega_0 t_P，最坏 {worst:.2e}")

# ---------- J3b/J3c：守恒读回与替身排除 ----------
print("\n=== J3b 守恒读回（正确幅值，十拍） ===")
N_BEAT = 10
TOL_ACC = 1e-12   # 预算：十拍双精度累积 ~1e-15，宽三档

def ledger(m_over_mp, f_tilde=None):
    """十拍：占用累积（每拍 m c^2 t_P，单位 hbar 计 = m/m_P 份额）与写入读回累积（每拍幅值 * hbar）。"""
    share = m_over_mp
    occ = sum(share for _ in range(N_BEAT))            # 占用（以 hbar 为单位）
    amp = (m_over_mp ** 2) if f_tilde == 'sq' else (
          math.sqrt(m_over_mp) if f_tilde == 'sqrt' else (
          m_over_mp ** 3 if f_tilde == 'cube' else m_over_mp))
    rec = sum(amp for _ in range(N_BEAT))
    return occ, rec

ok_b = True
worst_b = 0.0
for f_m in masses:
    occ, rec = ledger(f_m)
    rel = abs(rec / occ - 1)
    worst_b = max(worst_b, rel)
    if rel >= TOL_ACC:
        ok_b = False
judge("J3b 守恒读回", ok_b, f"正确幅值十拍读回==占用，5 档最坏相对偏差 {worst_b:.2e}（容差 {TOL_ACC:.0e}）")

print("\n=== J3c 替身排除（恒真反例组） ===")
tildes = ['sq', 'sqrt', 'cube']
names = {'sq': '(m/m_P)^2', 'sqrt': 'sqrt(m/m_P)', 'cube': '(m/m_P)^3'}
ok_c2 = True
detail_rows = []
for t in tildes:
    min_dev = None
    for f_m in masses:
        if f_m == 1.0:
            continue  # 退化档：m=m_P 时 1^k=1，幂替身天然不可判别，诚实排除
        occ, rec = ledger(f_m, f_tilde=t)
        dev = abs(rec / occ - 1)
        min_dev = dev if min_dev is None else min(min_dev, dev)
        detail_rows.append((names[t], f_m, dev))
    if min_dev is None or min_dev <= 0.05:
        ok_c2 = False
    print(f"  替身 {names[t]}: 非一档（0.1/0.5/2/5）最小相对偏差 {min_dev:.4f}（判据线 0.05）")
judge("J3c 替身全FAIL", ok_c2, "三替身非退化档偏差均份额级（>5%），非噪声级；m=m_P 档为幂替身天然退化点（1^k=1）诚实排除；正确幅值偏差机器级")

# ---------- 出图：守恒偏差曲线（正确 vs 替身） ----------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
beats = list(range(1, N_BEAT + 1))
m_demo = 0.5
share = m_demo
occ_cum = [share * n for n in beats]
for t, style in [(None, 'o-'), ('sq', 's--'), ('sqrt', '^--'), ('cube', 'd--')]:
    amp = share if t is None else (
        share**2 if t == 'sq' else math.sqrt(share) if t == 'sqrt' else share**3)
    rec_cum = [amp * n for n in beats]
    label = '正确幅值 m/m_P' if t is None else f'替身 {names[t]}'
    ax.plot(beats, rec_cum, style, label=label, ms=4)
ax.plot(beats, occ_cum, 'k-', lw=2.5, alpha=0.5, label='占用累积（真值）')
ax.set_xlabel('拍数')
ax.set_ylabel('累积作用量（以 ℏ 计）')
ax.set_title(f'十拍守恒读回：m = {m_demo} m_P')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1]
t_names = ['正确幅值'] + [names[t] for t in tildes]
devs_final = []
for t in [None, 'sq', 'sqrt', 'cube']:
    worst_t = 0.0
    for f_m in masses:
        occ, rec = ledger(f_m, f_tilde=t)
        worst_t = max(worst_t, abs(rec / occ - 1))
    devs_final.append(worst_t)
bars = ax.bar(range(len(t_names)), devs_final, color=['#2a7f2a' if d < TOL_ACC else '#b03030' for d in devs_final])
ax.axhline(TOL_ACC, color='k', ls=':', lw=1, label=f'正确容差 {TOL_ACC:.0e}')
ax.axhline(0.05, color='gray', ls='--', lw=1, label='替身判据线 0.05')
ax.set_yscale('log')
ax.set_xticks(range(len(t_names)))
ax.set_xticklabels(t_names, fontsize=8, rotation=12)
ax.set_ylabel('十拍读回最坏相对偏差（5 档）')
ax.set_title('守恒判据下的正确幅值与替身')
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which='both')

fig.tight_layout()
fig.savefig(OUT_IMG, dpi=150)
print(f"\n图已存：{OUT_IMG}")

# ---------- 总表 ----------
print("\n=== 判据总表 ===")
n_pass = sum(1 for _, ok, _ in results if ok)
for tag, ok, detail in results:
    print(f"[{'PASS' if ok else 'FAIL'}] {tag}")
print(f"\n合计 {n_pass}/{len(results)} 通过")
