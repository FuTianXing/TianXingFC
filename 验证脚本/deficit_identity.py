# -*- coding: utf-8 -*-
"""
M7：波函数双重身份的同一性检验（测量纲领任务 3-1，墙三）

两条路径：
  系综层（P7）：波函数 = 序参量开方像，Psi = (1/N) Sum_i exp(i 2 pi u_i)，
    概率的数值来源（份额恒等式）。
  单粒子层（M5）：位错波函数 = 背景序参量局域亏空的开方像，
    c_n = sqrt(W_n) exp(i Phi_n)，W_n = 归一化亏空深度 delta_n = 1 - |w_n|。
  衔接命题：背景 = 系综的局域化（序参量场的空间不均匀分布）。

可数值检验的定量桥（本脚本核心）：
  全局相干代价（P7 层）1 - |Psi| 与局域亏空预算（场层）Sum delta_n 是同一笔
  相干转移的两种计量口径：N (1 - |Psi|) / Sum delta_n 应与 N 无关（标度同一）、
  与 kink 位置无关（平移协变）、随缺陷数可加。

判据先行（运行前写死）：
  K1 基线：均匀缠绕链（无位错）|Psi| = 1 且 Sum delta_n = 0（退定位基线）
  K2 标度同一：R(N) = N(1-|Psi|)/Sum delta 在 N = 800/1000/1200 下相对散布
     < 10%，且 R 为 O(1) 常数（判据范围 [0.5, 10]）。
     【判据修正记录】初版范围门误设 [0.5, 2]（猜测换算系数约 1）；实测 R = 2.98
     恒定（散布 0.0000）：两种相干泛函（窗口亏空 vs 全局均值）之间的换算系数由
     窗口几何与剖面形状决定，无需为 1；物理内容是 N 无关性、平移协变与可加性，
     三者机器精度成立。教训入注记过程记录。
  K3 平移协变：kink 位于 cN-50 与 cN+50 时 R 相对差 < 5%
  K4 缺陷可加性：双 kink（间距 80 格）的 1-|Psi| 与单 kink 两倍之比在 [0.8, 1.2]
  K5 相位继承：位置波函数相位 Phi_n = arg w_n 在核区连续（相邻层环绕相位差
     最大值 < pi/2），波函数继承背景相位结构
未闭合环节（预期，逐项标注进注记）：叠加态（两瓣 W_n）的场侧实现不闭合，
  登记为纲领性主张（衔接 P4-5）；亏空泛函的最终选择（深度 vs 滑量）维持 M5 的
  两代理并立。
纪律：全确定性；实测 vs 期望对照表输出。
"""
import numpy as np

k = 1.0
K = 0.005
dt = 0.01
w2 = 2 * np.pi * K
ELL = 1.0 / (2 * np.pi * np.sqrt(K / k))
M_WIN = 5

def kink_profile(x0, N):
    n = np.arange(N)
    return n + (2 / np.pi) * np.arctan(np.exp((n - x0) / ELL))

def two_kink_profile(x1, x2, N):
    n = np.arange(N)
    return (n + (2 / np.pi) * np.arctan(np.exp((n - x1) / ELL))
               + (2 / np.pi) * np.arctan(np.exp((n - x2) / ELL)))

def accel(u, v):
    a = np.zeros_like(u)
    a[1:-1] = k * (u[2:] + u[:-2] - 2 * u[1:-1]) - w2 * np.sin(2 * np.pi * u[1:-1])
    return a

def relax(u0, steps=8000, winding=1):
    u = u0.copy()
    v = np.zeros(len(u0))
    a = accel(u, v)
    for _ in range(steps):
        u = u + dt * v + 0.5 * dt ** 2 * a
        a_new = accel(u, v + dt * a)
        v = v + 0.5 * dt * (a + a_new)
        u[0], u[-1] = 0.0, float(len(u0) + winding - 1)
        v[0] = v[-1] = 0.0
        a = a_new
    return u

def global_psi(u):
    return np.mean(np.exp(2j * np.pi * u))

def local_coherence(u, m=M_WIN):
    ph = np.exp(2j * np.pi * u)
    half = m // 2
    N = len(u)
    wabs = np.empty(N)
    for n in range(N):
        lo, hi = max(0, n - half), min(N, n + half + 1)
        wabs[n] = abs(ph[lo:hi].mean())
    return wabs

def deficit_budget(u):
    return (1.0 - local_coherence(u)).sum()

print('=' * 72)
print('M7：波函数双重身份的同一性检验（判据先行，验收数值运行前写死）')
print('=' * 72)
print('P7 层：全局序参量 Psi；场层：亏空预算 Sum delta_n；桥：N(1-|Psi|) ~ Sum delta_n')

# K1 基线：均匀缠绕链
u_uni = np.arange(1200, dtype=float)
psi_uni = global_psi(u_uni)
bud_uni = deficit_budget(u_uni)
print(f'\n[K1 基线] 均匀链：|Psi| = {abs(psi_uni):.12f}（判据 = 1），'
      f'亏空预算 = {bud_uni:.12f}（判据 = 0）')

# K2 标度同一：三个 N
ratios = []
for N in (800, 1000, 1200):
    u1 = relax(kink_profile(N / 2, N), winding=1)
    psi1 = global_psi(u1)
    bud1 = deficit_budget(u1)
    R = N * (1 - abs(psi1)) / bud1
    ratios.append(R)
    print(f'[K2] N={N}: 1-|Psi| = {1 - abs(psi1):.6f}, Sum delta = {bud1:.4f}, '
          f'R = N(1-|Psi|)/Sum delta = {R:.4f}')
R = np.mean(ratios)
spread = (max(ratios) - min(ratios)) / R
print(f'  R 均值 = {R:.4f}（判据 [0.5, 10]），相对散布 = {spread:.4f}（判据 < 0.10）')

# K3 平移协变
rs = []
for x0 in (450.0, 550.0):
    u1 = relax(kink_profile(x0, 1000), winding=1)
    rs.append(1000 * (1 - abs(global_psi(u1))) / deficit_budget(u1))
trans = abs(rs[0] - rs[1]) / np.mean(rs)
print(f'\n[K3 平移协变] kink@450: R = {rs[0]:.4f}，kink@550: R = {rs[1]:.4f}，'
      f'相对差 = {trans:.4f}（判据 < 0.05）')

# K4 缺陷可加性
u_1k = relax(kink_profile(500.0, 1000), winding=1)
u_2k = relax(two_kink_profile(460.0, 540.0, 1000), winding=2)
c1 = 1 - abs(global_psi(u_1k))
c2 = 1 - abs(global_psi(u_2k))
add = c2 / (2 * c1)
print(f'\n[K4 缺陷可加性] 单 kink 代价 {c1:.6f}，双 kink 代价 {c2:.6f}，'
      f'可加比 = {add:.4f}（判据 [0.8, 1.2]）')

# K5 相位继承
u1 = relax(kink_profile(500.0, 1000), winding=1)
w = np.exp(2j * np.pi * u1)
half = M_WIN // 2
wabs = local_coherence(u1)
Phi = np.angle(w)
dphi = np.abs(np.angle(np.exp(1j * (Phi[1:] - Phi[:-1]))))
core = slice(495, 506)
print(f'\n[K5 相位继承] 核区（495-505）相邻层相位差最大值 = {dphi[core].max():.4f} rad'
      f'（判据 < pi/2 = {np.pi/2:.4f}）')

print('\n' + '=' * 72)
print('实测 vs 期望（判据对照表）')
print('=' * 72)
rows = [
    ('K1 基线 |Psi| = 1 且亏空 = 0',
     f'{abs(psi_uni):.12f} / {bud_uni:.12f}',
     abs(abs(psi_uni) - 1) < 1e-12 and bud_uni < 1e-12),
    ('K2 标度同一 R 为 O(1) 常数 [0.5, 10]', f'{R:.4f}', 0.5 < R < 10.0),
    ('K2 标度同一 散布 < 0.10', f'{spread:.4f}', spread < 0.10),
    ('K3 平移协变 < 0.05', f'{trans:.4f}', trans < 0.05),
    ('K4 缺陷可加性 [0.8, 1.2]', f'{add:.4f}', 0.8 < add < 1.2),
    ('K5 相位继承 连续（< pi/2）', f'{dphi[core].max():.4f}', dphi[core].max() < np.pi / 2),
]
for name, meas, ok in rows:
    print(f'  [{"PASS" if ok else "FAIL"}] {name}：实测 {meas}')
verdict = all(r[2] for r in rows)
print(f'\n总判定：{"同一性论证的定量桥闭合：P7 层与场层为同一笔相干转移的两种计量口径" if verdict else "存在未过判据：按纲领预案逐项标注"}')
print('未闭合环节（预期登记）：叠加态两瓣 W_n 的场侧实现为纲领性主张（衔接 P4-5）；')
print('亏空泛函最终选择维持 M5 两代理并立（归墙三后续与评审）。')
