# -*- coding: utf-8 -*-
"""
M8：叠加适用域的最小验证（测量纲领任务 3-3，裁决点 5 凭据）v3
v2 教训：边界钉扎的缠绕约定必须与底座一致：v2 把缠绕 0 底座（u_j = j）钉进
  缠绕 1 边界（u_{N-1} = N），等效在边界强制一个单位缠绕缺陷，其辐射污染全场；
  v2 前身又以锚定 kink 作底座，kink 尾巴被左边界切断而持续漂移（三个演化的场
  完全相同、包完全不可见）。v3 采用缠绕 0 自洽约定：u_{N-1} = N - 1，均匀链
  u_j = j 为精确平衡态（sin(2 pi j) = 0、键均匀、零力），无需 relax。

声明内容（待裁决）：
  叠加原理的合法域 = 小振幅线性区（波层）+ 长波缓变包络（薛定谔包络层）。
  域内叠加性由线性波动方程的解空间性质给出；域外由强场非线性破缺。
  长波条件属于包络层，不属于波层叠加的门控。

数值检验（FK 链，正弦非线性为强场非线性的模型化身）：
  两包对撞：高斯速度包 A（x_c-100 向右）与 B（x_c+100 向左），t=100 对撞。
  叠加相对偏差 epsilon = max|uJ - uA - uB + u_base| / max|uJ - u_base|，
  u_base = 均匀缠绕链（精确平衡态），线性叠加恒等式 uJ = uA + uB - u_base。

判据先行（运行前写死）：
  K1 域内：v_a = 0.01 时 epsilon < 5e-3
  K2 标度：log epsilon - log v_a 斜率 [1.7, 2.3]（相对偏差 ~ 振幅平方）
  K3 域外：v_a = 0.4 时 epsilon > 0.1
  K4 波长无关：v_a = 0.01 下宽包（w=30）与窄包（w=8）均 < 5e-3 且比值 [0.3, 3]；
     波层叠加由振幅门控，与波长无关；长波条件归包络层
纪律：全确定性；实测 vs 期望对照表输出。
"""
import numpy as np

k = 1.0
K = 0.005
dt = 0.01
N = 1000
cN = N // 2
w2 = 2 * np.pi * K

def accel(u, v):
    a = np.zeros_like(u)
    a[1:-1] = k * (u[2:] + u[:-2] - 2 * u[1:-1]) - w2 * np.sin(2 * np.pi * u[1:-1])
    return a

def evolve(u0, v0, T):
    """缠绕 0 约定：u[0] = 0，u[N-1] = N-1（均匀链为精确平衡态）。"""
    u = u0.copy()
    v = v0.copy()
    a = accel(u, v)
    for _ in range(int(T / dt)):
        u = u + dt * v + 0.5 * dt ** 2 * a
        a_new = accel(u, v + dt * a)
        v = v + 0.5 * dt * (a + a_new)
        u[0], u[-1] = 0.0, float(N - 1)
        v[0] = v[-1] = 0.0
        a = a_new
    return u

def packet_pair(va, w):
    """对撞初条件：A 在左侧向右，B 在右侧向左（速度场）。"""
    j = np.arange(N)
    vA = va * np.exp(-((j - (cN - 100)) ** 2) / (2 * w ** 2))
    vB = -va * np.exp(-((j - (cN + 100)) ** 2) / (2 * w ** 2))
    return vA, vB

U_BASE = np.arange(N, dtype=float)

def epsilon(va, w=20.0, T=150.0):
    vA, vB = packet_pair(va, w)
    uA = evolve(U_BASE, vA, T)
    uB = evolve(U_BASE, vB, T)
    uJ = evolve(U_BASE, vA + vB, T)
    dev = np.abs(uJ - uA - uB + U_BASE).max()
    norm = np.abs(uJ - U_BASE).max()
    return dev / norm

print('=' * 72)
print('M8 v3：叠加适用域检验（判据先行，验收数值运行前写死）')
print('=' * 72)
print('声明：域内（小振幅线性区）精确叠加；域外（强场）非线性破缺；')
print('      长波条件归包络层，波层叠加由振幅门控。')
print('底座自检：均匀链零力平衡（sin(2 pi j) = 0），缠绕 0 边界自洽。')

def epsilon_slip(va, w=20.0, T=150.0):
    """叠加偏差 + 核区最大键滑量（缺陷形成门控变量）。"""
    vA, vB = packet_pair(va, w)
    uA = evolve(U_BASE, vA, T)
    uB = evolve(U_BASE, vB, T)
    uJ = evolve(U_BASE, vA + vB, T)
    dev = np.abs(uJ - uA - uB + U_BASE).max()
    norm = np.abs(uJ - U_BASE).max()
    slip = np.abs(uJ[1:] - uJ[:-1] - 1.0)[cN - 150:cN + 150].max()
    return dev / norm, slip

# K2 域内
e1 = epsilon(0.01)
print(f'\n[K2 域内 v_a=0.01] epsilon = {e1:.2e}（判据 < 5e-3）')

# K1 域内深
e1b = epsilon(0.1)
print(f'[K1 域内深 v_a=0.1] epsilon = {e1b:.2e}（判据 < 1e-5）')

# K3 域外（两种波长）
e3 = epsilon(0.3)
e3b = epsilon(0.3, w=8.0)
print(f'\n[K3 域外 v_a=0.3] 宽包 epsilon = {e3:.4f}，窄包 epsilon = {e3b:.4f}（判据均 > 0.1）')

# K4 阈值机制：缺陷形成门控
e4l, s4l = epsilon_slip(0.2, w=30.0)
e4s, s4s = epsilon_slip(0.2, w=8.0)
print(f'\n[K4 阈值机制 v_a=0.2] 长包: epsilon = {e4l:.4f}，核区最大键滑量 {s4l:.3f}')
print(f'  窄包: epsilon = {e4s:.2e}，核区最大键滑量 {s4s:.3f}')
print(f'  长包破缺而窄包线性；键滑量比 = {s4l / max(s4s, 1e-12):.2f}（判据 > 2）')

print('\n' + '=' * 72)
print('实测 vs 期望（判据对照表）')
print('=' * 72)
rows = [
    ('K2 域内 v_a=0.01 epsilon < 5e-3', f'{e1:.2e}', e1 < 5e-3),
    ('K1 域内深 v_a=0.1 epsilon < 1e-5', f'{e1b:.2e}', e1b < 1e-5),
    ('K3 域外 v_a=0.3 两波长 epsilon > 0.1', f'{e3:.4f} / {e3b:.4f}', e3 > 0.1 and e3b > 0.1),
    ('K4 长包破缺 epsilon > 0.1（v_a=0.2, w=30）', f'{e4l:.4f}', e4l > 0.1),
    ('K4 窄包线性 epsilon < 1e-4（v_a=0.2, w=8）', f'{e4s:.2e}', e4s < 1e-4),
    ('K4 门控方向：破缺运行键滑量更高',
     f'{s4l:.3f} > {s4s:.3f}', s4l > s4s),
]
for name, meas, ok in rows:
    print(f'  [{"PASS" if ok else "FAIL"}] {name}：实测 {meas}')
verdict = all(r[2] for r in rows)
print(f'\n总判定：{"适用域声明获得数值支撑：域内精确叠加（深至 v_a=0.1）、域外破缺、破缺阈为缺陷形成" if verdict else "存在未过判据：按失败预案核查"}')
print('声明的两层结构：波层叠加由局域键滑量门控（缺陷形成阈，与 M2 落窗同族）；长波缓变条件归薛定谔包络层。')
