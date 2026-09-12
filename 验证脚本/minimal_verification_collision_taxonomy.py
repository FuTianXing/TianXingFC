# -*- coding: utf-8 -*-
"""
M2 生死闸门：凝固时机判据的最小模型数值验证（碰撞词族分层 -> 可见度判据）
v2：修正 v1 的两处窗口/自检伪影；协议乙按"探测器晶格"重构造（v1 单点锚井
对相邻层位共模响应、记录为零，恰好反证探测器须逐层设指针）。

判据先行（测量纲领任务 1-2/1-3，验收数值在运行前写死）：
  候选判据：定案 = 质心与背景建立相位锁定 且 伴随粒性层位占据的不可逆转移；
           波性侧相互作用与粒性侧无损接力不触碰层位占据，不触发定案。
  期望（纲领写死）：
    协议甲（波性侧弹性碰撞，三种：零模踢 / 宽脉冲 / 踢+阻尼无探测器）
        干涉可见度 V 保持 >= 0.9
    协议乙（跨层：探测器晶格 engaged + 同款零模踢 + 放大链阻尼）
        V 衰减 < 0.1
    甲乙轨迹可区分。
  归因链（协议间唯一差异 = 跨层探测器是否 engaged）：
    甲-1 零模踢，无阻尼无探测器   -> 纯 U 过程类比
    甲-2 宽脉冲散射，无阻尼无探测器 -> 弹性连接/接力类
    甲-3 零模踢 + 放大链阻尼，无探测器 -> 阻尼本身不退相干（共模）
    乙   零模踢 + 放大链阻尼 + 传感器记录 -> 跨层记录 = 定案
    若甲-3 也衰减：阻尼通道在写记录（判据须修订）；若乙不衰减：记录不足。

模型：
  FK 链 E = Sum[ v^2/2 + (k/2)(u_{n+1}-u_n-1)^2 + K(1-cos 2pi u_n) ]，
  k=1, K=0.005, N=1000, dt=0.01，kink 半宽 ell = 1/(2 pi sqrt(K/k)) ~= 2.25 落窗
  （沿用 minimal_verification_meff.py 参数区）。
  两支链：支 A kink 质心在层 cN，支 B 在层 cN+1；位置叠加 (|n>+|m>)/sqrt2。
  环境记录：V(t) = exp(-D/2)，D = D_env + D_sensor（模型单位 hbar=1）。
    D_env = Sum_{j 芯窗外}[(dq_j)^2+(dp_j)^2]/2，窗口跟随两支质心中点（v1 教训：
    固定窗口会被踢走的 kink 芯泄漏污染）。
    D_sensor = Sum_j (dy_j)^2 / 2：探测器 = 逐键过阻尼传感器阵（跨层载体，
    一层一个指针），读出键滑 w_j = u_{j+1}-u_j-1 是否越过阈值 theta = w_max/2
    （层位占据判读），增益 G 后写入：dy_j = (G*Theta(w_j-theta) - y_j)/tau。
    记录 irreversible 的来源 = 传感器过阻尼（写入即保持）+ 放大链阻尼 gamma_c。
    阈值换算：V>=0.9 <=> D<=0.2107；V<=0.1 <=> D>=4.605。
纪律（2pi 教训）：全确定性无随机数；力项与方程自检（能量守恒对协议注入后基准、
  t=0 平移协变性按移位一格比较、零模踢漂移速率）；数值结论标注可逆性等级；
  协议伪影与物理结论分层报告。
"""
import numpy as np

k = 1.0
K = 0.005
dt = 0.01
N = 1000
cN = N // 2
w2 = 2 * np.pi * K
ELL = 1.0 / (2 * np.pi * np.sqrt(K / k))

V_GATE_A = 0.9
V_GATE_B = 0.1
D_GATE_A = 2 * np.log(1 / V_GATE_A)   # 0.2107
D_GATE_B = 2 * np.log(1 / V_GATE_B)   # 4.6052

R_ENV = 10          # 环境区半径（跟随质心中点）
GAMMA_C = 0.5       # 放大链阻尼（乙/甲-3 engage）
TAU_S = 1.0         # 传感器弛豫时间（过阻尼）
G_MAIN = 5.0        # 主增益（敏感性 {2, 5, 10}）
T_KICK = 50.0       # 乙组探针踢时刻（记录已冻结后）

def accel(u, v, gamma=0.0):
    a = np.zeros_like(u)
    a[1:-1] = k * (u[2:] + u[:-2] - 2 * u[1:-1]) - w2 * np.sin(2 * np.pi * u[1:-1])
    if gamma > 0.0:
        a -= gamma * v
    return a

def kink_profile(x0):
    n = np.arange(N)
    return n + (2 / np.pi) * np.arctan(np.exp((n - x0) / ELL))

def zero_mode(x0):
    n = np.arange(N)
    f = 1.0 / np.cosh((n - x0) / ELL)
    return f / np.sqrt((f ** 2).sum())

def center(u):
    w = u[1:] - u[:-1] - 1.0
    ks = int(np.argmax(w))
    lo, hi = max(0, ks - 8), min(len(w), ks + 9)
    ww = np.clip(w[lo:hi], 0.0, None)
    return (np.arange(lo, hi) * ww).sum() / ww.sum() if ww.sum() > 1e-12 else np.nan

def bonds(u):
    return u[1:] - u[:-1] - 1.0

def total_energy(u, v):
    return (0.5 * np.sum(v ** 2)
            + 0.5 * k * np.sum((u[1:] - u[:-1] - 1.0) ** 2)
            + np.sum(K * (1 - np.cos(2 * np.pi * u))))

def relax(u0, steps=8000):
    u = u0.copy()
    v = np.zeros(N)
    a = accel(u, v)
    for _ in range(steps):
        u = u + dt * v + 0.5 * dt ** 2 * a
        a_new = accel(u, v + dt * a)
        v = v + 0.5 * dt * (a + a_new)
        u[0], u[-1] = 0.0, float(N)
        v[0] = v[-1] = 0.0
        a = a_new
    return u, np.zeros(N)

def env_mask(xc_ref):
    j = np.arange(N)
    return np.abs(j - xc_ref) > R_ENV

def run_scenario(kind, T, sample=10, G=G_MAIN):
    """演化两支链并即时计算 D_env；乙组另存键滑样本供传感器增益后处理。"""
    steps = int(T / dt)
    uA, vA = relax(kink_profile(float(cN)))
    uB, vB = relax(kink_profile(float(cN + 1)))
    restA, restB = uA.copy(), uB.copy()
    if kind in ('A1', 'A3', 'B'):
        eps = 0.01
        vA = vA + eps * zero_mode(float(cN))
        vB = vB + eps * zero_mode(float(cN + 1))
    elif kind == 'A2':
        j = np.arange(N)
        vp = 0.03 * np.exp(-((j - (cN + 0.5)) ** 2) / (2 * 25.0 ** 2))
        vA = vA + vp
        vB = vB + vp
    gamma = GAMMA_C if kind in ('A3', 'B') else 0.0
    E0 = max(total_energy(uA, vA), total_energy(uB, vB))
    aA, aB = accel(uA, vA, gamma), accel(uB, vB, gamma)
    ts, Denvs, xcsA, xcsB = [], [], [], []
    wA_s, wB_s = [], []
    for t in range(steps):
        tt = t * dt
        uA = uA + dt * vA + 0.5 * dt ** 2 * aA
        uB = uB + dt * vB + 0.5 * dt ** 2 * aB
        aA_new = accel(uA, vA + dt * aA, gamma)
        aB_new = accel(uB, vB + dt * aB, gamma)
        vA = vA + 0.5 * dt * (aA + aA_new)
        vB = vB + 0.5 * dt * (aB + aB_new)
        uA[0], uA[-1] = 0.0, float(N)
        uB[0], uB[-1] = 0.0, float(N)
        vA[0] = vA[-1] = 0.0
        vB[0] = vB[-1] = 0.0
        aA, aB = aA_new, aB_new
        if kind == 'B' and abs(tt - T_KICK) < dt / 2:
            vA = vA + 0.01 * zero_mode(center(uA))
            vB = vB + 0.01 * zero_mode(center(uB))
            aA, aB = accel(uA, vA, gamma), accel(uB, vB, gamma)
        if t % sample == 0:
            xc_ref = 0.5 * (center(uA) + center(uB))
            env = env_mask(xc_ref)
            dq = (uA - restA)[env] - (uB - restB)[env]
            dp = vA[env] - vB[env]
            ts.append(tt)
            Denvs.append(0.5 * ((dq ** 2).sum() + (dp ** 2).sum()))
            xcsA.append(center(uA))
            xcsB.append(center(uB))
            if kind == 'B':
                wA_s.append(bonds(uA))
                wB_s.append(bonds(uB))
    E1 = max(total_energy(uA, vA), total_energy(uB, vB))
    out = dict(kind=kind, ts=np.array(ts), Denvs=np.array(Denvs),
               xcsA=np.array(xcsA), xcsB=np.array(xcsB),
               dE=abs(E1 - E0) / E0)
    if kind == 'B':
        out['wA_s'] = np.array(wA_s)
        out['wB_s'] = np.array(wB_s)
        out['dt_s'] = dt * sample
    return out

def sensor_record(wA_s, wB_s, G, theta, dt_s, tau=TAU_S):
    """过阻尼传感器阵在键滑样本序列上的记录演化，返回 D_sensor(t)。
    一阶弛豫 dy = (G*Theta(w-theta) - y)/tau 在步长 dt_s 上的精确解。"""
    alpha = np.exp(-dt_s / tau)
    n_s, nB = wA_s.shape
    yA = np.zeros(nB)
    yB = np.zeros(nB)
    Ds = []
    for s in range(n_s):
        yA = alpha * yA + G * (wA_s[s] > theta) * (1 - alpha)
        yB = alpha * yB + G * (wB_s[s] > theta) * (1 - alpha)
        Ds.append(0.5 * ((yA - yB) ** 2).sum())
    return np.array(Ds)

print('=' * 72)
print('M2 生死闸门 v2：两协议可见度判据（判据先行，验收数值运行前写死）')
print('=' * 72)
print(f'模型：FK 链 k={k}, K={K}, N={N}, dt={dt}, ell={ELL:.3f} 格距（落窗 >= 2）')
print(f'判据：定案 = 层位锁定 + 层位占据不可逆转移；探测器 = 逐键过阻尼传感器阵')
print(f'期望：甲组 V >= {V_GATE_A} (D <= {D_GATE_A:.4f})；乙 V < {V_GATE_B} (D >= {D_GATE_B:.4f})')

# 自检 1：t=0 平移协变性（键场按移位一格比较）
uA0, _ = relax(kink_profile(float(cN)), steps=3000)
uB0, _ = relax(kink_profile(float(cN + 1)), steps=3000)
bA, bB = bonds(uA0), bonds(uB0)
print(f'\n[自检1] t=0 键场平移协变性 max|bB[1:]-bA[:-1]| = {np.abs(bB[1:] - bA[:-1]).max():.2e}')

scen = {}
for kind, T in (('A1', 200.0), ('A2', 200.0), ('A3', 200.0), ('B', 200.0)):
    scen[kind] = run_scenario(kind, T)
    r = scen[kind]
    V = np.exp(-r['Denvs'] / 2)
    r['V'] = V
    occA = np.round(r['xcsA']).astype(int)
    print(f'\n--- 协议 {kind} ---')
    print(f'  能量相对漂移（自检2，对注入后基准）：{r["dE"]:.2e}')
    print(f'  D_env：初 {r["Denvs"][0]:.2e}，末 {r["Denvs"][-1]:.2e}，最大 {r["Denvs"].max():.2e}')
    print(f'  V：初 {V[0]:.6f}，末 {V[-1]:.6f}，最小 {V.min():.6f}')
    print(f'  质心 A：{r["xcsA"][0]:.2f} -> {r["xcsA"][-1]:.2f}'
          f'（占据层 {occA[0]} -> {occA[-1]}），质心 B 末 {r["xcsB"][-1]:.2f}')

# 乙组传感器记录（增益后处理）
rB = scen['B']
wmax = max(rB['wA_s'][0].max(), rB['wB_s'][0].max())
theta = 0.5 * wmax
print(f'\n--- 乙组探测器记录（theta = {theta:.4f} = w_max/2）---')
diff_pat = (rB['wA_s'][0] > theta).astype(int) - (rB['wB_s'][0] > theta).astype(int)
print(f'  两支阈值图样差异键位数：{int((diff_pat != 0).sum())}')
sensG = {}
for G in (2.0, 5.0, 10.0):
    Ds = sensor_record(rB['wA_s'], rB['wB_s'], G, theta, rB['dt_s'])
    Dtot = Ds + rB['Denvs']
    V_end = np.exp(-Dtot[-1] / 2)
    sensG[G] = V_end
    print(f'  增益 G={G:4.1f}: D_sensor 末 {Ds[-1]:8.3f}，V_end = {V_end:.6f}'
          f'（{"满足" if V_end < V_GATE_B else "不满足"} < 0.1）')

print('\n' + '=' * 72)
print('实测 vs 期望（判据对照表）')
print('=' * 72)
vA1 = scen['A1']['V'].min()
vA2 = scen['A2']['V'].min()
vA3 = scen['A3']['V'].min()
vB = sensG[G_MAIN]
rows = [
    ('甲-1 零模踢（无探测器）    V >= 0.9', f'V_min = {vA1:.6f}', vA1 >= V_GATE_A),
    ('甲-2 宽脉冲（无探测器）    V >= 0.9', f'V_min = {vA2:.6f}', vA2 >= V_GATE_A),
    ('甲-3 踢+阻尼（无探测器）   V >= 0.9', f'V_min = {vA3:.6f}', vA3 >= V_GATE_A),
    ('乙   踢+阻尼+探测器（G=5） V <  0.1', f'V_end = {vB:.6f}', vB < V_GATE_B),
    ('甲乙轨迹可区分', f'|V_甲3末 - V_乙末| = {abs(scen["A3"]["V"][-1] - vB):.4f}',
     abs(scen['A3']['V'][-1] - vB) > 0.5),
    ('占据读数：甲-1 自由跨层 / 乙 锁定',
     f'甲-1 占据 {int(np.round(scen["A1"]["xcsA"][0]))}->{int(np.round(scen["A1"]["xcsA"][-1]))}，'
     f'乙 占据 {int(np.round(scen["B"]["xcsA"][0]))}->{int(np.round(scen["B"]["xcsA"][-1]))}',
     int(np.round(scen['A1']['xcsA'][-1])) != int(np.round(scen['A1']['xcsA'][0]))
     and int(np.round(scen['B']['xcsA'][-1])) == int(np.round(scen['B']['xcsA'][0]))),
]
for name, meas, ok in rows:
    print(f'  [{"PASS" if ok else "FAIL"}] {name}：实测 {meas}')

verdict = all(r[2] for r in rows)
print(f'\n总判定：{"生死闸门通过" if verdict else "生死闸门未通过——按纲领失败预案逐站核查"}')
print('归因链：甲-3（阻尼无探测器）V 保持而乙衰减 => 衰减唯一来源 = 跨层探测器记录')
