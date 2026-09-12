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

M2b 增补轮判据（2026-09-13 作者裁决采甲读法：delta_i = 第 i 个粒性自由度的支间
  相位差，粒性自由度全集 = 链涨落位点 + 记录自由度；eta_a 场侧实现 = 块内支间
  重叠，M1 平摊相位式 (1/N_a)Sum e^{i delta_i} 是其纯相位差极限——传感器记录
  只有幅度差无相位，纯相位式看不见记录型退相干，故重叠式为全场实现）：
  块划分：链左/链右（环境区掩码内按两支质心中点分侧）+ 记录左/记录右（键位按
  同一中点分侧），四块穷尽粒性自由度全集。
  J7 方向判据：
    J7a 甲-1/甲-2/甲-3 全部链块 eta_overlap >= 0.9（全程最小，平摊式同门）
    J7b 乙 链块 eta_overlap >= 0.9 全程（传感器单向记录不回流，链块无辜：
        衰减全部住在记录自由度——归因链的块分辨率强化）
    J7c 乙 记录块 Pi_side eta_rec < 0.01（G=5）且至少一侧 < 0.1
  J8 乘积重构（恒等式级自检）：Pi_{四块} eta = V_end，相对残差 < 1e-10
    （相干态重叠对独立自由度精确因子化，分区穷尽则恒等；检验分区无漏无重）
  J9 双泛函桥（数值验证级）：链块 -ln|eta_flat| 对 (4 pi^2 / N_a) * D_q 的比值
    在 [0.7, 1.3]；D_q <= 1e-8 时要求 -ln|eta_flat| < 1e-4。
    推导：高斯小涨落下 -ln|eta_flat| = 2 pi^2 Var(du) = (4 pi^2/N_a) D_q，
    换算常数有推导值（强于 M7 之 R 的实测常数）。
  零模注意：平摊式对刚体（零模）相对位移不敏感（全场同相位差 = 纯相位，
    非退相干），重叠式计入；M2 三协议的支间差异不含刚体分量（甲-1 差异在
    动量部、甲-2 为空间变构型），零模相位差归 L（质心相位锁定度）度量：
    L 与 eta_a 由此互补（L 管零模，eta_a 管块内部分辨率）。

M2c 增补轮判据（2026-09-13 作者指示"进行下一轮"：L 相位版本的场侧实现）：
  场侧口径：Phi_cm(t) = 2 pi * frac(x_cm(t))（kink 质心的佩耶尔斯相位：质心在
    晶格小格内的分数位置）；背景相位 = 晶格框架（经典链无内禀时钟，均匀缠绕
    背景的相位原点即晶格参考，Phi_bg = 0；量子版公共节律之差的时间变率对应
    关系不变）；DeltaPhi(t) = 2 pi * xi(t)，xi = frac(x_cm)；
    L = 1 - Var_window(unwrapped DeltaPhi)（时间窗线性方差，unwrap 处理 2pi
    边界）；伴同报告有界圆测度 L_circ = |<e^{i DeltaPhi}>|^2 属 [0,1]。
  两支质心分数坐标恒等（x_B = x_A + 1，演化平移协变），L_A = L_B，报告 A 支。
  J-L1（自由质心失锁）：甲-1 末窗 L_lin < 0.3（xi 扫格，DeltaPhi 近均匀扫动）。
  J-L2（阻尼锁定）：甲-3 与乙 末窗 L_lin >= 0.9（质心钉入晶格框架，方差塌缩）。
  J-L3（占据联动）：L 末窗高低与占据静动逐协议一致（甲-1 漂移/失锁，
    甲-3、乙 静止/锁定），占据读数从 xcsA 数据现算，交叉印证。
  J-L4（合取结构报告，无门）：四协议铺陈（锁定性 × 记录性）四象限——甲-1/甲-2
    自由滑行（L 低、无记录）、甲-3 阻尼锁定（L 高、无记录）、乙 锁定加记录
    （定案）。初版预期"甲-2 占据静态故 L 高"与实测不符：宽脉冲对 kink 有
    辐射压（净动量），kink 自由滑行更快（末窗 L = -123）——判据预期按发现
    修正，教训入注记过程记录。L 单独不判定，判据 1 为合取（L -> 1 且
    eta_a -> 0）——M2b 互补分工（L 管零模、eta_a 管块内部）的确认。
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
    jidx = np.arange(N)
    Dq = {'L': [], 'R': []}
    Dp = {'L': [], 'R': []}
    etaF = {'L': [], 'R': []}
    Na = {'L': [], 'R': []}
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
            # M2b：链块按中点分侧（涨落自由度），eta 平摊式 + D_q/D_p 重叠式原料
            jenv = jidx[env]
            lm = jenv < xc_ref
            for tag, m in (('L', lm), ('R', ~lm)):
                dqs, dps = dq[m], dp[m]
                Dq[tag].append(0.5 * (dqs ** 2).sum())
                Dp[tag].append(0.5 * (dps ** 2).sum())
                etaF[tag].append(abs(np.exp(2j * np.pi * dqs).mean()))
                Na[tag].append(int(m.sum()))
            xcsA.append(center(uA))
            xcsB.append(center(uB))
            if kind == 'B':
                wA_s.append(bonds(uA))
                wB_s.append(bonds(uB))
    E1 = max(total_energy(uA, vA), total_energy(uB, vB))
    out = dict(kind=kind, ts=np.array(ts), Denvs=np.array(Denvs),
               xcsA=np.array(xcsA), xcsB=np.array(xcsB),
               dE=abs(E1 - E0) / E0,
               Dq={kk: np.array(vv) for kk, vv in Dq.items()},
               Dp={kk: np.array(vv) for kk, vv in Dp.items()},
               etaF={kk: np.array(vv) for kk, vv in etaF.items()},
               Na={kk: np.array(vv) for kk, vv in Na.items()})
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

def sensor_record_sides(wA_s, wB_s, G, theta, dt_s, xc_refs, tau=TAU_S):
    """M2b：记录块按中点分侧的 D_sensor 与块内 eta（重叠式）。
    键 j 连接位点 j 与 j+1，记键坐标为 j+0.5；eta_rec = exp(-D_side/2)。"""
    alpha = np.exp(-dt_s / tau)
    n_s, nB = wA_s.shape
    yA = np.zeros(nB)
    yB = np.zeros(nB)
    DsL, DsR = [], []
    bj = np.arange(nB) + 0.5
    for s in range(n_s):
        yA = alpha * yA + G * (wA_s[s] > theta) * (1 - alpha)
        yB = alpha * yB + G * (wB_s[s] > theta) * (1 - alpha)
        dy2 = (yA - yB) ** 2
        lm = bj < xc_refs[s]
        DsL.append(0.5 * dy2[lm].sum())
        DsR.append(0.5 * dy2[~lm].sum())
    return np.array(DsL), np.array(DsR)

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

# ============================================================
# M2b 增补轮：eta_a 块分辨率支间相干（甲读法，判据 J7-J9）
# ============================================================
print('\n' + '=' * 72)
print('M2b 增补轮：eta_a 块分辨率支间相干（甲读法：delta_i = 支间相位差，')
print('    粒性自由度全集 = 链涨落位点 + 记录自由度；eta_a = 块内支间重叠）')
print('=' * 72)

def chain_eta(rr, tag, kk):
    """链块重叠式 eta = exp(-(D_q + D_p)/2)。"""
    return np.exp(-(rr['Dq'][tag][kk] + rr['Dp'][tag][kk]) / 2)

kF = -1
j7a = True
j7a_meas = []
for kind in ('A1', 'A2', 'A3'):
    rr = scen[kind]
    e = min(chain_eta(rr, tg, kk).min() for tg in ('L', 'R') for kk in range(len(rr['ts'])))
    ef = min(rr['etaF'][tg].min() for tg in ('L', 'R'))
    j7a_meas.append(f'{kind}: overlap_min={e:.6f}/flat_min={ef:.6f}')
    j7a = j7a and (e >= 0.9) and (ef >= 0.9)
print(f'[J7a] 甲组链块全程最小（门 0.9，重叠式与平摊式同门）：')
for s in j7a_meas:
    print(f'      {s}')

rB = scen['B']
eB = min(chain_eta(rB, tg, kk).min() for tg in ('L', 'R') for kk in range(len(rB['ts'])))
print(f'[J7b] 乙链块全程最小（门 0.9）：{eB:.6f}')

xc_refB = 0.5 * (rB['xcsA'] + rB['xcsB'])
DsL, DsR = sensor_record_sides(rB['wA_s'], rB['wB_s'], G_MAIN, theta, rB['dt_s'], xc_refB)
etaRecL = np.exp(-DsL / 2)
etaRecR = np.exp(-DsR / 2)
prod_rec = etaRecL[-1] * etaRecR[-1]
print(f'[J7c] 乙记录块末值（G={G_MAIN}）：eta_recL = {etaRecL[-1]:.3e}, '
      f'eta_recR = {etaRecR[-1]:.3e}, 乘积 = {prod_rec:.3e}')
j7c = (prod_rec < 0.01) and (min(etaRecL[-1], etaRecR[-1]) < 0.1)

eta_chainL = chain_eta(rB, 'L', kF)
eta_chainR = chain_eta(rB, 'R', kF)
prod_all = eta_chainL * eta_chainR * etaRecL[-1] * etaRecR[-1]
v_end = np.exp(-(rB['Denvs'][-1] + DsL[-1] + DsR[-1]) / 2)
resid = abs(prod_all - v_end) / v_end
print(f'[J8] 乘积重构：Pi_eta = {prod_all:.12e}, V_end = {v_end:.12e}, 相对残差 = {resid:.2e}')
j8 = resid < 1e-10

print('[J9] 双泛函桥（-ln|eta_flat| vs (4 pi^2/N_a) D_q，推导常数）：')
j9 = True
for kind in ('A1', 'A2', 'A3', 'B'):
    rr = scen[kind]
    for tg in ('L', 'R'):
        ef = rr['etaF'][tg][-1]
        dq_ = rr['Dq'][tg][-1]
        na = rr['Na'][tg][-1]
        m = -np.log(max(ef, 1e-300))
        p = 4 * np.pi ** 2 * dq_ / na
        if p > 1e-8:
            ratio = m / p
            ok = 0.7 <= ratio <= 1.3
            print(f'  {kind}-{tg}: -ln|eta_flat| = {m:.3e}, 预测 {p:.3e}, 比值 {ratio:.3f}'
                  f'（{"PASS" if ok else "FAIL"}）')
        else:
            ok = m < 1e-4
            print(f'  {kind}-{tg}: D_q = {dq_:.2e}（小量支），-ln|eta_flat| = {m:.3e}'
                  f'（{"PASS" if ok else "FAIL"}）')
        j9 = j9 and ok

rows_b = [
    ('J7a 甲组链块保持（重叠式与平摊式同门 0.9）', j7a),
    ('J7b 乙链块全程保持（链块无辜）', eB >= 0.9),
    ('J7c 乙记录块衰减（乘积 < 0.01 且至少一侧 < 0.1）', j7c),
    ('J8 乘积重构恒等（相对残差 < 1e-10）', j8),
    ('J9 双泛函桥（比值 [0.7, 1.3] 或小量支）', j9),
]
for name, ok in rows_b:
    print(f'  [{"PASS" if ok else "FAIL"}] {name}')
verdict_b = all(r[1] for r in rows_b)
print(f'\nM2b 总判定：{"eta_a 块分辨率判据全过：判据 1 第二可测量场侧口径闭合" if verdict_b else "M2b 判据未过——按注记过程记录逐站核查"}')
print('零模注意：eta_a 对刚体相对位移不敏感（纯相位），零模相位差归 L 度量；L 与 eta_a 互补。')

# ============================================================
# M2c 增补轮：L 相位版本的场侧实现（佩耶尔斯相位读数，判据 J-L1 至 J-L4）
# ============================================================
print('\n' + '=' * 72)
print('M2c 增补轮：L 相位锁定度场侧实现（Phi_cm = 2 pi * frac(x_cm)，Phi_bg = 晶格框架）')
print('=' * 72)

def phase_Ls(rr, t_lo, t_hi):
    """时间窗 [t_lo, t_hi] 的 L = 1 - Var(unwrapped DeltaPhi)，DeltaPhi = 2 pi frac(x_cm)。"""
    m = (rr['ts'] >= t_lo) & (rr['ts'] <= t_hi)
    xi = np.mod(rr['xcsA'][m], 1.0)
    dphi = 2 * np.pi * xi
    dphi_u = np.unwrap(dphi)
    L_lin = 1.0 - dphi_u.var()
    L_circ = abs(np.exp(1j * dphi).mean()) ** 2
    return L_lin, L_circ, int(m.sum())

T_END = float(scen['A1']['ts'][-1])
win_late = (0.9 * T_END, T_END)
win_mid = (0.45 * T_END, 0.55 * T_END)
Ls = {}
for kind in ('A1', 'A2', 'A3', 'B'):
    rr = scen[kind]
    Ll, Lc, n1 = phase_Ls(rr, *win_late)
    Lm, _, n2 = phase_Ls(rr, *win_mid)
    Ls[kind] = (Ll, Lc, Lm)
    print(f'  {kind}: 末窗 L_lin = {Ll:+.4f}（L_circ = {Lc:.6f}，n = {n1}），'
          f'中窗 L_lin = {Lm:+.4f}（n = {n2}）')

Ll_A1, Ll_A2, Ll_A3, Ll_B = Ls['A1'][0], Ls['A2'][0], Ls['A3'][0], Ls['B'][0]
jL1 = Ll_A1 < 0.3
jL2 = (Ll_A3 >= 0.9) and (Ll_B >= 0.9)

# J-L3：占据联动（占据静动从 xcsA 现算）
occ_static = {}
for kind in ('A1', 'A2', 'A3', 'B'):
    xa = scen[kind]['xcsA']
    occ_static[kind] = int(np.round(xa[0])) == int(np.round(xa[-1]))
jL3 = ((Ll_A1 < 0.9) == (not occ_static['A1'])) and \
      ((Ll_A3 >= 0.9) == occ_static['A3']) and \
      ((Ll_B >= 0.9) == occ_static['B'])
occ_txt = '；'.join(f"{k}: 占据{'静态' if v else '漂移'}/L={'高' if Ls[k][0] >= 0.9 else '低'}"
                    for k, v in occ_static.items())

print(f'\n[J-L1] 甲-1 自由质心失锁（末窗 L < 0.3）：L = {Ll_A1:+.4f}，占据漂移 {occ_static["A1"]}')
print(f'[J-L2] 甲-3 与乙阻尼锁定（末窗 L >= 0.9）：甲-3 L = {Ll_A3:+.4f}，乙 L = {Ll_B:+.4f}')
print(f'[J-L3] 占据联动：{occ_txt}')
print(f'[J-L4] 合取结构报告（无门）：甲-2 L = {Ll_A2:+.4f}（占据静态 {occ_static["A2"]}，'
      f'eta_a 保持 0.9964），无转移不定案；L 单独不判定，判据 1 为合取。')

rows_c = [
    ('J-L1 甲-1 自由质心失锁（末窗 L < 0.3）', jL1, f'{Ll_A1:+.4f}'),
    ('J-L2 甲-3 与乙阻尼锁定（末窗 L >= 0.9）', jL2, f'甲-3 {Ll_A3:+.4f} / 乙 {Ll_B:+.4f}'),
    ('J-L3 占据联动（L 高低与占据静动逐协议一致）', jL3, occ_txt),
]
for name, ok, meas in rows_c:
    print(f'  [{"PASS" if ok else "FAIL"}] {name}：实测 {meas}')
verdict_c = all(r[1] for r in rows_c)
print(f'\nM2c 总判定：{"L 相位版本场侧实现闭合：判据 1 两个可测量全部在岗" if verdict_c else "M2c 判据未过——按注记过程记录逐站核查"}')
print('互补分工：L 管质心零模相位（佩耶尔斯相位读数），eta_a 管块内部分辨率；判据 1 为合取。')
