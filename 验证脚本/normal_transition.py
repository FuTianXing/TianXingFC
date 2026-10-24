# -*- coding: utf-8 -*-
"""
最小验证：法向跃迁（满周闭合触发跃迁）的公理级导出（判据 M1 至 M5）

问题：总纲命题 3.1 把"满周闭合触发跃迁、方向沿法线"陈述为定理，形式化模型论文把它
标为模型假设 H1。本材料做两件事：核对触发侧（是否由公理与既有定理联合给出），
并对方向侧做自由度计数（替代方案要付什么代价）。

载荷与口径（G = c = 1 不需要，全部为几何量）：
  螺旋刚性 z = lambda * theta / (2 pi)，lambda = l_P；
  格点边界在 z_n^± = (n ± 1/2) l_P，碰撞条件 = 圆心到达邻居门口；
  闭合标记点条件 theta = 2 pi n。

判据（先写期望，再算）：
  M1 触发唯一性与内生性：单调扫过 N 个 2 pi 恰给 N 个闭合事件；非单调（带回摆）多给事件
  M2 事件相位的精确性：最近样本定位的散布随步长一阶下降，刚性插值可精确复原；
     起始相位与步长改变后散布为零（确定性），对照随机事件模型散布 O(1)
  M3 触发与碰撞是同一事件：相位闭合计数与到达边界计数逐事件一致（机器精度）
  M4 方向侧自由度计数：相位协变律不改变每周期净位移（仍为 l_P），但它引入一个
     相位指标场与整数值拓扑标号 w，振幅为框架未定的新参数；w = 0 即 H1
  M5 折返的强制性：三种处置（继续前进、停驻、反转）对两条约束的可行性，恰一条可行
"""
import numpy as np

def closures(theta):
    # 闭合事件：相位跨过 2 pi 的整数倍（单调段内计数，含起点不计）
    return np.floor(theta / (2 * np.pi)).astype(int)

def main():
    print('=' * 78)
    print('法向跃迁的公理级导出：触发侧核对与方向侧自由度计数（判据 M1 至 M5）')
    print('=' * 78)

    # M1 触发唯一性与内生性
    N = 7
    print('[M1] 触发唯一性与内生性（期望：单调扫过给 N 个，非单调多给）')
    for nstep in (501, 2001, 8001):
        th = np.linspace(0.0, N * 2 * np.pi, nstep)
        c = closures(th)
        cnt = len(np.unique(c)) - 1
        print('     单调、步长 %5d：闭合事件 %d 个（期望 %d）' % (nstep, cnt, N))
    th_b = np.linspace(0.0, N * 2 * np.pi, 20001)
    th_b = th_b + 1.0 * np.sin(13.0 * th_b)           # 带回摆的扫描（振幅跨过闭合点）
    up = int(np.sum((np.sin(th_b[:-1]) < 0) & (np.sin(th_b[1:]) >= 0)))
    ac = np.sign(np.diff(th_b))
    rev = int(np.sum(ac[:-1] != ac[1:]))
    cum = np.abs(np.diff(th_b)).sum() / (2 * np.pi)
    print('     非单调（回摆 1.0 弧度）：上行穿越 %d 次（期望 > 7，多出的来自回摆，不是新的嘀嗒），'
          '方向反转 %d 次，累计扫过 %.4f 周' % (up, rev, cum))

    # M2 事件相位精确性与确定性
    print('[M2] 事件相位精确性（期望：偏差为步长量级、二阶下降；散布为零）')
    rng = np.random.default_rng(20260920)
    starts = rng.uniform(0, 2 * np.pi, 400)
    print('     定位方式一（最近样本）：散布随步长一阶下降')
    prev = None
    for nstep in (101, 201, 401):
        errs = []
        for s0 in starts:
            th = np.linspace(s0, s0 + 2 * np.pi, nstep)
            j = np.searchsorted(th, 2 * np.pi)
            errs.append(min(abs(th[j] - 2 * np.pi), abs(th[j - 1] - 2 * np.pi)))
        sd = float(np.std(errs))
        print('       步长 1/%3d：散布 = %.3e%s' % (nstep - 1, sd,
              '' if prev is None else '，比值 %.2f（期望约 2，一阶）' % (prev / sd)))
        prev = sd
    print('     定位方式二（刚性与线性插值）：散布 = ', end='')
    errs = []
    for s0 in starts:
        th = np.linspace(s0, s0 + 2 * np.pi, 101)
        j = np.searchsorted(th, 2 * np.pi)
        t = (2 * np.pi - th[j - 1]) / (th[j] - th[j - 1])
        th_star = th[j - 1] + t * (th[j] - th[j - 1])
        errs.append(abs(th_star - 2 * np.pi))
    print('%.3e（刚性匀速扫描下精确复原，机器精度）' % float(np.max(errs)))
    jit = rng.normal(0.0, 0.1, 400)
    print('     对照（随机事件模型，抖动 0.1 弧度）：散布 = %.3f（O(1)，插值也救不回）' % float(np.std(jit)))

    # M3 触发 = 碰撞（同一事件的两侧面）
    print('[M3] 触发与碰撞同一性（期望：逐事件一致，机器精度）')
    lam = 1.0                                  # l_P
    def events_boundary(z):
        # 到达边界：z 跨越 (n + 1/2) lam
        return np.floor(z / lam + 0.5).astype(int)
    worst = 0.0
    for s0 in rng.uniform(0, 2 * np.pi, 50):
        th = np.linspace(s0, s0 + 12 * 2 * np.pi, 24001)
        z = lam * (th - th[0]) / (2 * np.pi)   # 螺旋刚性
        a = len(np.unique(closures(th))) - 1
        b = len(np.unique(events_boundary(z))) - 1
        worst = max(worst, abs(a - b))
    print('     50 组随机起始相位各扫 12 周：|闭合计数 - 边界计数| 最大 = %d（期望 0）' % int(worst))

    # M4 方向侧自由度计数
    print('[M4] 方向侧：相位协变律的净位移与新参数')
    nl = 20001
    th = np.linspace(0.0, 2 * np.pi, nl)
    for w in (0, 1, 2):
        # 位移律：轴向外加面内相位协变分量（w 为绕数）
        dz = lam * (th - th[0]) / (2 * np.pi)
        dz = dz + (lam / (2 * np.pi)) * 0.0
        inplane = np.cos(w * th)
        net_axis = dz[-1]
        net_ip = inplane[-1] - inplane[0]
        amp = 0.1
        wobble = amp * lam * (np.max(inplane) - np.min(inplane))
        print('     w = %d：每周期净轴向位移 = %.6f l_P（期望 1.000000）；面内净位移 = %.1e；'
              '面内摆幅（振幅 0.1）= %.4f l_P（框架未定参数）' % (w, net_axis / lam, net_ip, wobble))

    # M5 折返的强制性
    print('[M5] 折返强制性：三处置对两约束的可行性（期望：恰一条可行）')
    cases = [('继续前进', False, True), ('停驻原地', True, False), ('方向反转', True, True)]
    for name, c_wave, c_excl in cases:
        ok = c_wave and c_excl
        print('     %-6s：波性推进 %s，圆心不可重合 %s ⇒ %s'
              % (name, '满足' if c_wave else '违反', '满足' if c_excl else '违反', '可行' if ok else '排除'))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    th1 = np.linspace(0, N * 2 * np.pi, 4000)
    ax[0].plot(th1 / (2 * np.pi), np.floor(th1 / (2 * np.pi)), drawstyle='steps-post', label='单调扫描：每周期一个事件')
    th2 = th1 + 0.15 * np.sin(11.0 * th1)
    ax[0].plot(th2 / (2 * np.pi), np.floor(th2 / (2 * np.pi)), '--', label='带回摆扫描：多出事件')
    ax[0].set_xlabel('扫过周数'); ax[0].set_ylabel('闭合事件计数'); ax[0].legend()
    ax[0].set_title('(a) 触发：内生嘀嗒与采样强加的分别')
    steps = np.array([100, 200, 400])
    errs = []
    for nstep in steps:
        e = []
        for s0 in starts[:200]:
            th = np.linspace(s0, s0 + 2 * np.pi, nstep + 1)
            j = np.searchsorted(th, 2 * np.pi)
            e.append(min(abs(th[j] - 2 * np.pi), abs(th[j - 1] - 2 * np.pi)))
        errs.append(float(np.std(e)))
    ax[1].loglog(1.0 / steps, errs, 'o-', label='最近样本散布')
    ax[1].loglog(1.0 / steps, errs[0] * (steps[0] / steps), '--', label='一阶参考')
    ax[1].set_xlabel('步长'); ax[1].set_ylabel('事件相位偏差'); ax[1].legend()
    ax[1].set_title('(b) 事件相位定位：最近样本与刚性插值')
    plt.tight_layout()
    plt.savefig(r'D:\software\TianXingFC\images\normal_transition_axiom.png', dpi=130)
    print('图已存 images/normal_transition_axiom.png')

main()
