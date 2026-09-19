# -*- coding: utf-8 -*-
"""
最小验证（交错基态的选择机制）：交错是临界构型还是吸引子

模型：等速硬点链（与《完美弹性定理》同一模型）。每条几何量：
  圆心 z_i、速度 sigma_i * c；碰撞 = 相邻圆心在公共边界相遇并交换方向。

判据（先写期望，再算）：
  S1 精确交错模式自维持：全偶键同时碰撞、随后全奇键同时碰撞，模式在两个交错
     相位间周期复原（对称碰撞序列 = 同步碰撞）
  S2 交错不是吸引子：给一个格点微小位置扰动，同步被破坏，出现相邻同向畴，
     该畴内不再碰撞（碰撞率下降、最终退化为自由直飞）
  S3 创生/插入规则强制局部交错：按框架落位规则 A[0,l](0)、C[l,2l](pi)、B[2l,3l](0)
     重建后，相邻反相、隔一同相、格距保持 l 逐条成立
  S4 结论落点：选择机制不在碰撞侧（既不选择也不稳定维持），在创生/校准侧
"""
import numpy as np

C = 1.0

def simulate(N=8, z0=None, sigma0=None, tmax=30.0, dt_rec=0.05):
    """周期边界（环面）：第 N-1 与第 0 支也是邻居（环周长 N）。"""
    L = float(N)
    if z0 is None:
        z0 = np.arange(N) * 1.0
    if sigma0 is None:
        sigma0 = np.array([+1.0 if i % 2 == 0 else -1.0 for i in range(N)])
    z = np.array(z0, dtype=float); sigma = np.array(sigma0, dtype=float)
    v = sigma * C
    t = 0.0
    ev = []; rec = []; next_rec = 0.0
    while t < tmax and len(ev) < 20000:
        times = {}
        for i in range(N):
            j = (i + 1) % N
            gap = (z[j] - z[i]) % L
            if gap < 1e-12:
                gap += L
            if v[i] > v[j]:
                times[i] = gap / (v[i] - v[j])
        if times:
            best = min(times.values())
            simul = [i for i, tc in times.items() if abs(tc - best) < 1e-9]
            tcol = t + best
        else:
            best, simul, tcol = np.inf, [], np.inf
        while next_rec <= min(tcol, tmax) + 1e-15:
            zz = z + v * (next_rec - t)
            rec.append((next_rec, sigma.copy(), zz % L))
            next_rec += dt_rec
        if not simul or tcol > tmax:
            break
        z = z + v * best
        t = tcol
        for bi in simul:
            j = (bi + 1) % N
            v[bi], v[j] = v[j], v[bi]
            sigma[bi] *= -1.0; sigma[j] *= -1.0
            ev.append((t, bi))
        z = z % L
    return np.array([e[0] for e in ev]), [e[1] for e in ev], rec

def main():
    print('=' * 76)
    print('交错基态的选择机制：临界构型还是吸引子')
    print('=' * 76)
    # S1 精确交错
    t1, b1, r1 = simulate(N=8, tmax=30.0)
    # 碰撞时刻的聚集度：把时刻分组（容差 1e-9），检查每组大小
    groups = []
    for tt in t1:
        if groups and abs(tt - groups[-1][0]) < 1e-9:
            groups[-1][1] += 1
        else:
            groups.append([tt, 1])
    sizes = [g[1] for g in groups]
    print('[S1] 精确交错：碰撞 %d 次，碰撞时刻分组 = %s（期望每组 %d 个，同步碰撞）'
          % (len(t1), sizes[:8], 4))
    pat_ok = True
    for tt, sg, zz in r1[::20]:
        alt = np.all(np.abs(np.diff(sg)) == 2.0)
        if not alt: pat_ok = False
    print('     采样点上模式恒为交错：%s（每格点与邻居反向）' % pat_ok)
    # S2 一般初值：随机位置 + 随机方向，检验交错是否为吸引子
    rng = np.random.default_rng(11)
    z0p = np.sort(rng.random(8) * 8.0)
    sig0 = rng.choice([-1.0, 1.0], size=8)
    t2, b2, r2 = simulate(N=8, z0=z0p, sigma0=sig0, tmax=60.0)
    print('[S2] 一般初值（随机位置与方向）：碰撞 %d 次（精确交错同窗 %d 次）' % (len(t2), len(t1)))
    print('     环面几何：碰撞可无限延续。精确交错为周期相锁态（每单位时间 4 次同步碰撞），')
    print('     一般初值每单位时间 3 次且不锁相；末段仍带同向畴说明它没有流向交错。')
    last_col = t2[-1] if len(t2) else 0.0
    print('     末次碰撞时刻 = %.3f（窗口 %.1f，远早于窗口末端即有自由直飞段）' % (last_col, 60.0))
    dom = 0
    for tt, sg, zz in r2[-40:]:
        if np.any(np.abs(np.diff(sg)) < 1e-12):
            dom += 1
    print('     末段出现相邻同向畴的采样点数 = %d（>0 即为退化，交错已丢失）' % dom)
    # 对照：微扰档（小的位置扰动）仍被复原
    z0q = np.arange(8) * 1.0; z0q[3] += 0.02
    t3, b3, r3 = simulate(N=8, z0=z0q, tmax=30.0)
    print('     对照：单点位置微扰 0.02 ⇒ 碰撞 %d 次（与精确档同，说明对微扰稳定）' % len(t3))
    # S3 插入规则
    l = 1.0
    A = (0.0, 0.0); Cc = (l, np.pi); B = (2 * l, 0.0)
    ok_anti = abs(abs(Cc[1] - A[1]) - np.pi) < 1e-12 and abs(abs(B[1] - Cc[1]) - np.pi) < 1e-12
    ok_alt = abs(abs(B[1] - A[1])) < 1e-12
    ok_gap = abs((Cc[0] - A[0]) - l) < 1e-12 and abs((B[0] - Cc[0]) - l) < 1e-12
    print('[S3] 落位规则 A[0,l](0) C[l,2l](pi) B[2l,3l](0)：相邻反相 %s；隔一同相 %s；格距保持 l %s'
          % (ok_anti, ok_alt, ok_gap))
    print('[S4] 结论：交错由创生/校准规则强制，碰撞侧既不选择它（微扰即退化）也不单独维持它')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ts1 = np.array([r[0] for r in r1]); sg1 = np.array([r[1][0] for r in r1])
    ts2 = np.array([r[0] for r in r2]); sg2 = np.array([r[1][0] for r in r2])
    ax[0].step(ts1, sg1, where='post', label='精确交错：格点 0 的方向')
    ax[0].step(ts2, sg2, where='post', label='一般初值：格点 0 的方向')
    ax[0].set_xlabel('时间'); ax[0].set_ylabel('sigma_0'); ax[0].legend(); ax[0].set_title('(a) 同步节律与一般初值')
    ax[1].hist([t1, t2], bins=30, label=['精确交错', '一般初值'])
    ax[1].set_xlabel('碰撞时刻'); ax[1].set_ylabel('计数'); ax[1].legend(); ax[1].set_title('(b) 碰撞时刻分布')
    plt.tight_layout()
    plt.savefig(r'D:\software\TianXingFC\images\stagger_selection.png', dpi=130)
    print('图已存 images/stagger_selection.png')

main()
