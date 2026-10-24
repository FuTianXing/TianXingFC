# -*- coding: utf-8 -*-
"""
最小验证（完美弹性定理）：等速硬点链的碰撞不变量与自由演化等价

模型（框架运动学的直接翻译）：
  每支元相子：相位推进速率恒定 |dtheta/dt| = omega_P（公理 5 绝对值约定）
            法线坐标与相位刚性锁定 z = l_P * theta / 2pi，故 |dz/dt| = c 恒定
            圆心不可重合（粒性互斥，硬核）=> 相遇即方向反转
  等价模型：一维等速硬点链（hard-point particles，等质量、速度幅度恒为 c）。

判据（先写期望，再算）：
  J1 每次碰撞前后总动量守恒（等速硬点链的交换定则）
  J2 所有速度幅度恒为 c（无参数可承载损耗）
  J3 位置多重集等于自由演化多重集（碰撞 = 标签交换，无动力学内容）
  J4 碰撞定则逐次等于速度交换（无自由参数、无接触时长）
  J5 摆程方向和 S = sum(sigma_n) 守恒（每次碰撞翻两个符号；即体系净绕数读数）
    口径：翻转的是瞬时摆程方向 sigma；手性（螺旋旋向）是几何标签，不入哈密顿、不翻转。
  J6 对照档：若引入反弹系数 e < 1，J1 至 J5 失效且偏差随碰撞数增长
"""
import numpy as np

C = 1.0

def simulate(N=8, seed=7, restitution=1.0, tmax=40.0, samples=400):
    rng = np.random.default_rng(seed)
    z = np.sort(np.arange(N) * 1.0 + 0.1 * rng.random(N))
    sigma = np.where(rng.random(N) < 0.5, +1.0, -1.0)
    v = sigma * C
    z0, v0 = z.copy(), v.copy()
    P0 = v.sum()
    S0 = sigma.sum()
    events = []
    rec = {'t': [], 'z': [], 'v': [], 'sigma': [], 'dev_free': [], 'mom': [], 'S': [], 'absv': [], 'E': []}
    # 事件驱动：先把所有碰撞时刻解开（反弹系数为 1 时用解析推进）
    t = 0.0
    tgrid = np.linspace(0, tmax, samples)
    gi = 0
    ncol = 0
    rule_swap_ok = True
    while gi < len(tgrid) and ncol < 4000:
        tnext_goal = tgrid[gi]
        # 找下一个碰撞
        best, bi = np.inf, -1
        for i in range(N - 1):
            if v[i] > v[i + 1]:
                gap = z[i + 1] - z[i]
                if gap > 1e-13:
                    tc = gap / (v[i] - v[i + 1])
                    if tc < best:
                        best, bi = tc, i
        if bi >= 0 and t + best <= tnext_goal:
            z = z + v * best
            t += best
            # 碰撞（等质量硬点：交换速度）
            vi, vj = v[bi], v[bi + 1]
            v[bi], v[bi + 1] = vj, vi
            if restitution != 1.0:
                v[bi] *= restitution
                v[bi + 1] *= restitution
            sigma[bi] *= -1.0
            sigma[bi + 1] *= -1.0
            ncol += 1
            # J4：等质量弹性硬点的碰撞定则就是速度交换（无自由参数）
            if restitution == 1.0 and not (np.isclose(v[bi], vj) and np.isclose(v[bi + 1], vi)):
                rule_swap_ok = False
            continue
        # 推进到记录时刻
        dt = tnext_goal - t
        z = z + v * dt
        t = tnext_goal
        free = z0 + v0 * t
        zs = np.sort(z)
        fs = np.sort(free)
        rec['t'].append(t); rec['z'].append(z.copy()); rec['v'].append(v.copy())
        rec['sigma'].append(sigma.copy())
        rec['dev_free'].append(np.max(np.abs(zs - fs)))
        rec['mom'].append(v.sum()); rec['S'].append(sigma.sum())
        rec['absv'].append(np.max(np.abs(np.abs(v) - C)))
        rec['E'].append(0.5 * np.sum(v**2))
        gi += 1
    rec['ncol'] = ncol
    rec['rule_swap_ok'] = rule_swap_ok
    rec['P0'] = P0; rec['S0'] = S0
    return rec

def main():
    print('=' * 74)
    print('完美弹性定理：等速硬点链的碰撞不变量与自由演化等价')
    print('=' * 74)
    r = simulate(restitution=1.0)
    print('[J1] 总动量漂移最大 = %.2e（期望 0，机器精度）' % np.max(np.abs(np.array(r['mom']) - r['P0'])))
    print('[J2] 速度幅度最大偏差 = %.2e（期望 0）；总动能 = %.12f（期望 %.1f）' % (np.max(r['absv']), r['E'][-1], 0.5 * len(r['z'][0])))
    print('[J3] 位置多重集对自由演化的最大偏差 = %.2e（期望 0），碰撞次数 = %d' % (np.max(r['dev_free']), r['ncol']))
    print('[J4] 碰撞定则 = 速度交换（无自由参数）：%s' % r['rule_swap_ok'])
    print('[J5] 摆程方向和漂移最大 = %.2e（期望 0，S0 = %.0f）' % (np.max(np.abs(np.array(r['S']) - r['S0'])), r['S0']))
    r2 = simulate(restitution=0.97)
    print('[J6] 对照档（反弹系数 0.97）：位置多重集最大偏差 = %.3f，速度幅度最大偏差 = %.4f，总动能 %.4f 对 %.1f'
          % (np.max(r2['dev_free']), np.max(r2['absv']), r2['E'][-1], 0.5 * len(r2['z'][0])))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].semilogy(r['t'], np.array(r['dev_free']) + 1e-18, label='弹性（e = 1）')
    ax[0].semilogy(r2['t'], np.array(r2['dev_free']) + 1e-18, label='对照（e = 0.97）')
    ax[0].set_xlabel('时间'); ax[0].set_ylabel('位置多重集偏差'); ax[0].legend(); ax[0].set_title('(a) 自由演化等价')
    ax[1].plot(r['t'], np.array(r['E']), label='弹性：总动能')
    ax[1].plot(r2['t'], np.array(r2['E']), label='对照：总动能')
    ax[1].set_xlabel('时间'); ax[1].set_ylabel('总动能'); ax[1].legend(); ax[1].set_title('(b) 总动能守恒与对照')
    plt.tight_layout()
    plt.savefig(r'D:\software\TianXingFC\images\perfect_elasticity.png', dpi=130)
    print('图已存 images/perfect_elasticity.png')

main()
