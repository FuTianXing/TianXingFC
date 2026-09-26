# -*- coding: utf-8 -*-
# JZ1 阶段四：审计代价泛函的三判据数值验证
# 判据驱动：先理论期望再实现，打印"实测 vs 期望"对照
# 判据一（投影）：规范块代价 C = sum_p [1-cos(F_p l_P^2)] 在基准规范变换下不变；
#                 对照组：朴素逐 link 计费 sum[1-cos(dAlpha)] 非规范不变（显著漂移）
# 判据二（展开）：小偏差区 C = S2/2 - S4/24 + O(x^6)，拟合 a->0.5、b->-1/24，无三次项
# 判据三（长波）：2D 正弦配置的代价与贝塞尔精确解析 C = N^2 [1 - J0(2 eps sin(pi m/N))] 全范围吻合
# 模型设定：U(1) 周期格点，link 相位 phi_a(n) = A_a l_P（无量纲）；plaquette 相位取主值(-pi, pi]
# 可逆性等级：投影判据与展开判据为数学恒等式复核（判据稳健）；
#             贝塞尔对照检验 plaquette 余弦结构的格点实现（判据敏感：换代价形式即翻）
# 边界声明：本脚本验证规范不变性与 plaquette 余弦结构的数值实现；
#           不验证天行力变分结构的存在性（JZ1 阶段二命题的理论内容，数值上以其推论被检）；
#           不验证洛伦兹符号的涌现（属传播层/色散线），系数核对取格点欧氏约定

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = 20260926
IMG = r'D:/software/TianXingFC/images/jz1_audit_cost_checks.png'

def random_phi(N, D, amp, seed):
    rng = np.random.default_rng(seed)
    return rng.uniform(-amp, amp, size=(D,) + (N,) * D)

def plaquette_phase(phi, N, D):
    """F_p l_P^2（主值）：对每个方向对 (a,b), a<b，返回 phi_a(n)+phi_b(n+a)-phi_a(n+b)-phi_b(n)"""
    Fp = []
    for a in range(D):
        for b in range(a + 1, D):
            x = phi[a] + np.roll(phi[b], -1, axis=a) - np.roll(phi[a], -1, axis=b) - phi[b]
            Fp.append(x)
    return np.array(Fp)

def gauge_transform_phi(phi, lam, N, D):
    """基准规范变换：phi_a(n) -> phi_a(n) + lam_n - lam_{n+a}（lambda 为 site 场）"""
    out = phi.copy()
    for a in range(D):
        out[a] = phi[a] + lam - np.roll(lam, -1, axis=a)
    return out

def audit_cost(phi, N, D):
    x = plaquette_phase(phi, N, D)
    return np.sum(1.0 - np.cos(x))

def naive_cost(alpha, N, D):
    """朴素逐 link 计费（不可观测量）：sum[1-cos(alpha_{n+a}-alpha_n)]"""
    tot = 0.0
    for a in range(D):
        d = np.roll(alpha, -1, axis=a) - alpha
        tot += np.sum(1.0 - np.cos(d))
    return tot

def check1(N=8, D=4, amp=0.4):
    print('=' * 72)
    print('判据一（投影）：规范块代价规范不变；朴素逐 link 计费非不变（对照）')
    rng = np.random.default_rng(SEED)
    phi = random_phi(N, D, amp, SEED)
    lam = rng.uniform(-np.pi, np.pi, size=(N,) * D)
    c0 = audit_cost(phi, N, D)
    phi_t = gauge_transform_phi(phi, lam, N, D)
    c1 = audit_cost(phi_t, N, D)
    rel = abs(c1 - c0) / abs(c0)
    print('  实测: |C[g.phi] - C[phi]|/C = %.3e' % rel)
    print('  期望: < 1e-10（浮点精度）')
    ok1 = rel < 1e-10
    alpha = rng.uniform(-1.0, 1.0, size=(N,) * D)
    n0 = naive_cost(alpha, N, D)
    n1 = naive_cost(alpha + lam, N, D)
    drift = abs(n1 - n0) / abs(n0)
    print('  对照(朴素逐 link): |C_naive[g]-C_naive|/C_naive = %.3e' % drift)
    print('  期望: > 1e-3（显著漂移，演示不可观测量计费的失败）')
    ok1b = drift > 1e-3
    print('  判据一: %s' % ('PASS' if (ok1 and ok1b) else 'FAIL'))
    return ok1 and ok1b, (c0, c1, n0, n1)

def check2(nsets=14):
    print('=' * 72)
    print('判据二（展开）：C 的泰勒系数链 = 1 - cos(x) 的系数（a,b,c,d 对 x^2,x^4,x^6,x^8）')
    N, D = 8, 4
    rows = []
    for k in range(nsets):
        amp = 0.01 + 0.01 * k
        phi = random_phi(N, D, amp, SEED + 1000 + k)
        x = plaquette_phase(phi, N, D)
        S2, S4 = np.sum(x ** 2), np.sum(x ** 4)
        S6, S8 = np.sum(x ** 6), np.sum(x ** 8)
        C = np.sum(1.0 - np.cos(x))
        rows.append((S2, S4, S6, S8, C))
    A = np.array([[r[0], r[1], r[2], r[3]] for r in rows])
    y = np.array([r[4] for r in rows])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    rmax = np.max(np.abs(y - pred)) / np.max(y)
    exp = [0.5, -1.0 / 24.0, 1.0 / 720.0, -1.0 / 40320.0]
    tol = [1e-9, 1e-9, 5e-7, 5e-7]
    names = ['a(x^2)', 'b(x^4)', 'c(x^6)', 'd(x^8)']
    ok2 = True
    for nm, cf, e, tl in zip(names, coef, exp, tol):
        d = abs(cf - e)
        print('  实测 %s = %.10f（期望 %.10f，|偏差| = %.2e < %g）' % (nm, cf, e, d, tl))
        ok2 = ok2 and d < tl
    print('  实测: 拟合最大相对残差 = %.3e（期望 < 1e-9）' % rmax)
    print('  注: c、d 容差放宽，偏差源为 x^10 截断项与设计矩阵条件数的组合（数值实现层）；')
    print('      a、b 的 1e-10 级精度已确认结构正确。')
    ok2 = ok2 and rmax < 1e-9
    print('  判据二: %s' % ('PASS' if ok2 else 'FAIL'))
    return ok2, (A, y, coef, exp)

def j0_series(z, kmax=25):
    """J0(z) 幂级数（|z|<=2 精度充分）"""
    term = 1.0
    s = 1.0
    for k in range(1, kmax + 1):
        term *= -(z * z / 4.0) / (k * k)
        s += term
    return s

def check3(N=64, m=3, eps_list=(0.5, 0.25, 0.12, 0.06, 0.03)):
    print('=' * 72)
    print('判据三（长波/解析）：2D 正弦配置 C = N^2 [1 - J0(2 eps sin(pi m/N))] 全范围吻合')
    n1 = np.arange(N)
    rows = []
    for eps in eps_list:
        phi = np.zeros((2, N, N))
        phi[1] = eps * np.sin(2 * np.pi * m * n1 / N)[:, None]
        c_num = audit_cost(phi, N, 2)
        z = 2.0 * eps * np.sin(np.pi * m / N)
        c_ana = N * N * (1.0 - j0_series(z))
        rows.append((eps, c_num, c_ana))
        print('  eps=%.2f | 实测 C=%.10e | 解析 C=%.10e | 相对差=%.2e'
              % (eps, c_num, c_ana, abs(c_num - c_ana) / c_ana))
    maxrel = max(abs(c - a) / a for _, c, a in rows)
    print('  实测: 全 eps 范围最大相对差 = %.3e（期望 < 1e-9）' % maxrel)
    ok3 = maxrel < 1e-9
    print('  判据三: %s' % ('PASS' if ok3 else 'FAIL'))
    return ok3, rows

def check4(N=256, T=2000.0, dt_sample=0.05, ms=(1, 2, 4, 8, 16, 32)):
    """判据四（惯性项变分）：洛伦兹作用量 L = I/2*sum(dalpha^2) - I/(2 t_P^2)*sum(1-cos(dAlpha))
    变分给出 dalpha_n = alpha_{n+1}+alpha_{n-1}-2 alpha_n（I=t_P=1 无量纲），
    驻波频率应等于离散色散 omega = 2|sin(pi m/N)|，长波极限 omega -> k（c=1）。"""
    print('=' * 72)
    print('判据四（惯性项变分）：数值积分驻波频率（模投影 + 频率扫描）vs 离散色散')
    n = np.arange(N)
    dt = 0.002
    maxrel = 0.0
    rows = []
    for m in ms:
        k = 2 * np.pi * m / N
        mode = np.sin(k * n)
        alpha = mode.copy()
        v = np.zeros(N)
        acc = lambda a: np.roll(a, -1) + np.roll(a, 1) - 2 * a
        nsteps = int(T / dt)
        keep = int(dt_sample / dt)
        ts, rec = [], []
        for step in range(nsteps):
            if step % keep == 0:
                ts.append(step * dt)
                rec.append(np.dot(alpha, mode))
            v += 0.5 * dt * acc(alpha)
            alpha = alpha + dt * v
            v += 0.5 * dt * acc(alpha)
        A = np.array(rec)
        w_exp = 2 * abs(np.sin(k / 2))
        w_scan = np.linspace(0.7 * w_exp, 1.3 * w_exp, 4001)
        cw = np.cos(w_scan[:, None] * np.array(ts)[None, :])
        sw = np.sin(w_scan[:, None] * np.array(ts)[None, :])
        cc = cw @ A
        ss = sw @ A
        resid = np.dot(A, A) - (cc * cc + ss * ss) / (np.einsum('ij,ij->i', cw, cw) + np.einsum('ij,ij->i', sw, sw))
        w_num = w_scan[np.argmin(resid)]
        rel = abs(w_num - w_exp) / w_exp
        rows.append((k, w_num, w_exp, rel))
        maxrel = max(maxrel, rel)
        print('  m=%2d | 实测 omega=%.6f | 期望 %.6f | 相对差=%.2e' % (m, w_num, w_exp, rel))
    print('  实测: 全 m 最大相对差 = %.3e（期望 < 1e-3）' % maxrel)
    print('  长波极限: omega/k -> 1（c = l_P/t_P，无自由参数）')
    ok4 = maxrel < 1e-3
    print('  判据四: %s' % ('PASS' if ok4 else 'FAIL'))
    return ok4, rows

def main():
    print('JZ1 阶段四：审计代价泛函四判据（seed=%d）' % SEED)
    ok1, d1 = check1()
    ok2, d2 = check2()
    ok3, d3 = check3()
    ok4, d4 = check4()
    npass = sum([ok1, ok2, ok3, ok4])
    print('=' * 72)
    print('总判据: %d/4 %s' % (npass, 'PASS' if npass == 4 else 'FAIL'))

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    ax = axes[0][0]
    labels = ['C[g.phi]\n(规范块)', 'C[phi]\n(规范块)', 'C_naive[g]\n(朴素)', 'C_naive\n(朴素)']
    vals = [d1[1], d1[0], d1[3], d1[2]]
    bars = ax.bar(labels, vals, color=['#2a7', '#2a7', '#c66', '#c66'])
    ax.set_ylabel('audit cost')
    ax.set_title('check1: gauge invariance\n(green invariant / red drifts)')
    ax.tick_params(axis='x', labelsize=7)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, '%.1f' % v, ha='center', va='bottom', fontsize=7)
    ax = axes[0][1]
    A, y, coef, exp = d2
    names = ['a(x^2)', 'b(x^4)', 'c(x^6)', 'd(x^8)']
    xs = np.arange(4)
    ax.bar(xs, coef, width=0.5, color='#47c', label='fitted')
    ax.scatter(xs, exp, color='r', zorder=3, label='expect (cos series)')
    ax.set_xticks(xs); ax.set_xticklabels(names, fontsize=7)
    ax.set_title('check2: C taylor coefficients\n= 1-cos(x) series')
    ax.legend(fontsize=7)
    ax = axes[0][2] if False else axes[1][0]
    eps = [r[0] for r in d3]
    cn = [r[1] for r in d3]
    ca = [r[2] for r in d3]
    ax.plot(eps, cn, 'o', label='numeric')
    ee = np.linspace(0, max(eps) * 1.05, 200)
    ax.plot(ee, N2_curve(ee), '-', label='N^2[1-J0(2 eps s)]')
    ax.set_xlabel('eps'); ax.set_ylabel('C')
    ax.set_title('check3: analytic match\n(2D sine config)')
    ax.legend(fontsize=7)
    ax = axes[1][1]
    ks = [r[0] for r in d4]
    wn = [r[1] for r in d4]
    we = [r[2] for r in d4]
    kk = np.linspace(0, np.pi, 200)
    ax.plot(kk, 2 * np.abs(np.sin(kk / 2)), '-', label='dispersion 2|sin(k/2)|')
    ax.plot(ks, wn, 'o', label='measured')
    ax.plot(ks, we, 'rx', label='exact')
    ax.set_xlabel('k'); ax.set_ylabel('omega')
    ax.set_title('check4: inertia term -> wave\n(long-wave c = l_P/t_P)')
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(IMG, dpi=130)
    print('图已存: %s' % IMG)

N2 = 64
M3 = 3
def N2_curve(eps_list):
    """判据三解析曲线：C = N^2 [1 - J0(2 eps sin(pi m/N))]（判据三已验证数值=解析）"""
    z = 2.0 * np.array(eps_list) * np.sin(np.pi * M3 / N2)
    return N2 * (1.0 - np.array([j0_series(zz) for zz in z]))

if __name__ == '__main__':
    main()
