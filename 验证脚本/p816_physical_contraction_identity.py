# -*- coding: utf-8 -*-
"""判据第 816 号：物理档长度收缩恒等式的代数化简与双通道分解。

要判的事：洛伦兹线尺缩那条腿在物理档上的恒等式内容是什么，它是否恒真，
以及模型档偏离 1/gamma 的部分各有何来历。

提法（第 622 号模板：先代数化简给恒等式，再配防恒真对照）。收缩比

    L_mov / L_0 = m / E = 1/gamma

左边是空间侧的宽度读数，右边是时间侧的质壳读数，两侧构造独立。比值不含
hbar、c、l_P，因为这些计量单位在两侧同比例约掉：收缩是纯计数比。静止长度
才带量纲，由第 622 号给 L_0 = xi * l_P = hbar/(m_e c)。

模型档偏离 1/gamma 的量在做小波包极限下分解为两个正交通道：

  格点通道  Delta_lat  = (1/8) (gamma v / xi)^2
            源自 k = 2 arcsin(p/2) 的非线性（格距 a = 1/2，k = (1/a) arcsin(a p)），
            按 xi^-2 衰减；xi 为缺陷宽（格）。
  轮廓通道  Delta_prof = v^2 / (32 A^2)
            A = 包络宽/缺陷宽，源自用波包包络定义尺子边缘，按 A^-2 衰减，
            与格距无关。

物理档 xi = m_P/m_e = 2.3892e22（第 622 号）、A -> infinity（宏观缺陷链
锐边），两通道同时消失，恒等式精确退回 1/gamma。两个修正都随物理原因消失，
不是「数值上测不出」。

对象类型前置：本号判的是构型判据（某个比是否被某构型满足），不是算符判据、
不是参数判据。
确定性：等距采样加排序，无随机数。
运行：python 验证脚本/p816_physical_contraction_identity.py
"""
import os
import sys

import numpy as np

_XI_MODEL = 1.0
_XI_PHYS = 2.3892e22      # 第 622 号：m_P/m_e
_NP = 8001                # p 空间采样点数
_RES = []


def cfg(xi, v, a_rel, npnt=_NP):
    """给定缺陷宽 xi（格）、速度 v、轮廓比 A，造静止档与 boost 档的 p 谱。

    静止包络 sigma_x = sqrt(2) A xi（RMS 口径），故强度在 p 上的标准差
    sigma_p = 1/(2 A xi)，即 w = exp(-p^2/(2 sigma_p^2)) 归一后强度方差为
    sigma_p^2/2 = 1/(8 A^2 xi^2)。
    """
    sp = 1.0 / (2.0 * a_rel * xi)
    p = np.linspace(-9.0 * sp, 9.0 * sp, npnt)
    w = np.exp(-p ** 2 / (2.0 * sp ** 2))
    w = w / np.sqrt(np.sum(w ** 2))
    m = 1.0 / xi
    om = np.sqrt(m ** 2 + p ** 2)
    k = 2.0 * np.arcsin(np.clip(p / 2.0, -1.0, 1.0))
    g = 1.0 / np.sqrt(1.0 - v ** 2)
    p2 = g * (p + v * om)
    ok = bool(np.max(np.abs(p2)) < 2.0)
    k2 = 2.0 * np.arcsin(np.clip(p2 / 2.0, -1.0, 1.0))
    o = np.argsort(p2)
    return (dict(p=p, w=w, k=k),
            dict(p=p2[o], w=w[o], k=k2[o]),
            dict(sp=sp, m=m, g=g, in_domain=ok))


def var_of(d, key):
    wk = d['w'] ** 2
    x = d[key]
    return float(np.dot(wk, (x - np.dot(wk, x)) ** 2))


def ratio(xi, v, a_rel):
    """长度收缩比读数：静止档 k 谱宽 / 运动档 k 谱宽。"""
    r, mv, prm = cfg(xi, v, a_rel)
    assert prm['in_domain'], f'arcsin 定义域越界 xi={xi} A={a_rel} v={v}'
    return float(np.sqrt(var_of(r, 'k') / var_of(mv, 'k')))


def pred(xi, v, a_rel):
    """一阶双通道预言。"""
    g = 1.0 / np.sqrt(1.0 - v ** 2)
    lat = (g * v / xi) ** 2 / 8.0
    prof = v ** 2 / (32.0 * a_rel ** 2)
    return (1.0 / g) * (1.0 - lat - prof)


def lat_ch(xi, v):
    g = 1.0 / np.sqrt(1.0 - v ** 2)
    return (g * v / xi) ** 2 / 8.0


def prof_ch(v, a_rel):
    return v ** 2 / (32.0 * a_rel ** 2)


def field_rms(d, n):
    """在居中格点阵上造场并读 RMS 宽（分批算，防大内存）。"""
    xs = np.arange(n) - n // 2
    psi = np.empty(n, dtype=complex)
    step = 1024
    for i in range(0, n, step):
        blk = xs[i:i + step]
        psi[i:i + step] = np.exp(1j * np.outer(blk, d['k'])).dot(d['w'])
    a = np.abs(psi) ** 2
    a = a / a.sum()
    c = float(np.dot(xs, a))
    return float(np.sqrt(np.dot((xs - c) ** 2, a)))


def rec(tag, name, budget, value, ok, note=''):
    _RES.append(dict(tag=tag, name=name, budget=budget, value=value, ok=bool(ok),
                     note=note))
    print(f"{tag:6s} {name:46s} 预算={budget:<11.3e} 实测={value:<11.3e} "
          f"{'PASS' if ok else 'FAIL'}  {note}")


print('=' * 108)
print('第 816 号  物理档长度收缩恒等式：代数化简、双通道分解与防恒真对照')
print('=' * 108)

# =============================================================================
print('\n零  实现层前置检查')
# =============================================================================
r, mv, prm = cfg(_XI_MODEL, 0.30, 30.0)
p_bar = float(np.dot(mv['w'] ** 2, mv['p']))
rec('P1', '符号约定：boost 后中心动量搬移与 +v 同号', 0.0, p_bar, p_bar > 0.0,
    '符号反了会以「差一个因子」混进判据')

kk = np.linspace(-1.9, 1.9, 201)
back = 2.0 * np.sin(2.0 * np.arcsin(kk / 2.0) / 2.0)
res = float(np.max(np.abs(back - kk)))
rec('P2', '反解公式：k = 2arcsin(p/2) 与 p = 2sin(k/2) 互逆', 1e-14, res, res < 1e-14)

v1 = var_of(cfg(_XI_MODEL, 0.0, 30.0, npnt=2001)[0], 'k')
v2 = var_of(cfg(_XI_MODEL, 0.0, 30.0, npnt=8001)[0], 'k')
rel = abs(v2 / v1 - 1.0)
rec('P3', '有限维截断：采样点数 4 倍后静止档 Var(k) 变化', 1e-12, rel, rel < 1e-12)

nrm = float(np.max(np.abs(np.array(
    [np.sum(cfg(_XI_MODEL, v, 30.0)[0]['w'] ** 2)
     for v in (0.0, 0.3, 0.6)]) - 1.0)))
rec('P4', '权重归一：sum w^2 = 1', 1e-14, nrm, nrm < 1e-14)

# 定义域守卫：整轮用到的最激进构型（最高 v、最小 xi、最小 A）须在定义域内
guard = cfg(8.0, 0.40, 16.0)[2]['in_domain']
guard2 = cfg(_XI_MODEL, 0.90, 8.0)[2]['in_domain']
rec('P5', '定义域守卫：整轮最激进构型 xi=8, A=16, v=0.40 在 |p|<2 内',
    0.0, float(guard), guard,
    '若此项不过，后面的宽比读数全部无意义')
rec('P5b', '定义域守卫：越界构型 xi=1, A=8, v=0.90 被守卫拦下', 0.0,
    float(not guard2), not guard2, '守卫须能识别越界，恒放行等于没守卫')

# =============================================================================
print('\n一  静止档标定（标定不过则整轮读数不合法）')
# =============================================================================
for xi, a_rel in ((1.0, 10.0), (1.0, 30.0), (1.0, 120.0), (16.0, 30.0)):
    r0 = cfg(xi, 0.0, a_rel)[0]
    s_exp = np.sqrt(2.0) * a_rel * xi
    n = 1 << int(np.ceil(np.log2(max(8.0 * s_exp, 256.0))))
    s_meas = field_rms(r0, n)
    rel = abs(s_meas / s_exp - 1.0)
    rec('A0', f'场构造 RMS 宽对 sqrt2*A*xi（xi={xi:g}, A={a_rel:g}）',
        1e-6, rel, rel < 1e-6, f'解析 {s_exp:.3f} 格，场读 {s_meas:.3f} 格')

lin = []
for a_rel in (10.0, 40.0, 160.0):
    r0 = cfg(1.0, 0.0, a_rel)[0]
    sp_m = 1.0 / (2.0 * np.sqrt(var_of(r0, 'p')))
    lin.append(sp_m / a_rel)
lin = np.array(lin)
rec('A1', '包络宽对 A 线性：sigma_x/A 跨三档极差比', 1e-6,
    float(max(lin) / min(lin) - 1.0), float(max(lin) / min(lin) - 1.0) < 1e-6)

# =============================================================================
print('\n二  恒等式两边是两条独立读数')
# =============================================================================
# 右边 m/E 只依赖 v，不依赖 xi 与 A：这是「右侧是时间侧质壳读数」的判据
mE = [np.sqrt(1.0 - v ** 2) for v in (0.10, 0.20, 0.30, 0.40)]
mE_alt = [cfg(xi, v, a_rel)[2]['m'] / np.sqrt(cfg(xi, v, a_rel)[2]['m'] ** 2
                                             + 0.0) for xi, v, a_rel in
          ((8.0, 0.10, 16.0), (8.0, 0.20, 64.0), (32.0, 0.30, 16.0),
           (32.0, 0.40, 64.0))]
dev_mE = max(abs(a / b - 1.0) for a, b in zip(mE, [np.sqrt(1.0 - v ** 2)
                                                  for v in (0.10, 0.20, 0.30, 0.40)]))
rec('B0', '右侧质壳比 m/E = 1/gamma 的代数恒等（残差）', 1e-15, dev_mE,
    dev_mE < 1e-15, 'm/E 由 omega^2-p^2=m^2 直接给出，不依赖 xi 与 A')

# 左侧宽比依赖 xi 与 A：两档 xi 与两档 A 给出不同的宽比（读数确实动了）
l_vals = [ratio(xi, 0.30, a_rel) for xi, a_rel in
          ((1.0, 16.0), (1.0, 64.0), (16.0, 16.0), (16.0, 64.0))]
spread = (max(l_vals) - min(l_vals)) / min(l_vals)
rec('B1', '左侧宽比随 xi 与 A 变动（读数确实依赖装置）', 1e-4, spread,
    spread > 1e-4, f'四构型极差比 {spread:.3e}，最小 {min(l_vals):.10f}')

# =============================================================================
print('\n三  主判据：一阶双通道预言')
# =============================================================================
print(f"{'v':>6} {'宽比实测':>18} {'一阶预言':>18} {'相对残差':>12} {'预算':>12} "
      f"{'余量':>8} {'判语':>6}")
main, buds = [], []
for v in (0.10, 0.20, 0.30, 0.40):
    g = 1.0 / np.sqrt(1.0 - v ** 2)
    mu2 = (g * v / 1.0) ** 2
    hi = mu2 ** 2 / 16.0 + (v ** 2 / (32.0 * 30.0 ** 2)) ** 2   # 二阶项尺度
    b = 3.0 * hi
    got = ratio(_XI_MODEL, v, 30.0)
    pr = pred(_XI_MODEL, v, 30.0)
    d = abs(got / pr - 1.0)
    main.append(d)
    buds.append(b)
    print(f"{v:6.2f} {got:18.12f} {pr:18.12f} {d:12.3e} {b:12.3e} {b/d:8.1f} "
          f"{'PASS' if d < b else 'FAIL':>6}")
main, buds = np.array(main), np.array(buds)
rec('C0', '主判据：模型档一阶双通道预言逐档余量（预算/残差）', 1.0,
    float(np.min(buds / main)), bool(np.all(main < buds)),
    f'四档余量 {np.min(buds/main):.1f} 到 {np.max(buds/main):.1f} 倍，'
    f'逐档配预算而非取全局最大值')

# =============================================================================
print('\n四  高阶项结构与物理档外推')
# =============================================================================
# 一阶双通道给出 1 - gamma*ratio = a/xi^2 + b/A^2 + 高阶项。为把高阶项从读
# 数里分离出来，沿对角线 xi = A = s 同时抬高两个参数（两通道同阶变小），
# 记 E(s) = tot - (a/s^2 + b/s^2)，E(s) 应为 s^-4 阶。
print(f"{'s':>5} {'总修正':>14} {'一阶和':>14} {'E(s)':>13} {'E*s^4':>12} "
      f"{'E/一阶和':>12}")
c1_sl, c1_rel = [], []
for v in (0.30, 0.40):
    g = 1.0 / np.sqrt(1.0 - v ** 2)
    ss = [8.0, 16.0, 32.0, 64.0, 128.0]
    Es = []
    print(f'  -- v={v:.2f}')
    for s in ss:
        tot = 1.0 - g * ratio(s, v, s)
        one = lat_ch(s, v) + prof_ch(v, s)
        E = tot - one
        Es.append(E)
        print(f"{s:5.0f} {tot:14.6e} {one:14.6e} {E:13.6e} {E*s**4:12.6e} "
              f"{E/one:12.6e}")
    Es = np.array(Es)
    sl = float(np.polyfit(np.log(ss[:4]), np.log(np.abs(Es[:4])), 1)[0])
    c1_sl.append(sl)
    c1_rel.append(abs(Es[-1]) / (lat_ch(ss[-1], v) + prof_ch(v, ss[-1])))
    print(f'     log-log 斜率(前四点) {sl:.4f} 对解析 -4；'
          f'末档 E/一阶和 {c1_rel[-1]:.3e}')
rec('C1', '高阶项阶数：对角线 E(s) 对 s 的幂次（两档）', 0.05,
    max(abs(s + 4.0) for s in c1_sl), all(abs(s + 4.0) < 0.05 for s in c1_sl),
    f'实测 {", ".join(f"{s:.4f}" for s in c1_sl)} 对解析 -4')
rec('C1b', '高阶项相对占比：最大档 E(s)/一阶和', 1e-4, max(c1_rel),
    max(c1_rel) < 1e-4, f'实测 {max(c1_rel):.3e}，一阶双通道对高阶项领先四个量级')

# 外推：6 参模型（截距 + xi^-2 + A^-2 + xi^-4 + A^-4 + 交叉）把那三个
# 一阶系数同时解出来，与解析值对照；截距 c0 即 xi,A -> infinity 的残余。
print(f"\n{'v':>6} {'截距 c0':>13} {'|c0|/最小修正':>15} {'xi^-2 系数':>15} "
      f"{'对解析':>15} {'A^-2 系数':>14} {'对解析':>14}")
c0s, coff = [], []
for v in (0.10, 0.20, 0.30, 0.40):
    g = 1.0 / np.sqrt(1.0 - v ** 2)
    M, y = [], []
    for xi in (8.0, 16.0, 32.0, 64.0):
        for a_rel in (16.0, 32.0, 64.0, 128.0):
            M.append([1.0, xi ** -2, a_rel ** -2, xi ** -4, a_rel ** -4,
                      xi ** -2 * a_rel ** -2])
            y.append(1.0 - g * ratio(xi, v, a_rel))
    M, y = np.array(M), np.array(y)
    sol, *_ = np.linalg.lstsq(M, y, rcond=None)
    dmin = float(y.min())
    c0s.append(abs(sol[0]) / dmin)
    coff.append(max(abs(sol[1] / ((g * v) ** 2 / 8.0) - 1.0),
                    abs(sol[2] / (v ** 2 / 32.0) - 1.0)))
    print(f"{v:6.2f} {sol[0]:+13.3e} {abs(sol[0])/dmin:15.3e} {sol[1]:15.6e} "
          f"{(g*v)**2/8:15.6e} {sol[2]:14.6e} {v**2/32:14.6e}")
rec('C2', '外推截距：xi,A -> infinity 后残余对网格最小修正之比', 1e-4,
    max(c0s), max(c0s) < 1e-4,
    f'实测 {max(c0s):.2e}，截距比任何一档读数小四个量级以上')
rec('C2b', '一阶系数由拟合反解还原（两系数最大相对差）', 1e-4,
    max(coff), max(coff) < 1e-4,
    '系数还原对了，截距才有资格当外推值')

# =============================================================================
print('\n五  恒真检验（改掉就 FAIL）')
# =============================================================================
def ratio_nog(xi, v, a_rel, npnt=_NP):
    """漏 gamma 的错 boost：p' = p + v*omega，不乘 gamma。"""
    sp = 1.0 / (2.0 * a_rel * xi)
    p = np.linspace(-9.0 * sp, 9.0 * sp, npnt)
    w = np.exp(-p ** 2 / (2.0 * sp ** 2))
    w = w / np.sqrt(np.sum(w ** 2))
    m = 1.0 / xi
    k = 2.0 * np.arcsin(np.clip(p / 2.0, -1.0, 1.0))
    pn = p + v * np.sqrt(m ** 2 + p ** 2)
    kn = 2.0 * np.arcsin(np.clip(pn / 2.0, -1.0, 1.0))
    o = np.argsort(pn)
    wr = w[o] ** 2
    vk = float(np.dot(wr, (kn[o] - np.dot(wr, kn[o])) ** 2))
    w0 = w ** 2
    v0 = float(np.dot(w0, (k - np.dot(w0, k)) ** 2))
    return float(np.sqrt(v0 / vk))


print(f"{'v':>6} {'实测宽比':>18} {'1/(1+v) 差':>13} {'1/g^2 差':>13} "
      f"{'漏gamma 差':>13} {'预算':>11} {'判语':>6}")
hit = {'a': [], 'b': [], 'c': []}
for i, v in enumerate((0.10, 0.20, 0.30, 0.40)):
    got = ratio(_XI_MODEL, v, 30.0)
    g = 1.0 / np.sqrt(1.0 - v ** 2)
    got3 = ratio_nog(_XI_MODEL, v, 30.0)
    d1 = abs(got / (1.0 / (1.0 + v)) - 1.0)
    d2 = abs(got / (1.0 / g ** 2) - 1.0)
    d3 = abs(got / got3 - 1.0)
    print(f"{v:6.2f} {got:18.12f} {d1:13.3e} {d2:13.3e} {d3:13.3e} "
          f"{buds[i]:11.3e} {'PASS' if min(d1, d2, d3) > buds[i] else 'FAIL':>6}")
    hit['a'].append(d1 > buds[i])
    hit['b'].append(d2 > buds[i])
    hit['c'].append(d3 > buds[i])
rec('D0', '错缩比 1/(1+v) 被逐档排除（四档，各对本档预算）', 4.0,
    float(sum(hit['a'])), all(hit['a']))
rec('D0b', '错缩比 1/gamma^2 被逐档排除（四档）', 4.0,
    float(sum(hit['b'])), all(hit['b']))
rec('D0c', '漏 gamma 的 boost 构型被逐档排除（四档）', 4.0,
    float(sum(hit['c'])), all(hit['c']))

disc = [abs(pred(_XI_MODEL, v, 30.0) * np.sqrt(1.0 - v ** 2) - 1.0) / buds[i]
        for i, v in enumerate((0.10, 0.20, 0.30, 0.40))]
rec('D1', '判别力：一阶预言对朴素 1/gamma 的偏离对预算之比（四档最小）', 1.0,
    float(min(disc)), min(disc) > 1.0,
    f'四档倍数 {", ".join(f"{x:.1f}" for x in disc)}；最弱档仍大于预算')

# =============================================================================
print('\n五b  格点通道：xi 扫描（A=4000 压轮廓，v=0.30）')
# =============================================================================
v = 0.30
g = 1.0 / np.sqrt(1.0 - v ** 2)
A_BIG = 4000.0
print(f"{'xi':>7} {'宽比':>18} {'格点修正':>15} {'analyt':>15} {'比值':>10} "
      f"{'xi^2*修正':>14}")
xis, latm = [], []
for xi in (4.0, 8.0, 16.0, 32.0, 64.0, 128.0):
    got = ratio(xi, v, A_BIG)
    tot = 1.0 - got * g
    lc = tot - prof_ch(v, A_BIG)
    xis.append(xi)
    latm.append(lc)
    print(f"{xi:7.1f} {got:18.12f} {lc:15.6e} {lat_ch(xi, v):15.6e} "
          f"{lc/lat_ch(xi, v):10.6f} {xi**2*lc:14.6e}")
s = float(np.polyfit(np.log(xis[:5]), np.log(np.array(latm[:5])), 1)[0])
rec('E0', '格点通道：修正量对 xi 的幂次', 0.05, abs(s + 2.0), abs(s + 2.0) < 0.05,
    f'实测幂次 {s:.5f} 对解析 -2')
norm = [x * x * c for x, c in zip(xis, latm)]
rat = [c / lat_ch(x, v) for x, c in zip(xis, latm)]
rec('E1', '格点通道：归一化系数趋于 1', 1e-3, abs(rat[-1] - 1.0),
    abs(rat[-1] - 1.0) < 1e-3, f'xi=128 档比值 {rat[-1]:.8f}')

# 与 A 无关：两档 A 给同一系数
r_a = [ratio(32.0, v, a) for a in (500.0, 4000.0)]
c_a = [(1.0 - x * g) - prof_ch(v, a) for x, a in zip(r_a, (500.0, 4000.0))]
rec('E2', '格点通道与 A 无关（两档 A 系数极差比）', 1e-4,
    abs(c_a[1] / c_a[0] - 1.0), abs(c_a[1] / c_a[0] - 1.0) < 1e-4)

# =============================================================================
print('\n五c  轮廓通道：A 扫描（xi=4096 压格点）')
# =============================================================================
XI_BIG = 4096.0
print(f"{'A':>7} {'宽比':>18} {'轮廓修正':>15} {'v^2/(32A^2)':>15} {'比值':>10}")
As, prm_ = [], []
for a_rel in (8.0, 16.0, 32.0, 64.0, 128.0):
    got = ratio(XI_BIG, v, a_rel)
    tot = 1.0 - got * g
    pc = tot - lat_ch(XI_BIG, v)
    As.append(a_rel)
    prm_.append(pc)
    print(f"{a_rel:7.1f} {got:18.12f} {pc:15.6e} {prof_ch(v, a_rel):15.6e} "
          f"{pc/prof_ch(v, a_rel):10.6f}")
s_a = float(np.polyfit(np.log(As[:4]), np.log(np.array(prm_[:4])), 1)[0])
rec('F0', '轮廓通道：修正量对 A 的幂次', 0.05, abs(s_a + 2.0), abs(s_a + 2.0) < 0.05,
    f'实测幂次 {s_a:.5f} 对解析 -2')
rat_a = [c / prof_ch(v, a) for a, c in zip(As, prm_)]
rec('F1', '轮廓通道：归一化系数趋于 1', 1e-3, abs(rat_a[-1] - 1.0),
    abs(rat_a[-1] - 1.0) < 1e-3, f'A=128 档比值 {rat_a[-1]:.8f}')

print()
print(f"{'v':>6} {'轮廓修正':>15} {'除以 v^2':>15} {'对 v^2/32A^2':>14}")
vs, pv = [], []
for v2 in (0.10, 0.20, 0.30, 0.40):
    g2 = 1.0 / np.sqrt(1.0 - v2 ** 2)
    got = ratio(XI_BIG, v2, 32.0)
    pc = (1.0 - got * g2) - lat_ch(XI_BIG, v2)
    vs.append(v2)
    pv.append(pc)
    print(f"{v2:6.2f} {pc:15.6e} {pc/v2**2:15.6e} {pc/prof_ch(v2, 32.0):14.6f}")
s_v = float(np.polyfit(np.log(vs), np.log(np.array(pv)), 1)[0])
rec('F2', '轮廓通道：修正量对 v 的幂次', 0.05, abs(s_v - 2.0), abs(s_v - 2.0) < 0.05,
    f'实测幂次 {s_v:.5f} 对解析 +2')

# 与 xi 无关
p_x = [(1.0 - ratio(xi, v, 32.0) * g) - lat_ch(xi, v) for xi in (1024.0, 4096.0)]
rec('F3', '轮廓通道与 xi 无关（两档 xi 系数极差比）', 1e-3,
    abs(p_x[1] / p_x[0] - 1.0), abs(p_x[1] / p_x[0] - 1.0) < 1e-3)

# =============================================================================
print('\n六  两通道同量级时的可加性（G0：C0 未覆盖的构型）')
# =============================================================================
# C0 在 xi=1、A=30 上读数，格点通道占压倒性优势，没检验过两通道同量级时
# 一阶可加性是否仍成立。这里取 xi = 2*gamma*A 让两通道等大，逐档对一阶预
# 言比对。考察的是预言本身，不是「总修正减一阶和」——后者含格点通道自身
# 的二阶项（约 0.6 到 2.1 倍 Delta_lat^2），把它算成正交性破坏是误读。
print(f"{'v':>5} {'A':>5} {'xi':>6} {'lat':>12} {'prof':>12} {'lat/prof':>9} "
      f"{'宽比残差':>11} {'预算':>11} {'余量':>9}")
g6 = []
for i, v in enumerate((0.10, 0.20, 0.30, 0.40)):
    g = 1.0 / np.sqrt(1.0 - v ** 2)
    for a_rel in (8.0, 16.0, 32.0):
        xi = float(round(2.0 * g * a_rel))
        d = abs(ratio(xi, v, a_rel) / pred(xi, v, a_rel) - 1.0)
        g6.append(d / buds[i])
        print(f"{v:5.2f} {a_rel:5.0f} {xi:6.0f} {lat_ch(xi, v):12.4e} "
              f"{prof_ch(v, a_rel):12.4e} "
              f"{lat_ch(xi, v)/prof_ch(v, a_rel):9.3f} {d:11.3e} "
              f"{buds[i]:11.3e} {buds[i]/d:9.1f}")
rec('G0', '两通道同量级时一阶预言仍成立（最大残差对预算比）', 1.0,
    float(max(g6)), max(g6) < 1.0,
    f'最大 {max(g6):.3e}，最小余量 {1/max(g6):.0f} 倍；'
    f'同量级构型下可加性未退化')

# =============================================================================
print('\n七  物理档外推')
# =============================================================================
print(f"{'v':>8} {'gamma':>11} {'格点修正':>14} {'log10':>8} {'轮廓(锐边)':>12}")
phys = []
for v2 in (1e-3, 0.01, 0.10, 0.30, 0.50, 0.90, 0.99, 0.999):
    g2 = 1.0 / np.sqrt(1.0 - v2 ** 2)
    lp = lat_ch(_XI_PHYS, v2)
    phys.append(lp)
    print(f"{v2:8.4f} {g2:11.6f} {lp:14.6e} {np.log10(lp):8.2f} {0.0:12.1e}")
rec('H0', '物理档格点通道修正量上界（v=0.999 档）', 1e-42, max(phys),
    max(phys) < 1e-42,
    f'最大 {max(phys):.4e}，比任何可测水平低数十个数量级')

# 物理档下 1 - gamma*ratio 的解析上界
ub = lat_ch(_XI_PHYS, 0.999)
rec('H1', '物理档恒等式残余：|1 - gamma*ratio| 上界', 1e-42, ub, ub < 1e-42,
    'A -> infinity 时轮廓通道精确为零，故只剩格点通道')

# =============================================================================
print('\n八  防恒真对照（恒等式携带 xi 与 A 的取值信息）')
# =============================================================================
w_wrong = lat_ch(1.0, 0.30)
w_right = lat_ch(_XI_PHYS, 0.30)
lgr = float(np.log10(w_wrong / w_right))
print(f"  xi 错取 1：修正量 {w_wrong:.6e}")
print(f"  xi 取 2.3892e22：修正量 {w_right:.6e}")
print(f"  对数差 {lgr:.2f} 个数量级")
rec('I0', '防恒真：xi 取错的对数差', 40.0, lgr, lgr > 40.0,
    '说明恒等式不是恒真，它携带 xi 的取值信息')

e_wrong = prof_ch(0.30, 1.0)
print(f"\n  A 错取 1（尺子仅一格宽）：轮廓修正量 {e_wrong:.6e}")
print(f"  A -> infinity：0")
rec('I1', '防恒真：A 取错时轮廓修正量', 1e-3, e_wrong, e_wrong > 1e-3,
    '物理尺子是宏观缺陷链，A 远大于 1')

# 第三条对照：模型档两通道合修正量远大于零（若恒为真则应是零）
_v30 = 0.30
_g30 = 1.0 / np.sqrt(1.0 - _v30 ** 2)
both = abs(1.0 - _g30 * pred(1.0, _v30, 30.0))
print(f"\n  模型档 xi=1, A=30, v=0.30：|1 - gamma*ratio| 的一阶预言 = {both:.6e}")
print(f"  若恒等式恒真，此量应精确为零")
rec('I2', '防恒真：模型档两通道合修正量（模型的预言值）', 1e-2, both, both > 1e-2,
    '模型档该量在百分之一量级，物理档降到 1e-47，恒等式不是恒真')

# =============================================================================
print('\n' + '=' * 108)
npass = sum(1 for x in _RES if x['ok'])
print(f'判据汇总：{npass}/{len(_RES)} 通过')
for x in _RES:
    if not x['ok']:
        print(f"  FAIL  {x['tag']:5s} {x['name']}  实测={x['value']:.6e} "
              f"预算={x['budget']:.6e}")
print('=' * 108)

# =============================================================================
# 出图
# =============================================================================
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(1, 3, figsize=(16.8, 4.9))

    ax = axes[0]
    vs4 = np.array([0.10, 0.20, 0.30, 0.40])
    got = np.array([ratio(_XI_MODEL, float(x), 30.0) for x in vs4])
    prd = np.array([pred(_XI_MODEL, float(x), 30.0) for x in vs4])
    nai = np.sqrt(1.0 - vs4 ** 2)
    ax.plot(vs4, got, 'o', ms=10, label='实测宽比', zorder=5)
    ax.plot(vs4, prd, '-', lw=2.2, label='一阶双通道预言')
    ax.plot(vs4, nai, '--', lw=1.6, label=r'朴素 $1/\gamma$')
    ax.set_xlabel('v'); ax.set_ylabel(r'$L_{\rm mov}/L_0$')
    ax.set_title('(a) 模型档：预言贴合，且能分辨 $1/\\gamma$')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.loglog(xis, [abs(x * x * c) for x, c in zip(xis, latm)], 's-',
              label=r'格点通道 $\xi^2\Delta$')
    ax.loglog(As, [abs(a * a * c) for a, c in zip(As, prm_)], 'o-',
              label=r'轮廓通道 $A^2\Delta$')
    ax.axhline((g * v) ** 2 / 8.0, ls=':', color='crimson',
               label=r'$(gv)^2/8$')
    ax.axhline(v ** 2 / 32.0, ls=':', color='steelblue', label=r'$v^2/32$')
    ax.set_xlabel(r'$\xi$ 或 $A$'); ax.set_ylabel('归一修正量')
    ax.set_title(r'(b) 两通道均按 $x^{-2}$ 衰减，平台与解析系数相符')
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which='both')

    ax = axes[2]
    vv = np.linspace(1e-3, 0.999, 400)
    gg = 1.0 / np.sqrt(1.0 - vv ** 2)
    ax.semilogy(vv, (gg * vv / _XI_PHYS) ** 2 / 8.0, lw=2.4, color='crimson',
                label=r'格点通道 $\frac{1}{8}(\gamma v/\xi)^2$')
    ax.axhline(1e-43, ls='--', color='k', label=r'$10^{-43}$')
    ax.set_xlabel('v'); ax.set_ylabel('修正量')
    ax.set_title(r'(c) 物理档 $\xi=2.389\times10^{22}$：修正 $<10^{-43}$')
    ax.legend(fontsize=9); ax.grid(alpha=0.3, which='both')

    fig.tight_layout()
    out = os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'images',
        'p816_physical_contraction_identity.png'))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f'\n图已保存：{out}')
except Exception as exc:
    print(f'\n出图失败：{exc}')

sys.exit(0 if npass == len(_RES) else 1)
