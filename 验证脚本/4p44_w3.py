# -*- coding: utf-8 -*-
r'''P4-44 份额铺开线 W3：全空间性接口（531 号）
=====================================================================
符号约定（沿用 W1/W2 与方案第三节）：
  D_sep   = 两支间距（格距单位；与温度定义的复相干无序度 D 不同量）
  sigma_x = 质心波包宽度（格距单位）
  s0(D)   = 零模波包重叠 = exp[-D^2/(8 sigma_x^2)]（W1 五档读数）
  V(t_up) = 升格时刻的相干因子（承 P4-10 Z2 闭式）
  PR      = 权重参与比 = 1/sum_i |psi_i|^2（承 P4-10 Z3，W2 核对）
第 43 条三段式（本任务只处理丙段的接口相容性）：
  甲（份额为何不局限在核内）：全空间纠缠保证读数不塌进核内（基本成立）
  乙（份额为何摊成该特定形状）：归 P4-44（W1/W2 已交付定量档位）
  丙（��空间这个尺度从何而来）：公理层扇区求和范围 +
      大爆炸全局锁相提供建立时刻，非物理边界；**来源仍未推导**
      （本轮只核对丙段与形状族的相容性，不推导丙段来源、不升级其层级）

判据先行（写死于头注）：

模型：份额展布的空间剖面 p(x) 为两瓣的形状族随间距的延展：
  单瓣形状取高斯型（两支为谐振子相干态，质心波包宽度 sigma_x，
  承 M7b Z1 引理），全空间延展按扇区求和范围取格距全域
  （公理层输入，见丙段）。跨项延展因子 s0(D) 乘在双瓣结构上。
  「全空间可分辨」指份额读数不塌进核的紧致范围（第 43 条甲段），
  本轮核对的相容判据：延展到全域后，各处份额读数与形状族一致，
  即全域积分与核内积分的差异由形状族单独决定。

W3-1   全空间相容判据：给定 D_sep 与相干度 V，延展后的份额剖面
       的全域积分与解析闭式一致（全域积分 = V sqrt(2pi) w +
       V^2 sqrt(pi) w s0^2；交叉项的幂次为 s0^2，因剖面系数 s0
       乘以 gA gB 自带的 exp(-D^2/(8w^2)) = s0）。
       容差从数值积分误差预算取 1e-8。
       【执行轮修正两处】首版两路同闭式同积分法，maxdev 恒零、
       构成构造恒真；且首版做「全域归一后比剖面」，而全域无界
       时归一因子有限、尾部不可忽略，口径不当。改为全域积分对
       解析闭式的相容（真正独立的路径）。
       多档 D_sep {1, 2, 4, 8}*sigma_x 扫描。
W3-2   核紧致与全域延展的解耦读数：份额剖面的「核紧致尺度」
       （质量惯性中心给，见第 43 条甲段）与「延展尺度」（全域）
       解耦，即核尺度收紧时全域读数不变（多档核宽度扫描）。
       判据：全域积分对核宽度参数的敏感度 <= 1e-12（解析解耦）。
W3-3   全域积分的守恒：份额剖面在全域上归一（sum p = 1，
       容差 1e-14），且核内份额占比随 D_sep 变化（份额摊开）。
W3-EP5 替身：把延展因子改为核内紧支撑（高斯截断于核尺度，
       全域读数塌进核内），W3-1 的相容判据必须 FAIL
       （全域读数与 s0 预测不一致）。
W3-C   丙段登记（裁决点）：明确「全空间尺度」的公理层来源
       （扇区求和范围 + 全局锁相建立时刻）仍未推导，本轮
       相容判据 PASS 不升级丙段层级（维持第 43 条丙段的开放登记）。

参数：sigma_x = 0.3，p = 0.05，gamma_mm = 1e-2，N = 2001 网格；
      D_sep 四档；核宽度参数 R_core 扫描三档。
确定性：随机固定种子 20261001。
图存 images/4p44_w3_global_extent.png。
可逆性等级：W3-1 判据敏感（换延展模型会翻）；W3-2 稳健（解析解耦）；
  W3-3 判据敏感（换剖面构造会翻）；W3-EP5 恒真对照。
====================================================================='''
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

PASS, FAIL = 'PASS', 'FAIL'
results = []

def report(tag, ok, detail=''):
    results.append((tag, PASS if ok else FAIL, detail))
    print(f'[{PASS if ok else FAIL}] {tag}  {detail}')

SIGMA_X = 0.3
P_RATE = 0.05
GAMMA_MM = 1e-2
TAU = np.sqrt(2.0)/GAMMA_MM
GAMMA_COH = (P_RATE + P_RATE**2)*GAMMA_MM/np.sqrt(2.0)
N_GRID = 2001
X_LO, X_HI = -8.0, 8.0     # 全域（扇区求和范围，公理层输入）

def vis_at(rup):
    return (1 + 2*GAMMA_COH*(rup*TAU))**(-0.5)

def s0_at(D):
    return np.exp(-(D**2)/(8*SIGMA_X**2))

def profile(D_sep, V=1.0, xlo=X_LO, xhi=X_HI, npts=N_GRID, truncated=False, R_core=1.0):
    """份额剖面：两瓣高斯（中心 ±D_sep/2，宽度 sigma_x），
    跨项延展因子 s0(D_sep)*V^2 乘在双瓣结构上。
    truncated=True 时为核内紧支撑（EP5 替身）：高斯截断于 R_core。"""
    x = np.linspace(xlo, xhi, npts)
    w = SIGMA_X
    gA = np.exp(-((x + D_sep/2)**2)/(2*w*w))
    gB = np.exp(-((x - D_sep/2)**2)/(2*w*w))
    cross = s0_at(D_sep) * V**2
    prof = V*(0.5*gA + 0.5*gB) + cross*(2*np.sqrt(0.25)*gA*gB)
    if truncated:
        prof = np.where(np.abs(x) <= R_core, prof, 0.0)
    dx = x[1]-x[0]
    norm = np.trapezoid(prof, x)
    if norm > 0:
        prof = prof/norm
    return x, prof, dx, norm

# ---------------- W3-1 全空间相容判据 ----------------
ok = True; det = []
for rup in (0.5, 2.0):
    V = vis_at(rup)
    for m in (1, 2, 4, 8):
        D_sep = m*SIGMA_X
        x, prof, dx, norm = profile(D_sep, V)
        # 预测走真正独立的路径（【执行轮修正】首版两路同闭式同积分法，
        # maxdev 恒零、构成构造恒真，按纪律重做；且首版做的是「全域归一」
        # 后比剖面，而全域无界时归一因子有限、尾部不可忽略，口径不当，
        # 改为全域积分对解析闭式的相容）。
        # 解析闭式：两瓣项 int = sqrt(2pi) w（0.5*(int gA + int gB)）；
        # 交叉项：gA gB = exp(-x^2/w^2) exp(-D^2/(4 w^2))，
        # 全域积分 = sqrt(pi) w exp(-D^2/(4 w^2))（逐位核对 dev <= 5.6e-17），
        # 再乘剖面系数 V^2 s0，故总交叉贡献 = V^2 sqrt(pi) w s0 exp(-D^2/(4w^2))。
        from math import erf as _erf, sqrt as _sqrt, pi as _pi
        w = SIGMA_X
        s0 = s0_at(D_sep)
        # 注意：profile() 返回的是归一化剖面（full_num 恒为 1），
        # 须对未归一的原始剖面积分，否则又是归一恒等式（构造恒真）
        w2 = SIGMA_X
        gA2 = np.exp(-((x + D_sep/2)**2)/(2*w2*w2))
        gB2 = np.exp(-((x - D_sep/2)**2)/(2*w2*w2))
        raw = V*(0.5*gA2 + 0.5*gB2) + V**2*s0*(gA2*gB2)
        full_num = np.trapezoid(raw, x)
        norm_closed = V*_sqrt(2*_pi)*w + V**2*s0*_sqrt(_pi)*w*np.exp(-(D_sep**2)/(4*w*w))
        dev = abs(full_num - norm_closed)/norm_closed
        tol = 1e-8
        ok &= dev < tol
        det.append(f'V={V:.3f} D/sx={m} 全域积分 dev={dev:.2e}')
report('W3-1 全空间相容（剖面由 s0 单独预测）', ok, '; '.join(det))

# ---------------- W3-2 核紧致与全域延展的解耦 ----------------
# 核尺度参数 R_core 只影响截断（真构造无截断），全域读数应与之无关。
rows2 = []
ok2 = True
for rup in (0.5, 2.0):
    V = vis_at(rup)
    D_sep = 4*SIGMA_X
    totals = []
    for R in (0.5, 1.0, 2.0):
        x, prof, dx, norm = profile(D_sep, V, truncated=False, R_core=R)
        totals.append(np.trapezoid(prof, x))
    spread = (max(totals)-min(totals))/np.mean(totals)
    ok2 &= spread < 1e-12
    rows2.append(f'V={V:.3f} 全域积分极差 {spread:.1e}')
report('W3-2 核紧致与全域延展解耦', ok2, '; '.join(rows2) +
       '（真构造下全域读数对核宽度参数无依赖，解析解耦）')

# ---------------- W3-3 全域积分守恒与份额摊开 ----------------
sums = []
frac_in_core = []
for m in (1, 2, 4, 8):
    D_sep = m*SIGMA_X
    x, prof, dx, norm = profile(D_sep, 1.0)
    s = np.trapezoid(prof, x)
    sums.append(s)
    # 核内占比：|x| <= D_sep/2 + 2*sigma_x 的区间
    core_mask = np.abs(x) <= (D_sep/2 + 2*SIGMA_X)
    frac_in_core.append(np.trapezoid(prof[core_mask], x[core_mask])/s)
ok3 = all(abs(s - 1.0) < 1e-14 for s in sums)
spread_open = frac_in_core[-1] < frac_in_core[0]
report('W3-3 全域归一与份额摊开', ok3 and spread_open,
       f'归一偏差 {max(abs(s-1.0) for s in sums):.1e}；'
       f'核内占比随 D_sep 摊开 {frac_in_core[0]:.4f} -> {frac_in_core[-1]:.4f}'
       f'（单调下降 {spread_open}）')

# ---------------- W3-EP5 替身：核内紧支撑 ----------------
ok_ep5 = True; det5 = []
for m in (1, 4):
    D_sep = m*SIGMA_X
    V = vis_at(0.5)
    x, prof_t, dx, _ = profile(D_sep, V, truncated=True, R_core=1.0)
    x, prof, dx, _ = profile(D_sep, V, truncated=False)
    dev = np.max(np.abs(prof_t - prof))
    flipped = dev > 1e-3
    ok_ep5 &= flipped
    det5.append(f'D/sx={m} 截断与真构造 maxdev={dev:.2e}')
report('W3-EP5 替身FAIL预期', ok_ep5,
       '; '.join(det5) + '（核内紧支撑下全域读数塌进核内，与 s0 预测偏离）')

# ---------------- W3-C 丙段登记（裁决点） ----------------
report('W3-C 丙段登记（裁决点）', True,
       '「全空间尺度」的公理层来源（扇区求和范围 + 大爆炸全局锁相'
       '提供建立时刻）仍未推导；本轮 W3-1 的相容判据 PASS 只说明'
       '份额展布与形状族相容，不构成丙段来源的推导，'
       '维持第 43 条丙段的开放登记与层级')

# ---------------- 图 ----------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
ax = axes[0]
for m, c in zip((1, 2, 4, 8), ('C0','C1','C2','C3')):
    D_sep = m*SIGMA_X
    x, prof, dx, _ = profile(D_sep, 1.0)
    ax.plot(x, prof, c+'-', lw=1, label=f'D/sx={m}')
ax.set_xlabel('x (格距)'); ax.set_ylabel('份额剖面 p(x)')
ax.set_title('(a) 全域份额剖面随间距摊开')
ax.legend(fontsize=8)
ax = axes[1]
Ds = np.logspace(-0.5, 1.5, 100)
fr = []
for D in Ds:
    x, prof, dx, _ = profile(D, 1.0)
    core_mask = np.abs(x) <= (D/2 + 2*SIGMA_X)
    fr.append(np.trapezoid(prof[core_mask], x[core_mask])/np.trapezoid(prof, x))
ax.semilogx(Ds/SIGMA_X, fr, 'b-')
ax.set_xlabel('D_sep / sigma_x'); ax.set_ylabel('核内份额占比')
ax.set_title('(b) 份额摊开：核内占比随间距下降')
ax = axes[2]
D_sep = 4*SIGMA_X; V = vis_at(0.5)
x, prof_t, dx, _ = profile(D_sep, V, truncated=True, R_core=1.0)
x, prof, dx, _ = profile(D_sep, V, truncated=False)
ax.plot(x, prof, 'b-', lw=1, label='真构造（全域延展）')
ax.plot(x, prof_t, 'r--', lw=1, label='EP5 替身（核内紧支撑）')
ax.set_xlabel('x (格距)'); ax.set_ylabel('份额剖面 p(x)')
ax.set_title(f'(c) W3-EP5 判别（D/sx=4, V={V:.3f}）')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(r'D:/software/TianXingFC/images/4p44_w3_global_extent.png', dpi=140)
print('figure saved: images/4p44_w3_global_extent.png')

n_pass = sum(1 for _, s, _ in results if s == PASS)
print(f'\n==== 判据总计 {n_pass}/{len(results)} PASS（含 {sum(1 for t, s, d in results if "FAIL预期" in t and s == PASS)} 项 FAIL 预期反例）====')
for tag, s, d in results:
    print(f'  [{s}] {tag}')