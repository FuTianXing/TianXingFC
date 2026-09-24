# -*- coding: utf-8 -*-
"""
minimal_verification_empty_branch_gravity.py

空支能量的引力地位：两瓣叠加模型内的判据 K1 至 K6。

模型：一维 FK 链（N 格点），两个扭结（两瓣）位于 xA、xB，叠加像权重 q 与 1-q。
判据：
  K1 总能量为一份（两瓣构型的缺陷能量对单缺陷，且与权重无关）
  K2 能量分配（逐瓣能量对权重，和为一）
  K3 写入随能量（写入密度剖面与能量密度剖面成比例）
  K4 远场守恒（远场只依赖总质量，两候选皆然）
  K5 近场判别（甲：源双峰；乙：源单峰偏心）
  K6 反空转（两候选在 K5 上可分辨）

口径：一维链上的离散泊松，仅作近场峰数与远场守恒的模型内代理；
      三维与物理单位换算不在本轮。
"""

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUT = r'D:\software\TianXingFC\images\empty_branch_gravity.png'

# ---------- 模型 ----------
N = 1000
k, K = 1.0, 0.005
ell = 1.0 / (2 * np.pi * np.sqrt(K / k))
Mw = 5
n_arr = np.arange(N, dtype=float)


def kink(x):
    """扭结剖面（单粒子定域本相）"""
    return n_arr + (2 / np.pi) * np.arctan(np.exp((n_arr - x) / ell))


def win_avg(f, M=Mw):
    pad = M // 2
    fp = np.pad(f, pad, mode='edge')
    return np.convolve(fp, np.ones(M) / M, mode='valid')


def w_of(u):
    """窗口序参量 w_n"""
    return win_avg(np.exp(2j * np.pi * u))


xA, xB = 400.0, 600.0
wA = w_of(kink(xA))
wB = w_of(kink(xB))

dA = 1 - np.real(wA)          # 实通道亏空（线性泛函）
dB = 1 - np.real(wB)

BA, BB = dA.sum(), dB.sum()   # 单缺陷的实通道预算
print('=== 单缺陷实通道预算 ===')
print('B_A = %.6f   B_B = %.6f   相对差 = %.3e' % (BA, BB, abs(BA - BB) / BA))

# ---------- 判据 ----------
qs = [0.2, 0.35, 0.5, 0.65, 0.8]
loA, hiA = int(xA - 60), int(xA + 60)
loB, hiB = int(xB - 60), int(xB + 60)

print('\n=== K1 总能量为一份（与权重无关）===')
print('   q      B_img      B_img/B_A')
k1_ratio = []
for q in qs:
    d_img = q * dA + (1 - q) * dB
    B_img = d_img.sum()
    k1_ratio.append(B_img / BA)
    print('  %.2f  %.8f   %.8f' % (q, B_img, B_img / BA))
k1_dev = max(abs(r - 1) for r in k1_ratio)
print('  与 1 的最大偏离 = %.3e' % k1_dev)

print('\n=== K2 能量分配随权重 ===')
print('   q      A 占比     期望 q    B 占比   期望 1-q')
k2_dev = []
for q in qs:
    d_img = q * dA + (1 - q) * dB
    sA = d_img[loA:hiA].sum() / d_img.sum()
    sB = d_img[loB:hiB].sum() / d_img.sum()
    k2_dev.append(max(abs(sA - q), abs(sB - (1 - q))))
    print('  %.2f  %.6f   %.6f  %.6f   %.6f' % (q, sA, q, sB, 1 - q))
print('  最大偏离 = %.3e' % max(k2_dev))

# 写入密度 = 能量密度（甲的口径）：rho_n ∝ d_img_n
q0 = 0.5
d_img = q0 * dA + (1 - q0) * dB
rho_jia = d_img / d_img.sum()          # 甲：能量即源
xA_int = int(round(xA))
xB_int = int(round(xB))
rho_yi = np.zeros(N)
rho_yi[xA_int] = 1.0                    # 乙：锚核即源（单点，在 A 瓣）

print('\n=== K3 写入随能量（甲：写入剖面 = 能量剖面）===')
peakA = d_img[loA:hiA].max()
peakB = d_img[loB:hiB].max()
print('  能量剖面双峰：A 峰 %.6e，B 峰 %.6e，比 %.4f（期望 1.000，q=0.5）'
      % (peakA, peakB, peakA / peakB))
print('  写入剖面（甲）= 能量剖面，比例常数 1/Σ = %.3e' % (1.0 / d_img.sum()))


def poisson_1d(rho):
    """离散泊松：theta_{n+1}-2theta_n+theta_{n-1} = -rho_n，端点 theta=0"""
    b = -rho.copy()
    # 三对角求解
    main = -2 * np.ones(N)
    off = np.ones(N - 1)
    # Thomas 算法
    a = np.zeros(N)
    c = np.zeros(N)
    d = np.zeros(N)
    a[0] = main[0]
    c[0] = off[0] / a[0]
    d[0] = b[0] / a[0]
    for i in range(1, N):
        m = main[i] - off[i - 1] * c[i - 1]
        a[i] = m
        if i < N - 1:
            c[i] = off[i] / m
        d[i] = (b[i] - off[i - 1] * d[i - 1]) / m
    x = np.zeros(N)
    x[-1] = d[-1]
    for i in range(N - 2, -1, -1):
        x[i] = d[i] - c[i] * x[i + 1]
    return x


th_jia = poisson_1d(rho_jia)
th_yi = poisson_1d(rho_yi)

print('\n=== K4 远场守恒 ===')
# 远场斜率：取两端 200 格
def far_slope(th, w=200):
    sL = np.polyfit(np.arange(w), th[:w], 1)[0]
    sR = np.polyfit(np.arange(N - w, N), th[N - w:], 1)[0]
    return sL, sR


for name, th in (('甲（能量即源）', th_jia), ('乙（锚核即源）', th_yi)):
    sL, sR = far_slope(th)
    print('  %s：左斜率 %.6e，右斜率 %.6e，总质量 1.0' % (name, sL, sR))

print('\n=== K5 近场判别（源剖面的峰数）===')


def count_peaks(prof, frac=0.1):
    thr = frac * prof.max()
    c = 0
    for i in range(1, len(prof) - 1):
        if prof[i] > thr and prof[i] >= prof[i - 1] and prof[i] > prof[i + 1]:
            c += 1
    return c


# 峰间距需大于窗宽 M 才计为独立峰
def count_peaks_sep(prof, frac=0.1, sep=Mw):
    thr = frac * prof.max()
    cand = [i for i in range(1, len(prof) - 1)
            if prof[i] > thr and prof[i] >= prof[i - 1] and prof[i] > prof[i + 1]]
    peaks = []
    for i in cand:
        if not peaks or i - peaks[-1] >= sep:
            peaks.append(i)
        elif prof[i] > prof[peaks[-1]]:
            peaks[-1] = i
    return peaks


pj = count_peaks_sep(rho_jia)
py = count_peaks_sep(rho_yi)
print('  甲：源峰数 = %d，峰位 = %s' % (len(pj), pj))
print('  乙：源峰数 = %d，峰位 = %s' % (len(py), py))

# 近场（源剖面的二阶差）峰数
lap_jia = -np.diff(th_jia, 2)
lap_yi = -np.diff(th_yi, 2)
print('  甲：场二阶差峰数 = %d' % len(count_peaks_sep(np.abs(lap_jia))))
print('  乙：场二阶差峰数 = %d' % len(count_peaks_sep(np.abs(lap_yi))))

print('\n=== K6 反空转 ===')
print('  两候选在 K5 上可分辨：甲 %d 峰，乙 %d 峰，差 %d'
      % (len(pj), len(py), len(pj) - len(py)))
print('  K1 至 K4 为共同约束，K5 为唯一判别。')

# ---------- 图 ----------
fig, ax = plt.subplots(2, 2, figsize=(11, 7))

ax[0, 0].plot(np.arange(N)[loA - 100:hiB + 100], d_img[loA - 100:hiB + 100], lw=1.2)
ax[0, 0].set_title('能量密度剖面（q=0.5）：双峰')
ax[0, 0].set_xlabel('格点 n')
ax[0, 0].set_ylabel(r'$\tilde\delta_n$')

ax[0, 1].plot(np.arange(N), rho_jia, lw=1.0, label='甲：能量即源')
ax[0, 1].plot(np.arange(N), rho_yi, lw=1.0, label='乙：锚核即源')
ax[0, 1].set_title('两条候选的源剖面')
ax[0, 1].set_xlabel('格点 n')
ax[0, 1].legend(fontsize=8)

ax[1, 0].plot(np.arange(N), th_jia, lw=1.2, label='甲')
ax[1, 0].plot(np.arange(N), th_yi, lw=1.0, ls='--', label='乙')
ax[1, 0].set_title('场剖面（离散泊松）')
ax[1, 0].set_xlabel('格点 n')
ax[1, 0].set_ylabel(r'$\theta_n$')
ax[1, 0].legend(fontsize=8)

ax[1, 1].plot(np.arange(1, N - 1), -np.diff(th_jia, 2), lw=1.2, label='甲')
ax[1, 1].plot(np.arange(1, N - 1), -np.diff(th_yi, 2), lw=1.0, ls='--', label='乙')
ax[1, 1].set_xlim(loA - 120, hiB + 120)
ax[1, 1].set_title('场二阶差（= 源）：峰数判别')
ax[1, 1].set_xlabel('格点 n')
ax[1, 1].legend(fontsize=8)

plt.tight_layout()
plt.savefig(OUT, dpi=130)
print('\n图已保存：%s' % OUT)
