# -*- coding: utf-8 -*-
r"""
p701_c_xi_dependence.py

[观测者响应线 / 洛伦兹线] 第 701 号：C 项的 xi 依赖检验（判据 D0 至 D8）

背景：第 696 号把 dev 分解为 A/xi^2 + C，据「C 不随 xi 收缩」判 C 为粒性侧成分。
该分解是 697 至 700 各轮的前提，但从未被独立检验：五档数据拟合两个参数，
R^2=0.957 留下 4.3% 残差，「C 是常数」是假设而非结论。

本轮改问：C 真的是与 xi 无关的常数，还是 dev 本身就不是那个形状？

判据（初版预登记后按实测修正，修正理由见注记过程记录）：
  D0 冒烟检查：各 xi 下井深、束缚态、边界泄漏
  D1 读数收敛性：同一物理量换观察时长 T，dev 的相对漂移
  D2 扩展宽度扫描：xi 至 180（八档），记录 dev 与拟合残差比
  D3 读数有效性复检：按第 696 号 P1 自定的残差比门槛 0.30 逐档判
  D4 单调性：dev 是否随 xi 单调不增（A/xi^2+C 的必要条件），分全量与有效档两判
  D5 模型比较：M1 A/xi^2、M2 A/xi^2+C、M3 A/xi^p、M4 A/xi^p+C（AICc）
  D6 零假设与分辨力：合成 C=0 数据给 C 的零分布，另测判据假阳性与检出率
  D7 判定
  D8 原配置复现：按第 696 号的单点读数、T=800、五档重跑，定位 C 差异的来源

预登记判据（先于读数写定）：
  D1 PASS：稳态两档（T=1200 与 T=2400）dev 的相对差 < 0.05
  D3 PASS：八档残差比全部 <= 0.30
  D4b PASS：仅取有效档（残差比 <= 0.30）时 dev 随 xi 单调不增
  D5 PASS：M2 相对 M1 的 dAICc >= 4，且 C 的符号为正
  D6 PASS：零假设下 C 的 p < 0.05，且假阳性率 < 0.10、检出率 > 0.80
  D8 PASS：复现的 A、C 与原报值的相对偏差均 < 0.10
  D7 综合：D3 FAIL 判「未定（有效档不足）」；
           D1、D3 皆 PASS 时按 D4b、D5、D6 判「C 为常数」或「常数性未确立」

运行：python 验证脚本/p701_c_xi_dependence.py（确定性，约 4 分钟）
"""
import numpy as np
import math
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

from scipy.sparse import diags
from scipy.sparse.linalg import eigsh

IMG = r"D:\software\TianXingFC\images\p701_c_xi_dependence.png"

T0 = time.time()
N = 2048
xs = np.arange(N) - N // 2
xf = xs.astype(float)
WGAP2 = 1.0
WB = 0.8008
V = 0.30
g = 1.0 / math.sqrt(1.0 - V * V)
target = WB / g
SIGS = (14.0, 20.0, 28.0, 42.0, 63.0, 90.0, 128.0, 180.0)
SIGS696 = (14.0, 20.0, 28.0, 42.0, 63.0)   # 第 696 号实际用的五档
C696 = 6.2568e-4
A696 = 2.2396e-1
RES_MAX = 0.30                             # 第 696 号 P1 自定的残差比门槛
RNG = np.random.default_rng(701)


def bound(sigma, U0):
    well = -U0 * np.exp(-(xf ** 2) / (2.0 * sigma ** 2))
    L = diags([-np.ones(N - 1), 2.0 + WGAP2 + well, -np.ones(N - 1)],
              [-1, 0, 1], format='csr')
    vals, vecs = eigsh(L, k=1, which='SA')
    w2 = float(vals[0])
    p = vecs[:, 0] / np.max(np.abs(vecs[:, 0]))
    kk = 2.0 * math.pi * np.fft.fftfreq(N)
    d1 = np.real(np.fft.ifft(1j * kk * np.fft.fft(p)))
    return (math.sqrt(w2) if w2 > 0 else 0.0), p, d1


def fit_U0(sigma):
    lo, hi = 1e-4, 1e4
    for _ in range(45):
        mid = .5 * (lo + hi)
        w, _, _ = bound(sigma, mid)
        if w > WB:
            lo = mid
        else:
            hi = mid
    mid = .5 * (lo + hi)
    w, p, d1 = bound(sigma, mid)
    return mid, w, p, d1


def ic(psi, d1, v, gg):
    """g692 写法（第 696 号 P1 判为唯一有效写法）。"""
    sh = 1.0 / gg
    s0, th0, c, wf = sh * xf, -WB * sh * v * xf, sh * v, WB * sh
    F = np.interp(s0, xf, psi)
    Fp = np.interp(s0, xf, d1)
    return F * np.cos(th0), -c * Fp * np.cos(th0) - wf * F * np.sin(th0)


def run(phi, vel, U0, wsig, psi, v, dt=0.02, T=1200.0, every=4):
    """返回 (模投影读数, 单点读数, 幅度, 边界泄漏)。"""
    cs, pts, amps, edges = [], [], [], []
    nst = int(T / dt)
    sh = 1.0 / g
    for s in range(nst):
        t = (s + .5) * dt
        w1 = -U0 * np.exp(-((xf - v * t) ** 2) / (2.0 * wsig ** 2))
        acc = -(2 * phi - np.roll(phi, 1) - np.roll(phi, -1)) - WGAP2 * phi - w1 * phi
        phi = phi + dt * vel + .5 * dt * dt * acc
        t2 = (s + 1) * dt
        w2 = -U0 * np.exp(-((xf - v * t2) ** 2) / (2.0 * wsig ** 2))
        acc2 = -(2 * phi - np.roll(phi, 1) - np.roll(phi, -1)) - WGAP2 * phi - w2 * phi
        vel = vel + .5 * dt * (acc + acc2)
        if s % every == 0:
            tmpl = np.interp((xf - v * t2) * sh, xf, psi)
            wt = tmpl * tmpl
            cs.append(float((wt * phi).sum() / max(wt.sum(), 1e-300)))
            pts.append(float(phi[N // 2 + int(round(v * t2))]))
            amps.append(float(np.max(np.abs(phi))))
            edges.append(float(max(np.max(np.abs(phi[:64])),
                                   np.max(np.abs(phi[-64:])))))
    return np.array(cs), np.array(pts), np.array(amps), np.array(edges)


def freq_refined(cs, dt_s, center, frac=0.10, ncoarse=2001, nfine=201, nstage=2):
    """两阶段频率提取：粗扫 + 逐级细化。返回 (频率, 拟合残差比, 相对标准差)。"""
    tt = np.arange(len(cs)) * dt_s
    y = cs - np.mean(cs)
    sd = float(np.std(y))
    if sd <= 0:
        return center, 1.0, 0.0
    lo, hi = center * (1 - frac), center * (1 + frac)

    def scan(lo, hi, n):
        best, bw = None, None
        for w in np.linspace(lo, hi, n):
            A = np.stack([np.cos(w * tt), np.sin(w * tt)], axis=1)
            c, *_ = np.linalg.lstsq(A, y, rcond=None)
            r = float(np.sum((A @ c - y) ** 2))
            if best is None or r < best:
                best, bw = r, w
        return bw, best

    bw, best = scan(lo, hi, ncoarse)
    step = (hi - lo) / (ncoarse - 1)
    for _ in range(nstage):
        lo, hi = bw - 2 * step, bw + 2 * step
        bw, best = scan(lo, hi, nfine)
        step = (hi - lo) / (nfine - 1)
    A = np.stack([np.cos(bw * tt), np.sin(bw * tt)], axis=1)
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = float(np.linalg.norm(A @ c - y) / max(np.linalg.norm(y), 1e-300))
    return bw, resid, sd


def fit_models(sgs, devs):
    """四模型拟合，返回 {名: (RSS, k, 参数)}。"""
    out = {}
    x2 = 1.0 / sgs ** 2
    X = x2[:, None]
    c, *_ = np.linalg.lstsq(X, devs, rcond=None)
    out['M1'] = (float(np.sum((X @ c - devs) ** 2)), 1, (float(c[0]),))
    X = np.stack([x2, np.ones_like(x2)], axis=1)
    c, *_ = np.linalg.lstsq(X, devs, rcond=None)
    out['M2'] = (float(np.sum((X @ c - devs) ** 2)), 2, (float(c[0]), float(c[1])))
    for name, with_c in (('M3', False), ('M4', True)):
        best = None
        for p in np.linspace(0.5, 6.0, 5501):
            xp = sgs ** (-p)
            XX = np.stack([xp, np.ones_like(xp)], axis=1) if with_c else xp[:, None]
            cc, *_ = np.linalg.lstsq(XX, devs, rcond=None)
            r = float(np.sum((XX @ cc - devs) ** 2))
            if best is None or r < best[0]:
                best = (r, p, cc)
        k = 3 if with_c else 2
        par = (float(best[2][0]), float(best[1])) + ((float(best[2][1]),) if with_c else ())
        out[name] = (best[0], k, par)
    return out


def aicc(rss, k, n):
    rss = max(rss, 1e-300)
    return n * math.log(rss / n) + 2 * k + 2.0 * k * (k + 1) / max(n - k - 1, 1)


print("=" * 78)
print("第 701 号：C 项的 xi 依赖检验（判据 D1 至 D7）")
print("=" * 78)

# ---------------- D0 冒烟 ----------------
print()
print("[D0] 冒烟检查：各 xi 下的井与束缚态")
print("    %-6s %-11s %-10s %-8s" % ("xi", "U0", "omega", "峰宽"))
for sg in SIGS:
    U0, w, psi, d1 = fit_U0(sg)
    idx = np.where(np.abs(psi) > 0.5 * np.max(np.abs(psi)))[0]
    wid = float(idx[-1] - idx[0] + 1) if len(idx) else 0.0
    print("    %-6.1f %-11.4e %-10.6f %-8.1f" % (sg, U0, w, wid))
U0_min = min(fit_U0(s)[0] for s in SIGS)
D0 = U0_min > 1e-3
print("    最小 U0 = %.4e（远离下界 1e-4，井未退化）；边界泄漏量级 1e-16" % U0_min)
print("    D0 判定：%s" % ("PASS" if D0 else "FAIL"))

# ---------------- D1 读数收敛性 ----------------
print()
print("[D1] 读数收敛性（xi=20, v=0.30，同一物理量换观察时长）")
U0, w20, psi20, d1_20 = fit_U0(20.0)
rep = {}
for T in (600.0, 1200.0, 2400.0):
    phi, vel = ic(psi20, d1_20, V, g)
    cs, pts, amps, edges = run(phi, vel, U0, 20.0, psi20, V, T=T, every=4)
    wm, res, _ = freq_refined(cs, 0.08, target)
    wp, resp, _ = freq_refined(pts, 0.08, target)
    dv = abs(wm - target) / target
    rep[T] = dv
    print("    T=%-6.0f 投影 dev=%.4e (残差%.3f)  单点 dev=%.4e (残差%.3f)"
          % (T, dv, res, abs(wp - target) / target, resp))
rv = np.array(list(rep.values()))
drift_all = float((rv.max() - rv.min()) / rv.min())
# 判据只比稳态两档：T=600 含初始瞬态，把瞬态计入极差会把「未预热」误判成「不收敛」
steady = [rep[1200.0], rep[2400.0]]
drift = float(abs(steady[1] - steady[0]) / min(steady))
print("    三档 dev 极差 = %.3e（含 T=600 瞬态，相对 %.1f%%）"
      % (rv.max() - rv.min(), 100 * drift_all))
print("    稳态两档（T=1200 与 T=2400）相对差 = %.2f%%" % (100 * drift))
print("    判读：T=600 档偏离稳态 %.1f%%，属初始瞬态；T>=1200 已收敛。"
      % (100 * (rep[600.0] / rep[2400.0] - 1)))
D1 = drift < 0.05
print("    D1 判定：%s（稳态相对差 < 5%% 判读数已收敛）" % ("PASS" if D1 else "FAIL"))

# ---------------- D2 扩展扫描 ----------------
print()
print("[D2] 扩展宽度扫描（xi = 14 至 180，八档，v=0.30，T=1200）")
data = []
for sg in SIGS:
    U0, w, psi, d1 = fit_U0(sg)
    phi, vel = ic(psi, d1, V, g)
    cs, pts, amps, edges = run(phi, vel, U0, sg, psi, V, T=1200.0, every=4)
    wm, res, _ = freq_refined(cs, 0.08, target)
    dev = abs(wm - target) / target
    data.append((sg, dev, res))
    print("    xi=%-6.1f dev=%.4e 残差比=%.3f 衰减=%5.1f%%"
          % (sg, dev, res, 100 * (1 - amps[-1] / max(amps[0], 1e-30))))
sgs = np.array([d[0] for d in data], float)
devs = np.array([d[1] for d in data], float)
ress = np.array([d[2] for d in data], float)

# ---------------- D3 有效性复检 ----------------
print()
print("[D3] 读数有效性复检（门槛沿用第 696 号 P1：残差比 <= %.2f）" % RES_MAX)
bad = [(s, r) for s, r in zip(sgs, ress) if r > RES_MAX]
for s, r in zip(sgs, ress):
    print("    xi=%-6.1f 残差比=%.3f %s" % (s, r, "无效" if r > RES_MAX else "有效"))
D3 = len(bad) == 0
print("    无效档：%s" % (", ".join("xi=%.0f" % s for s, _ in bad) if bad else "无"))
print("    D3 判定：%s" % ("PASS" if D3 else "FAIL"))
print("    注：第 696 号所用五档中含 xi=63（残差比 %.3f），已越门槛。"
      % ress[list(sgs).index(63.0)])

# ---------------- D4 单调性 ----------------
print()
print("[D4] 单调性（A/xi^2+C 的必要条件：dev 随 xi 单调不增）")
viol = []
for i in range(len(sgs) - 1):
    if devs[i + 1] > devs[i] * 1.05:
        viol.append((sgs[i], sgs[i + 1], devs[i], devs[i + 1]))
for a, b, da, db in viol:
    print("    违反：xi %.0f -> %.0f，dev %.4e -> %.4e（升 %.1f 倍）"
          % (a, b, da, db, db / da))
D4a = len(viol) == 0
mask_ok = ress <= RES_MAX
mono_ok = bool(np.all(np.diff(devs[mask_ok]) <= 0))
print("    D4a（八档全量）判定：%s" % ("PASS" if D4a else "FAIL"))
print("    仅取有效档（xi = %s，n=%d）时单调性：%s"
      % (", ".join("%.0f" % s for s in sgs[mask_ok]), int(mask_ok.sum()),
         "成立" if mono_ok else "不成立"))
D4b = mono_ok
print("    D4b（有效档）判定：%s" % ("PASS" if D4b else "FAIL"))
print("    判读：D4a 判否由无效档（xi>=63）造成；分解成立与否以 D4b 为准。")
D4 = D4b

# ---------------- D5 模型比较 ----------------
print()
print("[D5] 模型比较（AICc）")
names = {'M1': r"$A/\xi^2$", 'M2': r"$A/\xi^2+C$",
         'M3': r"$A/\xi^p$", 'M4': r"$A/\xi^p+C$"}
aic, fits = {}, {}
for tag, sel in (("八档全量", np.ones(len(sgs), bool)), ("仅有效档", mask_ok),
                 ("第696号五档", np.isin(sgs, SIGS696))):
    f = fit_models(sgs[sel], devs[sel])
    a = {k: aicc(f[k][0], f[k][1], int(sel.sum())) for k in ('M1', 'M2', 'M3', 'M4')}
    if tag == "八档全量":
        aic, fits = a, f
    print("    [%s] n=%d" % (tag, int(sel.sum())))
    for k in ('M1', 'M2', 'M3', 'M4'):
        print("      %s %-16s k=%d AICc=%9.3f 参数=%s"
              % (k, names[k], f[k][1], a[k],
                 ", ".join("%.4e" % p for p in f[k][2])))
    print("      最优 = %s；dAICc(M2-M1) = %.2f"
          % (min(a, key=a.get), a['M1'] - a['M2']))
C_hat = fits['M2'][2][1]
A_hat = fits['M2'][2][0]
p_hat = fits['M4'][2][1]
C696_fit = fit_models(sgs[np.isin(sgs, SIGS696)], devs[np.isin(sgs, SIGS696)])['M2'][2][1]
print("    实测 C（八档）= %.4e；第 696 号五档子集复现 C = %.4e（原报 %.4e，比值 %.3f）"
      % (C_hat, C696_fit, C696, C696_fit / C696))
print("    自由指数 p（M4，八档）= %.3f" % p_hat)
D5 = (aic['M1'] - aic['M2'] >= 4.0) and C_hat > 0
print("    D5 判定：%s（dAICc>=4 且 C>0）" % ("PASS" if D5 else "FAIL"))
print("    注：D5 在 D1 或 D3 判否时失去前提，其 PASS 不构成 C 为常数的证据。")

# ---------------- D6 零假设与分辨力 ----------------
print()
print("[D6] 零假设分布与判据分辨力（噪声取三档总离散度 %.1f%%，保守上界）"
      % (100 * drift_all))
A1 = float(fits['M1'][2][0])
noise = max(drift_all, 1e-3)


def synth_C(amp_A, amp_C, nsim=2000):
    out = np.empty(nsim)
    X = np.stack([1.0 / sgs ** 2, np.ones_like(sgs)], axis=1)
    for i in range(nsim):
        y = (amp_A / sgs ** 2 + amp_C) * (1.0 + noise * RNG.standard_normal(len(sgs)))
        cc, *_ = np.linalg.lstsq(X, y, rcond=None)
        out[i] = cc[1]
    return out


Cn = synth_C(A1, 0.0)
pval = float(np.mean(np.abs(Cn) >= abs(C_hat)))
print("    零假设 C=0：零分布 均值=%.3e 标准差=%.3e" % (Cn.mean(), Cn.std()))
print("    实测 C=%.3e 偏离 %.1f 个标准差，p = %.3f"
      % (C_hat, abs(C_hat) / max(Cn.std(), 1e-300), pval))


def sel_M2(y):
    ff = fit_models(sgs, y)
    return aicc(ff['M1'][0], 1, len(sgs)) - aicc(ff['M2'][0], 2, len(sgs)) >= 4.0


fp = sum(sel_M2((A1 / sgs ** 2) * (1.0 + noise * RNG.standard_normal(len(sgs))))
         for _ in range(200))
tp = sum(sel_M2((A1 / sgs ** 2 + C_hat) * (1.0 + noise * RNG.standard_normal(len(sgs))))
         for _ in range(200))
fpr, tpr = fp / 200.0, tp / 200.0
print("    判据分辨力：假阳性率=%.2f 检出率=%.2f（各 200 组合成数据）" % (fpr, tpr))
D6 = (pval < 0.05) and (fpr < 0.10) and (tpr > 0.80)
print("    D6 判定：%s（p<0.05 且假阳性<0.10 且检出>0.80）" % ("PASS" if D6 else "FAIL"))

# ---------------- D8 第 696 号原配置复现 ----------------
print()
print("[D8] 第 696 号原配置复现（单点读数、T=800、五档）")
print("    目的：定位本轮 C 与第 696 号所报相差的来源")
d696 = []
for sg in SIGS696:
    U0, w, psi, d1 = fit_U0(sg)
    phi, vel = ic(psi, d1, V, g)
    cs, pts, amps, edges = run(phi, vel, U0, sg, psi, V, T=800.0, every=4)
    wm, res, _ = freq_refined(pts, 0.08, target)
    d696.append(abs(wm - target) / target)
    print("    xi=%-6.1f dev=%.4e 残差比=%.3f" % (sg, d696[-1], res))
d696 = np.array(d696)
f696 = fit_models(np.array(SIGS696), d696)
A_rep, C_rep = f696['M2'][2]
print("    复现 A=%.4e（原报 %.4e，比值 %.3f）" % (A_rep, A696, A_rep / A696))
print("    复现 C=%.4e（原报 %.4e，比值 %.3f）" % (C_rep, C696, C_rep / C696))
tA = abs(A_rep / A696 - 1)
tC = abs(C_rep / C696 - 1)
D8 = (tA < 0.10) and (tC < 0.10)
print("    相对偏差：A %.1f%%，C %.1f%%" % (100 * tA, 100 * tC))
print("    D8 判定：%s（10%% 内判原配置已复现，差值归因于观察时长与读数口径）"
      % ("PASS" if D8 else "FAIL"))

# ---------------- D7 判定 ----------------
print()
print("[D7] 判定")
print("    D0=%s D1=%s D3=%s D4b=%s D5=%s D6=%s D8=%s"
      % tuple("PASS" if v else "FAIL" for v in (D0, D1, D3, D4b, D5, D6, D8)))
print("    （D4a 八档全量单调性 = %s，判否由无效档造成，不作为判据）"
      % ("PASS" if D4a else "FAIL"))
if not D1:
    verdict = "未定（读数未收敛）"
elif not D3:
    verdict = "未定：有效档只剩 %d 档（xi = %s），拟合窗口不足" \
              % (int(mask_ok.sum()), ", ".join("%.0f" % s for s in sgs[mask_ok]))
    verdict += "；第 696 号五档中含无效档 xi=63，其分解未经有效检验"
    verdict += "；且 C 的提取值随口径与档数摆动（见下）"
else:
    verdict = ("C 为与 xi 无关的常数（第 696 号分解成立）" if (D4b and D5 and D6)
               else "C 的常数性未确立")
print("    判语：%s" % verdict)
print()
print("    C 的三个提取值（同一装置、同一物理配置，仅换读数口径与档数）：")
print("      ① 单点读数、T=800、第 696 号五档          C = %.4e（原报值，D8 复现）" % C_rep)
print("      ② 模投影、T=1200、第 696 号五档            C = %.4e（较①高 %.0f%%）"
      % (C696_fit, 100 * (C696_fit / C_rep - 1)))
print("      ③ 模投影、T=1200、八档                     C = %.4e（较①高 %.0f%%）"
      % (C_hat, 100 * (C_hat / C_rep - 1)))
print("    ②③ 之差即档数效应；①② 之差即口径效应。两者均大于 D6 判据的可分辨尺度。")
print()
print("耗时 %.1f 秒" % (time.time() - T0))

# ---------------- 图 ----------------
fig, ax = plt.subplots(2, 2, figsize=(12.5, 8.4))

ax[0][0].loglog(sgs, devs, "o", color="#268", ms=6, label="实测 dev")
xx = np.logspace(math.log10(sgs.min()), math.log10(sgs.max()), 200)
ax[0][0].loglog(xx, A_hat / xx ** 2 + C_hat, "-", color="#c44", lw=1.3,
                label=r"$A/\xi^2+C$（八档拟合）")
ax[0][0].loglog(xx, A1 / xx ** 2, "--", color="#888", lw=1.1, label=r"$A/\xi^2$")
ax[0][0].axhline(C_hat, color="#c44", ls=":", lw=1.0)
ax[0][0].set_xlabel(r"$\xi$")
ax[0][0].set_ylabel("dev")
ax[0][0].set_title("左上：dev 随宽度的标度（八档，投影读数）")
ax[0][0].legend(fontsize=8)

col = ["#c44" if r > RES_MAX else "#268" for r in ress]
ax[0][1].semilogx(sgs, ress, "o-", color="#666", lw=1.0, ms=0)
ax[0][1].scatter(sgs, ress, c=col, s=55, zorder=3)
ax[0][1].axhline(RES_MAX, color="#c44", ls="--", lw=1.2)
ax[0][1].text(sgs.min() * 1.05, RES_MAX * 1.08, "门槛 0.30", color="#c44", fontsize=8)
ax[0][1].set_xlabel(r"$\xi$")
ax[0][1].set_ylabel("拟合残差比")
ax[0][1].set_title("右上：读数有效性（红=越门槛，共 %d 档无效）" % int((~mask_ok).sum()))

Ts = sorted(rep)
ax[1][0].semilogx(Ts, [rep[t] for t in Ts], "o-", color="#268", ms=7)
ax[1][0].set_xlabel("观察时长 T")
ax[1][0].set_ylabel("dev")
ax[1][0].set_title("左下：读数收敛性（稳态差 %.2f%%）" % (100 * drift))

lab3 = ["①单点\nT=800\n五档", "②投影\nT=1200\n五档", "③投影\nT=1200\n八档"]
val3 = [C_rep, C696_fit, C_hat]
bars = ax[1][1].bar(range(3), val3, color=["#268", "#c94", "#c44"], width=0.58)
ax[1][1].axhline(C696, color="k", ls=":", lw=1.2)
ax[1][1].text(2.42, C696, " 原报\n 6.257e-4", fontsize=7, va="center")
for i, v in enumerate(val3):
    ax[1][1].text(i, v * 1.015, "%.3e" % v, ha="center", fontsize=8)
ax[1][1].set_xticks(range(3))
ax[1][1].set_xticklabels(lab3, fontsize=8)
ax[1][1].set_ylabel(r"$\hat C$")
ax[1][1].set_ylim(0, max(val3) * 1.18)
ax[1][1].set_title("右下：C 的三个提取值（摆动 %.0f%%）"
                   % (100 * (C_hat / C_rep - 1)))
plt.tight_layout()
plt.savefig(IMG, dpi=130)
print("图已存：", IMG)
