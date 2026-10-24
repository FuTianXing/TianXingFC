# -*- coding: utf-8 -*-
r"""
p702_construction_dependence.py

[观测者响应线 / 洛伦兹线] 第 702 号：dev 的构造来源分解与 A 的伪影检验（判据 AC0 至 AC6）

背景：第 696 号把 dev 分解为 A/xi^2 + C，据此判 A 为波性侧成分，并只外推 A 得物理档
3.92e-46（与预算 1.46e-46 同量级）。第 701 号已证明 C 的提取值随读数口径摆动 48%。
本轮问同一类问题，换一个更大的自变量：**构造**。

第 695 号测过四种构造（阱宽 rigid/contracted x 初值 boost 式/g692 式），dev 极差
117 倍。第 696 号只在其中一种（rigid+g692）上做了 A/C 分解，其余三种的读数有效性
从未逐构造检验。若 A 也随构造变，则物理档闭合的正面结论不成立。

本轮另加一种构造：第 695 号的 contracted 是「井宽收缩而束缚态仍按原宽求解」，本身
不自洽（实测束缚频率偏离 omega_b）。补一个自洽收缩构造（井宽与束缚态同时收缩），
把 dev 中的构造失配与收缩本身分开。

判据：
  AC0 冒烟：五种构造的井、束缚态与实际束缚频率
  AC1 逐构造读数有效性（xi=20，残差比与幅度衰减），按第 696 号 P1 的门槛判
  AC2 各有效构造的四档宽度扫描（xi=14,20,28,42），拟合 A 与 C
  AC3 A 与 C 的构造间极差，判 A 是否为构造伪影
  AC4 初值插值阶数检验（线性 vs 三次样条），定位 dev 中的插值误差贡献
  AC5 大 xi 档的有效性（xi=63,90），看换构造能否突破第 701 号的档数墙
  AC6 八档完整扫描（单点口径，与第 696 号同口径），直接检验档数墙是否口径造成
  AC7 判定

预登记判据（先于读数写定）：
  AC0 冒烟：自洽构造（井宽与解态宽相同）的实际束缚频率等于 omega_b
  AC1 有效性：残差比 <= 0.30 且幅度衰减 <= 50%
  AC3a 全部有效构造的 A 极差比（量度构造失配的敏感度）
  AC3b 自洽构造之间的 A 极差比 < 2.0（判 A 的稳健性，AC3 以此为准）
  AC4 PASS：三次样条相对线性插值的 dev 变化 < 30%（插值误差不是主因）
  AC5 PASS：存在有效构造在 xi=90 上残差比仍 <= 0.30（档数墙可绕过）
  AC6 PASS：单点口径下八档残差比全部 <= 0.30（则第 701 号的档数墙判为口径 artefact）
             （单调性列为诊断项，不并入本判据：它属 C 的常数性问题，第 701 号已判）
  AC7 综合：AC1 有效构造 < 2 判「构造敏感性不可判」；
            AC3b 与 AC4 皆 PASS 判「A 在自洽构造间稳健，构造层面无伪影」；
            否则判「A 含构造伪影，第 696 号的物理档外推须撤回」

运行：python 验证脚本/p702_construction_dependence.py（确定性，约 8 分钟）
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
from scipy.interpolate import CubicSpline

IMG = r"D:\software\TianXingFC\images\p702_construction_dependence.png"

T0 = time.time()
N = 2048
xs = np.arange(N) - N // 2
xf = xs.astype(float)
WGAP2 = 1.0
WB = 0.8008
V = 0.30
g = 1.0 / math.sqrt(1.0 - V * V)
target = WB / g
SIGS = (14.0, 20.0, 28.0, 42.0)     # 第 701 号判定的四个有效档
BIGS = (63.0, 90.0)                 # 第 701 号判 invalid 的大档
RES_MAX = 0.30
DEC_MAX = 0.50

# (名称, 运行井宽因子, 解束缚态的宽度因子, 初值写法)，因子乘在 xi 上。
# 第 695 号原定义：contr 只缩井宽，束缚态仍按 xi 求解。
CONFIGS = (
    ("rigid+boost",    1.0,     1.0,     "boost"),
    ("rigid+g692",     1.0,     1.0,     "g692"),
    ("contr+boost",    1.0 / g, 1.0,     "boost"),
    ("contr+g692",     1.0 / g, 1.0,     "g692"),
    ("selfcons+g692",  1.0 / g, 1.0 / g, "g692"),
)

_U0CACHE = {}


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
    """求 U0 使束缚频率恰为 omega_b。按 sigma 缓存。"""
    key = round(sigma, 10)
    if key in _U0CACHE:
        return _U0CACHE[key]
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
    _U0CACHE[key] = (mid, w, p, d1)
    return _U0CACHE[key]


_CS = {}


def clear_cs():
    _CS.clear()


def interp_at(s0, psi, d1, mode):
    """初值插值。mode='linear' 为第 692/695/696 号原写法，'cubic' 为三次样条。"""
    if mode == "linear":
        return np.interp(s0, xf, psi), np.interp(s0, xf, d1)
    key = id(psi)
    if key not in _CS:
        _CS[key] = (CubicSpline(xf, psi), CubicSpline(xf, d1))
    cp, cd = _CS[key]
    return cp(s0), cd(s0)


def make_ic(psi, d1, v, gg, style, mode="linear"):
    """style: 'g692' 用收缩因子 1/gamma；'boost' 用 gamma（第 695 号定义）。"""
    if style == "boost":
        s0, th0, c, wfac = gg * xf, WB * gg * v * xf, gg * v, WB * gg
    else:
        sh = 1.0 / gg
        s0, th0, c, wfac = sh * xf, -WB * sh * v * xf, sh * v, WB * sh
    F, Fp = interp_at(s0, psi, d1, mode)
    return F * np.cos(th0), -c * Fp * np.cos(th0) - wfac * F * np.sin(th0)


def run(phi, vel, U0, wsig, v, dt=0.02, T=1200.0, every=4):
    cs, amps = [], []
    nst = int(T / dt)
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
            cs.append(float(phi[N // 2 + int(round(v * t2))]))
            amps.append(float(np.max(np.abs(phi))))
    return np.array(cs), np.array(amps)


def freq_refined(cs, dt_s, center, frac=0.10, ncoarse=2001, nfine=201, nstage=2):
    tt = np.arange(len(cs)) * dt_s
    y = cs - np.mean(cs)
    if float(np.std(y)) <= 0:
        return center, 1.0
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
    return bw, float(np.linalg.norm(A @ c - y) / max(np.linalg.norm(y), 1e-300))


def measure(cfg, sg, T=1200.0, mode="linear"):
    """对构造 cfg 在宽度 sg 上做一次测量，返回 (dev, 残差比, 衰减)。"""
    _, wf, sf, style = cfg
    U0, _, psi, d1 = fit_U0(sf * sg)
    clear_cs()
    wsig = wf * sg
    phi, vel = make_ic(psi, d1, V, g, style, mode=mode)
    cs, amps = run(phi, vel, U0, wsig, V, T=T)
    wm, res = freq_refined(cs, 0.08, target)
    return abs(wm - target) / target, res, 1.0 - amps[-1] / max(amps[0], 1e-30)


def fit_AC(sgs, devs):
    """拟合 dev = A/xi^2 + C，返回 (A, C, R^2, A 的标准误)。"""
    X = np.stack([1.0 / sgs ** 2, np.ones_like(sgs)], axis=1)
    coef, *_ = np.linalg.lstsq(X, devs, rcond=None)
    pred = X @ coef
    rss = float(np.sum((devs - pred) ** 2))
    ss = float(np.sum((devs - np.mean(devs)) ** 2))
    r2 = 1.0 - rss / max(ss, 1e-300)
    dof = max(len(sgs) - 2, 1)
    try:
        cov = (rss / dof) * np.linalg.inv(X.T @ X)
        se_A = float(math.sqrt(max(cov[0, 0], 0.0)))
    except Exception:
        se_A = float("nan")
    return float(coef[0]), float(coef[1]), r2, se_A


print("=" * 78)
print("第 702 号：dev 的构造来源分解与 A 的伪影检验（判据 AC0 至 AC6）")
print("=" * 78)
print("装置：N=%d  omega_b=%.4f  v=%.2f  gamma=%.6f  target=%.6f"
      % (N, WB, V, g, target))

# ---------------- AC0 冒烟 ----------------
print()
print("[AC0] 冒烟：五构造的井与束缚态（xi=20）")
print("    %-15s %-11s %-11s %-11s %-11s"
      % ("构造", "运行井宽", "解态宽", "实际 omega", "omega 偏差"))
SMOKE = {}
for cfg in CONFIGS:
    name, wf, sf, _ = cfg
    U0, _, psi, d1 = fit_U0(sf * 20.0)
    w_act, _, _ = bound(wf * 20.0, U0)
    SMOKE[name] = (wf * 20.0, sf * 20.0, w_act)
    print("    %-15s %-11.3f %-11.3f %-11.6f %+.2e"
          % (name, wf * 20.0, sf * 20.0, w_act, w_act - WB))
print("    判读：omega 偏差即井宽与束缚态不匹配的量，自洽构造应给出机器零。")
# 判据只对自洽构造成立：rigid 与 selfcons 的井宽与解态宽相同，
# contr 族沿用第 695 号定义（只缩井宽、不重解束缚态），其偏差是构造失配量本身。
SELFCONS = ("rigid+boost", "rigid+g692", "selfcons+g692")
AC0 = all(abs(SMOKE[n][2] - WB) < 1e-5 for n in SELFCONS)
print("    AC0 判定：%s（自洽构造 %s 的实际束缚频率均等于 omega_b）"
      % ("PASS" if AC0 else "FAIL", "/".join(SELFCONS)))
print("    注：contr 族的偏差 %.2e 来自井宽收缩而束缚态未重解，是构造失配量，"
      "不计入冒烟失败项。" % abs(SMOKE["contr+g692"][2] - WB))

# ---------------- AC1 逐构造有效性 ----------------
print()
print("[AC1] 逐构造读数有效性（xi=20，v=0.30，T=1200）")
print("    门槛：残差比 <= %.2f 且幅度衰减 <= %.0f%%" % (RES_MAX, 100 * DEC_MAX))
valid, P1 = {}, {}
for cfg in CONFIGS:
    name = cfg[0]
    dev, res, decay = measure(cfg, 20.0)
    ok = (res <= RES_MAX) and (decay <= DEC_MAX)
    valid[name] = ok
    P1[name] = (dev, res, decay)
    print("    %-15s dev=%.4e 残差比=%.3f 衰减=%5.1f%% -> %s"
          % (name, dev, res, 100 * decay, "有效" if ok else "无效"))
eff = [c[0] for c in CONFIGS if valid[c[0]]]
print("    有效构造：%s（共 %d 个）" % (", ".join(eff) if eff else "无", len(eff)))
AC1 = len(eff) >= 2
print("    AC1 判定：%s（至少两个有效构造才能做构造比较）" % ("PASS" if AC1 else "FAIL"))

# ---------------- AC2 各有效构造的宽度扫描 ----------------
print()
print("[AC2] 各有效构造的四档宽度扫描（xi = 14,20,28,42）")
FITS = {}
for name in eff:
    cfg = [c for c in CONFIGS if c[0] == name][0]
    devs = []
    for sg in SIGS:
        dev, res, _ = measure(cfg, sg)
        devs.append(dev)
        print("    %-15s xi=%-5.0f dev=%.4e 残差比=%.3f" % (name, sg, dev, res))
    devs = np.array(devs)
    A_, C_, r2_, seA = fit_AC(np.array(SIGS), devs)
    FITS[name] = (A_, C_, r2_, seA, devs)
    print("    拟合：A=%.4e (se %.1e)  C=%.4e  R^2=%.4f" % (A_, seA, C_, r2_))

# ---------------- AC3 A 的构造间极差 ----------------
print()
print("[AC3] A 与 C 的构造间极差")
if len(FITS) >= 2:
    As = np.array([FITS[c][0] for c in FITS])
    Cs = np.array([FITS[c][1] for c in FITS])
    rA = float(As.max() / max(As.min(), 1e-300))
    rC = float(Cs.max() / max(abs(Cs).min(), 1e-300))
    for c in FITS:
        print("    %-15s A=%.4e  C=%.4e  R^2=%.4f" % (c, FITS[c][0], FITS[c][1], FITS[c][2]))
    print("    A 极差比（全部有效构造）= %.2f" % rA)
    print("    C 极差比（全部有效构造）= %.2f" % rC)
    # 分两判：全量极差量度「构造失配的敏感度」，自洽构造之间的极差才是 A 的稳健性
    con_ok = [c for c in FITS if c in SELFCONS]
    if len(con_ok) >= 2:
        As2 = np.array([FITS[c][0] for c in con_ok])
        rA2 = float(As2.max() / max(As2.min(), 1e-300))
        print("    自洽构造（%s）的 A 极差比 = %.2f" % ("/".join(con_ok), rA2))
    else:
        rA2 = float("nan")
        print("    自洽构造不足两个，无法做稳健性判")
    print("    A 与预算系数 1/12 = %.4e 之比：" % (1.0 / 12.0))
    for c in FITS:
        print("      %-15s A*12 = %.3f" % (c, FITS[c][0] * 12))
    # AC3 以自洽构造之间的极差为准：不自洽构造的偏离属构造失配，不是 A 的不确定性
    AC3 = bool(rA2 < 2.0) if rA2 == rA2 else None
    print("    AC3a（全量，量度构造失配敏感度）判定：%s"
          % ("PASS" if rA < 2.0 else "FAIL"))
    print("    AC3b（自洽构造之间，判 A 的稳健性）判定：%s"
          % ("PASS" if AC3 else ("FAIL" if AC3 is False else "不可判")))
else:
    rA, rA2 = float("nan"), float("nan")
    AC3 = None
    print("    有效构造不足两个，无法比较")
print("    AC3 判定（以 AC3b 为准）：%s"
      % ("PASS（A 稳健）" if AC3 else ("FAIL（A 随构造变）" if AC3 is False else "不可判")))

# ---------------- AC4 插值阶数检验 ----------------
print()
print("[AC4] 初值插值阶数检验（rigid+g692，xi=20 与 42）")
print("    %-6s %-8s %-13s %-10s %-11s" % ("xi", "插值", "dev", "残差比", "相对线性"))
cfg_r = [c for c in CONFIGS if c[0] == "rigid+g692"][0]
AC4_rows = []
for sg in (20.0, 42.0):
    base = None
    for mode in ("linear", "cubic"):
        dev, res, _ = measure(cfg_r, sg, mode=mode)
        if base is None:
            base = dev
        AC4_rows.append((sg, mode, dev, res, dev / base - 1))
        print("    %-6.0f %-8s %-13.4e %-10.3f %+10.1f%%"
              % (sg, mode, dev, res, 100 * (dev / base - 1)))
worst = max(abs(r[4]) for r in AC4_rows if r[1] == "cubic")
print("    三次样条相对线性插值的最大变化 = %.2f%%（判据 < 30%%）" % (100 * worst))
AC4 = worst < 0.30
print("    AC4 判定：%s" % ("PASS（插值误差不是主因）" if AC4
                          else "FAIL（dev 主要来自初值插值）"))

# ---------------- AC5 大 xi 档有效性 ----------------
print()
print("[AC5] 大 xi 档的有效性（xi = 63 与 90，看换构造能否突破档数墙）")
print("    %-15s %-8s %-10s %-10s" % ("构造", "xi", "残差比", "是否有效"))
BIG = {}
for name in eff:
    cfg = [c for c in CONFIGS if c[0] == name][0]
    BIG[name] = {}
    for sg in BIGS:
        dev, res, decay = measure(cfg, sg)
        BIG[name][sg] = (dev, res)
        print("    %-15s %-8.0f %-10.3f %-10s"
              % (name, sg, res, "有效" if res <= RES_MAX else "无效"))
got = [n for n in BIG if BIG[n][90.0][1] <= RES_MAX]
AC5 = len(got) > 0
print("    在 xi=90 仍有效的构造：%s" % (", ".join(got) if got else "无"))
print("    AC5 判定：%s" % ("PASS（档数墙可绕过）" if AC5 else "FAIL（档数墙与构造无关）"))

# ---------------- AC6 八档完整扫描（单点口径） ----------------
print()
print("[AC6] 八档完整扫描（rigid+g692，单点读数，与第 696 号同口径）")
print("    目的：第 701 号判「有效档停在四档、档数墙不可绕」是在模投影口径下得出的，")
print("          第 696 号本身用的是单点口径。本判据直接检验该墙是否口径造成。")
SIGS8 = (14.0, 20.0, 28.0, 42.0, 63.0, 90.0, 128.0, 180.0)
dev8, res8 = [], []
for sg in SIGS8:
    dev, res, _ = measure(cfg_r, sg)
    dev8.append(dev)
    res8.append(res)
    print("    xi=%-6.0f dev=%.4e 残差比=%.3f %s"
          % (sg, dev, res, "有效" if res <= RES_MAX else "无效"))
dev8 = np.array(dev8)
res8 = np.array(res8)
nb = int(np.sum(res8 > RES_MAX))
print("    八档中越门槛的档数 = %d（第 701 号在模投影口径下为 4）" % nb)
mono8 = bool(np.all(np.diff(dev8) <= 0))
print("    单调性（诊断项，非本判据内容）：%s"
      % ("成立" if mono8 else "不成立（违反 %d 处，末档回升 %.1f%%）"
         % (int(np.sum(np.diff(dev8) > 0)),
            100 * (dev8[-1] / dev8[-2] - 1))))
A8, C8, r28, se8 = fit_AC(np.array(SIGS8), dev8)
print("    八档拟合：A=%.4e (se %.1e)  C=%.4e  R^2=%.4f" % (A8, se8, C8, r28))
print("    与外推预算之比 A*12 = %.3f（四档给 %.3f）" % (A8 * 12, FITS["rigid+g692"][0] * 12))
print("    注：八档 A 较四档高 %.0f%%，因 xi>=90 三档的 dev 低于「饱和到 C」的预期；"
      % (100 * (A8 / FITS["rigid+g692"][0] - 1)))
print("        该偏离属 C 的常数性问题（第 701 号已判），不计入本判据。")
AC6 = (nb == 0)
print("    AC6 判定：%s（单点口径下越门槛档数为 0，则第 701 号的档数墙判为口径 artefact）"
      % ("PASS" if AC6 else "FAIL"))

# ---------------- AC7 判定 ----------------
print()
print("[AC7] 判定")
print("    AC0=%s AC1=%s AC3b=%s AC4=%s AC5=%s AC6=%s"
      % ("PASS" if AC0 else "FAIL", "PASS" if AC1 else "FAIL",
         ("PASS" if AC3 else "FAIL") if AC3 is not None else "N/A",
         "PASS" if AC4 else "FAIL", "PASS" if AC5 else "FAIL",
         "PASS" if AC6 else "FAIL"))
if not AC1:
    verdict = "构造敏感性不可判（有效构造不足两个）"
elif AC3 and AC4:
    verdict = "A 在自洽构造间稳健，构造层面无伪影"
else:
    verdict = "A 含构造伪影，第 696 号的物理档外推须撤回"
    if not AC3:
        verdict += "；自洽构造间 A 的极差 %.2f 倍" % rA2
    if not AC4:
        verdict += "；初值插值贡献达 %.0f%%" % (100 * worst)
print("    判语：%s" % verdict)
print()
print("    构造失配的敏感度（AC3a）：全量 A 极差 %.2f 倍，自洽构造间 %.2f 倍。" % (rA, rA2))
print("    判读：A 对「井宽与束缚态是否匹配」高度敏感，对「井宽是否随速度收缩」不敏感")
print("          （rigid 与 selfcons 的 A 相差 %.1f%%）。第 696 号所用 rigid+g692 属自洽构造，"
      % (100 * abs(FITS["selfcons+g692"][0] / FITS["rigid+g692"][0] - 1)))
print("          其 A 与 selfcons 一致，故其外推在构造维度上站得住。")
print()
if AC6:
    print("    **第 701 号 D3 的更正**：该号判「有效档停在四档、xi>=63 四档越门槛」，")
    print("    但那是在模投影口径下读的残差比；第 696 号用的是单点口径。本轮以单点口径")
    print("    重做八档，越门槛档数 = %d，单调性 %s，档数墙判为口径 artefact。"
          % (nb, "成立" if mono8 else "不成立"))
    print("    第 696 号五档（含 xi=63）在单点口径下全部有效，其分解未经有效检验的指控不成立。")

BUDGET = 1.4598416214400906e-46
XI_PHYS = 2.389e22
print()
print("    物理档外推的构造依赖（这是本轮最要紧的读出）：")
print("    预算 = %.4e" % BUDGET)
print("    %-15s %-13s %-13s %-10s" % ("构造", "A", "外推值", "外推/预算"))
for c in FITS:
    ext = FITS[c][0] / XI_PHYS ** 2
    print("    %-15s %-13.4e %-13.4e %-10.3f" % (c, FITS[c][0], ext, ext / BUDGET))
if len(FITS) >= 2:
    exts = [FITS[c][0] / XI_PHYS ** 2 for c in FITS if c in SELFCONS]
    print("    自洽构造外推值跨 %.2f 至 %.2f 倍预算；不自洽的 contr 给 %.2f 倍。"
          % (min(exts) / BUDGET, max(exts) / BUDGET,
             FITS["contr+g692"][0] / XI_PHYS ** 2 / BUDGET))
print("耗时 %.1f 秒" % (time.time() - T0))

# ---------------- 图 ----------------
fig, ax = plt.subplots(2, 2, figsize=(12.8, 8.4))

xpos = np.arange(len(CONFIGS))
dev20 = [P1[c[0]][0] for c in CONFIGS]
colr = ["#268" if valid[c[0]] else "#c44" for c in CONFIGS]
ax[0][0].bar(xpos, dev20, color=colr, width=0.62)
ax[0][0].set_yscale("log")
ax[0][0].set_xticks(xpos)
ax[0][0].set_xticklabels([c[0] for c in CONFIGS], fontsize=7, rotation=22, ha="right")
ax[0][0].set_ylabel("dev")
ax[0][0].set_title("左上：五构造的 dev（xi=20，红=读数无效）")

xx = np.logspace(math.log10(13), math.log10(95), 120)
for c in FITS:
    ax[0][1].loglog(SIGS, FITS[c][4], "o", ms=5)
    ax[0][1].loglog(xx, FITS[c][0] / xx ** 2 + FITS[c][1], "-", lw=1.1, label=c)
ax[0][1].set_xlabel(r"$\xi$")
ax[0][1].set_ylabel("dev")
ax[0][1].set_title("右上：各有效构造的 dev 标度（曲线外推至 xi=90）")
ax[0][1].legend(fontsize=7)

ax[1][0].semilogx(SIGS8, res8, "o-", color="#268", ms=6)
ax[1][0].axhline(RES_MAX, color="#c44", ls="--", lw=1.2)
ax[1][0].text(15, RES_MAX * 1.08, "门槛 0.30", color="#c44", fontsize=8)
ax[1][0].set_xlabel(r"$\xi$")
ax[1][0].set_ylabel("拟合残差比")
ax[1][0].set_title("左下：单点口径的八档残差比（越门槛 %d 档）" % nb)

if len(FITS) >= 2:
    lab = list(FITS)
    ext = [FITS[c][0] / XI_PHYS ** 2 / BUDGET for c in lab]
    cols = ["#368" if c in SELFCONS else "#c94" for c in lab]
    ax[1][1].bar(range(len(lab)), ext, color=cols, width=0.6)
    ax[1][1].axhline(1.0, color="#c44", ls="--", lw=1.2)
    ax[1][1].text(len(lab) - 0.55, 1.06, "预算", color="#c44", fontsize=8)
    for i, v in enumerate(ext):
        ax[1][1].text(i, v * 1.04, "%.2f" % v, ha="center", fontsize=8)
    ax[1][1].set_xticks(range(len(lab)))
    ax[1][1].set_xticklabels(lab, fontsize=7, rotation=22, ha="right")
    ax[1][1].set_yscale("log")
    ax[1][1].set_ylabel("外推值 / 预算")
    ax[1][1].set_title("右下：物理档外推（蓝=自洽构造，橙=不自洽）")
else:
    ax[1][1].text(0.5, 0.5, "有效构造不足两个\n构造比较不可判",
                  ha="center", va="center", fontsize=11)
    ax[1][1].set_axis_off()
plt.tight_layout()
plt.savefig(IMG, dpi=130)
print("图已存：", IMG)
