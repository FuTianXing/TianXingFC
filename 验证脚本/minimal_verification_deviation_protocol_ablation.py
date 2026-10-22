"""第 811 号：偏离度标度幂次的协议消融（相对第 767 号与第 810 号读数之差）

起因
  第 810 号在 (k, x) 二维网格上给出偏离度对无量纲组 x = K/k 的幂次 +0.4834
  （标准误 0.0536），并把第 767 号的 +1.705 记为需重估的旧值。两者的自变量
  并非只差常数因子，须先把两者换算到同一自变量上：第 767 号对 1/ell 拟合，
  而 1/ell = 2*pi*sqrt(x)，故 dev 随 (1/ell)^p 变化等价于随 x^(p/2) 变化。
  第 767 号的 +1.705 换算到 x 上是 +0.853，与第 810 号的 +0.4834 相差
  1.76 倍。这是两个真实不同的读数，其差只能来自采样协议。本轮只动协议，
  不动判据量，且全部幂次一律对 x 拟合以免记法混淆。

判据量（逐字继承，不改）
  相对论形状偏离度 dev = rms(R) / mean(|dE|)，其中 R = dE - M*g1，
  g1 = c_s^2 * (1/sqrt(1 - (v/c_s)^2) - 1)，M = sum(g1*dE)/sum(g1^2)。
  本脚本的 deviation 函数直接继承第 810 号的逐字实现。

理论期望（先立，后实现）
  若幂次是物理的，则在判据量定义固定的前提下，仅切换采样协议不应把幂次
  改变到超过协议散布容差。若幂次是协议的，则存在某个协议元素，其单独切换
  造成的幂次偏移显著大于其余元素；本轮不预设是哪一个元素，读数给出答案。
  判否不等于判负：若不存在唯一主导元素，则结论是「幂次差不由单一协议元素
  支配」，这本身是有效结论。

消融矩阵（两个 k 档，K 档随 k 取）
  基线（第 810 号协议）：自相似窗口 W = NW*ell、静置阻尼 gamma = 0.05、
    静置步数按 sqrt(1/K) 缩放并翻倍取快照、速度档以相对速度 beta = v/c_s
    定义。
  A 切回固定窗口 W = 30 格。
  B 切回无阻尼静置。
  C 切回固定静置步数 12000（第 767 号的步数，不翻倍）。
  D 全第 767 号协议（固定窗口 + 无阻尼 + 固定步数 + 绝对速度档）。
  E 只切速度档口径：绝对速度档 v0 = b（相对档为 v0 = b*c_s）。
  替身协议：只微调拟合区间，物理协议不动，用作恒真检验。

  两个 k 档各有用途。k = 1 时链信号速度 c_s = 1，绝对速度档与相对速度档
  逐位相同，「速度档口径」这一协议元素在此时恒等失效，故 k = 1 只用于
  与第 767 号读数同条件复现（其原读数即在 k = 1 上取得）。k = 4 时
  c_s = 2，绝对档 0.20 对应 beta = 0.10，与相对档明显不同，四个协议元素
  全部有效，主消融与主导元素定位在此进行。K 档按 k 取：k = 1 用
  0.00125、0.0025、0.005、0.01（与第 767 号的可用四档相同）；k = 4 用
  0.005、0.01、0.02、0.04（对应 x = 0.00125 至 0.01，与第 810 号 k = 4
  档的可用四档相同）。

判据
  P1 基线复现：k = 4 档基线幂次与第 810 号 k = 4 分档读数 +0.4813 之差
     在容差内。
  P1-b 第 767 号复现：k = 1 档 D 协议对 x 的幂次与第 767 号读数换算到 x
     后的等效值 +0.853 之差是否在容差内。注意比较目标须先换算：第 767 号
     报的是对 1/ell 的幂 +1.705，其对 x 的等效幂为 +1.705/2 = +0.853。
     此条单列，不计入总分。
  P2 主导元素定位：在 k = 4 档上，A、B、C、E 四个单元素切换的幂次偏移中，
     最大者超过偏移下限，且与其余三者中位者之比达到区分力门槛。PASS 只
     表示「存在唯一主导元素」，不表示是哪一个。四个单元素档各自只改动一
     个元素，与基线相比的偏移量即可归因到该元素；D 档同时改动四个元素，
     其偏移只作描述读数，并与四个单元素偏移之和作可加性对照。
  P3 恒真替身：k = 4 档替身协议的幂次偏移远小于真实元素偏移。

  dev 的跨协议散布作为描述读数报出，不作硬判否依据：固定窗口在 K 变化
  下覆盖的物理内容随之改变，dev 的协议差异是设计内的，把 dev 散布设为
  门槛会把「协议设计差异」误判为「读数不稳」。

容差预算
  幂次偏移下限取 0.05 绝对值；主导元素区分力门槛取 3 倍；基线复现容差
  取 0.10，第 767 号复现容差取 0.10。基线容差约两个标准误（810 号实测
  幂次标准误 0.0536，三档 k 分读 +0.4847、+0.4841、+0.4813）；第 767 号
  复现容差由该轮自身的跨协议散布换算到 x 幂后取定：其对 1/ell 的幂散布
  21.9%（+1.509 至 +1.881）换算到 x 幂为 +0.7545 至 +0.9405，相对中心
  +0.853 的半宽约 0.093，故取 0.10。两项容差均宽于实测散布，用于容纳
  协议切换本身的偏移。

模型与跨轮声明
  链参数为位点势深 K、键刚度 k、格距 a；以 a 为长度单位时无量纲组只剩
  x = K/k。本轮只做协议消融，不重开机制归属。读数与第 767、810 号不逐位
  可比，只在本轮内部自洽比较。

可逆性等级：判据稳健（判据量定义逐字继承，协议轴显式枚举）。
"""

import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

IMG = r"D:\\software\\TianXingFC\\images\\deviation_group_collapse_773.png"

N = 1400
cN = 700
dt = 0.01

XSETS = (0.00125, 0.0025, 0.005, 0.01, 0.02)
KKS = (0.25, 1.0, 4.0)

# 相对速度档（beta = v/c_s）
BETA_SETS = {
    "基准五档": (0.20, 0.35, 0.50, 0.65, 0.80),
    "宽七档": (0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95),
    "窄四档": (0.25, 0.40, 0.55, 0.70),
}

NW = 26.65          # 窗口自相似系数，W = NW * ell
GAMMA = 0.05        # 静置阶段阻尼系数

# 探测协议三档：只改窗口半径，速度档集合固定，用于 J6 协议稳定性
NW_PROTOS = (0.90 * NW, NW, 1.10 * NW)

# 静置收敛残差的无量纲门槛（残差除以剖面跨度 N）
CONV_REL_MAX = 1e-6

# 无量纲相对论质量参数 M*c_s^2/E_rest 的可接受区间
MCS2_LO, MCS2_HI = 0.5, 2.0

# 变体敏感度用的速度档（偏离度至少需三点）
SENS_BETAS = (0.35, 0.50, 0.65)

# 窗口协议之间偏离度读数的容许相对极差
NW_SPREAD_MAX = 0.20
REL_REF = 0.02      # 松弛步数的参考 K
NREL_REF = 12000    # 参考点松弛步数
NB0 = 60000         # k=1 的推进步数（行程 600*beta 格）

k = 1.0
K = 0.0
w2 = 0.0
c_s = 1.0
ell = 1.0
Wd = 30


# ---- 第 811 号消融矩阵新增常量 ----
KS_BY_K = {
    1.0: (0.00125, 0.0025, 0.005, 0.01),
    4.0: (0.005, 0.01, 0.02, 0.04),
}
K_MAIN = 4.0
W_FIXED = 30
NREL_FIXED = 12000
DEV_SPREAD_MAX = 0.20
OFFS_MIN = 0.05
DOMINANCE_RATIO = 3.0
BASE_REPRO_TOL = 0.10
P767_REPRO_TOL = 0.10
P767_X_EQ = 0.853
FIT_LO, FIT_HI = 1.0 / 3.0, 2.0 / 3.0
FIT_LO_FAKE, FIT_HI_FAKE = 0.32, 0.68
IMG_811 = r"D:\software\TianXingFC\images\deviation_protocol_ablation_811.png"

PROTOCOLS = {
    "基线(810协议)": dict(window="self", gamma=GAMMA, relax="scaled", speed="rel"),
    "A_固定窗口": dict(window="fixed", gamma=GAMMA, relax="scaled", speed="rel"),
    "B_无阻尼": dict(window="self", gamma=0.0, relax="scaled", speed="rel"),
    "C_固定静置步数": dict(window="self", gamma=GAMMA, relax="fixed", speed="rel"),
    "D_767全协议": dict(window="fixed", gamma=0.0, relax="fixed", speed="abs"),
    "E_绝对速度档": dict(window="self", gamma=GAMMA, relax="scaled", speed="abs"),
}

def set_params(kv, Kv, nw=NW):
    """设参数并按半宽自相似地定窗口半径，nw 为窗口自相似系数。"""
    global k, K, w2, c_s, ell, Wd
    k = float(kv)
    K = float(Kv)
    w2 = 2 * np.pi * K
    c_s = np.sqrt(k)
    ell = 1.0 / (2 * np.pi * np.sqrt(K / k))
    Wd = max(10, int(round(nw * ell)))


def accel(u, v):
    a = np.zeros_like(u)
    a[1:-1] = k * (u[2:] + u[:-2] - 2 * u[1:-1]) - w2 * np.sin(2 * np.pi * u[1:-1])
    return a


def kink_profile(x0):
    n = np.arange(N)
    el = 1.0 / (2 * np.pi * np.sqrt(K / k))
    return n + (2 / np.pi) * np.arctan(np.exp(np.clip((n - x0) / el, -700.0, 700.0)))


def center(u, half=8):
    w = u[1:] - u[:-1] - 1.0
    ks = int(np.argmax(w))
    lo, hi = max(0, ks - half), min(len(w), ks + half + 1)
    ww = np.clip(w[lo:hi], 0.0, None)
    return (np.arange(lo, hi) * ww).sum() / ww.sum() if ww.sum() > 1e-12 else np.nan


def window_energy(u, v, xc, Wd):
    n = np.arange(N)
    ms = np.abs(n - xc) <= Wd
    b = np.arange(N - 1)
    mb = np.abs(b + 0.5 - xc) <= Wd
    E = 0.5 * np.sum(v[ms] ** 2) + np.sum(K * (1 - np.cos(2 * np.pi * u[ms])))
    E += 0.5 * k * np.sum((u[b[mb] + 1] - u[b[mb]] - 1.0) ** 2)
    return E


def relax_kink(nsteps, snap_at=0, gamma=GAMMA):
    """Verlet 静置，返回 (u, v, 快照)。快照取自第 snap_at 步。

    gamma 为每步速度衰减系数，仅作用于静置阶段；推进阶段不加阻尼。
    """
    u = kink_profile(cN * 1.0)
    v = np.zeros(N)
    a = accel(u, v)
    snap = None
    for t in range(nsteps):
        u = u + dt * v + 0.5 * dt ** 2 * a
        a_new = accel(u, v + dt * a)
        v = v + 0.5 * dt * (a + a_new)
        if gamma:
            v = v * (1.0 - gamma * dt)
        u[0], u[-1] = 0.0, float(N)
        v[0] = v[-1] = 0.0
        a = a_new
        if snap_at and t + 1 == snap_at:
            snap = u.copy()
    return u, np.zeros(N), snap


def run_boost(u_rest, v0, steps, Wd, sample=20):
    """无阻尼推进，返回采样时间、质心位置、窗能量。"""
    u = u_rest.copy()
    v = -v0 * np.gradient(u_rest - np.arange(N))
    v[0] = v[-1] = 0.0
    a = accel(u, v)
    ts, xs, Ews = [], [], []
    for t in range(steps):
        u = u + dt * v + 0.5 * dt ** 2 * a
        a_new = accel(u, v + dt * a)
        v = v + 0.5 * dt * (a + a_new)
        u[0], u[-1] = 0.0, float(N)
        v[0] = v[-1] = 0.0
        a = a_new
        if t % sample == 0:
            xc = center(u)
            ts.append(t * dt)
            xs.append(xc)
            Ews.append(window_energy(u, v, xc, Wd))
            if not np.isnan(xc) and xc > N - 100:
                break
    return np.array(ts), np.array(xs), np.array(Ews)


def collect_all(u_rest, E_rest_w, betas):
    """对给定 beta 集合各推进一次，返回以 beta 为键的行表。"""
    steps = max(6000, int(NB0 / np.sqrt(k)))
    out = {}
    for b in betas:
        v0 = b * c_s
        ts, xs, Ews = run_boost(u_rest, v0, steps, Wd)
        n1, n2 = len(ts) // 3, 2 * len(ts) // 3
        ok = ~np.isnan(xs[n1:n2])
        if ok.sum() < 5:
            continue
        v_meas = np.polyfit(ts[n1:n2][ok], xs[n1:n2][ok], 1)[0]
        Ew_avg = np.mean(Ews[n2:]) if n2 < len(Ews) else np.mean(Ews[n1:])
        out[b] = (b, v_meas, Ew_avg - E_rest_w)
    return out


def measure(u_rest, betas, nw):
    """按给定窗口协议与速度档测一行表，返回 (M, dev, v_max, E_rest_w)。

    窗口半径随协议改变，故静止窗口能量须按该窗口重算。
    """
    global Wd
    Wd = max(10, int(round(nw * ell)))
    x0 = center(u_rest)
    E_rest_w = window_energy(u_rest, np.zeros(N), x0, Wd)
    rows = collect_all(u_rest, E_rest_w, betas)
    sel = [rows[b] for b in betas if b in rows]
    if len(sel) < 3:
        return float("nan"), float("nan"), float("nan"), E_rest_w
    arr = np.array(sel)
    M, dev, res = deviation(arr[:, 1], arr[:, 2])
    return M, dev, float(arr[:, 1].max()), E_rest_w


def deviation(v, dE):
    """返回 (M, 偏离度 rms 相对量, 相对论拟合相对残差)。

    口径与第 767、771、772 号逐字一致，便于比较判据量本身。
    """
    g1 = c_s ** 2 * (1.0 / np.sqrt(1.0 - (v / c_s) ** 2) - 1.0)
    M = float(np.sum(g1 * dE) / np.sum(g1 ** 2))
    R = dE - M * g1
    dev = float(np.sqrt(np.mean(R ** 2)) / np.mean(np.abs(dE)))
    res = float(np.max(np.abs(R)) / np.max(np.abs(dE)))
    return M, dev, res


def loglin(x, y):
    """对数线性拟合，返回 (斜率, 截距, 残差平方和, 斜率标准误)。"""
    lx, ly = np.log(x), np.log(y)
    slope, intercept = np.polyfit(lx, ly, 1)
    pred = slope * lx + intercept
    rss = float(np.sum((ly - pred) ** 2))
    n = len(lx)
    if n > 2:
        s2 = rss / (n - 2)
        sxx = float(np.sum((lx - lx.mean()) ** 2))
        se = float(np.sqrt(s2 / sxx)) if sxx > 0 else float("inf")
    else:
        se = float("inf")
    return float(slope), float(intercept), rss, se


def Mvals_ratio(M, E):
    return M / E if E > 0 else float("inf")


def relax_steps():
    """静置步数：按格点能隙 2*pi*sqrt(K) 的时间尺度 sqrt(1/K) 缩放。"""
    return max(3000, int(NREL_REF * np.sqrt(REL_REF / K)))


def protocol_devs(u_rest, E_rest_w, rows_by_beta, betas):
    """按一组速度档算偏离度，返回 (M, dev, v_max)。"""
    sel = [rows_by_beta[b] for b in betas if b in rows_by_beta]
    if len(sel) < 3:
        return float("nan"), float("nan"), float("nan")
    arr = np.array(sel)
    M, dev, res = deviation(arr[:, 1], arr[:, 2])
    return M, dev, float(arr[:, 1].max())


def dev_with_window(raws, fit_lo, fit_hi):
    """按给定拟合区间由原始采样序列算 (dev, M, res, v_max)。

    须先保证全局 c_s 与该档一致（deviation 依赖它）。
    """
    rows = []
    for b, ts, xs, Ews, E_rest_w in raws:
        n1, n2 = int(len(ts) * fit_lo), int(len(ts) * fit_hi)
        ok = ~np.isnan(xs[n1:n2])
        if ok.sum() < 5:
            continue
        v_meas = np.polyfit(ts[n1:n2][ok], xs[n1:n2][ok], 1)[0]
        Ew_avg = np.mean(Ews[n2:]) if n2 < len(Ews) else np.mean(Ews[n1:])
        rows.append((b, v_meas, Ew_avg - E_rest_w))
    if len(rows) < 3:
        return float("nan"), float("nan"), float("nan"), float("nan")
    arr = np.array(rows)
    M, dev, res = deviation(arr[:, 1], arr[:, 2])
    return dev, M, res, float(arr[:, 1].max())


def run_protocol(proto, Kv, kval):
    """按给定协议在单档参数上测量，返回读数字典（含各速度档原始采样序列）。"""
    global Wd
    set_params(kval, Kv)
    if proto["window"] == "fixed":
        Wd = W_FIXED
    else:
        Wd = max(10, int(round(NW * ell)))
    if proto["relax"] == "scaled":
        nrel = relax_steps()
        u_rest, v_rest, snap = relax_kink(2 * nrel, snap_at=nrel,
                                         gamma=proto["gamma"])
        conv = (float(np.max(np.abs(u_rest - snap)))
                if snap is not None else float("nan"))
    else:
        nrel = NREL_FIXED
        u_rest, v_rest, snap = relax_kink(nrel, snap_at=0,
                                         gamma=proto["gamma"])
        conv = float("nan")
    x0 = center(u_rest)
    E_rest_w = window_energy(u_rest, np.zeros(N), x0, Wd)
    betas = BETA_SETS["基准五档"]
    steps = max(6000, int(NB0 / np.sqrt(k)))
    raws = []
    for b in betas:
        v0 = b * c_s if proto["speed"] == "rel" else b
        ts, xs, Ews = run_boost(u_rest, v0, steps, Wd)
        raws.append((b, ts, xs, Ews, E_rest_w))
    dev, M, res, vmax = dev_with_window(raws, FIT_LO, FIT_HI)
    if dev != dev:
        return None
    return dict(dev=dev, M=M, res=res, vmax=vmax,
                conv=conv, W=Wd, nrel=nrel, E_rest=E_rest_w, raws=raws)


def fit_slope(devs, kval=1.0):
    """对 x = K/k 拟合幂次，返回 (斜率, 标准误, 点数)。"""
    ks = sorted(devs.keys())
    xs = np.array(ks) / kval
    ys = np.array([devs[k] for k in ks])
    sl, ic, rss, se = loglin(xs, ys)
    return sl, se, len(xs)


def main():
    print("== 第 811 号：偏离度幂次的协议消融 ==")
    results = {}
    for kval, klist in KS_BY_K.items():
        res_k = {}
        for pname, proto in PROTOCOLS.items():
            devs = {}
            convs = []
            raws_by_K = {}
            for Kv in klist:
                r = run_protocol(proto, Kv, kval)
                if r is None:
                    print(f"  [警告] 协议 {pname} k={kval} K={Kv} 速度档不足")
                    continue
                devs[Kv] = r["dev"]
                convs.append(r["conv"])
                raws_by_K[Kv] = r["raws"]
            if len(devs) < 3:
                print(f"  [警告] 协议 {pname} k={kval} 可用档不足")
                continue
            sl, se, npts = fit_slope(devs, kval)
            res_k[pname] = dict(devs=devs, slope=sl, se=se, convs=convs,
                                proto=proto, npts=npts, raws_by_K=raws_by_K)
            print(f"\n[协议] k={kval} {pname}  {proto}")
            for Kv in sorted(devs):
                print(f"    K={Kv:.5f}  x={Kv / kval:.6f}"
                      f"  dev={devs[Kv] * 100:.4f}%")
            print(f"    幂次 dev ∝ x^{sl:+.4f}"
                  f"（标准误 {se:.4f}，{npts} 点；"
                  f"折合对 1/ell 的幂 {2 * sl:+.4f}）")
            cs = [c for c in convs if c == c]
            if cs:
                print(f"    静置收敛残差 最大 {max(cs):.3e}")
        results[kval] = res_k

    base_name = "基线(810协议)"
    kmain = K_MAIN
    if kmain not in results or base_name not in results.get(kmain, {}):
        print("主消融档读数不足，终止")
        return
    base_slope = results[kmain][base_name]["slope"]
    print(f"\n[基线] k={kmain} {base_name} 幂次 {base_slope:+.4f}"
          f"（折合对 1/ell 的幂 {2 * base_slope:+.4f}）")

    # ---- 替身协议：只微调拟合区间，物理协议不动，复用基线的原始采样序列 ----
    fake_devs = {}
    for Kv in KS_BY_K[kmain]:
        raws = results[kmain][base_name]["raws_by_K"].get(Kv)
        if raws is None:
            continue
        set_params(kmain, Kv)
        d, M, res, vmax = dev_with_window(raws, FIT_LO_FAKE, FIT_HI_FAKE)
        if d == d:
            fake_devs[Kv] = d
    if len(fake_devs) >= 3:
        fsl, fse, fn = fit_slope(fake_devs, kmain)
    else:
        fsl, fse, fn = float("nan"), float("nan"), 0
    fake_off = abs(fsl - base_slope)
    print(f"\n[替身协议] 拟合区间微调 {FIT_LO_FAKE:.2f}–{FIT_HI_FAKE:.2f}"
          f"（基线 {FIT_LO:.4f}–{FIT_HI:.4f}）")
    print(f"  幂次 {fsl:+.4f}，相对基线偏移 {fake_off:+.4f}")

    # ---- 偏移与主导元素（主消融档） ----
    offsets = {}
    for pname, res in results[kmain].items():
        offsets[pname] = res["slope"] - base_slope
    print("\n[各协议幂次偏移，相对基线]")
    for pname in offsets:
        print(f"  {pname:16s} 幂次 {results[kmain][pname]['slope']:+.4f}"
              f"  偏移 {offsets[pname]:+.4f}")

    single = ["A_固定窗口", "B_无阻尼", "C_固定静置步数", "E_绝对速度档"]
    single_off = {p: abs(offsets[p]) for p in single if p in offsets}
    dom_pname, dom_val, dom_ratio = None, 0.0, 0.0
    if len(single_off) == len(single):
        dom_pname = max(single_off, key=single_off.get)
        dom_val = single_off[dom_pname]
        others = sorted(v for kk, v in single_off.items() if kk != dom_pname)
        med_other = others[len(others) // 2]
        dom_ratio = dom_val / med_other if med_other > 0 else float("inf")
    print(f"\n[单元素切换的幂次偏移] {single_off}")
    print(f"  最大者 {dom_pname} = {dom_val:.4f}，与其余中位之比 {dom_ratio:.2f}")

    # ---- 描述读数：D 档四元素合成的偏移与四个单元素偏移之和作可加性对照 ----
    if "D_767全协议" in offsets and len(single_off) == len(single):
        s_sum = sum(offsets[p] for p in single)
        print(f"\n[可加性对照] D 档四元素合成偏移 {offsets['D_767全协议']:+.4f}，"
              f"四个单元素偏移之和 {s_sum:+.4f}，"
              f"差 {offsets['D_767全协议'] - s_sum:+.4f}")

    # ---- 描述读数：dev 的跨协议散布 ----
    print("\n[描述读数] 同档跨协议 dev 相对极差")
    spreads = []
    for Kv in KS_BY_K[kmain]:
        vals = [results[kmain][p]["devs"][Kv] for p in results[kmain]
                if Kv in results[kmain][p]["devs"]]
        if len(vals) >= 2:
            sp = (max(vals) - min(vals)) / np.median(vals)
            spreads.append(sp)
            print(f"  k={kmain} K={Kv:.5f}: {sp * 100:.3f}%")

    # ---- 判据 ----
    print("\n[判据]")
    p1_base = abs(base_slope - 0.4813) <= BASE_REPRO_TOL
    print(f"  P1 基线复现：k={kmain} 基线幂次 {base_slope:+.4f} 对第 810 号 "
          f"k=4 读数 +0.4813，差 {abs(base_slope - 0.4813):.4f}"
          f"（门槛 {BASE_REPRO_TOL}）：{'PASS' if p1_base else 'FAIL'}")

    d1 = results.get(1.0, {}).get("D_767全协议")
    if d1 is None:
        p1_767 = False
        print("  P1-b 第 767 号复现：k=1 档 D 协议读数缺失")
    else:
        p1_767 = abs(d1["slope"] - P767_X_EQ) <= P767_REPRO_TOL
        print(f"  P1-b 第 767 号复现：k=1 D 档对 x 的幂次 {d1['slope']:+.4f}，"
              f"折合对 1/ell 的幂 {2 * d1['slope']:+.4f}；"
              f"第 767 号读数 +1.705 换算到 x 为 {P767_X_EQ:+.3f}，"
              f"差 {abs(d1['slope'] - P767_X_EQ):.4f}"
              f"（门槛 {P767_REPRO_TOL}）：{'PASS' if p1_767 else 'FAIL'}")

    p2 = (dom_val >= OFFS_MIN and dom_ratio >= DOMINANCE_RATIO)
    print(f"  P2 主导元素定位：最大偏移 {dom_val:.4f}（门槛 {OFFS_MIN}），"
          f"区分力比 {dom_ratio:.2f}（门槛 {DOMINANCE_RATIO}）："
          f"{'PASS' if p2 else 'FAIL'}（PASS 只表示存在唯一主导元素，"
          f"本轮读数指向 {dom_pname}）")

    p3 = fake_off <= dom_val / DOMINANCE_RATIO if dom_val > 0 else False
    print(f"  P3 恒真替身：替身偏移 {fake_off:.4f}，真实主导偏移 "
          f"{dom_val:.4f} 的三分之一为 {dom_val / DOMINANCE_RATIO:.4f}："
          f"{'PASS' if p3 else 'FAIL'}")

    npass = sum([p1_base, p2, p3])
    print(f"\n总判据 {npass}/3（P1-b 单列，不计入总分）")

    # ---- 出图 ----
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for pname, res in results[kmain].items():
        ks = sorted(res["devs"].keys())
        xs = np.array(ks) / kmain
        ys = np.array([res["devs"][kk] for kk in ks]) * 100
        ax[0].plot(xs, ys, marker="o", label=pname)
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("x = K/k")
    ax[0].set_ylabel("dev (%)")
    ax[0].set_title(f"(a) 各协议下的偏离度（k = {kmain}）")
    ax[0].legend(fontsize=8)

    names = list(results[kmain].keys())
    sls = [results[kmain][n]["slope"] for n in names]
    ax[1].barh(range(len(names)), sls, color="steelblue")
    ax[1].axvline(P767_X_EQ, color="firebrick", ls="--",
                  label="767 号 +1.705 折合 x 幂 +0.853")
    ax[1].axvline(0.4813, color="darkgreen", ls="--",
                  label="810 号 k=4 +0.4813")
    ax[1].set_yticks(range(len(names)))
    ax[1].set_yticklabels(names, fontsize=8)
    ax[1].set_xlabel("幂次（对 x = K/k）")
    ax[1].set_title("(b) 各协议给出的幂次")
    ax[1].legend(fontsize=8)

    off_names = list(offsets.keys())
    off_vals = [offsets[n] for n in off_names]
    ax[2].barh(range(len(off_names)), off_vals, color="indianred")
    ax[2].set_yticks(range(len(off_names)))
    ax[2].set_yticklabels(off_names, fontsize=8)
    ax[2].axvline(0, color="black", lw=0.8)
    ax[2].set_xlabel("幂次偏移（相对基线）")
    ax[2].set_title("(c) 协议元素各自的偏移")
    fig.tight_layout()
    fig.savefig(IMG_811, dpi=150)
    print(f"\n图已存 {IMG_811}")


if __name__ == "__main__":
    main()
