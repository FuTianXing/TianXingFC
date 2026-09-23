# -*- coding: utf-8 -*-
r"""弹性通道的代价归属：M2 缓增的轨迹分解与时间反演可逆性（判据 S1 至 S5）

沿用 M2 的模型与协议定义（FK 链 k=1、K=0.005、N=1000、dt=0.01、kink 半宽 ell、
环境区半径 10、甲-1 零模踢、甲-2 宽脉冲、甲-3 加阻尼）。本脚本只做 M2 未做的两件事：
  (1) 输出 D_env(t) 轨迹并检验其与遍历距离的关系；
  (2) 时间反演协议（v -> -v, a -> -a 后原算法回推同样步数），检验弹性通道代价是否可逆。
velocity-Verlet 无阻尼时严格可逆，故反演残差给出不可逆成分的量级。
确定性：无随机源。图存 D:/software/TianXingFC/images/elastic_reversibility.png
"""
import numpy as np

RES = []
FIG = "D:/software/TianXingFC/images/elastic_reversibility.png"
k, K, dt, N = 1.0, 0.005, 0.01, 1000
cN = N // 2
w2 = 2 * np.pi * K
ELL = 1.0 / (2 * np.pi * np.sqrt(K / k))
R_ENV, GAMMA_C = 10, 0.5
HALF = int(100.0 / dt)
FULL = int(200.0 / dt)


def report(tag, name, meas, exp, note="", grade="通过"):
    RES.append((tag, name, meas, exp, note, grade))


def accel(u, v, gamma=0.0):
    a = np.zeros_like(u)
    a[1:-1] = k * (u[2:] + u[:-2] - 2 * u[1:-1]) - w2 * np.sin(2 * np.pi * u[1:-1])
    if gamma > 0.0:
        a -= gamma * v
    return a


def kink_profile(x0):
    n = np.arange(N)
    return n + (2 / np.pi) * np.arctan(np.exp((n - x0) / ELL))


def zero_mode(x0):
    n = np.arange(N)
    f = 1.0 / np.cosh((n - x0) / ELL)
    return f / np.sqrt((f ** 2).sum())


def center(u):
    w = u[1:] - u[:-1] - 1.0
    ks = int(np.argmax(w))
    lo, hi = max(0, ks - 8), min(len(w), ks + 9)
    ww = np.clip(w[lo:hi], 0.0, None)
    return (np.arange(lo, hi) * ww).sum() / ww.sum() if ww.sum() > 1e-12 else np.nan


def relax(u0, steps=8000):
    u = u0.copy(); v = np.zeros(N)
    a = accel(u, v)
    for _ in range(steps):
        u = u + dt * v + 0.5 * dt ** 2 * a
        a_new = accel(u, v + dt * a)
        v = v + 0.5 * dt * (a + a_new)
        u[0], u[-1] = 0.0, float(N)
        v[0] = v[-1] = 0.0
        a = a_new
    return u, np.zeros(N)


_CACHE = {}


def rest_profiles():
    if "rest" not in _CACHE:
        _CACHE["rest"] = (relax(kink_profile(float(cN)))[0], relax(kink_profile(float(cN + 1)))[0])
    return _CACHE["rest"]


def initial_state(kind):
    if ("init", kind) not in _CACHE:
        uA, vA = relax(kink_profile(float(cN)))
        uB, vB = relax(kink_profile(float(cN + 1)))
        if kind in ("A1", "A3"):
            vA = vA + 0.01 * zero_mode(float(cN))
            vB = vB + 0.01 * zero_mode(float(cN + 1))
        elif kind == "A2":
            j = np.arange(N)
            vp = 0.03 * np.exp(-((j - (cN + 0.5)) ** 2) / (2 * 25.0 ** 2))
            vA = vA + vp
            vB = vB + vp
        _CACHE[("init", kind)] = (uA, vA, uB, vB)
    uA, vA, uB, vB = _CACHE[("init", kind)]
    return uA.copy(), vA.copy(), uB.copy(), vB.copy()


def d_env(uA, vA, uB, vB, restA, restB):
    xc = 0.5 * (center(uA) + center(uB))
    j = np.arange(N)
    env = np.abs(j - xc) > R_ENV
    dq = (uA - restA)[env] - (uB - restB)[env]
    dp = vA[env] - vB[env]
    return 0.5 * ((dq ** 2).sum() + (dp ** 2).sum())


def run(kind, steps, gamma=0.0, reverse=False, sample=10):
    restA, restB = rest_profiles()
    uA, vA, uB, vB = initial_state(kind)
    aA, aB = accel(uA, vA, gamma), accel(uB, vB, gamma)
    ts, Ds, xs = [], [], []

    def leg(n):
        nonlocal uA, vA, uB, vB, aA, aB
        for t in range(n):
            uA = uA + dt * vA + 0.5 * dt ** 2 * aA
            uB = uB + dt * vB + 0.5 * dt ** 2 * aB
            aAn = accel(uA, vA + dt * aA, gamma)
            aBn = accel(uB, vB + dt * aB, gamma)
            vA = vA + 0.5 * dt * (aA + aAn)
            vB = vB + 0.5 * dt * (aB + aBn)
            uA[0], uA[-1] = 0.0, float(N)
            uB[0], uB[-1] = 0.0, float(N)
            vA[0] = vA[-1] = 0.0
            vB[0] = vB[-1] = 0.0
            aA, aB = aAn, aBn
            if t % sample == 0:
                ts.append(len(ts) * dt * sample)
                Ds.append(d_env(uA, vA, uB, vB, restA, restB))
                xs.append(center(uA))

    leg(steps)
    if reverse:
        vA = -vA; vB = -vB; aA = -aA; aB = -aB
        leg(steps)
    return dict(ts=np.array(ts), D=np.array(Ds), xs=np.array(xs),
                Dend=d_env(uA, vA, uB, vB, restA, restB))


def main():
    print("=" * 100)
    print("弹性通道的代价归属：轨迹分解与时间反演可逆性（S1 至 S5）")
    print("=" * 100)
    ref = {"A1": 0.0025, "A2": 0.0433, "A3": 1.5e-6}
    traj = {}
    for kind in ("A1", "A2", "A3"):
        traj[kind] = run(kind, FULL, gamma=(GAMMA_C if kind == "A3" else 0.0))
        print("协议 %-3s：D_env 末值 %.6e（M2 发表值 %.1e，相对偏差 %.2f%%）"
              % (kind, traj[kind]["D"][-1], ref[kind], 100 * abs(traj[kind]["D"][-1] - ref[kind]) / ref[kind]))
    dev = max(abs(traj[q]["D"][-1] - ref[q]) / ref[q] for q in ref)
    report("S1", "三协议 D_env 末值对 M2 发表值的最大相对偏差", dev, 0.05, "复现口径校验",
           "通过" if dev < 0.05 else "未过")

    print()
    print("轨迹与遍历距离")
    print("-" * 100)
    slopes = {}
    for kind in ("A1", "A2"):
        r = traj[kind]
        trav = np.abs(r["xs"] - r["xs"][0])
        slopes[kind] = float(np.dot(trav, r["D"]) / np.dot(trav, trav))
        mono = float(np.mean(np.diff(r["D"]) > -1e-14))
        print("%-3s：末遍历 %.2f 格距，D_env 末 %.4e，过原点斜率 %.4e，单调步占比 %.3f"
              % (kind, trav[-1], r["D"][-1], slopes[kind], mono))
        report("S2", "%s 每格距代价（轨迹斜率）" % kind, slopes[kind], "-",
               "端点比值 %.2e" % (r["D"][-1] / trav[-1]), "读数")
        report("S2", "%s D_env 单调步占比" % kind, mono, "-", "低于 1 则含一次性成分", "读数")

    print()
    print("时间反演协议（前向 100 段 + v->-v、a->-a 回推同长；GAMMA=%s）" % GAMMA_C)
    print("-" * 100)
    rev = {}
    for kind, gam, lab in (("A2", 0.0, "弹性 A2"), ("A1", 0.0, "弹性 A1"), ("A2", GAMMA_C, "加阻尼反证")):
        r = run(kind, HALF, gamma=gam, reverse=True)
        rev[lab] = r
        h = len(r["D"]) // 2
        print("%-10s：D 起步 %.3e，半程 %.4e，反演后 %.4e，V 终 %.10f"
              % (lab, r["D"][0], r["D"][h - 1], r["Dend"], np.exp(-r["Dend"] / 2)))
    def ratio(lab):
        r = rev[lab]; h = len(r["D"]) // 2
        return float(r["Dend"] / max(r["D"][h - 1], 1e-300))

    r_el, r_dm = ratio("弹性 A2"), ratio("加阻尼反证")
    v_ir = 1.0 - float(np.exp(-rev["弹性 A2"]["Dend"] / 2))
    report("S3", "弹性 A2 反演恢复比（残差/半程）", r_el, 1e-5, "可逆通道应回到初值量级",
           "通过" if r_el < 1e-5 else "未过")
    report("S3", "弹性 A2 反演后 V 的不可恢复部分", v_ir, 1e-8, "相干恢复到 7 位有效数字",
           "通过" if v_ir < 1e-8 else "未过")
    report("S4", "加阻尼通道反演恢复比（反证）", r_dm, "-", "阻尼破坏可逆性", "读数")
    report("S5", "两通道恢复比之比（反证/弹性）", r_dm / max(r_el, 1e-300), "-",
           "比值远大于 1 说明判别有牙齿", "读数")

    print()
    print("=" * 100)
    print("判据 | 判据量 | 实测 | 期望 | 备注 | 结论")
    print("=" * 100)
    for tag, name, meas, exp, note, grade in RES:
        m = ("%.6g" % meas) if isinstance(meas, float) else str(meas)
        e = ("%.6g" % exp) if isinstance(exp, float) else str(exp)
        print("%-5s | %s | %s | %s | %s | %s" % (tag, name, m, e, note, grade))
    print("=" * 100)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(2, 2, figsize=(11.5, 8.0))

    for kind, c in (("A1", "#4c72b0"), ("A2", "#dd8452"), ("A3", "#55a868")):
        ax[0, 0].semilogy(traj[kind]["ts"], np.clip(traj[kind]["D"], 1e-30, None), "-", color=c, label=kind)
    ax[0, 0].set_xlabel("时间（模型单位）"); ax[0, 0].set_ylabel("D_env")
    ax[0, 0].set_title("S1：三协议 D_env(t) 轨迹"); ax[0, 0].legend(fontsize=8)

    for kind, c in (("A1", "#4c72b0"), ("A2", "#dd8452")):
        trav = np.abs(traj[kind]["xs"] - traj[kind]["xs"][0])
        ax[0, 1].loglog(np.clip(trav, 1e-3, None), np.clip(traj[kind]["D"], 1e-30, None), ".", ms=2, color=c, label=kind)
        tt = np.linspace(1e-3, trav.max(), 50)
        ax[0, 1].loglog(tt, slopes[kind] * tt, "--", color=c)
    ax[0, 1].set_xlabel("累计遍历格距"); ax[0, 1].set_ylabel("D_env")
    ax[0, 1].set_title("S2：代价对遍历距离（虚线为过原点斜率）"); ax[0, 1].legend(fontsize=8)

    for lab, c in (("弹性 A2", "#4c72b0"), ("加阻尼反证", "#c44e52")):
        r = rev[lab]
        n = len(r["D"])
        ax[1, 0].semilogy(range(n), np.clip(r["D"], 1e-30, None), "-", color=c, label=lab)
        ax[1, 0].axvline(n // 2, color="gray", ls=":", lw=1.0)
    ax[1, 0].set_xlabel("采样步（虚线处做时间反演）"); ax[1, 0].set_ylabel("D_env")
    ax[1, 0].set_title("S3/S4：时间反演可逆性"); ax[1, 0].legend(fontsize=8)

    labs = ["弹性 A1", "弹性 A2", "加阻尼反证"]

    def rat(lab):
        r = rev[lab]; h = len(r["D"]) // 2
        return float(r["Dend"] / max(r["D"][h - 1], 1e-300))

    vals = [max(rat(q), 1e-30) for q in labs]
    ax[1, 1].bar(labs, vals, color=["#4c72b0", "#4c72b0", "#c44e52"])
    ax[1, 1].set_yscale("log"); ax[1, 1].axhline(1e-5, color="gray", ls="--", lw=1.0, label="通过阈值")
    ax[1, 1].set_ylabel("反演恢复比（残差/半程）")
    ax[1, 1].set_title("S5：恢复比对照（反证有牙齿）"); ax[1, 1].legend(fontsize=8)

    fig.tight_layout(); fig.savefig(FIG, dpi=140)
    print("图已保存：", FIG)


if __name__ == "__main__":
    main()
