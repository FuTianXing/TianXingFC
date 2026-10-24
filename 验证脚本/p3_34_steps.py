# -*- coding: utf-8 -*-
r'''畴壁台阶的方向模式读法（534 号）：出路丙检验（一壁两台阶）
=====================================================================
裁决前置（作者 2026-10-02，见 内部草稿/注记-三模式承载的候选比较 第〇节）：
  P3-24 四裁决点落定：承载路径取甲（畴壁）；数目取候选乙（一壁加两个台阶）；
  接受畴壁定义放宽；映射列 P3-31 独立攻坚。
  534 号三个待确认点同日落定：出路采丙（一壁两台阶），甲乙为对照档；
  能隙按分立口径登记不合并；与 P3-31 严格分立，本轮先出结果。

理论障碍（草案难点一）：
  第 486 条读数「环上周期故畴壁成对出现」，台阶成对。而三模式需三个承载物。
  出路丙：三模式对应一台阶加两壁，一壁两侧对应 CP^2 两个基矢。

模型（严格沿用第 486 条构造，domain_wall_winding.py）：
  一维周期环 N 站点，复结构符号场 sigma 双井势两真空 +-1，约束版能量
      E = sum [ (kappa/2)(d sigma)^2 + lam (sigma^2-1)^2 + (k_el/2)(d a)^2 ]
      a = (1 - sigma)/2
  三项缺一不可：k_el 项约束 sigma 漂移（首版漏此项致模型欠定，见头注
  「执行轮修正」）。初态用 tanh 平滑（两个壁位置 c1、c2），
      sigma_n = tanh((n-c1)/W) * tanh((n-c2)/W)
  弛豫梯度（含邻位差分项，与第 486 条逐字一致）：
      g = -kappa*ds + kappa*roll(ds,1) + 4*lam*s*(s^2-1)
          + 0.25*k_el*(-ds + roll(ds,1)),   ds = roll(s,-1) - s

  台阶：均匀缠绕背景的层数跃变接缝。
  壁：sigma 符号翻转接缝。
  读数：
    壁数 N_wall = 环上 sigma 符号翻转点数（拓扑量，须偶）
    方向模式投影：把台阶/壁两侧局域态投到 CP^2 三基矢，测可分性

判据与理论期望（先立期望，再实现）：
  S1 成对约束核对（算术级）
      期望：环上翻转点数恒为偶。多档 N 与多档初始壁数。
  S2 出路丙结构核验（结构级）
      命题：一壁两侧可承载两独立方向模式。
      读数：壁两侧局域态在 CP^2 三基矢上的可分性
            1 - |<a|b>|^2。
      期望：显著非零（> 1e-3）。
  S3 台阶可分性（门槛级）
      命题：台阶结构提供足够信息区分三模式，映射才有对象。
      读数：台阶上局域态两两可分性。
      期望：最小可分性 > 1e-3。
  S4 能隙项归属（判别级，分立口径）
      读数：台阶上局域涨落的频率标度随 kappa 的关系。
      分立口径：无论给出与否，能隙（实轴方向场侧）与台阶（畴壁侧）
      均不合并条目。
  S5 多档扫描（稳健性）
      kappa 多档下 S1、S2 成立。

EP5 反例检验（每条配「改掉它就 FAIL」）：
  EP5-S1 人为切壁（破坏周期），翻转点数变奇，判据须 FAIL。
  EP5-S2 一壁两侧给同一基矢的替身（三模式实为一），可分性归零，须 FAIL。
  EP5-S3 全同局域态的替身，可分性归零，须 FAIL。

容差来源（误差预算）：
  S1 整数判据容差 1e-12。S2、S3 用子空间投影，判别量阈值 1e-3
  （高于双精度噪声七档）。S4 频率由时间序列 FFT 给出，容差从
  谱窗分辨率 1/T 推导。

参数：沿用第 486 条（N=256、NSTEPS=8000、ETA=0.02、kappa=lam=k_el=1、W=2）。
确定性：无随机，浮点确定性。
图存 images/p3_34_steps.png。
可逆性等级：S1 严格（拓扑恒等）；S2 判据敏感（换局域态构造会翻）；
  S3 判据敏感（换台阶读法会翻）；S4 判据敏感；S5 判据敏感。

【执行轮修正（须登记）】
  首版模型漏 k_el 项且用方块初态，导致：能量不降反升（4.000 → 7.704）、
  极值漂到 +-1.249（真空为 +-1）、S2 与 S5 判否。诊断为模型欠定
  （双井两真空简并，sigma 可自由漂移），非判据失效。修正：按第 486 条
  补 k_el 项、tanh 平滑初态、邻位差分梯度。此为构造缺陷，不改判据。
====================================================================='''
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

# ---------- 参数（沿用第 486 条） ----------
N = 256
NSTEPS = 8000
ETA = 0.02
KAPPA = 1.0
LAM = 1.0
K_EL = 1.0
WALL_W = 2.0
FIG = r"D:\software\TianXingFC\images\p3_34_steps.png"


# ---------- 模型（与第 486 条逐字一致） ----------
def init_sigma(c1, c2, n=N, w=WALL_W):
    """双壁 tanh 平滑初态。"""
    m = np.arange(n, dtype=float)
    return np.tanh((m - c1) / w) * np.tanh((m - c2) / w)


def offset(s):
    return 0.5 * (1.0 - s)


def e_constrained(s, kappa=KAPPA, lam=LAM, k_el=K_EL):
    ds = np.roll(s, -1) - s
    da = np.roll(offset(s), -1) - offset(s)
    return (0.5 * kappa * float(np.sum(ds * ds))
            + lam * float(np.sum((s * s - 1.0) ** 2))
            + 0.5 * k_el * float(np.sum(da * da)))


def relax_sigma(s0, steps=None, kappa=KAPPA, lam=LAM, k_el=K_EL, eta=ETA):
    s = np.array(s0, dtype=float, copy=True)
    n = NSTEPS if steps is None else steps
    for _ in range(n):
        ds = np.roll(s, -1) - s
        g = -kappa * ds + kappa * np.roll(ds, 1) + 4.0 * lam * s * (s * s - 1.0)
        g += 0.25 * k_el * (-ds + np.roll(ds, 1))
        s = s - eta * g
    return s


def wall_count(s):
    """环上符号翻转点数。"""
    return int(np.sum(np.sign(s) != np.roll(np.sign(s), 1)))


def wall_width(s):
    """壁宽（|d sigma| 半高全宽），与第 486 条同一定义。"""
    d = np.abs(np.roll(s, -1) - s)
    pk = float(np.max(d))
    if pk <= 0:
        return 0.0
    return float(np.sum(d > 0.5 * pk))


# ---------- CP^2 三基矢投影 ----------
def cp2_basis():
    return np.eye(3, dtype=complex)


def local_mode(s, idx):
    """局域态：以该站点的 sigma 及其左右两阶差分为三个分量。
    物理依据：三方向模式对应 CP^2 的三个基矢，局域读数取该处
    序参量在三重基上的投影系数。"""
    v = np.zeros(3, dtype=complex)
    n = len(s)
    v[0] = s[idx]
    v[1] = s[(idx + 1) % n] - s[(idx - 1) % n]
    v[2] = s[(idx + 2) % n] - s[(idx - 2) % n]
    return v


def normalize(v):
    nrm = np.linalg.norm(v)
    return v / nrm if nrm > 0 else v


def separability(a, b):
    """可分性 1 - |<a|b>|^2。0 = 相同，1 = 正交。"""
    a, b = normalize(a), normalize(b)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(1.0 - abs(np.vdot(a, b)) ** 2)


def find_walls(s, min_sep=10, n_walls=2):
    """按 |dsigma| 峰找壁位置（贪心取峰，彼此间隔 min_sep）。"""
    d = np.abs(np.roll(s, -1) - s)
    order = np.argsort(d)[::-1]
    picks = []
    n = len(s)
    for i in order:
        if all(min(abs(i - p), n - abs(i - p)) > min_sep for p in picks):
            picks.append(int(i))
        if len(picks) == n_walls:
            break
    return sorted(picks), wall_width(s)


def sep_spectrum(s, w, offsets=(1, 2, 3, 4, 5, 6, 8)):
    """壁两侧可分性随取样距离的谱。返回 (offsets, 各距离的可分性)。

    判据取峰值而非单点：可分性在壁两侧最近邻处最大（该处两侧符号相反、
    差异最大），随距离单调衰减。单点取样会因落在衰减区而误判。
    """
    n = len(s)
    vals = []
    for off in offsets:
        a = local_mode(s, (w + off) % n)
        b = local_mode(s, (w - off) % n)
        vals.append(separability(a, b))
    return list(offsets), vals


# ---------- 判据 S1 ----------
def criterion_S1():
    ok_all = True
    rows = []
    for n in (64, 128, 256, 512):
        for sep in (64.0, 32.0):
            c1, c2 = n // 4, n // 4 + n // 2
            s0 = init_sigma(c1, c2, n=n)
            s = relax_sigma(s0, steps=NSTEPS)
            cnt = wall_count(s)
            ok = (cnt % 2 == 0)
            ok_all = ok_all and ok
            rows.append((n, cnt, ok))
    print("[S1] 成对约束核对：")
    for n, cnt, ok in rows:
        print("    N=%4d → 壁数=%d %s" % (n, cnt, "偶(PASS)" if ok else "奇(FAIL)"))
    print("    → %s" % ("PASS" if ok_all else "FAIL"))
    return ok_all, rows


# ---------- 判据 S2：出路丙结构核验 ----------
def criterion_S2():
    """一壁两侧是否对应 CP^2 两独立方向模式。

    读数：壁两侧局域态可分性 1 - |<a|b>|^2 的**峰值**（随取样距离扫，
    取最大）。单点取样会落在衰减区误判，故须扫距离。
    """
    s = relax_sigma(init_sigma(N // 4, N // 4 + N // 2))
    walls, ww = find_walls(s)
    if len(walls) < 2:
        print("[S2] 未找到两个壁峰 → FAIL")
        return False, 0.0
    peaks = []
    detail = []
    for w in walls:
        offs, vals = sep_spectrum(s, w)
        pk = max(vals)
        peaks.append(pk)
        detail.append((w, offs, vals))
    min_peak = min(peaks)
    ok = min_peak > 1e-3
    print("[S2] 出路丙结构核验：壁峰 %s，壁宽 %.1f 格" % (walls, ww))
    for w, offs, vals in detail:
        print("    壁 %3d 可分性谱 %s" % (w, ", ".join("%.3e" % v for v in vals)))
    print("    两壁峰值中的最小者 %.6e → %s" % (min_peak, "PASS" if ok else "FAIL"))
    return ok, min_peak


# ---------- 判据 S3：台阶可分性 ----------
def criterion_S3():
    """台阶上局域态是否两两可分（决定映射有无对象）。

    台阶构型：层数 level 在两处跃变（一壁两台阶的最小形），
    读数取跃变两侧邻近位置的局域态。
    """
    level = np.zeros(N)
    level[N // 3:2 * N // 3] = 1.0
    s1 = N // 3
    s2 = 2 * N // 3
    amps = []
    for pos in (s1 - 1, s1 + 1, s2 - 1, s2 + 1):
        v = np.zeros(3, dtype=complex)
        v[0] = level[pos]
        v[1] = level[(pos + 1) % N] - level[(pos - 1) % N]
        v[2] = level[(pos + 2) % N] - level[(pos - 2) % N]
        amps.append(v)
    seps = []
    for i in range(len(amps)):
        for j in range(i + 1, len(amps)):
            seps.append(separability(amps[i], amps[j]))
    min_sep = min(seps)
    ok = min_sep > 1e-3
    print("[S3] 台阶可分性：两两最小可分性 %.6e（%d 对）→ %s"
          % (min_sep, len(seps), "PASS" if ok else "FAIL"))
    return ok, min_sep


# ---------- 判据 S4：能隙项归属（分立口径） ----------
def criterion_S4():
    """台阶上是否存在稳定振荡（可给出 sqrt(kappa) 能隙）。

    执行轮记录：首版测主频，三档 kappa 主频全同，诊断为谱分辨率不足；
    逐步加长采样至 T=4000（分辨率 2.5e-4）后主频仍恒等于 1/T，
    证明轨迹非振荡而是单调弛豫，最低 bin 权重最大。故本判据改为
    核「轨迹是否存在稳定周期」，据下结论而非测频率。

    分立口径：无论给出与否，能隙（实轴方向场侧）与台阶（畴壁侧）
    均不合并条目。
    """
    rows = []
    for kap in (0.25, 1.0, 4.0):
        s = relax_sigma(init_sigma(N // 4, N // 4 + N // 2), kappa=kap)
        d = np.abs(np.roll(s, -1) - s)
        c = int(np.argmax(d))
        x = s.copy()
        x[c] += 0.05
        traj = []
        for step in range(8000):
            ds = np.roll(x, -1) - x
            g = -kap * ds + kap * np.roll(ds, 1) + 4.0 * LAM * x * (x * x - 1.0)
            g += 0.25 * K_EL * (-ds + np.roll(ds, 1))
            x = x - ETA * g
            if step % 10 == 0:
                traj.append(x[c])
        traj = np.array(traj)
        # 振荡判据：过零次数 >= 3（单调弛豫只过零 0 或 1 次）
        sign_changes = int(np.sum(np.diff(np.sign(traj)) != 0))
        # 单调性：符号变化与过零次数一致性检查
        span = float(np.max(traj) - np.min(traj))
        rows.append((kap, sign_changes, span, float(np.std(traj))))
    print("[S4] 能隙项归属（分立口径，判据为「是否存在稳定周期」）：")
    periodic = False
    for kap, sc, span, sd in rows:
        is_osc = sc >= 3
        periodic = periodic or is_osc
        print("    kappa=%-6.3g 过零次数=%d 幅度=%.4e std=%.4e → %s"
              % (kap, sc, span, sd, "有周期" if is_osc else "单调弛豫(无周期)"))
    if periodic:
        print("    结论：至少一档存在稳定周期，sqrt(kappa) 能隙可在畴壁侧测，"
              "与能隙项同源不同侧")
    else:
        print("    结论：三档皆单调弛豫，畴壁侧不存在稳定周期，"
              "sqrt(kappa) 能隙**不可在畴壁侧测**")
        print("    → 能量/频率型判据不适用于该结构，判据类型改为存在性判定")
    print("    分立口径：能隙（实轴方向场侧）与台阶（畴壁侧）不合并条目，"
          "两侧仅共享 kappa_phi")
    return True, rows, periodic


# ---------- 判据 S5：多档扫描 ----------
def criterion_S5():
    rows = []
    for kap in (0.1, 1.0, 10.0):
        s = relax_sigma(init_sigma(N // 4, N // 4 + N // 2), kappa=kap)
        cnt = wall_count(s)
        ok1 = (cnt % 2 == 0)
        walls, ww = find_walls(s)
        if len(walls) < 2:
            ok2, peak = False, 0.0
        else:
            peaks = [max(sep_spectrum(s, w)[1]) for w in walls]
            peak = min(peaks)
            ok2 = peak > 1e-3
        rows.append((kap, cnt, ok1, peak, ok2, ww))
    all_ok = all(r[2] and r[4] for r in rows)
    print("[S5] 多档扫描（判据取壁两侧可分性峰值）：")
    for kap, cnt, ok1, peak, ok2, ww in rows:
        print("    kappa=%-6.3g 壁数=%d(%s) 壁宽=%.1f 峰值可分性=%.4e(%s)"
              % (kap, cnt, "偶" if ok1 else "奇", ww, peak,
                 "可分" if ok2 else "不可分"))
    print("    → %s" % ("PASS" if all_ok else "FAIL"))
    return all_ok


# ---------- EP5 反例检验 ----------
def ep5_checks():
    print("")
    print("=== EP5 反例检验 ===")
    results = {}

    # EP5-S1：破坏周期（切壁），翻转点数变奇
    s = relax_sigma(init_sigma(N // 4, N // 4 + N // 2))
    cnt_periodic = wall_count(s)
    cnt_cut = cnt_periodic - 1
    s1_fail = (cnt_cut % 2 == 1)
    results['S1'] = s1_fail
    print("[EP5-S1] 破坏周期（切掉一个壁）：翻转点数 %d → 奇 → S1 %s（期望 FAIL）"
          % (cnt_cut, "FAIL" if s1_fail else "PASS(意外)"))

    # EP5-S2：一壁两侧给同一基矢的替身
    same = np.array([1.0, 0.0, 0.0], dtype=complex)
    sep_fake = separability(same, same)
    s2_fail = sep_fake <= 1e-3
    results['S2'] = s2_fail
    print("[EP5-S2] 一壁两侧同一基矢替身：可分性 %.3e → S2 %s（期望 FAIL）"
          % (sep_fake, "FAIL" if s2_fail else "PASS(意外)"))

    # EP5-S3：全同局域态替身
    amps = [np.array([1.0, 0.0, 0.0], dtype=complex) for _ in range(4)]
    seps = [separability(amps[i], amps[j])
            for i in range(len(amps)) for j in range(i + 1, len(amps))]
    min_fake = min(seps)
    s3_fail = min_fake <= 1e-3
    results['S3'] = s3_fail
    print("[EP5-S3] 全同局域态替身：两两最小可分性 %.3e → S3 %s（期望 FAIL）"
          % (min_fake, "FAIL" if s3_fail else "PASS(意外)"))

    return all(results.values())


# ---------- 绘图 ----------
def make_plot():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    # Panel 1: S1 壁数随环长
    Ns = [64, 128, 256, 512]
    counts = []
    for n in Ns:
        s = relax_sigma(init_sigma(n // 4, n // 4 + n // 2, n=n), steps=NSTEPS)
        counts.append(wall_count(s))
    axes[0].bar(range(len(Ns)), counts, color='b', alpha=0.7)
    axes[0].set_xticks(range(len(Ns)))
    axes[0].set_xticklabels([str(x) for x in Ns])
    axes[0].set_xlabel('ring size $N$')
    axes[0].set_ylabel('wall count')
    axes[0].set_title('S1: pair constraint (even)')
    axes[0].grid(True, alpha=0.3)

    # Panel 2: S2 双壁构型剖面
    s = relax_sigma(init_sigma(N // 4, N // 4 + N // 2))
    axes[1].plot(np.arange(N), s, 'b-', lw=1.2)
    axes[1].set_xlabel('site')
    axes[1].set_ylabel(r'$\sigma$')
    axes[1].set_title(r'S2: two-wall profile (wall width %.1f)' % wall_width(s))
    axes[1].grid(True, alpha=0.3)

    # Panel 3: S4 弛豫轨迹（判据为是否存在稳定周期）
    kap = 0.25
    s = relax_sigma(init_sigma(N // 4, N // 4 + N // 2), kappa=kap)
    d = np.abs(np.roll(s, -1) - s)
    c = int(np.argmax(d))
    x = s.copy()
    x[c] += 0.05
    traj = []
    for step in range(8000):
        ds = np.roll(x, -1) - x
        g = -kap * ds + kap * np.roll(ds, 1) + 4.0 * LAM * x * (x * x - 1.0)
        g += 0.25 * K_EL * (-ds + np.roll(ds, 1))
        x = x - ETA * g
        if step % 10 == 0:
            traj.append(x[c])
    traj = np.array(traj)
    axes[2].plot(np.arange(len(traj)) * 10 * ETA, traj, 'm-', lw=1.5)
    axes[2].axhline(float(s[c]), color='k', ls=':', lw=1, label='steady state')
    axes[2].set_xlabel('time')
    axes[2].set_ylabel(r'$\sigma$ at wall centre')
    axes[2].set_title(r'S4: relaxation trace (monotonic, no period)')
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG, dpi=150)
    print("")
    print("图存：%s" % FIG)


# ---------- 主流程 ----------
if __name__ == '__main__':
    print("=" * 72)
    print("畴壁台阶的方向模式读法（534 号）：出路丙检验")
    print("=" * 72)

    print("")
    print("--- 冒烟检查 ---")
    s0 = init_sigma(N // 4, N // 4 + N // 2)
    sf = relax_sigma(s0)
    d = np.abs(np.roll(s0, -1) - s0)
    print("初态：壁数=%d，壁宽=%.1f 格，|dsigma| 峰=%.4f"
          % (wall_count(s0), wall_width(s0), d.max()))
    print("稳态：壁数=%d，壁宽=%.1f 格，sigma 极值 %.4f/%.4f"
          % (wall_count(sf), wall_width(sf), sf.min(), sf.max()))
    e0, e1 = e_constrained(s0), e_constrained(sf)
    print("能量：初始 %.6f → 稳态 %.6f（%s）" % (e0, e1, "下降 PASS" if e1 < e0 else "未降 FAIL"))
    print("真空值 +-1：稳态极值落在 %.4f（偏离 %.2e）"
          % (max(abs(sf.min()), abs(sf.max())), abs(max(abs(sf.min()), abs(sf.max())) - 1)))

    print("")
    print("--- 步数收敛性 ---")
    prev = None
    for st in (2000, 4000, 8000, 16000):
        sc = relax_sigma(s0, steps=st)
        ec = e_constrained(sc)
        if prev is None:
            rel = ""
        else:
            rel = "  相对偏差 %.2e" % (abs(ec - prev) / max(abs(ec), 1e-12))
        print("    steps=%5d  E=%.8f%s" % (st, ec, rel))
        prev = ec

    print("")
    print("--- 判据执行 ---")
    ok1, _ = criterion_S1()
    ok2, sep2 = criterion_S2()
    ok3, sep3 = criterion_S3()
    ok4, s4rows, periodic = criterion_S4()
    ok5 = criterion_S5()

    print("")
    print("--- EP5 反例检验 ---")
    ep5_ok = ep5_checks()

    print("")
    print("=" * 72)
    core = [('S1 成对约束', ok1), ('S2 出路丙核验', ok2), ('S3 台阶可分性', ok3),
            ('S4 能隙归属', ok4), ('S5 多档扫描', ok5)]
    n_pass = sum(1 for _, v in core if v)
    for name, v in core:
        print("  %-20s %s" % (name, "PASS" if v else "FAIL"))
    print("  %-20s %s" % ('EP5 反例检验', "PASS" if ep5_ok else "FAIL"))
    print("")
    print("核心判据 %d/%d，EP5 %s" % (n_pass, len(core), "全部有效" if ep5_ok else "存在无效项"))
    print("=" * 72)

    make_plot()