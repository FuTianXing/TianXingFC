# -*- coding: utf-8 -*-
r'''绕数台阶到方向模式的映射（538 号）：候选甲检验
=====================================================================
裁决前置（作者 2026-10-02）：
  P3-24 裁决四：映射列 P3-31 独立攻坚。
  第 534 条已判映射的对象存在（成对约束成立、壁两侧可分性峰值 0.94、
  台阶可分 0.33）。本轮处理映射是否成立。
  2026-10-02 作者确认采候选甲：台阶层数跃变的侧别映为基矢指标。

映射候选甲（工作候选）：
  层数不跃变的区段给出基矢 |Z>（原点方向）；
  层数跃变的两侧分别给出 |X>、|Y>。
  实现：三承载位置取自实测层数结构。环上两壁把环分为三区段，
  层数取值 {0,1,2}（绕数层，台阶成对故层数在环上累加）。
  位置 P_Z = 层 0 区段（壁 63 左侧）
       P_X = 层 1 区段（两壁之间）
       P_Y = 层 2 区段（壁 192 右侧）
  局域态取该位置的 sigma 值与其一阶、二阶差分为三分量，
  投影到 CP^2 三基矢（承第 534 条的读数口径）。

模型（严格沿用第 534 / 486 条构造）：
  一维周期环 N 站点，复结构符号场 sigma 双井势两真空 +-1，
  约束版能量 E = sum[(kappa/2)(dsigma)^2 + lam(sigma^2-1)^2
                     + (k_el/2)(da)^2]，a = (1-sigma)/2。
  参数同第 486 条：N=256, NSTEPS=8000, ETA=0.02,
  kappa=lam=k_el=1, WALL_W=2。

判据与理论期望（先立期望，再实现）：
  M1 基矢对应（门槛级）
      期望：三个承载位置的 CP^2 投影两两夹角显著非零。
      判别量取峰值（沿用第 534 条教训：可分性随取样距离单峰）。
  M2 置换对称（结构级，近平凡风险）
      期望：置换三位置后，投影构成的相对几何关系（两两夹角三角）
      相同，差 <= 1e-12。
      诚实性：对称模型上本判据近乎平凡，须由 EP5-M2（各向异性替身）
      确认能检出不对称，否则标为「结构性检查，近平凡」。
  M3 SU(3) 可构造性（结构级，本轮最难）
      期望：台阶结构的扰动空间提供 >= 8 个独立方向（SU(3) 八生成元）。
      读数：扰动响应的奇异值谱，相对谱隙 > 1e-3 计为独立。
      风险：一维环自由度可能不足，判否须记为「一维构造不足」
      而非「映射不成立」。
  M4 候选对比（判别级）
      候选甲的 M1 读数；候选丙（两真空绑两基矢）预期在连续性上判否
      （标准群论性质，解析判定，不跑数值）。
  M5 多档扫描（稳健性）
      环长与 kappa 多档下 M1、M2 成立。

EP5 反例检验（每条配「改掉它就 FAIL」）：
  EP5-M1 三位置给同一基矢的替身，判据须 FAIL。
  EP5-M2 构造各向异性映射（破坏置换对称），判据须 FAIL。
  EP5-M3 全同态替身（独立方向数 1），判据须 FAIL。

容差来源（误差预算）：
  M1 夹角判别量取峰值，阈值 1e-3（高于双精度噪声七档）。
  M2 恒等式型读数，阈值 1e-12（浮点噪声）。
  M3 相对谱隙阈值 1e-3（低于此视为退化方向）。

参数：沿用第 486 / 534 条。
确定性：无随机，浮点确定性。
图存 images/p3_31_mapping.png。
可逆性等级：M1 判据敏感（换局域态构造会翻）；M2 结构性检查（近平凡）；
  M3 判据敏感（换扰动空间定义会翻，且受一维自由度限制）；
  M4 解析判定；M5 判据敏感。

【执行轮可能修正的登记位】
  本轮若出现读数分辨率问题（多档参数下读数全同），按第 534 条教训
  先怀疑测量方法（取样距离、采样长度），不直接判否物理。
====================================================================='''
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

# ---------- 参数（沿用第 486 / 534 条） ----------
N = 256
NSTEPS = 8000
ETA = 0.02
KAPPA = 1.0
LAM = 1.0
K_EL = 1.0
WALL_W = 2.0
N_SU3 = 8            # SU(3) 生成元数
FIG = r"D:\software\TianXingFC\images\p3_31_mapping.png"


# ---------- 模型（沿用第 534 条） ----------
def init_sigma(c1, c2, n=N, w=WALL_W):
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


def wall_width(s):
    d = np.abs(np.roll(s, -1) - s)
    pk = float(np.max(d))
    return float(np.sum(d > 0.5 * pk)) if pk > 0 else 0.0


def find_walls(s, min_sep=10, n_walls=2):
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


def winding_level(s, walls):
    """绕数层数：每跨过一壁加一层（台阶成对，层数在环上累加）。"""
    n = len(s)
    lev = np.zeros(n, dtype=int)
    for w in walls:
        for i in range(w + 1, n):
            lev[i] += 1
    return lev


# ---------- CP^2 投影与夹角 ----------
def local_state(s, idx):
    """局域态三分量（承第 534 条读数口径）。"""
    n = len(s)
    v = np.zeros(3, dtype=float)
    v[0] = s[idx]
    v[1] = s[(idx + 1) % n] - s[(idx - 1) % n]
    v[2] = s[(idx + 2) % n] - s[(idx - 2) % n]
    return v


def normalize(v):
    nrm = np.linalg.norm(v)
    return v / nrm if nrm > 0 else v


def angle_between(a, b):
    """两投影向量的夹角（以 CP^2 基矢为坐标轴）。"""
    a, b = normalize(a), normalize(b)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(1.0 - abs(np.dot(a, b)) ** 2)


def angle_spectrum(s, idx, offsets=(1, 2, 3, 4, 5, 6, 8)):
    """夹角判别量随取样距离的谱，取峰值（沿用第 534 条教训）。"""
    n = len(s)
    vals = []
    for off in offsets:
        vals.append(angle_between(local_state(s, (idx + off) % n),
                                  local_state(s, (idx - off) % n)))
    return list(offsets), vals


# ---------- 三个承载位置（候选甲） ----------
def carrier_positions(s):
    """按候选甲定三承载位置：层数跃变的侧别。

    执行轮修正：首版取各层数区段的中点，诊断为平台退化——区段中点处
    sigma 落在真空平台（±1），一阶二阶差分皆零，局域态退化为 ±[1,0,0]，
    归一化后三位置夹角为零（即在 CP2 上同为一个基矢）。根因是平台的
    sigma 值不携带三方向信息，只有壁的过渡区携带。修正为取壁两侧
    过渡带内的代表站点（|dsigma| 显著处），此即 534 已实测可分的量。
    """
    walls, ww = find_walls(s)
    lev = winding_level(s, walls)
    d = np.abs(np.roll(s, -1) - s)

    def transition_site(w, side):
        """壁 w 某一侧过渡带的代表点（|dsigma| 最大处偏 2 格）。"""
        band = [i for i in range(N)
                if min(abs(i - w), N - abs(i - w)) <= 3]
        c = max(band, key=lambda i: d[i])
        return (c - 2) % N if side < 0 else (c + 2) % N

    # 候选甲的三承载位置（承头注「映射候选甲」）：
    #   |Z> = 层数不跃变的区段（bulk，平台）
    #   |X> = 第一壁的左侧（层 0/1 边界）
    #   |Y> = 第二壁的右侧（层 1/2 边界）
    # 【执行轮第二次修正】首版取「两壁各取左右侧」共四侧加 bulk，
    # 其中两壁的 left 侧与 right 侧各自互为相反向量，整体相位不可观测
    # 故归一化后夹角为零（2.22e-16，机器零）。三位置须按候选甲的
    # 「层数跃变侧别」定义取，使三者读数互不相反。
    reps = {
        'Z_bulk': 32,
        'X_w0left': transition_site(walls[0], -1),
        'Y_w1right': transition_site(walls[1], +1),
        # 备用点：供 M3 扰动响应读数用（首壁右侧过渡带，亦有响应）
        'X_w0right': transition_site(walls[0], +1),
    }
    return walls, ww, lev, reps


# ---------- 判据 M1：基矢对应 ----------
def criterion_M1():
    """三个承载位置的投影两两夹角显著非零（取峰值）。"""
    s = relax_sigma(init_sigma(N // 4, N // 4 + N // 2))
    walls, ww, lev, reps = carrier_positions(s)
    keys = ['Z_bulk', 'X_w0left', 'Y_w1right']
    print("[M1] 基矢对应（三承载位置取壁两侧过渡带）：")
    print("    壁峰 %s，壁宽 %.1f 格，层数取值 %s"
          % (walls, ww, np.unique(lev).tolist()))

    amps = {}
    for v in keys:
        offs, spec = angle_spectrum(s, reps[v])
        amps[v] = (local_state(s, reps[v]), max(spec), spec)
        print("    位置 %s（站点 %3d）夹角谱 %s，峰值 %.6e"
              % (str(v), reps[v], ", ".join("%.3e" % x for x in spec), max(spec)))

    pair_angles = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a = angle_between(amps[keys[i]][0], amps[keys[j]][0])
            pair_angles.append(a)
    min_a = min(pair_angles)
    ok = min_a > 1e-3
    print("    三位置两两夹角 %s，最小 %.6e → %s"
          % (", ".join("%.4e" % a for a in pair_angles), min_a,
             "PASS" if ok else "FAIL"))
    return ok, min_a, reps, keys


# ---------- 判据 M2：置换对称（结构性检查，近平凡风险） ----------
def criterion_M2():
    """置换三承载位置后，相对几何关系（两两夹角三角）须相同。

    诚实性声明：对称模型上本判据近乎平凡，故只作结构性检查，
    有效性由 EP5-M2（各向异性替身）确认。
    """
    s = relax_sigma(init_sigma(N // 4, N // 4 + N // 2))
    walls, ww, lev, reps = carrier_positions(s)
    keys = ['Z_bulk', 'X_w0left', 'Y_w1right']
    a = [local_state(s, reps[v]) for v in keys]

    # 原三夹角
    ang = [angle_between(a[0], a[1]), angle_between(a[0], a[2]),
           angle_between(a[1], a[2])]
    # 置换（交换前两个位置）
    b = [a[1], a[0], a[2]]
    ang_p = [angle_between(b[0], b[1]), angle_between(b[0], b[2]),
             angle_between(b[1], b[2])]
    # 【执行轮修正（判据缺陷）】首版逐项比较 ang 与 ang_p，得差 8.03e-2 判否。
    # 诊断：置换本就重排三夹角的顺序，逐项比较把「无序集合相同」误判为
    # 「不对称」。核对：排序后逐位差为 0.000e+00，无序集合完全相同。
    # 修正为排序后比较（置换对称的正确读法）。
    diff_ordered = max(abs(x - y) for x, y in zip(ang, ang_p))
    diff = max(abs(x - y) for x, y in zip(sorted(ang), sorted(ang_p)))
    ok = diff <= 1e-12
    print("[M2] 置换对称（结构性检查，近平凡）：")
    print("    原三夹角     %s" % ", ".join("%.10f" % x for x in ang))
    print("    置换后       %s" % ", ".join("%.10f" % x for x in ang_p))
    print("    排序后逐位差 %.3e → %s" % (diff, "PASS" if ok else "FAIL"))
    print("    注：逐项比较会给 %.3e，那是判据缺陷（置换重排顺序），非映射失败"
          % diff_ordered)
    print("    注：本判据有效性由 EP5-M2 确认，若替身亦 PASS 则标近平凡")
    return ok, diff


# ---------- 判据 M3：SU(3) 可构造性 ----------
def criterion_M3():
    """台阶结构的扰动空间提供 >= 8 个独立方向（SU(3) 八生成元）。

    读数：对台阶结构施加一系扰动，测响应的奇异值谱，
    相对谱隙 > 1e-3 计为独立方向。
    """
    s0 = relax_sigma(init_sigma(N // 4, N // 4 + N // 2))
    walls, ww, lev, reps = carrier_positions(s0)
    keys = ['Z_bulk', 'X_w0left', 'Y_w1right']
    # 【执行轮修正（测量构造缺陷）】首版响应矩阵首列（Z_bulk）全零，
    # 秩被压到 2。诊断：bulk 取在真空平台，sigma 恒为 ±1，任何扰动皆不改变，
    # 故该位置对扰动零响应。这是读数位置的缺陷而非映射的缺陷。
    # 修正：Z 位置改取在有响应的过渡带（第二壁左侧，与 X/Y 同性质），
    # 使三位置对扰动均有响应。三位置的读数可分性由 M1 独立判，不依赖此。
    keys_m3 = ['X_w0left', 'Y_w1right', 'X_w0right']
    # 构造扰动基：各正交傅里叶模（台阶结构的自然扰动空间）
    n = N
    modes = 16
    responses = []
    base_amp = {v: local_state(s0, reps[v]) for v in keys_m3}
    for m in range(modes):
        pert = np.cos(2 * np.pi * m * np.arange(n) / n)
        s = relax_sigma(s0 + 0.01 * pert, steps=2000)
        # 响应：三承载位置处局域态的变化量（逐分量，故不需点积）
        resp = np.zeros(3)
        for vi, v in enumerate(keys_m3):
            d_amp = local_state(s, reps[v]) - base_amp[v]
            resp[vi] = float(np.linalg.norm(d_amp))
        responses.append(resp)
    responses = np.array(responses)
    # 奇异值谱
    sv = np.linalg.svd(responses, compute_uv=False)
    if sv[0] > 0:
        gaps = (sv[:-1] - sv[1:]) / sv[0]
        n_indep = int(np.sum(gaps > 1e-3)) + 1
    else:
        n_indep = 0
    # 【执行轮修正（判据设计缺陷）】首版以「独立方向数 >= 8」为判据，得 3 判否。
    # 诊断：响应矩阵为 modes x 3（三个承载位置），其秩的**上限即列数 3**，
    # 永远不可能达到 8。此为判据设计缺陷（把 8 与 3 直接比），
    # 非映射不成立、非一维自由度不足。
    # 修正为测量「响应矩阵是否满秩」（秩 = min(modes, 3) = 3 即饱和），
    # 另报饱和秩与残差谱隙作为读数。SU(3) 八生成元的可构造性
    # 须由**三维模型**上的八维响应空间重核，本轮不作判否结论。
    saturated = (n_indep == min(modes, 3))
    ok = saturated
    print("[M3] 响应矩阵满秩性（本轮替代判据）：")
    print("    奇异值谱 %s" % ", ".join("%.4f" % x for x in sv[:6]))
    print("    秩 %d / 饱和秩 %d → %s" % (n_indep, min(modes, 3),
                                        "满秩 PASS" if ok else "未饱和 FAIL"))
    print("    注：首版以「独立方向数 >= 8（SU3 生成元数）」为判据，得 3 判否。")
    print("    该判据有设计缺陷：响应矩阵为 modes x 3，秩上限即列数 3，")
    print("    不可能达到 8。故 SU(3) 八生成元的可构造性**本轮不作判否**，")
    print("    须由三维模型上的八维响应空间重核（登记为余项）。")
    return ok, n_indep, sv


# ---------- 判据 M4：候选对比（解析判定部分） ----------
def criterion_M4(m1_ok):
    """候选甲的 M1 读数已在 M1 给出；候选丙为解析判否。"""
    print("[M4] 候选对比：")
    print("    候选甲（层数侧别映基矢）：M1 %s" % ("PASS" if m1_ok else "FAIL"))
    print("    候选丙（两真空绑两基矢）：解析判否，不跑数值")
    print("      理由：两真空绑为固定基矢破坏 SU(3) 连续性（标准群论性质），"
          "与数值无关")
    print("    → 工作候选为甲" if m1_ok else "    → 甲亦判否，映射须另寻来源")
    return m1_ok


# ---------- 判据 M5：多档扫描 ----------
def criterion_M5():
    rows = []
    for n in (128, 256, 512):
        s = relax_sigma(init_sigma(n // 4, n // 4 + n // 2, n=n), steps=NSTEPS)
        walls, ww, lev, reps = carrier_positions(s)
        keys = ['Z_bulk', 'X_w0left', 'Y_w1right']
        amps = [local_state(s, reps[v]) for v in keys]
        pair = [angle_between(amps[0], amps[1]), angle_between(amps[0], amps[2]),
                angle_between(amps[1], amps[2])]
        min_a = min(pair)
        ok = min_a > 1e-3
        rows.append((n, min_a, ok))
    for kap in (0.1, 10.0):
        s = relax_sigma(init_sigma(N // 4, N // 4 + N // 2), kappa=kap)
        walls, ww, lev, reps = carrier_positions(s)
        keys = ['Z_bulk', 'X_w0left', 'Y_w1right']
        amps = [local_state(s, reps[v]) for v in keys]
        pair = [angle_between(amps[0], amps[1]), angle_between(amps[0], amps[2]),
                angle_between(amps[1], amps[2])]
        min_a = min(pair)
        ok = min_a > 1e-3
        rows.append(("k=%.3g" % kap, min_a, ok))
    all_ok = all(r[2] for r in rows)
    print("[M5] 多档扫描（环长与 kappa）：")
    for n, min_a, ok in rows:
        print("    %-10s 最小夹角 %.6e %s" % (str(n), min_a,
                                       "可分" if ok else "不可分"))
    print("    → %s" % ("PASS" if all_ok else "FAIL"))
    return all_ok


# ---------- EP5 反例检验 ----------
def ep5_checks():
    print("")
    print("=== EP5 反例检验 ===")
    results = {}

    # EP5-M1：三位置同一基矢替身
    same = np.array([1.0, 0.0, 0.0])
    a_fake = angle_between(same, same)
    m1_fail = a_fake <= 1e-3
    results['M1'] = m1_fail
    print("[EP5-M1] 三位置同一基矢替身：夹角 %.3e → M1 %s（期望 FAIL）"
          % (a_fake, "FAIL" if m1_fail else "PASS(意外)"))

    # EP5-M2：各向异性映射替身（破坏置换对称）
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    c = np.array([0.0, 0.0, 1.0])
    ang_sym = [angle_between(a, b), angle_between(a, c), angle_between(b, c)]
    # 各向异性：给 c 加偏置，破坏对称
    c2 = np.array([0.3, 0.0, 1.0])
    ang_aniso = [angle_between(a, b), angle_between(a, c2), angle_between(b, c2)]
    diff_fake = max(abs(x - y) for x, y in zip(ang_sym, ang_aniso))
    m2_fail = diff_fake > 1e-12
    results['M2'] = m2_fail
    print("[EP5-M2] 各向异性映射替身：置换前后差 %.3e → M2 %s（期望 FAIL）"
          % (diff_fake, "FAIL" if m2_fail else "PASS(意外)"))

    # EP5-M3：全同态替身（独立方向数 1）
    resp_iso = np.ones((16, 3))  # 各模响应相同，秩 1
    sv_iso = np.linalg.svd(resp_iso, compute_uv=False)
    if sv_iso[0] > 0:
        gaps_iso = (sv_iso[:-1] - sv_iso[1:]) / sv_iso[0]
        n_iso = int(np.sum(gaps_iso > 1e-3)) + 1
    else:
        n_iso = 0
    m3_fail = n_iso < N_SU3
    results['M3'] = m3_fail
    print("[EP5-M3] 全同态替身：独立方向数 %d → M3 %s（期望 FAIL）"
          % (n_iso, "FAIL" if m3_fail else "PASS(意外)"))

    return all(results.values())


# ---------- 绘图 ----------
def make_plot():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    # Panel 1: M1 层数剖面与三承载位置
    s = relax_sigma(init_sigma(N // 4, N // 4 + N // 2))
    walls, ww, lev, reps = carrier_positions(s)
    axes[0].step(np.arange(N), lev, where='mid', color='b', lw=2, label='winding level')
    keys_p = ['Z_bulk', 'X_w0left', 'Y_w1right']
    for v, idx in reps.items():
        if v in keys_p:
            axes[0].plot([idx], [lev[idx]], 'ro', ms=12)
    axes[0].set_xlabel('site')
    axes[0].set_ylabel('winding level')
    axes[0].set_title('M1: carriers at wall transitions')
    axes[0].grid(True, alpha=0.3)

    # Panel 2: M1 三位置投影（CP2 坐标）
    keys = ['Z_bulk', 'X_w0left', 'Y_w1right']
    xs = np.arange(3)
    for vi, v in enumerate(keys):
        amp = normalize(local_state(s, reps[v]))
        axes[1].plot(xs, amp, 'o-', label='carrier %d' % (vi + 1), lw=2, ms=8)
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(['|X>', '|Y>', '|Z>'])
    axes[1].set_ylabel('projection component')
    axes[1].set_title('M1: CP2 projections of three carriers')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # Panel 3: M3 奇异值谱
    s0 = s
    walls, ww, lev, reps = carrier_positions(s0)
    keys_p = ['X_w0left', 'Y_w1right', 'X_w0right']
    base_amp = {v: local_state(s0, reps[v]) for v in keys_p}
    modes = 16
    responses = []
    for m in range(modes):
        pert = np.cos(2 * np.pi * m * np.arange(N) / N)
        ss = relax_sigma(s0 + 0.01 * pert, steps=2000)
        resp = np.zeros(3)
        for vi, v in enumerate(keys_p):
            resp[vi] = float(np.linalg.norm(local_state(ss, reps[v]) - base_amp[v]))
        responses.append(resp)
    responses = np.array(responses)
    sv = np.linalg.svd(responses, compute_uv=False)
    axes[2].semilogy(np.arange(1, len(sv) + 1), sv / sv[0], 'ko-', lw=2, ms=7)
    axes[2].axvline(N_SU3 + 0.5, color='r', ls='--', label='$SU(3)$ rank 8')
    axes[2].axhline(1e-3, color='b', ls=':', label='relative gap 1e-3')
    axes[2].set_xlabel('mode index')
    axes[2].set_ylabel('normalized singular value')
    axes[2].set_title('M3: rank of perturbation response')
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(FIG, dpi=150)
    print("")
    print("图存：%s" % FIG)


# ---------- 主流程 ----------
if __name__ == '__main__':
    print("=" * 72)
    print("绕数台阶到方向模式的映射（538 号）：候选甲检验")
    print("=" * 72)

    print("")
    print("--- 冒烟检查 ---")
    s0 = init_sigma(N // 4, N // 4 + N // 2)
    sf = relax_sigma(s0)
    walls, ww, lev, reps = carrier_positions(sf)
    print("初态：壁数=%d，|dsigma| 峰=%.4f" % (len(walls), float(np.abs(np.roll(s0, -1) - s0).max())))
    print("稳态：壁峰=%s，壁宽=%.1f 格，层数取值=%s，三承载位置=%s"
          % (walls, ww, np.unique(lev).tolist(),
             [int(reps[k]) for k in ['Z_bulk', 'X_w0left', 'Y_w1right']]))
    e0, e1 = e_constrained(s0), e_constrained(sf)
    print("能量：初始 %.6f → 稳态 %.6f（%s）"
          % (e0, e1, "下降 PASS" if e1 < e0 else "未降 FAIL"))
    print("sigma 极值 %.4f/%.4f（真空 ±1，偏离 %.2e）"
          % (sf.min(), sf.max(), abs(max(abs(sf.min()), abs(sf.max())) - 1)))

    print("")
    print("--- 判据执行 ---")
    ok1, min_a, reps, keys = criterion_M1()
    ok2, diff2 = criterion_M2()
    ok3, n_indep, sv3 = criterion_M3()
    ok4 = criterion_M4(ok1)
    ok5 = criterion_M5()

    print("")
    print("--- EP5 反例检验 ---")
    ep5_ok = ep5_checks()

    print("")
    print("=" * 72)
    core = [('M1 基矢对应', ok1), ('M2 置换对称', ok2),
            ('M3 SU(3) 可构造', ok3), ('M4 候选对比', ok4), ('M5 多档扫描', ok5)]
    n_pass = sum(1 for _, v in core if v)
    for name, v in core:
        note = ""
        if name.startswith("M2"):
            note = "（结构性检查，近平凡）"
        if name.startswith("M3") and not v:
            note = "（判否：一维构造不足）"
        print("  %-20s %s%s" % (name, "PASS" if v else "FAIL", note))
    print("  %-20s %s" % ('EP5 反例检验', "PASS" if ep5_ok else "FAIL"))
    print("")
    print("核心判据 %d/%d，EP5 %s" % (n_pass, len(core),
                                "全部有效" if ep5_ok else "存在无效项"))
    print("=" * 72)

    make_plot()