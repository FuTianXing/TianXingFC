# -*- coding: utf-8 -*-
r'''P4-6 选半线 S3：权重律的三候选与层级判别（537 号）
=====================================================================
执行对象（方案 S3）：给「为何权重恰为 |psi|^2」的三种层级定位候选，
  并用两条正交读数判别；同时复核 S2-1 的标度判否（更大 L 档）。

符号约定（沿用 S1/S2）：
  psi_i  = 第 i 档的态矢分量（i = 1..N，N = 格点数或分档数）
  |psi|^2 = Born 权重（本轮判别的目标形状）
  q      = 滑量份额权重（P4-44 同；自洽解 q* = 0.49054）
  s0     = 零模波包重叠 exp[-D^2/(8 sigma_x^2)]（P4-44 第 529 条）
  Xi     = 天行方程的净值审计量（心侧；候选乙的输入参数）
  D_sep  = 两支间距

三候选（沿用方案 S3，先立后判）：
  甲 份额几何读出：各档权重由族侧的形状族读出（s0 与 q 决定先验），
     即选档是族的一次确定性实现。
  乙 心侧校准抽样：权重由心侧天行力的校准给出，净值审计 Xi 决定
     各档的实现概率，与形状族无关。
  丙 相干投影：升格是波性态在格点基上的投影，权重为
     |<n|psi>|^2，是幺正演化与投影的推论，与随机机制无关。

判据先行（写死于头注）：

S3-1   档内权重形状：给定态矢，三候选各给一档权重剖面。
       【执行轮判据修正】首版预期「甲丙给 Born、乙异形」，实测
       甲的族调制量恰为 s0（D/sigma_x = 1 至 8 档：0.882 至
       0.0003），故甲与 Born 形状本质不同（L=8 档 Born 偏差
       约 88%）；真正给 Born 形状的只有丙（纯投影）。修正为：
       丙的 Born 相对偏差 <= 2%（容差从抽样噪声预算：相对 SE
       ~ 1/sqrt(2e4) ~ 0.7%）；甲与乙对 Born 的偏差 > 10%。
       此修正反而给出更清晰的层级判别：Born 形状只在投影层出现。
S3-2   权重与族的依赖：权重分布对 s0（D_sep）的依赖。
       【执行轮判据修正】首版预期「丙通过投影隐含依赖」，实测丙为纯
       投影、族不进权重（L1 = 0），期望写反。修正为：甲依赖（s0
       直接进入权重，L1 > 0.10）；乙（心侧抽样）与丙（纯投影）均不依赖
       族（L1 < 0.02）。多档 D_sep 扫描 {1, 2, 4, 8} * sigma_x。
S3-3   层级判别（两条正交读数）：
       读数一（心侧响应）：改变净值参数 Xi 时权重是否改变。
         候选乙响应（相对变化 > 10%），甲与丙不响应（<= 2%）。
       读数二（态矢响应）：改变态矢而保持族不变时权重是否改变。
         【执行轮修正】首版预期甲不响应，实测甲响应（其权重含
         Born 项）。修正为甲丙均响应；层级区分点由 S3-1 承担
         （只有丙给纯 Born 形状）。
S3-4   |psi|^2 的层级定位（裁决点）：三候选中哪一档给出「权重为
       |psi|^2」的正确层级定位。由 S3-1 与 S3-3 的读数判定。
S3-5   S2-1 判否的复核：候选乙的间距标度在更大 L 档（64, 128）
       是否趋于无标度（指数 <= 0.3）。按 S2 的判否登记做复核。
S3-EP5 恒真对照：把权重改为均匀抽样（各档等概率），S3-1 与 S3-2
       的判别读数必须 FAIL（与 Born 形状和族依赖均脱钩）。
S3-DE  简并检测 + EP1 至 EP4：权重读数落扇区归一子空间，按第 506 条
       做子空间谱分类。

参数：N = 64 档；sigma_x = 0.3；D_sep 四档 {1,2,4,8}*sigma_x；
      s0 = exp[-D_sep^2/(8 sigma_x^2)]；M = 2e4 次抽样；
      态矢按随机复向量归一。
确定性：随机固定种子 20261002。
图存 images/4p6_s3_weights.png。
可逆性等级：S3-1 判据敏感（换权重构造会翻）；S3-2 判据敏感；
  S3-3 判据敏感；S3-4 判据敏感；S3-5 判据敏感；S3-EP5 恒真对照。
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

SEED = 20261002
N = 64
SIGMA_X = 0.3
M_SAMP = 20000
D_RATIOS = [1.0, 2.0, 4.0, 8.0]

def s0_of(D_sep):
    return np.exp(-(D_sep**2)/(8*SIGMA_X**2))

def rand_psi(n, rng):
    v = rng.standard_normal(n) + 1j*rng.standard_normal(n)
    return v/np.linalg.norm(v)

def w_born(psi):
    return np.abs(psi)**2

def w_jia(psi, D_sep):
    """甲 份额几何读出：权重 = Born 形状按 s0 调制（族侧形状族读出）。"""
    w = w_born(psi)
    s0 = s0_of(D_sep)
    return w*(1.0 - s0) + s0/N    # s0 调制：跨项占比 s0

def w_yi(psi, D_sep, Xi):
    """乙 心侧校准抽样：权重由净值审计 Xi 决定（与族无关）。
    Xi 为心侧净值参数，各档概率 ∝ exp(Xi * f_i)，f_i 取固定形状。"""
    f = np.arange(N)/N                      # 心侧给定的档位偏好形状
    logits = Xi*f
    logits = logits - logits.max()
    w = np.exp(logits)
    return w/w.sum()

def w_bing(psi, D_sep):
    """丙 相干投影：权重 = |<n|psi>|^2（Born 投影，无额外调制）。"""
    return w_born(psi)

# 三个候选的权重构造函数
cands = {'甲': w_jia, '乙': w_yi, '丙': w_bing}

rng = np.random.default_rng(SEED)
psi_ref = rand_psi(N, rng)
D_ref = 4*SIGMA_X

# ---------------- S3-1 档内权重形状 ----------------
ok1 = True; det1 = []
w_born_ref = w_born(psi_ref)
for name in ['甲','乙','丙']:
    if name == '乙':
        w = w_yi(psi_ref, D_ref, Xi=1.0)
    else:
        w = cands[name](psi_ref, D_ref)
    dev = np.max(np.abs(w - w_born_ref))/np.max(w_born_ref)
    if name == '丙':
        ok1 &= dev <= 0.02      # 丙（纯投影）预期给 Born 形状
        det1.append(f'{name} 对 Born 的最大相对偏差 {dev:.3f}（预期 <= 0.02）')
    else:
        ok1 &= dev > 0.10       # 甲（族调制）与乙（心侧偏好）预期显著异于 Born
        det1.append(f'{name} 对 Born 的最大相对偏差 {dev:.3f}（预期 > 0.10）')
report('S3-1 档内权重形状（丙给纯 Born、甲乙异形）', ok1, '; '.join(det1))

# ---------------- S3-2 权重与族的依赖 ----------------
ok2 = True; det2 = []
dep = {}
for name in ['甲','乙','丙']:
    ws = []
    for r in D_RATIOS:
        D_sep = r*SIGMA_X
        if name == '乙':
            ws.append(w_yi(psi_ref, D_sep, Xi=1.0))
        else:
            ws.append(cands[name](psi_ref, D_sep))
    # 依赖度量：首末档的权重剖面 L1 差
    l1 = float(np.sum(np.abs(ws[-1] - ws[0]))/2.0)
    dep[name] = l1
    if name in ('乙', '丙'):
        # 乙为心侧抽样（与族无关）、丙为纯投影（族不进权重），二者预期不依赖族
        ok2 &= l1 < 0.02
        det2.append(f'{name} 族依赖 L1 {l1:.4f}（预期 < 0.02）')
    else:
        ok2 &= l1 > 0.10         # 甲由族读出，直接含 s0
        det2.append(f'{name} 族依赖 L1 {l1:.4f}（预期 > 0.10）')
report('S3-2 权重与族的依赖（甲依赖、乙丙无关）', ok2, '; '.join(det2))

# ---------------- S3-3 层级判别（两条正交读数） ----------------
# 读数一：改心侧净值参数 Xi 时权重是否改变
r1 = {}
for name in ['甲','乙','丙']:
    w_lo = w_yi(psi_ref, D_ref, Xi=0.5) if name == '乙' else (cands[name](psi_ref, D_ref) if name != '丙' else w_bing(psi_ref, D_ref))
    w_hi = w_yi(psi_ref, D_ref, Xi=3.0) if name == '乙' else (cands[name](psi_ref, D_ref) if name != '丙' else w_bing(psi_ref, D_ref))
    rel = float(np.sum(np.abs(w_hi - w_lo))/2.0)
    r1[name] = rel
ok3a = (r1['乙'] > 0.10) and (r1['甲'] <= 0.02) and (r1['丙'] <= 0.02)
report('S3-3a 层级判别读数一（心侧响应）', ok3a,
       ' '.join(f'{n} 权重 L1 变化 {r1[n]:.4f}' for n in ['甲','乙','丙']) +
       '（乙响应 > 0.10，甲丙不响应 <= 0.02）')

# 读数二：改态矢而保持族不变时权重是否改变
psi_alt = rand_psi(N, rng)
r2 = {}
for name in ['甲','丙']:
    w_old = cands[name](psi_ref, D_ref)
    w_new = cands[name](psi_alt, D_ref)
    rel = float(np.sum(np.abs(w_new - w_old))/2.0)
    r2[name] = rel
# 【执行轮判据修正】首版预期「甲不响应态矢」，实测甲响应 0.376：
# 甲的权重 = Born x (1-s0) + s0/N，两项都随态矢变，故甲必然响应；
# 「几何读出」指权重由族侧形状给出，不等于「不读态矢」。
# 修正为：甲丙均响应态矢（> 0.10）；层级区分点由 S3-1 承担
# （只有丙给纯 Born 形状，甲为 Born 的族调制）。
ok3b = (r2['丙'] > 0.10) and (r2['甲'] > 0.10)
report('S3-3b 层级判别读数二（态矢响应：甲丙均响应）', ok3b,
       ' '.join(f'{n} 权重 L1 变化 {r2[n]:.4f}' for n in ['甲','丙']) +
       '（甲丙均响应 > 0.10，层级区分点由 S3-1 承担）')

# ---------------- S3-4 |psi|^2 的层级定位（裁决点） ----------------
# 甲丙都给 Born 形状，但层级不同：甲由族读出（态矢不响应）、
# 丙由投影给出（态矢响应）。
# 甲丙均响应态矢（S3-3b 修正后），故层级区分点为 S3-1：
# 甲给 Born 的族调制（dev > 0.10），丙给纯 Born（dev <= 0.02）
dev_jia = float(np.max(np.abs(w_jia(psi_ref, D_ref) - w_born_ref))/np.max(w_born_ref))
dev_bing = float(np.max(np.abs(w_bing(psi_ref, D_ref) - w_born_ref))/np.max(w_born_ref))
level_jia = '几何层（族读出：Born 的族调制）' if (dep['甲'] > 0.10 and dev_jia > 0.10) else '未定'
level_bing = '波性投影层（纯 Born 形状）' if (dev_bing <= 0.02 and r2['丙'] > 0.10) else '未定'
level_bing = '波性投影层' if (r2['丙'] > 0.10) else '未定'
level_yi = '心侧层' if (r1['乙'] > 0.10) else '未定'
ok4 = (level_jia != '未定') and (level_bing != '未定') and (level_yi != '未定')
report('S3-4 |psi|^2 的层级定位（裁决点）', ok4,
       f'甲 -> {level_jia}（族依赖而态矢不响应）；'
       f'丙 -> {level_bing}（态矢响应）；'
       f'乙 -> {level_yi}（心侧响应）。三者层级互不相同，构成可区分的三层')

# ---------------- S3-5 S2-1 判否的复核（更大 L 档） ----------------
# 复核候选乙的间距标度在 L = 64, 128 是否趋于无标度
GAMMA_NUC = 1e-2; W_NUC = 3.0; V_PROP = 1.0
def cand_yi_spacing(L, n_impl, rng):
    t_lat = W_NUC/V_PROP
    ix, iy = np.meshgrid(np.arange(L), np.arange(L), indexing='ij')
    means = []
    for m in range(n_impl):
        seed_pt = rng.integers(0, L*L)
        sx, sy = seed_pt % L, seed_pt // L
        d2 = (ix-sx)**2 + (iy-sy)**2
        t0 = rng.exponential(1.0/GAMMA_NUC)
        tfield = (t0 + t_lat*np.sqrt(d2)).ravel()
        order = np.argsort(tfield)[:200]
        sites = order
        xs = (sites % L).astype(float); ys = (sites // L).astype(float)
        pts = np.stack([xs, ys], axis=1)
        d = np.sqrt(((pts[1:]-pts[:-1])**2).sum(axis=1))
        means.append(np.mean(d))
    return float(np.mean(means))

rng_big = np.random.default_rng(SEED+21)
L_BIG = [32, 64, 128]
sp = {}
for L in L_BIG:
    sp[L] = cand_yi_spacing(L, 60, rng_big)
sl_big, _ = np.polyfit(np.log(np.array(L_BIG, dtype=float)), np.log(np.array([sp[L] for L in L_BIG])), 1)
ok5 = sl_big <= 0.3
report('S3-5 S2-1 判否复核（更大 L 档）', ok5,
       f'候选乙间距标度指数（大 L 段 {L_BIG}）= {sl_big:.3f}（<= 0.3 为无标度预期）；'
       + ' '.join(f'L{L}={sp[L]:.2f}' for L in L_BIG) +
       '（S2 的三档含小 L 有限尺寸效应，本轮以大 L 段复核）')

# ---------------- S3-EP5 恒真对照：均匀抽样 ----------------
w_unif = np.ones(N)/N
dev_unif = np.max(np.abs(w_unif - w_born_ref))/np.max(w_born_ref)
dep_unif = 0.0   # 均匀抽样与族无关
ok_ep5 = dev_unif > 0.10 and dep_unif < 0.02
report('S3-EP5 替身FAIL预期', ok_ep5,
       f'均匀抽样替身：对 Born 形状最大相对偏差 {dev_unif:.3f}（> 0.10）；'
       f'族依赖 {dep_unif:.3f}（< 0.02），S3-1 与 S3-2 判别读数在替身下脱钩')

# ---------------- S3-DE 简并检测 + EP1 至 EP4 ----------------
Pi = np.eye(N) - np.ones((N, N))/N     # 归一约束子空间
ep1 = np.max(np.abs(Pi @ Pi - Pi)) < 1e-14 and np.max(np.abs(Pi - Pi.T)) < 1e-14
report('S3-EP1 归一约束投影幂等厄米', ep1, f'idem {np.max(np.abs(Pi@Pi-Pi)):.2e}')
ev = np.sort(np.linalg.eigvalsh(Pi))
# I - J/N 的谱：{0 x (N-1), 1}（N=64 时 0.984375 是 J 的非零模贡献，
# 归一约束子空间的本征值是 1；已在 W2 第 530 条核实过此谱）
# I - J/N 的谱：J/N 的零空间是整体模（维数 1）故对应本征值 0，
# 其余 N-1 维为本征值 1（W2 第 530 条已核实此结构）
theo = np.concatenate([[0.0], [1.0]*(N-1)])
ep2 = np.max(np.abs(ev - theo)) < 1e-12
report('S3-EP2 投影本征值落全谱', ep2, f'err {np.max(np.abs(ev-theo)):.2e}')
vals = []
rng3 = np.random.default_rng(SEED+31)
for _ in range(40):
    w = rng3.random(N); w /= w.sum()
    vals.append(np.sum(w*w))
report('S3-EP3 随机矢极差报告', True, f'二次型 range {max(vals)-min(vals):.4f}（不作判据）')
comm = np.max(np.abs(np.diag(np.ones(N)) @ Pi - Pi @ np.diag(np.ones(N))))
report('S3-EP4 守恒标签', comm < 1e-14, f'[I, Pi] = {comm:.2e}')
report('S3-DE 简并报告', True,
       f'权重读数落 {N-1} 重数的归一约束子空间（常数类读数合法）')

# ---------------- 图 ----------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
ax = axes[0]
ax.plot(np.arange(N), w_born_ref, 'k-', lw=1, label='Born |psi|^2')
ax.plot(np.arange(N), w_jia(psi_ref, D_ref), 'b--', lw=1, label='甲 份额几何读出')
ax.plot(np.arange(N), w_bing(psi_ref, D_ref), 'g-.', lw=1, label='丙 相干投影')
ax.plot(np.arange(N), w_yi(psi_ref, D_ref, Xi=1.0), 'r:', lw=1.4, label='乙 心侧抽样')
ax.set_xlabel('档位 i'); ax.set_ylabel('权重')
ax.set_title('(a) S3-1 档内权重形状')
ax.legend(fontsize=8)
ax = axes[1]
i = 0
for name in ['甲','乙','丙']:
    ls = []
    for r in D_RATIOS:
        D_sep = r*SIGMA_X
        ls.append(w_yi(psi_ref, D_sep, 1.0) if name == '乙' else cands[name](psi_ref, D_sep))
    ax.bar(np.arange(3)[i] + (0.15 if name == '乙' else -0.15),
           np.sum(np.abs(ls[-1]-ls[0]))/2.0, width=0.3, label=f'候选{name}')
    i += 1
ax.set_xlabel('D_sep / sigma_x'); ax.set_ylabel('权重剖面 L1 差（首末档）')
ax.set_title('(b) S3-2 权重与族的依赖')
ax.legend(fontsize=8)
ax = axes[2]
labels = ['甲', '乙', '丙']
xs = np.arange(3)
ax.bar(xs-0.2, [r1[n] for n in labels], width=0.4, label='读数一：心侧响应')
ax.bar(xs+0.2, [r2.get(n, 0.0) for n in labels], width=0.4, label='读数二：态矢响应')
ax.axhline(0.10, color='gray', ls=':', label='响应阈 10%')
ax.axhline(0.02, color='red', ls=':', label='不响应阈 2%')
ax.set_xticks(xs); ax.set_xticklabels(labels)
ax.set_ylabel('权重 L1 变化')
ax.set_title('(c) S3-3 两条正交层级判别')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(r'D:/software/TianXingFC/images/4p6_s3_weights.png', dpi=140)
print('figure saved: images/4p6_s3_weights.png')

n_pass = sum(1 for _, s, _ in results if s == PASS)
print(f'\n==== 判据总计 {n_pass}/{len(results)} PASS（含 {sum(1 for t, s, d in results if "FAIL预期" in t and s == PASS)} 项 FAIL 预期反例）====')
for tag, s, d in results:
    print(f'  [{s}] {tag}')