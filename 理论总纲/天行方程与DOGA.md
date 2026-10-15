# 天行方程与 DOGA

> 来源：《[[统摄万物兴衰演化的天行方程]]》（傅天行，天行新学研究院，2026.05.19）
> 本文件是技能知识库的基准内容，未经确认不得修改。新思想请先写入 `内部草稿/`。

## 1. 天行方程（The Tianxing Equation）

全名"天行万物兴衰演化动力学方程"，是天行力学的核心动力学方程，对系统整段历史执行**净值审计**。注意：方程中的 $t, dt$ 是归一化后的无量纲时间（严格记为 $u$），对应各系统自身的特征时间。

**主方程**（波粒耦合的优美形式，充分体现心物波粒二象）：

$$
\boxed{ \Xi(T) = \tanh \left\{ \Re \left[ \left( \int_0^T \mathcal{W}(t) \otimes \mathcal{P}(t) \, dt \right)^2 \right] \right\}}
$$

**完整方程版**（展开波核与粒核）：

$$
\Xi(T) = \tanh\!\left\{ \Re\!\left[ \left( \int_0^{T} \sqrt{\frac{\tilde O \tilde G}{\tilde A}}\, \exp\!\left( -i\eta\pi \int_0^t \ln \tilde D \, d\tau \right) dt \right)^2 \right] \right\}
$$

**振幅相位版**：

$$
\Xi(T)=\tanh\left\{\Re\left[\left(\int_0^T \mathcal P(t)e^{i\Phi(t)}\,dt\right)^2\right]\right\}
$$

**简化版**（历史累积量形式）：

$$
\Xi(T)=\tanh\!\big(A(T)^2-B(T)^2\big), \quad Z(T)=\int_{T_0}^{T}\Psi(t)\,dt=A(T)+iB(T)
$$

## 2. 四个动态环节

1. **态函数定义**：波核 $\mathcal{W}$ 与粒核 $\mathcal{P}$ 耦合为相干耦合态 $\Psi(t)$，构成系统每一时刻的完整存在状态。
2. **历史累积**：每个事件以复矢量 $P(t)e^{i\Phi(t)}dt$ 录入复历史积分器；同向事件同向叠加增强，反向事件相互抵消侵蚀净值。
3. **终末清算**：$\Re[Z^2] = A^2 - B^2$，$A$ 为正功，$B$ 为负功，二者以平方形式放大参与终局审判。
4. **终极映射**：$\tanh$ 将净值压缩至 $(-1, +1)$；临界阈值约 $\pm 1.39$（$\tanh$ 饱和拐点），跨过即发生不可逆结构性跃迁：$+1$ 为兴，$-1$ 为衰，$0$ 为混沌未决。

## 3. 波性相干核 $\mathcal{W}(t)$

> **符号约定说明（2026 检查）**：偏航相位存在两处源文档符号差异。本知识库采用与《统摄万物兴衰演化的天行方程》及《理论纲领》一致的**负号约定**：$\Phi(t) = -\eta\pi\int_0^t \ln\tilde{D}\,d\tau$，即 $\Omega(t) = -\eta\pi\ln\tilde{D}(t)$。

波核表征系统相位与方向调谐能力，决定系统能否维持统一、协调与持续相干；在心物合一的复杂系统中体现为精神相位。

$$
\mathcal{W}(t) = e^{i\Phi(t)} = \exp \left( i \int_0^t \Omega(\tau) \, d\tau \right)
$$

- 累积相位：$\Phi(t) = \int_0^t \Omega(\tau)\, d\tau$
- 偏航频率：$\Omega(t) = -\eta \pi \ln \tilde D(t)$，$\eta$ 为敏感系数
- $\tilde D(t) \in (0,1]$ 度量演化方向与最优路径的瞬时契合程度；$\tilde D \to 1$ 时 $\Phi$ 恒定，贡献纯正功；$\tilde D$ 偏离 1 时负功不可逆积累
- 关键理解（《天行方程即圆之自转》）：旋转本身是基线而非波核，波核测量的是旋转偏离最优路径的程度；$\tilde D = 1$ 时 $\mathcal{W}(t) = e^{i\cdot 0} = 1$

## 4. 粒性实存核 $\mathcal{P}(t)$

粒核表征系统将结构稳定性与发展动能转化为实际产出的强度。

$$
\mathcal P(t)=\sqrt{\frac{\tilde O(t)\tilde G(t)}{\tilde A(t)}}
$$

## 5. 天行力与天行力定律

**[[天行力]]**定义为终末净值对方向对齐度的全历史变分导数：

$$
\mathcal{F}[D] \equiv -\frac{\delta \Xi(T)}{\delta D}
$$

负号确保顺向的天行力指向净值增长方向（经典力学"保守力是势能负梯度"的直接推广）。

**方向对齐率**（自主系统中特称相位调谐率）：

$$
\dot{D}_{\mathcal{F}} \equiv \frac{dD}{dt}\bigg|_{\mathcal{F}}
$$

**天行力定律**（方向对齐度完整动力学方程）：

$$
\frac{dD}{dt} = \alpha_D \cdot (1-D) \cdot m(t) - \beta_D \cdot D \cdot (1-m(t)) + \gamma \cdot \mathcal{F}[D](t)
$$

前两项为外部参考模板匹配度 $m(t)$ 的直接驱动项，第三项为天行力激发的全局净值反馈驱动项。物理解释：经典力是"桨"（效率学），天行力是"水流"（方向学）。

## 6. DOGA 四序参量

| 序参量 | 名称 | 含义 | 物理对应力 |
|---|---|---|---|
| $\tilde{D}$ | 方向对齐度 | 演化方向与最优路径的契合程度 | 引力 |
| $\tilde{O}$ | 秩序稳固度 | 结构稳定性 | 强核力 |
| $\tilde{G}$ | 增长动能 | 发展动能 | 电磁力 |
| $\tilde{A}$ | 异化熵增度 | 内耗与异化 | 弱核力 |

体用关系：天行力是体，DOGA 四力是用；四力属于物侧（被动规律），天行力属于心侧（能动根源）。

## 7. DOGA 耦合演化方程组（附件 D）

在无量纲化内部时间 $u$ 中，四序参量 $\tilde{D}, \tilde{O}, \tilde{G}, \tilde{A} \in (0,1]$ 满足闭合演化方程组：

$$
\begin{aligned}
\frac{d\tilde{D}}{du} &= \alpha_D \tilde{D}(1-\tilde{D}) \;-\; \beta_D \tilde{A}\tilde{D} \;+\; \gamma_D \tilde{O}\tilde{G} \\[6pt]
\frac{d\tilde{O}}{du} &= \alpha_O \tilde{O}(1-\tilde{O}) \;-\; \beta_O \tilde{A}\tilde{O} \;+\; \gamma_O \tilde{G}\tilde{D} \\[6pt]
\frac{d\tilde{G}}{du} &= \alpha_G \tilde{G}(1-\tilde{G}) \;-\; \beta_G \tilde{A}\tilde{G} \;+\; \gamma_G \tilde{O}\tilde{D} \\[6pt]
\frac{d\tilde{A}}{du} &= \alpha_A \tilde{A}(1-\tilde{A}) \;+\; \beta_A (1-\tilde{D})\tilde{O}\tilde{G} \;-\; \delta \tilde{A}\tilde{D}
\end{aligned}
$$

各项含义：
- **Logistic 自限项** $\alpha_X X(1-X)$：有限承载力约束，$X \to 1$ 时饱和
- **异化抑制项** $-\beta_X A X$（对 $D, O, G$）：内耗优先蛀空已有成就
- **正向交叉催化项** $+\gamma_X YZ$：$D, O, G$ 构成兴盛螺旋的正反馈闭环
- **异化生成项** $+\beta_A (1-D) O G$：方向偏离时系统被自身成功内在地制造异化（"其兴也勃焉，其亡也忽焉"的动力学机制）
- **异化清除项** $-\delta A D$：异化清除只能通过方向对齐 $D$ 的主动作用执行

三个设计原则：归一化封闭性、结构对称性（$D,O,G$ 正向三元组 vs $A$ 负向元）、方向枢纽性（$D$ 是异化生成抑制与清除执行的双重枢纽）。

**两个吸引子**：兴盛吸引子 $(1,1,1,0)$ 对应 $\Xi \to +1$；崩溃吸引子 $(0,0,0,1)$ 对应 $\Xi \to -1$。中部均衡点通常为鞍点，不存在稳定中庸平衡；动力学制造极化，天行方程审计极化。

> **2026-10-16 更正（判据第 648 号，7/7）**：严格地说，只有崩溃点 $(0,0,0,1)$ 是方程组的精确不动点；兴盛点 $(1,1,1,0)$ 的残差为 $\gamma_D$（非零），它须在**界域约束**（正向三元组严格钳在 $[0,1]$）下才成为可达吸引子。同轮更正：$\tilde A$ 的稳态为 $\max(0,1-\delta/\alpha_A)$，并非恒为零，$(1,1,1,0)$ 只是 $\delta\ge\alpha_A$ 档的特例。另：设计原则 1 所称"由 Logistic 自限项保证所有变量在 $(0,1]$ 区间内演化"在交叉催化项存在时不成立（越界阈值 $\gamma_c=\beta A$）。详见 [[验证脚本/注记-DOGA两吸引子的真伪与界域约束形式（判据第648号）|判据第 648 号注记]]。

**演化到结算闭环**：

$$
\text{DOGA耦合演化} \;\longrightarrow\; \text{历史路径} \;\longrightarrow\; \text{终末清算} \;\longrightarrow\; \Xi
$$

更高阶：$\Xi$ 反馈调校演化参数 $\alpha, \beta, \gamma, \delta$，实现"以史为鉴"的自适应求存。参数可依系统尺度标定（个体/企业/国家文明），骨架不变。


