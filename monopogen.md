# Single-nucleotide variant calling in single-cell sequencing data with Monopogen

非常棒的问题，这篇论文介绍的 **Monopogen** 工具确实是目前最前沿、适用于单细胞数据中 **体细胞 SNV（single-nucleotide variant）检测** 的方法之一。下面我将从三个方面来系统解析这篇文献的内容：

---

## 📘 一、文献主要内容概述

**论文标题：**  
*Single-nucleotide variant calling in single-cell sequencing data with Monopogen*  
**期刊：** Nature Biotechnology（2024年5月）

**研究目的：**  
开发一个高精度、高通用性的算法——**Monopogen**，能够从各种单细胞测序数据（包括 10x Genomics）中检测出：
- **种系突变（Germline SNVs）**
- **体细胞突变（Somatic SNVs）**

**研究亮点：**
- 不依赖 bulk WGS/WES 数据。
- 能用于低覆盖度、高噪声的 scRNA-seq、snRNA-seq、scATAC-seq、scDNA-seq。
- 特别适用于 **CHIP / 克隆性造血 / 癌症进化** 研究中的体细胞突变识别与克隆追踪。

---

## 🧠 二、Monopogen 的核心算法原理

Monopogen 包括两个主要模块：

### ✅ 1. **种系突变检测（Germline SNV calling）**
- 从 **pooled BAM（多个单细胞合并）** 中识别候选突变位点。
- 利用 **外部参考群体的 Linkage Disequilibrium（LD）信息**（如 1KG phase 3）对每个位点进行 **LD-based genotype refinement**，从而显著提升在稀疏数据中的基因型判定准确性。
- 准确率达 **95~97%**，远超 GATK/Samtools。

### ✅ 2. **体细胞突变检测（Putative somatic SNVs）**
主要包括以下几个子模块：

#### 🧩（1）SVM 模型过滤测序错误
- 用一系列 variant calling 特征训练 SVM 分类器：
  - 如：QS、BAF、VDB、BQB、MQSB、RPB、SGB 等
- 正样本：已知 germline SNV；
- 负样本：连续 de novo SNVs（>2个）但不含 germline。

#### 🧩（2）LD refinement 模型鉴别体细胞突变
- 创新地将 **LD 分析从人群层面拓展到了细胞群体层面！**
- 思路：
  - 如果 SNV 与周围 germline SNP 的 LD 很强 → 大概率是种系突变；
  - 如果 LD 弱，仅在部分细胞中共分离 → 更可能是体细胞突变；
- 用 **LD score ∈ [0, 0.5]** 衡量突变的体细胞可能性（值越大越可能是体细胞 SNV）。

#### 🧩（3）最终基因型估计
- 对于疑似体细胞 SNVs，使用 **Monovar** 在单细胞或 cluster 水平调用基因型。

---

## 🧪 三、Monopogen 是否可以直接用于 10x Genomics 数据进行 CHIP（体细胞突变）检测？

### ✅ 是的！它专为 **非定向捕获的单细胞数据（如 10x Genomics）** 设计。

你前面提到的限制（如：10x 不能定向扩增突变位点、表达量低、dropout多）都是 传统方法 的问题，而 **Monopogen 则是为了解决这些问题而设计的！**

---

### 📌 为什么 Monopogen 适用于 10x 数据？

| 挑战 | Monopogen 如何解决 |
|------|--------------------|
| **表达覆盖稀疏、非定向** | 用全基因组 LD 模型来“填补”低覆盖区域的基因型信息 |
| **体细胞突变仅存在于小群细胞中** | 用 **cell population-level LD cosegregation 模型** 鉴别体细胞突变 |
| **RNA测序错配率高** | 先用 SVM 模型去除 sequencing error，再用 LD 模型剥离 germline 与 somatic SNV |
| **突变不能直接定向 PCR** | 不需要 PCR，只用 BAM 文件，完全基于 reads 分布与 LD 特征 |
| **没有 WGS 对照** | 可在无 bulk 数据的背景下独立进行体细胞突变检测（并可用于 mtDNA、TCR 验证克隆） |

---

## 📊 四、Monopogen 在 CHIP 应用上的案例

论文中真实分析了：

### 🧬 CHIP 案例：
- 骨髓样本，9,346 个细胞，10x Genomics 平台（scRNA-seq）
- 搭配 MAESTER 技术同时捕获 mtDNA
- 用 Monopogen 检测到：
  - 284 个 putative somatic SNVs（CHIP候选）
  - 这些 SNV 在 mtDNA clone 或 TCR clone 中显著富集，支持其为克隆性变异
  - 不需要 WGS 或 GoT，直接从 10x BAM 中推断突变 + 克隆结构

---

## ✅ 总结：是否值得用于你自己的 10x + CHIP 数据？

| 维度 | 评价 |
|------|------|
| 📦 输入 | 单细胞 BAM 文件（10x Genomics 等） |
| 🧬 输出 | 每个位点的基因型、体细胞突变列表、克隆归属（可选） |
| 📊 应用 | CHIP 研究、肿瘤克隆演化、体细胞 SNV 热点分析、mtDNA-TCR 验证 |
| 💻 技术门槛 | Python 实现、可并行处理、大规模数据友好 |
| 💡 特点 | 不需 bulk WGS、不需定向 PCR，**完全算法层面解决突变分型问题** |
| 🌟 推荐指数 | ⭐⭐⭐⭐⭐（对你目前的研究目标来说，非常匹配） |

---

如果你感兴趣，我可以继续帮你：

- 部署 Monopogen 工具环境
- 准备 BAM 数据（从 Cell Ranger 输出开始）
- 整合 SNV 结果到 Seurat / Scanpy 分群结果中
- 挖掘 CHIP 克隆结构、推断影响基因等

是否想让我一步步帮你搭建和使用这个工具？我们可以从准备 BAM 文件或安装环境开始。

Monopogen 是专门为从稀疏、高噪声的单细胞测序数据中**识别种系与体细胞 SNVs**而设计的算法工具，它的设计核心是将 **人群层面的 LD 模型** 和 **细胞群体层面的等位基因共分离模式（allelic cosegregation）** 融合在一起，大大提升了在如 10x Genomics 这类平台上的变异检测能力。

---

## 🧠 一、Monopogen 的整体算法流程（简略图解）

```
BAM (10x / snRNA / scATAC / scDNA)
      │
      ▼
① 整体 pileup 扫描（寻找潜在SNV位点）
      │
      ▼
② 种系 SNV 检测（LD refinement with population reference）
      │
      ▼
③ 构建测序误差模型（用于后续 somatic SNV 排错）
      │
      ▼
④ SVM 过滤假阳性 SNV（去除伪突变、错配）
      │
      ▼
⑤ LD refinement @ 单细胞群体层面（检测 somatic SNV）
      │
      ▼
⑥ 单细胞基因型推断（使用 Monovar）
      │
      ▼
输出：高置信度的种系 SNV + 体细胞 SNV（可做克隆分析）
```

---

## 🔬 二、详细分步骤解释

---

### ✅ 步骤 1：候选 SNV 识别（de novo SNV scan）

- 从单细胞 BAM 文件中读取所有 reads。
- 合并所有细胞（pool）构建 pileup。
- 识别出在至少一个 read 中出现非参考等位基因的位置 → **候选 SNV 位点（de novo SNVs）**。

---

### ✅ 步骤 2：种系 SNV 检测（Germline SNV calling）

- 使用 Samtools 计算这些候选位点的基因型似然（Genotype Likelihood, GL）。
- 与 **外部种群参考面板（如 1000 Genomes phase 3）** 中的变异位点做匹配。
- 运用 **人群层面的 Linkage Disequilibrium (LD)**，将 low-confidence 的位点补齐、矫正，提升精度。

> 🧠 **LD refinement score 趋近于 0** → 是种系变异（高度与周围 SNP 共分离）

---

### ✅ 步骤 3：构建测序误差模型

- 对于 **始终与 LD 不一致的位点**，假设其为误差来源。
- 利用这些“discordant 位点”建立测序误差模型（估计假阳性概率）。

---

### ✅ 步骤 4：SVM 分类过滤假 SNVs（sequencing error）

- 输入特征包括：
  - QS（质量评分）
  - BAF（等位基因频率）
  - VDB（变异距离偏差）
  - BQB（碱基质量偏差）
  - MQSB（比对质量与链偏差）
  - RPB（read 位置偏差）
  - SGB（segregation-based metric）

- 训练：
  - 正类：LD 一致的已知种系突变
  - 负类：连片出现的 >2 个 de novo SNV（疑似错配）
- 分类出可信 de novo SNV → 进入下一步分析

---

### ✅ 步骤 5：体细胞 SNV 识别（Somatic SNV detection）

> ⭐ **创新点！**

- 核心思想：体细胞 SNV 通常只存在于部分细胞，**它与附近 germline SNP 不完全共分离**。
- 做法：
  1. 将 de novo SNV 与其邻近的 germline SNP 做 **haplotype phasing**
  2. 计算 **LD refinement score**（衡量其“偏离正常 LD 模式”的程度）
     - 越大 → 越有可能是体细胞 SNV
     - 趋近于 0 → 更可能是 germline SNV

- 可使用：
  - 两个位点 phasing（two-locus LD model）
  - 三个位点 phasing（three-locus LD model）
  - 后期可扩展多位点模型

---

### ✅ 步骤 6：单细胞/cluster 层面的基因型推断

- 使用 **Monovar** 工具在每个细胞水平上进行体细胞 SNV 的基因型推断。
- 输出：
  - 哪个细胞有突变，哪个没有
  - 可用于克隆追踪（如 mtDNA / TCR / lineage）

---

## 🧬 三、Monopogen 的输入与输出

| 输入 | 内容 |
|------|------|
| BAM 文件 | 来自 10x Genomics / Smart-seq / snRNA-seq / scATAC-seq 等 |
| 外部参考 | 1000 Genomes Phase 3（可替换为你自己的） |
| 细胞注释（可选） | 可关联 Seurat / Scanpy 的聚类信息做 downstream |

| 输出 | 用途 |
|------|------|
| Germline SNV 表 | 遗传背景、祖源分析 |
| Somatic SNV 表 | CHIP、克隆追踪、癌症变异分析 |
| SNV × Cell 表 | 可用于可视化、聚类、轨迹追踪 |

---

## 🎯 四、优势总结（相对于传统方法）

| 问题 | Monopogen 如何解决 |
|------|--------------------|
| 表达覆盖稀疏 | 用 population-level LD + SVM |
| 表达 dropout 多 | 用 phasing 识别真实共分离 |
| 难区分 germline/somatic | 用 LD refinement 识别出只在部分细胞中出现的体突 |
| 不需要定向 PCR | ✅ 只需 BAM 文件即可运行 |
| 平台适配性 | ✅ 支持 10x / Smart-seq2 / snATAC-seq / scDNA |
| 可以克隆追踪 | ✅ 与 mtDNA / TCR / Seurat 聚类结果匹配分析 |

---

## ✅ 想用的话我可以帮你做什么？

- 指导安装运行环境（Python 环境 + 依赖）
- 准备输入 BAM 数据（比如从 Cell Ranger 输出整理）
- 提供参数模板（适配 10x 或其他平台）
- 帮你将结果导入 Seurat / Scanpy 中展示突变分布与克隆结构

是否想我帮你写一个运行 Monopogen 的具体流程或脚本？我们可以从你的数据起步。

你问得非常到位！**LD refinement** 是 Monopogen 区分 **germline** 和 **somatic SNVs** 的核心算法部分之一，尤其是它的 **cell population-level linkage disequilibrium (LD) 分析**，非常创新且巧妙。下面我详细拆解这个算法的逻辑、数理思想和实际怎么运作。

---

## 🧠 一、为什么 LD 可以用来区分 germline 和 somatic SNVs？

**核心思路：**
- **Germline SNVs** 是在胚系就存在的，出现在 **所有细胞的同一条染色体上**，所以与周围 germline SNPs 一起遗传，形成稳定的 haplotype，**强 LD（共分离）**。
- **Somatic SNVs** 是在某一细胞后天突变产生的，仅存在于 **克隆子集细胞中**，所以它与周围 SNPs 只有在那部分细胞中共分离，**总体 LD 较弱**。

---

## 🧩 二、LD refinement 的计算流程

Monopogen 实现了两种模型：
- **Two-locus LD refinement**（简单）
- **Three-locus LD refinement**（更稳健）

### 🧬 输入：
- 所有单细胞中的：  
  - 已知 germline SNP 的 phased allele（来自外部参考或已调用）  
  - 候选 de novo SNV 的等位基因读数（通常很稀疏）

---

### ✅ 步骤1：将候选 SNV 与邻近的 germline SNP 一起做 *haplotype phasing*

对于每一个候选 SNV，考虑它周围最近的 1~2 个已知 SNP，比如：

```
位置        chr1:100     chr1:120     chr1:150
类型        germline     germline     SNV (未知)
样本A       A            G            A
样本B       A            G            A
样本C       A            T            A
样本D       A            G            T
```

你会发现：
- **候选 SNV（chr1:150）** 只有在一部分细胞中和 chr1:100~120 的等位基因形成“常见 haplotype”
- 如果它和附近 SNP 共现一致（共享 haplotype），可能是 germline；
- 如果只在少数细胞中共享 → **somatic**

---

### ✅ 步骤2：计算 LD refinement score

LD refinement score ∈ [0, 0.5]，越大越倾向 somatic：

#### ✏️ 定义（简化理解）：

> **LD score = 1 - average(co-occurrence rate of SNV with nearby SNP haplotypes)**

- 对所有细胞进行等位基因组合统计；
- 建立候选 SNV 的**等位基因分布条件概率**；
- 度量它与周围 SNP 的联合分布偏离程度；
- 结果：
  - **germline SNV**：LD score ≈ 0（与附近 SNPs 强 LD）
  - **somatic SNV**：LD score > 0（只在部分细胞中出现）

---

### ✅ 步骤3：使用 LD score 作为 somatic SNV 的判别标准

- Monopogen 设定阈值（例如 0.1~0.2）
- 大于阈值的 SNV → **putative somatic SNV**
- 也可以用于 rank 所有 SNV 的 somatic 可能性

---

## 📉 三、示意图理解（概念图）

```
所有细胞（共分离）：

SNP1---SNP2---SNV
 A       T      G
 A       T      G
 A       T      G
 A       T      G     => Germline（LD 高，score ≈ 0）

部分细胞（不共分离）：

SNP1---SNP2---SNV
 A       T      G
 A       T      G
 A       T      T
 A       T      T     => Somatic（LD 弱，score ↑）
```

---

## 🔬 四、实际结果中怎么验证有效性？

### ✅ 论文验证方式：
1. **在 TNBC（肿瘤）单细胞 DNA-seq** 中：
   - 与 matched WGS 对比：LD score 趋近 0 的果然是 germline；
   - LD score 显著上升的 → 多为 somatic
2. **在 bone marrow（CHIP 模型）中**：
   - 体细胞 SNV 多只存在于一个或多个 mtDNA 克隆中；
   - germline SNV 在所有细胞中稳定存在。

---

## 🧪 五、与传统方法比较的优势

| 特点 | Monopogen 的优势 |
|------|------------------|
| **无需定向测序突变位点** | 只要读到一个 allele 就能参与分析 |
| **无需 bulk 对照** | 通过 LD + cosegregation 分析即可判别 somatic |
| **抗稀疏** | 利用 pooled cell + LD 结构“补”缺失信息 |
| **可扩展** | 可用于 RNA/ATAC/DNA，甚至低覆盖数据 |

---

## ✅ 总结一句话：

> **Monopogen 的 LD refinement 算法** 通过衡量候选 SNV 与邻近 germline SNP 在单细胞群体中的“共分离程度”，构建出一个介于 0~0.5 的 LD score，用以判断其是普遍存在的 germline，还是克隆子集中出现的 somatic SNV，是**无 PCR、无 bulk 的突变检测新范式**。

---

如果你想看实际代码实现、算一个例子，或让 Monopogen 帮你在你自己的数据中调用体突，我们可以继续拆细一步一步来。要不要我举个简单例子算个 LD score 呢？

非常棒！你已经进入了 Monopogen 的核心数学机制：**haplotype phasing（单倍型相位）** 与 **LD score（连锁不平衡打分）**。下面我来深入解释这两个概念，并告诉你 Monopogen 中 LD score 是怎么计算出来的（逻辑 + 数学）。

---

## 🧬 一、什么是 haplotype phasing（单倍型相位）？

### ✅ 概念：
- 一个体细胞是 **二倍体（diploid）**：每个位点有两个等位基因，一个来自父亲，一个来自母亲。
- **Haplotype** 指的是一条染色体上的 **一串连续的等位基因组合**，例如：

```
位点1   位点2   位点3
 A       G       T   ← haplotype1
 G       A       C   ← haplotype2
```

### ✅ Phasing 的目的：
> 从观察到的 **杂合位点（heterozygous）** 中，推测它们分别属于哪一条染色体。

比如在 scRNA-seq 中读到某细胞在两个位点都是 A/G，但你不知道 A-A 和 G-G 是否在同一条链上，phasing 就是要推断这种关系。

---

## 🧠 二、Monopogen 的核心创新：用单细胞群体信息做 haplotype phasing

### 📌 传统 phasing：
- 基于家系（父母-子代）或长读段（long-read）或 linked-read 数据；
- 需要连续覆盖多个 SNP。

### 📌 Monopogen 的 phasing：
- 在单细胞群体中，用共现频率推断 phasing：
  > “如果 SNP A 和 SNP B 总是一起出现（在同一细胞内），那它们可能在同一个 haplotype 上。”

例如：

| Cell ID | SNP1 | SNP2 |
|---------|------|------|
| C1      | A    | G    |
| C2      | A    | G    |
| C3      | G    | A    |
| C4      | A    | G    |

→ 推测：`SNP1=A` 和 `SNP2=G` 是同一 haplotype

---

## 📊 三、LD score 的计算方法（简化公式 + 概念）

Monopogen 通过一个**统计量（LD refinement score）**来量化候选 SNV 与邻近 germline SNP 是否“经常一起出现”。

---

### 🧮 基本符号：

- 假设你有一个候选 SNV（S），和它周围的两个已知 SNP（G1, G2）。
- 你从所有细胞中提取这些位点的等位基因组成（基因型）。

### ✅ Step 1: 构建 haplotype 表

比如收集所有细胞中这三个位点的组合（只保留能读到的）：

| Cell | G1 | G2 | S |
|------|----|----|---|
| C1   | A  | T  | G |
| C2   | A  | T  | G |
| C3   | A  | T  | G |
| C4   | A  | T  | T |
| C5   | G  | T  | T |
| C6   | A  | T  | . |
| ...  | .. | .. | ..|

---

### ✅ Step 2: 计算 LD refinement score

Monopogen 实际上是在问：

> **这个 SNV 是否“稳定地”与某一 germline haplotype 共现？如果不是，那就是 somatic。**

#### ✏️ 简化理解（非精确公式）：

```math
LD_score = 1 - max_P(P(SNV = alt | G1, G2 = 某个haplotype组合))
```

- 意思是：如果这个 SNV 在所有某个 haplotype 中几乎总是 alt → 很可能是 germline
- 如果它只在某一小群体细胞中才有 alt，而大部分相同 haplotype 中都没有 → somatic

---

### ✅ 更数学的理解（简洁）：

Monopogen 用 **LD refinement score ∈ [0, 0.5]** 衡量“**haplotype一致性**”：

- **score → 0**：说明 SNV 与附近 SNP 形成高度一致 haplotype（germline 特征）
- **score ↑**：SNV 只在一部分细胞的 haplotype 中出现 → somatic 可能性高

> 在细胞群体中，这个打分类似于“某突变等位基因在 haplotype 背景下的非随机偏离程度”。

---

## 🧪 四、举个真实的例子（伪代码式）

你看到如下共现：

```
Haplotype:   A-T   A-T   A-T   G-T   A-T
SNV allele:   G     G     T     T     G
```

G/T 是候选 SNV 的两个等位基因：

- 在 `A-T` 背景下，出现了 G 和 T → 不是完美 LD
- 在 `G-T` 背景下只出现 T → 和 `A-T` 不一致

→ 说明这个 SNV 只在部分 haplotype 上出现 → LD score ↑ → **somatic 候选**

---

## 🧠 五、关键优点

- **不依赖连续测序覆盖**（不像 long-read 需要完全读穿）
- **不需要 bulk phasing 或定向 PCR**
- 可适用于表达非常稀疏的位点（即使只有单个细胞覆盖）

---

## ✅ 总结一句话：

> **Haplotype phasing** 是通过统计突变位点与邻近已知 SNPs 的共现模式，推测它们是否来自同一条染色体；而 **Monopogen 的 LD score** 就是在衡量这个共现是否广泛一致，**一致→germline，局部偏离→somatic**，这是无 WGS 和 PCR 情况下实现变异分类的核心创新。

---

是否想我用一个实际的例子，写成代码 + 演示一下 LD score 的计算过程？或者我们也可以直接在你自己的 BAM 文件上跑一个 mini 流程来看看。
太好了！我们来用一个 **简化的人工示例**，手动演示一下 **LD score 的计算过程**，帮助你完全理解 Monopogen 背后的逻辑。

---

## 🎯 目标

我们模拟一个小型数据集，展示：

- 候选 SNV 是否与周围 germline SNP 共分离（haplotype phasing）
- 如何通过这种共分离的“强/弱”，计算 **LD refinement score**
- 进而区分是 **germline 还是 somatic**

---

## 🧪 模拟数据集

假设我们有 6 个细胞，观测 3 个位点：

- **G1：germline SNP**
- **G2：germline SNP**
- **S：候选 SNV（我们要判断它是 germline 还是 somatic）**

| Cell | G1  | G2  | S   |
|------|-----|-----|-----|
| C1   | A   | T   | G   |
| C2   | A   | T   | G   |
| C3   | A   | T   | G   |
| C4   | A   | T   | .   |
| C5   | G   | C   | .   |
| C6   | G   | C   | G   |

---

## 🧬 第一步：统计 haplotype + S 的组合频率

我们统计一下不同 G1-G2 背景下 S 出现哪种等位基因：

| Haplotype (G1-G2) | S = G | S = . (缺失) |
|-------------------|-------|--------------|
| A-T               | 3     | 1            |
| G-C               | 1     | 1            |

---

## 🧠 第二步：计算每种 haplotype 中 S = G 的比例

我们希望看 G 是不是“专属于”某一个 haplotype：

| Haplotype | G 出现频率 |
|-----------|-------------|
| A-T       | 3 / 4 = 0.75 |
| G-C       | 1 / 2 = 0.5  |

如果 S 是 germline，我们期望它**只**出现在一个 haplotype 中且频率接近 1。

---

## 🧮 第三步：计算 LD score（简化版本）

我们定义：

```text
LD_score = 1 - max{P(S = G | haplotype)}
```

在这里：
- `P(S=G|A-T) = 0.75`
- `P(S=G|G-C) = 0.5`
- max = 0.75

所以：

```
LD_score = 1 - 0.75 = 0.25
```

---

## ✅ 第四步：结果解释

- **LD_score = 0.25**
- 如果我们设置的判断阈值是 0.1（论文默认约 0.1~0.2），那么这个 SNV 属于 **疑似体细胞突变（somatic SNV）**
- 如果 LD_score 很小（比如 0.01），说明 S 在某个 haplotype 上非常稳定地存在（支持 germline）

---

## 🧠 思维总结

- **核心逻辑：** 看候选 SNV 是否“强烈绑定”在某个 germline haplotype 上；
- **Monopogen 的聪明之处：** 把全体细胞当作“群体”，用类似群体遗传学的 **LD 共分离模型**，来代替传统需要 family trio、WGS、PCR 的办法。

---

## 🛠️ 进阶想法

- Monopogen 实际上还会考虑多种 phasing 情况、SNP 异常组合，并结合 SVM 模型进一步增强分类；
- 它支持三位点 LD 分析，更强健（比如 G1-G2-S 的三元组合）；
- 也可以引入 prior，如基因表达/测序质量等联合建模。

---

是否想我写一个 Python 小脚本把这个 LD score 自动算出来？也可以让你拿真实 BAM 结果算一下候选突变的 score。需要我帮你写吗？

非常好，你已经在深入挖掘 Monopogen 的关键组成部分了！下面我会从整体流程、特征设计、训练方式和实际判别逻辑上，**完整拆解 Monopogen 中的 SVM（支持向量机）模块**，也就是它用来剔除假突变（sequencing error）并挑选出真实突变的那一块。

---

## 🧠 一、SVM 的作用：过滤假突变（错误识别）

在稀疏、噪声极大的单细胞测序数据中，**错配、PCR错误、mapping错误**会造成大量“看起来像突变”的伪阳性 SNVs。

**Monopogen 的 SVM 模块**：
- 位于整个流程的中间
- 用来 **从 de novo SNV 中筛出“像真的”的突变**
- 被筛掉的通常是测序错配/假阳性

---

## 🔍 二、训练思路：弱监督 + 真实数据内构建正负样本

Monopogen 的巧妙之处在于：**它不需要事先标注好的真/假突变数据**，而是自己在数据中构造正负样本用于训练！

### ✅ 正样本（True SNVs）：
- 已知的 **germline SNP**，在外部数据库中有记录
- 假设这些是真的变异
- 用来代表真实突变的统计特征

### ✅ 负样本（False SNVs）：
- 连续的 **>2个 de novo SNV** 且未被确认的区域
- 通常是由于 mapping 错误或错配，误调用出来的一串“突变”
- 被当作负类

> ❗注意：这是 **“弱监督学习”**，因为样本是自动生成的，不需要人工标注

---

## 🧬 三、使用的特征（SVM 的输入）

每个候选 SNV 都会被提取出多个统计特征，作为 SVM 的输入。下表是论文中用到的特征集（非常经典的 variant calling 特征）：

| 特征名 | 含义 |
|--------|------|
| **QS** (Quality Score) | 变异的 Phred 质量分数 |
| **BAF** (B-allele frequency) | 变异等位基因频率 |
| **VDB** (Variant Distance Bias) | 变异等位基因在 reads 中分布是否偏斜 |
| **BQB** (Base Quality Bias) | 不同等位基因的碱基质量是否有显著差异 |
| **MQSB** (Mapping Quality Strand Bias) | 比对质量与正负链偏倚 |
| **RPB** (Read Position Bias) | 变异碱基出现在 read 的哪一段是否有偏差 |
| **SGB** (Segregation-based bias) | 是否只出现在某些细胞群体中、非均匀分布 |

这些特征大多数来自 samtools mpileup 或 bcftools 的 VCF 输出，也可直接从 BAM 中统计。

---

## 📊 四、SVM 分类器的训练与使用

### 🧠 模型结构：
- 使用标准 **支持向量机（SVM）** 分类器（可能是基于 scikit-learn 实现）
- 输入：每个 SNV 的 7 维特征向量
- 输出：标签（真实突变 / 噪声）

### 🔁 训练流程：
1. 构造正负样本集（如上）
2. 对特征进行标准化或归一化
3. 训练 SVM 模型（可能使用核函数如 RBF，或线性核）
4. 使用交叉验证选择超参数（如 C, γ）

---

### 🧪 应用流程：
对于每一个待判断的 de novo SNV：
- 抽取其 7 个特征
- 输入到 SVM 中
- 输出一个判别分数 / 0-1 概率
- 设置阈值（如 0.5）决定是否保留该 SNV

最终，只有通过了这个“第一道关卡”的候选 SNV 才会进入 **LD refinement 模块** 继续被判断是否为体细胞突变。

---

## ✅ 总结：SVM 模块的作用与贡献

| 作用 | 细节 |
|------|------|
| 🧹 初步清洗变异 | 去掉最常见的假突变（mapping 错误、碱基错配、PCR 错误） |
| 🎯 强化后续分析准确性 | 减少 LD 模块的 false positives，提高 somatic SNV specificity |
| 🔍 特征丰富，解释性强 | 大部分特征都是生物学可解释的 BAM 属性 |
| 💡 不依赖人工标注 | 自动构造正负样本，自适应不同平台和物种 |

---

## 📦 Bonus：你也可以单独用这个模块！

- 如果你不需要完整跑 Monopogen，也可以单独使用它的 **SVM + 特征提取模块** 来做变异质量过滤（SNV filtering）；
- 可扩展到其他数据类型如 ATAC-seq, snRNA-seq；

---

如果你想，我可以：

1. 给你写一个简单的 `Python` 版 SVM 特征提取 + 分类小例子；
2. 或者帮你调用真实 BAM 文件，提取 SNV + 特征，看看效果。

是否需要我演示一下 SVM 模型的训练或使用过程？我们可以用模拟数据跑一遍。