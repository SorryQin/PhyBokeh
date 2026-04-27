# PhyBokeh 项目完整框架详解

> 基于 SDXL 的物理引导生成式虚化（Bokeh）系统

---

## 目录

1. [项目总览](#1-项目总览)
2. [目录结构](#2-目录结构)
3. [整体架构图](#3-整体架构图)
4. [核心模块详解](#4-核心模块详解)
   - 4.1 [基础模型 SDXL](#41-基础模型-sdxl)
   - 4.2 [PISA 物理引导注意力处理器](#42-pisa-物理引导注意力处理器)
   - 4.3 [经典物理渲染器](#43-经典物理渲染器)
   - 4.4 [自定义 Tiled SDXL Pipeline](#44-自定义-tiled-sdxl-pipeline)
   - 4.5 [数据集模块](#45-数据集模块)
   - 4.6 [损失函数](#46-损失函数)
   - 4.7 [工具模块](#47-工具模块)
5. [训练流程详解](#5-训练流程详解)
6. [推理流程详解](#6-推理流程详解)
7. [数据准备流程](#7-数据准备流程)
8. [关键设计决策与创新点](#8-关键设计决策与创新点)
9. [超参数配置](#9-超参数配置)

---

## 1. 项目总览

PhyBokeh 是一个**物理引导的生成式虚化**系统，基于 **Stable Diffusion XL (SDXL)** 构建。其核心目标是将全聚焦（All-In-Focus, AIF）图像转化为具有真实浅景深（虚化/Bokeh）效果的图像，由深度/视差图和虚化强度控制。

**核心创新**：将**物理光学先验**（弥散圆 Circle of Confusion、遮挡关系、光线追踪散焦）直接注入扩散模型的注意力机制中，结合经典物理渲染器在线生成训练监督信号，无需预配对的训练数据。

**与传统扩散训练的区别**：PhyBokeh 不采用标准的"加噪-预测噪声"范式，而是使用**确定性多步编码-解码**流程：编码 AIF 图像 → 经固定时间步的 UNet 带物理引导注意力逐步处理 → 解码为虚化图像。损失直接在图像空间计算。

---

## 2. 目录结构

```
F:/PhyBokeh/
├── train.py                           # 主训练脚本
├── inference.py                       # 推理/评估脚本
├── dataset.py                         # 数据集类 (OnTheFlyDataset, TestDataset)
├── PISA_attn_processor.py             # 核心：物理引导自注意力处理器
├── optimization.py                    # 学习率调度器 (cosine, linear 等)
├── prepare_data.py                    # 预生成深度图 + 显著性掩码
├── utils_zcx.py                       # SSIM/MS-SSIM 指标 + Sinkhorn 归一化
├── wavelet_fix.py                     # AdaIN + 小波颜色修复 + 导向滤波
├── custom_diffusers/
│   ├── pipeline_sdxl.py               # 分块(Tiled) SDXL Pipeline，支持 disp_coc
│   └── attention_processor.py         # AttnProcessor2_0 兼容层
├── classical_renderer/
│   ├── mpi_multi_reverse.py           # MPI 光线追踪渲染器（圆形光圈）— 训练GT
│   ├── mpi_multi_reverse_qzr.py       # 扩展 MPI 渲染器（椭圆/猫眼 PSF）
│   ├── scatter.py                     # Scatter 渲染器（圆形光圈，CuPy CUDA）
│   └── scatter_ex.py                  # Scatter 渲染器（可调多边形光圈）
├── ckpt/                              # 注意力处理器快照（用于运行）
├── sdxl-base/                         # SDXL 1.0 基础模型权重
├── train_data/                        # 训练数据
│   ├── fg/                            # 前景图像 (RGBA)
│   └── bg/                            # 背景图像
├── test_data/                         # 推理测试数据
│   ├── input/                         # 输入图像
│   ├── depth/                         # 预计算深度图
│   └── mask/                          # 预计算显著性掩码
└── output/                            # 输出/日志目录
```

---

## 3. 整体架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PhyBokeh 整体架构                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │  前景图 (RGBA) │    │  背景图 (RGB) │    │  随机深度/焦面参数    │  │
│  └──────┬───────┘    └──────┬───────┘    └──────────┬───────────┘  │
│         │                   │                       │              │
│         └───────────────────┼───────────────────────┘              │
│                             ▼                                      │
│                 ┌───────────────────────┐                          │
│                 │  在线合成 AIF 图像      │  ← 合成全聚焦图像         │
│                 │  + 视差图 + 散焦强度    │                          │
│                 └───────────┬───────────┘                          │
│                             │                                      │
│              ┌──────────────┴──────────────┐                       │
│              ▼                             ▼                       │
│  ┌───────────────────┐         ┌─────────────────────┐            │
│  │ MPI 光线追踪渲染器 │         │  VAE Encoder        │            │
│  │ (物理GT生成)       │         │  + Gamma校正        │            │
│  └────────┬──────────┘         └──────────┬──────────┘            │
│           │                               │                        │
│           ▼                               ▼                        │
│  ┌───────────────────┐         ┌─────────────────────┐            │
│  │  虚化 GT 图像       │         │  输入 Latent (x₀)    │            │
│  │  (sRGB, [-1,1])   │         │  (4CH, H/8×W/8)     │            │
│  └────────┬──────────┘         └──────────┬──────────┘            │
│           │                               │                        │
│           │                   ┌───────────┘                        │
│           │                   │  固定3步去噪 [499, 300, 100]       │
│           │                   │  ┌──────────────────────────┐     │
│           │                   │  │  UNet + PISA Attention   │     │
│           │                   │  │  ┌────────────────────┐  │     │
│           │                   │  │  │ 输入:              │  │     │
│           │                   │  │  │  - latent x_t      │  │     │
│           │                   │  │  │  - timestep t      │  │     │
│           │                   │  │  │  - text_embeds     │  │     │
│           │                   │  │  │  - disp_coc (2CH)  │  │     │
│           │                   │  │  │  - pisa_strength   │  │     │
│           │                   │  │  └────────────────────┘  │     │
│           │                   │  │        │                  │     │
│           │                   │  │        ▼                  │     │
│           │                   │  │  model_pred = UNet(x,t)  │     │
│           │                   │  │  x = (x - √β·pred)/√α   │     │
│           │                   │  └──────────────────────────┘     │
│           │                   │                    │                │
│           │                   ▼                    ▼                │
│           │           ┌──────────────────────────────┐            │
│           │           │     VAE Decoder + 逆Gamma     │            │
│           │           └──────────────┬───────────────┘            │
│           │                          ▼                            │
│           │                ┌───────────────────┐                  │
│           └───────────────►│    预测虚化图像      │                  │
│                           └─────────┬─────────┘                  │
│                                     │                            │
│                                     ▼                            │
│                           ┌───────────────────┐                  │
│                           │  损失计算           │                  │
│                           │  MSE + Edge + LPIPS │                  │
│                           │  + GAN (可选)       │                  │
│                           └───────────────────┘                  │
│                                                                     │
│  可训练参数: LoRA (UNet) + 部分 VAE Encoder                         │
│  文本编码器: 固定 prompt → 预计算后删除                               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. 核心模块详解

### 4.1 基础模型 SDXL

PhyBokeh 以 **SDXL 1.0** 为基础模型，各子模型配置如下：

| 子模型 | 配置 | 说明 |
|--------|------|------|
| **UNet** | 3阶段 [320, 640, 1280]，Transformer层 [1, 2, 10]，注意力头维 [5, 10, 20]，4通道latent | 主干网络，全冻结 |
| **VAE** | 4阶段 [128, 256, 512, 512]，scaling_factor=0.13025，4通道latent | 编码器部分可训练 |
| **Text Encoder 1** | CLIP-ViT-L (768维, 12层) | 冻结，预计算后删除 |
| **Text Encoder 2** | CLIP-ViT-bigG (1280维, 32层, projection_dim=1280) | 冻结，预计算后删除 |
| **Scheduler** | DDPMScheduler (1000步, epsilon预测, scaled_linear beta) | 噪声调度器 |

#### LoRA 注入

UNet 全部参数冻结，通过 **PEFT LoRA** 注入低秩适配器：

```python
LoraConfig(
    r=8,                          # LoRA 秩
    lora_alpha=8,                 # 缩放因子 = rank
    init_lora_weights="gaussian", # 高斯初始化
    target_modules=[
        "to_q", "to_k", "to_v", "to_out.0",   # 注意力投影
        "proj_in", "proj_out",                   # Transformer块投影
        "ff.net.0.proj", "ff.net.2",            # 前馈网络
        "conv1", "conv2", "conv_shortcut",      # ResNet卷积
        "downsamplers.0.conv", "upsamplers.0.conv", # 下/上采样卷积
        "time_emb_proj",                         # 时间嵌入投影
    ],
)
```

#### VAE 部分训练

当 `--opt_vae=1` 时，仅解冻 VAE 编码器的三层：

- `encoder.conv_in`
- `encoder.mid_block`
- `encoder.conv_out`

学习率为 LoRA 学习率的 **0.1 倍**。

---

### 4.2 PISA 物理引导注意力处理器

**文件**: `PISA_attn_processor.py`

这是项目的**核心创新**，将物理光学先验注入 SDXL 的自注意力机制。

#### 4.2.1 安装位置

PISA 处理器**仅替换一个注意力块**：`unet.down_blocks.1.attentions.0`（第二个下采样阶段的第一个注意力块）。其余所有注意力块使用标准的 `AttnProcessor2_0`。

```python
def fn_recursive_attn_processor(name, module, processor):
    if hasattr(module, "set_processor") and ('down_blocks.1.attentions.0' in name.lower()):
        module.set_processor(processor)
    # 递归遍历子模块...
```

#### 4.2.2 AttnProcessorDistReciprocal 类

```python
class AttnProcessorDistReciprocal:
    def __init__(self, hard=3, supersampling_num=4, segment_num=4, train=True):
        self.hard = hard               # sigmoid 硬度，训练中递增
        self.supersampling_num = 4      # 超采样数（遮挡计算精度）
        self.segment_num = 4            # 光线采样段数
        self.train = train              # 训练/推理模式
        self.pisa_strength = 1.0        # PISA 混合比例
```

#### 4.2.3 处理流程

当调用注意力处理器时，完整流程如下：

**Step 1: 分辨率对齐**

将 `disp_coc`（2通道：视差 + 散焦强度）下采样到 UNet 特征图分辨率，同时保留3倍分辨率的高精度版本用于遮挡计算。

```python
len_hidden = int((disp_coc.shape[-1]*disp_coc.shape[-2]/hidden_states.shape[1])**0.5)
shape = [disp_coc.shape[-2]//len_hidden, disp_coc.shape[-1]//len_hidden]
disp_highres = F.interpolate(disp_coc, size=(shape[0]*3, shape[1]*3))
disp_coc = F.interpolate(disp_coc, size=shape)
```

**Step 2: 距离矩阵计算**

基于像素归一化坐标，构建**倒数距离矩阵**，乘以 `cutoff` 参数（默认 51.2）：

```python
index_i, index_j = torch.meshgrid(...)  # 归一化到 [-1+ε, 1-ε]
reci_dist_matrix = sqrt((i[:,None]-i[None,:])² + (j[:,None]-j[None,:])²) * cutoff
```

物理意义：该矩阵度量了图像中任意两个像素位置之间的"光学距离"，用于后续弥散圆掩码的计算。

**Step 3: 遮挡图 (Occlusion Map) 计算**

对于每个像素，沿该像素到相机的光线采样 `segment_num` 个点，每个点做 `supersampling_num` 次抖动超采样。判断光线路径上是否有前景物体遮挡：

```python
# 光线上均匀采样
ps = torch.linspace(0.1, 0.9, segment_num)
# 计算采样点的2D投影位置
P_locs = ((1-disp_ps) * disp_ravel) / (disp_ps * (1-disp_ravel)) * (index_ij_diff) + index_ij
# 超采样抖动
P_locs += (rand-0.5) * 2/shape[0]  # 抖动量与分辨率成反比
# 通过 grid_sample 查询实际视差
actual_disp_ps = F.grid_sample(disp, P_locs)
# 判断是否被遮挡：如果路径上某点的实际视差 > 采样视差，说明有遮挡
occ_map = mean(((actual_disp_ps - disp_ps > 0).sum(axis=1) > 0).float())
```

输出：`occ_map` 形状为 `(B, 1, H, W)`，值域 [0, 1]，表示每个像素作为"key"被遮挡的概率。

**Step 4: 自定义注意力计算 (customized_scaled_dot_product_attention)**

```python
def customized_scaled_dot_product_attention(query, key, value, weight_matrix, disp_coc,
                                            hard, train, occ_map, pisa_strength):
```

核心逻辑分为三条路径：

**(a) 标准 SDXL 注意力先验 (Vanilla)**：

```python
attn_weight_qk = Q @ K^T * scale_factor          # 标准QK点积
attn_weight_qk_exp = exp(attn_weight_qk - max)    # 数值稳定
attn_weight_vanilla = qk_exp / sum(qk_exp)        # softmax 归一化
```

**(b) 物理引导注意力 (PISA)**：

```python
# 弥散圆掩码：sigmoid(hard * (defocus_strength - distance_matrix))
# 当 defocus_strength > distance 时，像素在弥散圆内，贡献大
attn_weight_manual = sigmoid(hard * (disp_coc[:,1] - weight_matrix))

# 物理先验 × QK 分布
attn_weight_pisa = qk_exp * attn_weight_manual
attn_weight_pisa = attn_weight_pisa / sum(attn_weight_pisa)
```

物理含义：`disp_coc[:,1]` 是散焦强度（与弥散圆半径成正比），`weight_matrix` 是像素间的光学距离。当散焦强度大于距离时，两个像素处于同一弥散圆内，注意力权重增大。

**(c) 课程学习混合**：

```python
attn = pisa_strength * attn_pisa + (1 - pisa_strength) * attn_vanilla
attn = attn / sum(attn)  # 重新归一化
attn = attn * (1 - occ_map)  # 遮挡屏蔽
```

#### 4.2.4 PISA 课程策略

- `pisa_strength` 从 `pisa_ratio_start`（默认 1.0）线性衰减到 `pisa_ratio_end`（默认 0.0）
- **训练初期**：纯物理引导，模型从物理先验学习
- **训练后期**：纯 SDXL 学习注意力，模型已内化物理先验

```python
def compute_pisa_strength(global_step, max_steps, start_ratio, end_ratio):
    progress = min(max(global_step / (max_steps - 1), 0.0), 1.0)
    return start_ratio + (end_ratio - start_ratio) * progress
```

#### 4.2.5 硬度退火

`hard` 参数控制 sigmoid 的陡峭程度，每次前向传播自增 1：

```python
if self.hard < 1e6:
    self.hard += 1  # 逐步锐化弥散圆边界
```

- 训练初期 `hard` 较小 → sigmoid 平滑 → 弥散圆边界模糊
- 训练后期 `hard` 增大 → sigmoid 陡峭 → 弥散圆边界清晰
- **推理时** `hard=1e7` → sigmoid 接近阶跃函数 → 最清晰的 CoC 边界

#### 4.2.6 disp_coc 条件输入格式

`disp_coc` 是一个 **2 通道**张量，形状 `(B, 2, H, W)`：

| 通道 | 含义 | 值域 |
|------|------|------|
| 通道 0 | 视差 (disparity) | [0, 1]，近处大远处小 |
| 通道 1 | 散焦强度 × K / 10 | ≥ 0，值越大虚化越强 |

通过 UNet 的 `cross_attention_kwargs` 传递到 PISA 处理器。

---

### 4.3 经典物理渲染器

**目录**: `classical_renderer/`

训练时在线生成物理准确的虚化 GT 图像，无需预配对数据。

#### 4.3.1 MPI 光线追踪渲染器 (mpi_multi_reverse.py)

**类**: `ModuleRenderRT` — 主要训练渲染器

**原理**: Multi-Plane Image (MPI) + 光线追踪。对每个像素，在镜头光圈上均匀采样，追踪每条光线穿过各 MPI 层，进行前到后 alpha 合成。

**输入格式**：

| 参数 | 形状 | 说明 |
|------|------|------|
| `images` | `(B, L, 3, H, W)` | L层图像（前景、背景等），**线性空间** |
| `alphas` | `(B, L, 1, H, W)` | L层透明度 |
| `coffs` | `(B, L, 3)` | 每层平面系数 (a, b, c)，深度 = c/(1-ax-by) |
| `K` | `(B,)` | 虚化半径参数 |
| `depth_focus` | `(B,)` | 对焦深度 |
| `Eta` | `(B,)` | 渐晕参数（猫眼效应） |
| `samples_per_side` | int | 光圈采样网格边长（默认 51） |

**输出**：

| 参数 | 形状 | 说明 |
|------|------|------|
| `bokeh_cum` | `(B, 4, H, W)` | 累积 RGB + Alpha |
| `weight_cum` | `(B, 1, H, W)` | 累积权重（用于归一化） |

**CUDA 内核核心逻辑**：

```
对每个输出像素 (x, y):
    对光圈上每个采样点 (xs[i], ys[i]):
        检查渐晕（光学暗角）
        对每个 MPI 层 l (从前到后):
            计算光线与层平面的交点 (x_map, y_map)
            双线性插值获取颜色和透明度
            前到后 alpha 合成
            若当前层完全不透明 (alpha > 0.99), 停止追踪
    累积 bokeh_cum 和 weight_cum
```

**深度模型**：采用 3 系数平面模型 `depth = c / (1 - a·x - b·y)`，允许深度在图像平面上线性变化。

#### 4.3.2 Scatter 渲染器 (scatter.py)

**类**: `ModuleRenderScatter` — 用于推理验证

**原理**: 散射（scatter）而非收集（gather）。每个像素根据其散焦半径将颜色散射到邻域像素。

**输入**：

| 参数 | 形状 | 说明 |
|------|------|------|
| `image` | `(B, 3, H, W)` | 输入图像 |
| `defocus` | `(B, 1, H, W)` | 带符号的散焦图 |

**核心逻辑**：每个像素向半径内的邻居加权散射，权重为 `(0.5 + 0.5·tanh(4·(R-d))) / (R²+0.2)`，其中 R 是散焦半径，d 是到中心距离。同时通过 `atomicMax` 进行散焦膨胀（dilation）。

#### 4.3.3 其他渲染器

- `scatter_ex.py`: 支持可调多边形光圈（`poly_sides`, `init_angle`）
- `mpi_multi_reverse_qzr.py`: 支持逐像素**椭圆度**和**角度**图，模拟画面边缘的猫眼虚化

---

### 4.4 自定义 Tiled SDXL Pipeline

**文件**: `custom_diffusers/pipeline_sdxl.py`

**类**: `TiledStableDiffusionXLPipeline`

为低显存推理设计的分块去噪 Pipeline，核心修改：

#### 分块策略

- 默认 `TILE_SIZE = 64`（latent 空间），对应图像空间 512×512
- 当 latent 总面积 ≤ 2 × TILE_SIZE² 时，退化为非分块处理
- 分块时，块间有 75% 重叠（步长 = TILE_SIZE × 3/4），避免块边界伪影

#### 分块去噪流程

```
1. 计算所有 tile 的左上角坐标 (cropped_height_coord, cropped_width_coord)
2. 为每个 tile 计算对应的 add_time_ids（含裁剪坐标）
3. 为每个 tile 裁剪对应的 disp_coc 局部区域
4. 对每个 timestep:
    buffer_noise_pred = zeros
    gaussian_kernel = 2D 高斯权重
    对每个 tile:
        UNet(tile_latent, t, disp_coc_local, time_ids_local) × gaussian_kernel
        累积到 buffer_noise_pred
    noise_pred = buffer_noise_pred / gaussian_accumulation
5. Scheduler step 更新 latent
```

高斯权重核确保块中心贡献大、边缘贡献小，实现平滑拼接。

#### 兼容注意力处理器

`custom_diffusers/attention_processor.py` 中的 `AttnProcessor2_0` 是标准 SDXL 注意力的包装器，额外接受 `shape`, `cutoff`, `disp_coc` 参数但忽略它们，避免非 PISA 注意力块报错。

---

### 4.5 数据集模块

**文件**: `dataset.py`

#### 4.5.1 OnTheFlyDataset（训练用）

**核心思想**：无需预配对的 AIF-Bokeh 数据，训练对**在线合成**。

**完整数据生成流程**：

```
1. 随机选取前景 (RGBA) 和背景 (RGB) 图像
2. 紧密裁剪前景 → 随机缩放 → 随机位置合成到背景上
   → 生成 AIF 图像 (syn_image)
3. 为背景和前景分别生成随机平面深度:
   - 深度模型: depth = c / (1 - a·x - b·y)
   - 前景视差强制 > 背景视差（前景更近）
4. 随机选择对焦面（前景均值或背景均值视差）
5. 计算自适应模糊半径:
   K = max(4, round(12 × W/768))
   times = 2^(random×5-2)           # 随机因子 [0.25, 64]
   K_adapt = K / max(disp_focus, 1-disp_focus) × times
6. MPI 光线追踪渲染器生成物理 GT 虚化图像
7. 裁剪到 --resolution 大小（随机裁剪或中心裁剪）
8. 数据增强：水平/垂直翻转、随机亮度缩放
```

**输出字典**：

| Key | 形状 | 值域 | 说明 |
|-----|------|------|------|
| `pixel_values` | `(C, H, W)` | [-1, 1] | 虚化 GT 图像 |
| `aif` | `(C, H, W)` | [-1, 1] | 全聚焦输入图像 |
| `disparity` | `(1, H, W)` | [0, 1] | 视差图 |
| `defocus_strength` | `(1, H, W)` | [-1, 1] | 散焦强度（带符号） |
| `K` | scalar | ≥ 0 | 归一化虚化强度 K/10 |
| `original_size` | (W, H) | - | 原始尺寸 |
| `crop_top_left` | (y, x) | - | 裁剪偏移 |

**关键细节**：
- 渲染在**线性空间**进行（gamma=2.2），渲染后转回 sRGB
- 渲染画布比最终裁剪区域大 `4K` 像素（每边 2K），避免边缘效应
- `times` 的范围使虚化强度变化跨越约 8 个数量级

#### 4.5.2 TestDataset（推理用）

支持多种数据组织格式：

| 格式 | 目录结构 | 深度图格式 | 掩码 |
|------|---------|-----------|------|
| **EBB** | `input/`, `depth/`, `mask/` | .npy/.png/.jpg 智能匹配 | .png/.jpg |
| **folder** | 扁平目录 | `*_pred.npy` | `mask_portrait.jpg` |
| **pngdepth** | `original/`, `depth/` | PNG | 可选 |
| **blb** | 特定 benchmark | EXR | JSON参数 |

**对焦面估计**：

1. **有掩码时**：取掩码区域视差的中位数作为对焦视差
2. **有 Bokeh GT 时**：通过 Canny 边缘检测找到 AIF 与 Bokeh GT 的公共清晰边缘，取其视差中位数
3. 掩码为空时，默认使用图像中心区域

**散焦强度计算**：

```python
defocus_strength = disparity - focused_points  # 带符号
```

---

### 4.6 损失函数

#### 4.6.1 MSE 损失（主损失）

```python
loss_mse = mean((pred_im - bokeh_image)²)
loss = loss_mse * lambda_l2    # lambda_l2 默认 1.0
```

图像在 [-1, 1] 范围内计算。PSNR 监控：`PSNR = -10·log10(MSE/4)`。

#### 4.6.2 多尺度边缘损失（可选 `--edge`）

提取预测图像与 GT 在多个尺度（1x, 1/2x, 1/3x）的梯度差异，以 GT 和 AIF 的最大边缘幅度加权：

```python
nabla_pred = pred[..., :-1] - pred[..., 1:]
nabla_gt = gt[..., :-1] - gt[..., 1:]
nabla_aif = aif[..., :-1] - aif[..., 1:]
weight = 1 + max(|nabla_gt|, |nabla_aif|)
loss_edge = mean(|nabla_pred - nabla_gt| × weight)
```

设计动机：在边缘区域（尤其是虚化边缘）施加更强约束。

#### 4.6.3 LPIPS 感知损失（可选 `--lpips`）

```python
loss_lpips = LPIPS_VGG(pred_im, bokeh_image).mean()
loss += loss_lpips * lambda_lpips  # lambda_lpips 默认 0.1
```

使用 VGG 网络的 LPIPS 度量，保持训练中 VGG 冻结。

#### 4.6.4 GAN 判别器损失（可选 `--lambda_gan > 0`）

```python
# 生成器损失：欺骗判别器
lossG = Disc(0.5 + 0.5*pred, for_G=True).mean()
loss += lossG * lambda_gan       # lambda_gan 默认 0.1

# 判别器损失：区分真假
lossD_real = Disc(0.5 + 0.5*GT, for_real=True).mean()
lossD_fake = Disc(0.5 + 0.5*pred.detach(), for_real=False).mean()
loss_disc = lossD_real + lossD_fake
```

GAN 在 `gan_step`（默认 1000）步后加入，使用 vision-aided-gan 判别器。

---

### 4.7 工具模块

#### 4.7.1 wavelet_fix.py — 颜色修复工具

| 函数 | 功能 |
|------|------|
| `adain_color_correction` | AdaIN 颜色校正：匹配目标图像到源图像的均值和标准差 |
| `wavelet_color_fix` | 小波颜色修复：分解高频（内容）和低频（颜色），用源低频+目标高频重建 |
| `guided_filter` | O(1) 导向滤波（灰度/彩色版本），用于边缘保持平滑 |
| `sliding_filter` | 滑窗多项式颜色匹配，带高斯加权 |

#### 4.7.2 utils_zcx.py — 评估指标

| 函数 | 功能 |
|------|------|
| `ssim` / `MS_SSIM` | 结构相似度 / 多尺度结构相似度（PyTorch 实现） |
| `make_doubly_stochastic` | Sinkhorn 归一化：使矩阵双随机（行和=列和=1） |

#### 4.7.3 optimization.py — 学习率调度

支持：linear, cosine, cosine_with_restarts（含 `power` 衰减参数）, polynomial, constant, constant_with_warmup, piecewise_constant。

---

## 5. 训练流程详解

### 5.1 初始化阶段

```
1. 加载 SDXL 全部组件
   ├── Tokenizer 1 & 2
   ├── Text Encoder 1 & 2 (冻结)
   ├── VAE (冻结，可选部分训练)
   ├── UNet (冻结 + LoRA 注入)
   └── DDPMScheduler

2. 安装注意力处理器
   ├── 所有 UNet 块 → AttnProcessor2_0
   └── down_blocks.1.attentions.0 → AttnProcessorDistReciprocal

3. 创建 OnTheFlyDataset + DataLoader

4. 预计算文本嵌入
   prompt = "an excellent photo with a large aperture"
   ├── encoder_hidden_states_1 (CLIP-ViT-L, 倒数第二层)
   ├── encoder_hidden_states_2 (CLIP-ViT-bigG, 倒数第二层)
   └── encoder_output_2 (pooled output, 1280维)

5. 删除文本编码器和分词器（节省显存）
```

### 5.2 单步训练循环

```
对每个 batch:

    ┌─ 1. VAE 编码 ──────────────────────────────────────────┐
    │  aif_image (AIF, [-1,1])                                │
    │  → gamma_correction(aif, gamma=1.0) → float32           │
    │  → VAE.encode() → latent_dist.mode() × scaling_factor   │
    │  → input_latents (4CH, H/8×W/8)                         │
    └──────────────────────────────────────────────────────────┘

    ┌─ 2. 构建条件 ──────────────────────────────────────────┐
    │  disp_coc = cat([disparity, defocus_strength × K], dim=1)│
    │  add_time_ids = [original_size, crop_top_left, target_size]│
    │  encoder_hidden_states = cat([text_enc1, text_enc2], dim=-1)│
    │  pisa_strength = compute_pisa_strength(step, max_steps)  │
    └──────────────────────────────────────────────────────────┘

    ┌─ 3. 固定多步去噪 [499, 300, 100] ────────────────────┐
    │  x = input_latents                                       │
    │  for t in [499, 300, 100]:                               │
    │      model_pred = UNet(x, t, text_emb, disp_coc, pisa)  │
    │      α = alphas_cumprod[t]                               │
    │      β = 1 - α                                           │
    │      x = (x - √β × model_pred) / √α                     │
    │  pred_latent = x                                          │
    └──────────────────────────────────────────────────────────┘

    ┌─ 4. VAE 解码 ─────────────────────────────────────────┐
    │  pred_latent / scaling_factor → VAE.decode()              │
    │  → gamma_correction(pred, 1/gamma) → pred_im [-1,1]      │
    └──────────────────────────────────────────────────────────┘

    ┌─ 5. 损失计算与反向传播 ───────────────────────────────┐
    │  loss = λ_L2 × MSE(pred, GT)                             │
    │       + edge_loss (可选)                                  │
    │       + λ_lpips × LPIPS(pred, GT) (可选)                 │
    │       + λ_gan × G_loss (可选, step>gan_step)             │
    │  accelerator.backward(loss)                               │
    │  gradient clipping (max_grad_norm=1.0)                    │
    │  optimizer.step() + lr_scheduler.step()                   │
    │  (若 opt_vae: optimizer_vae.step() + lr_scheduler_vae)    │
    └──────────────────────────────────────────────────────────┘
```

### 5.3 检查点保存

每 `checkpointing_steps` 步保存：

```
checkpoint-{global_step}/
├── pytorch_lora_weights.safetensors   # LoRA 权重
└── vae.ckpt                           # 可训练 VAE 参数（仅 requires_grad=True 的参数）
```

### 5.4 日志与验证

- 每 `validation_steps` 步保存对比图（预测 | GT | AIF 并排）
- 每 20 步记录 loss、PSNR、LPIPS、学习率到 TensorBoard
- `hard` 参数自动递增（每次 UNet 前向传播 +1）

---

## 6. 推理流程详解

### 6.1 初始化

```
1. 加载 SDXL 基础模型 + 训练好的 LoRA + VAE 权重
2. 安装 PISA 处理器（推理配置）:
   hard=1e7, supersampling_num=5, segment_num=7, train=False
3. 创建 TestDataset + DataLoader
```

### 6.2 单张图像推理

```
对每张测试图像:
    对每个虚化强度 [0.2, 0.4, 0.6, 0.8, 1.0]:

        ┌─ 1. 构建条件 ─────────────────────────────────────┐
        │  defocus_strength = base_defocus × blur_strength   │
        │  amplify = K × upsample / 10                       │
        │  disp_coc = cat([disparity, |defocus| × amplify]) │
        └─────────────────────────────────────────────────────┘

        ┌─ 2. VAE 编码 ─────────────────────────────────────┐
        │  aif_image → VAE.encode() → latents × scaling      │
        └─────────────────────────────────────────────────────┘

        ┌─ 3. 固定 3 步去噪 [499, 300, 100] ───────────────┐
        │  for t in [499, 300, 100]:                          │
        │      pred = UNet(latents, t, text_emb, disp_coc)   │
        │      latents = scheduler.step(pred, t, latents)     │
        └─────────────────────────────────────────────────────┘

        ┌─ 4. VAE 解码 ─────────────────────────────────────┐
        │  latents / scaling → VAE.decode() → image [0,1]    │
        │  → clamp + ×255 → uint8 → PIL.Image                │
        │  → resize 回原始分辨率 → JPEG 保存                   │
        └─────────────────────────────────────────────────────┘

    输出: {filename}_blur0.20.jpg, _blur0.40.jpg, ..., _blur1.00.jpg
```

### 6.3 Tiled 推理（大图像）

当图像分辨率较大时，Pipeline 自动启用分块推理：

- 将 latent 分成 TILE_SIZE=64 的块
- 每个块独立通过 UNet，结果用高斯核加权融合
- 每个块使用局部裁剪的 `disp_coc` 和对应的 `add_time_ids`

---

## 7. 数据准备流程

**文件**: `prepare_data.py`

自动为测试图像生成深度和掩码：

```
1. 深度/视差预测
   使用 Depth-Anything-V2 (Small/Base/Large)
   → 预测单目深度
   → 归一化到 [0, 1]
   → 视差 = 1 - normalized_depth

2. 显著性掩码预测
   使用 BiRefNet (ZhengPeng7/BiRefNet)
   → 预测肖像/显著物体掩码
```

---

## 8. 关键设计决策与创新点

### 8.1 非标准扩散训练范式

**传统扩散训练**：加噪 → 预测噪声 → 去噪损失

**PhyBokeh**：AIF 编码 → 多步确定性变换 → 图像空间重建损失

这是"编码器式单步预测"范式，固定 3 步去噪不是从纯噪声出发，而是从 AIF 图像的 latent 出发，逐步注入虚化效果。

### 8.2 在线合成消除配对数据需求

MPI 光线追踪器从任意前景+背景合成物理准确的虚化训练对，使系统**自监督**于未配对的 RGBA 前景和背景图像。

### 8.3 最小干预的最大效果

PISA 处理器**仅替换一个注意力块**（`down_blocks.1.attentions.0`），就实现了物理先验的有效注入。这是一个有针对性的最小干预。

### 8.4 课程学习策略

- **PISA 强度衰减**：从纯物理到纯学习，平滑过渡
- **硬度退火**：从模糊边界到清晰边界
- **GAN 延迟引入**：前 1000 步不用 GAN，等基础模型稳定

### 8.5 固定文本条件

prompt "an excellent photo with a large aperture" 在训练开始时预计算并缓存，文本编码器随后被删除以节省显存。模型不依赖文本条件变化，文本仅提供"大光圈照片"的语义锚点。

---

## 9. 超参数配置

### 训练超参数（train.sh 典型配置）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `pretrained_model_name_or_path` | `./sdxl-base` | SDXL 基础模型路径 |
| `resolution` | 384 | 训练裁剪尺寸 |
| `render_base` | 384/768 | 在线合成画布尺寸 |
| `train_batch_size` | 1 | 批大小 |
| `max_train_steps` | 500 | 最大训练步数 |
| `learning_rate` | 1e-4 | LoRA 学习率 |
| `lr_scheduler` | cosine | 学习率调度 |
| `mixed_precision` | fp16 | 混合精度 |
| `rank` | 8 | LoRA 秩 |
| `opt_vae` | 0/1 | 是否训练 VAE 编码器 |
| `lambda_l2` | 1.0 | MSE 损失权重 |
| `lambda_lpips` | 0.1 | LPIPS 损失权重 |
| `lambda_gan` | 0.1 | GAN 损失权重（0=关闭） |
| `pisa_ratio_start` | 1.0 | PISA 初始混合比例 |
| `pisa_ratio_end` | 0.0 | PISA 最终混合比例 |
| `gamma` | 1.0 | Gamma 校正值 |
| `low_vram` | False | 低显存模式 |
| `gradient_checkpointing` | True | 梯度检查点 |
| `max_grad_norm` | 1.0 | 梯度裁剪阈值 |

### 推理超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `K` | 20 | 光圈参数，越大越模糊 |
| `upsample` | 1 | 上采样因子（推理前放大） |
| `blur_strength_list` | [0.2, 0.4, 0.6, 0.8, 1.0] | 虚化强度列表 |
| `hard` | 1e7 | 推理时 sigmoid 硬度 |
| `supersampling_num` | 5 | 推理时超采样数 |
| `segment_num` | 7 | 推理时光线采样段数 |
| `FIXED_DENOISE_TIMESTEPS` | [499, 300, 100] | 固定去噪时间步 |
