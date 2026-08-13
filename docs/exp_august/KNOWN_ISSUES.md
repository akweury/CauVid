# exp_august 主要问题整理

本文档依据 `src/exp_august/pipeline.py` 的实际调用链整理，范围覆盖 `exp_august`、其复用的 `exp_july/perception` 实现，以及 `exp_driving_videos/modules` 中被 August 激活的下游模块。

## 总览

`exp_august` 可以完成从驾驶视频到符号场景表示的 11 步线性流水线，但尚未实现目标设计中的 training-free analysis-by-synthesis loop。最重要的问题分为两类：一类影响论文主张，包括 Steps 5-9 的 world-hypothesis beam、forward verification、bounded repair 和 best-ever selection 尚未落地，以及物理状态缺少独立真实性验证；另一类影响工程可靠性，包括接口缺乏类型约束、公开阶段与真实计算边界不一致、Step 8 强制耦合评估、Step 10 尚未真正实现重要对象选择，以及多代实验代码和配置共存。

## P0-R：会削弱论文核心主张的研究方法缺口

### R1. 目标闭环尚未存在于可执行 runner 中

**现状**

目标设计把 Steps 5-9 定义为主要方法：构建 Top-K world hypotheses、前向预测其 mask/flow/depth/background signatures、诊断冲突、执行有界局部重估计，并通过统一规则保留 best-ever hypothesis。当前 `pipeline.py` 仍按一次性的 11 步顺序运行：公开 Step 5 是 ego-motion abstraction，Step 7 是 relative-motion handoff，Step 8 的局部重估计不是显式阶段，Step 9 scorer/beam/stop controller 不存在。

**影响**

- 流程图中的反馈箭头目前不是可复现算法；
- 无法进行 open-loop versus closed-loop 的核心消融；
- 无法证明修正后选择的是历史最优解释，而不是最后一次处理结果；
- 审稿人容易将系统视为 pretrained components 与手工规则的工程串联。

**建议**

- 先实现类型化 `WorldHypothesis`、immutable parent/child diff 和 diverse Top-K beam；
- 将 Step 6 forward prediction、Step 8 local solver、Step 9 feasibility/selection 做成独立可测试模块；
- 给每轮循环记录输入 beam、residual packet、repair proposal、child hypotheses、排名和停止原因；
- 在 runner 中增加明确的 iteration/compute budget，而不是在旧 Step 6 内隐式修复。

### R2. 当前验证可能形成 circular self-consistency

**现状**

现有设计主要使用同一批 mask、flow、depth 和 tracks 生成轨迹并检查轨迹。尚未实现冻结的 `EvidenceUsePlan`，也没有区分 solver 可拟合的 evidence 与只用于候选接受的 check-only evidence。

**影响**

- residual 下降可能只是模型更贴合自身输入，并不表示更接近真实 world state；
- 物理平滑可能掩盖错误 scale、pose、mask 或 identity association；
- 无法区分 `externally supported` 与 `self_consistency_only` 的结果。

**建议**

- 在 Step 3 manifest closure 后生成冻结的 `EvidenceUsePlan`；
- 使用 backward flow、未选 mask candidates、unmatched detections、固定时空验证点或 cue holdout 作为 check-only evidence；
- Step 8 optimizer 禁止读取当前 repair 的 check-only cues；
- Step 9 要求 check residual 改善或在冻结容差内不退化。

### R3. Temporal segmentation 不能单独验证真实物理状态

**现状**

最终评估重点是人工 temporal segmentation，但论文目标还包括相机/ego 运动、物体 3D 轨迹、速度、加速度和尺度。Segmentation 标签只能验证状态边界或类别，不能证明 metric trajectory 正确。

**影响**

- 即使 segmentation F1 提高，也不能推出速度或 3D trajectory 更接近真实值；
- 物理合理但错误的轨迹可能获得良好的分段结果；
- “恢复真实 world state”的论文主张缺乏直接证据。

**建议**

- 在至少一个 benchmark/subset 上加入 pose、trajectory、speed 或 scale reference；
- 报告每次 repair 的 internal-residual change 与 external-error change；
- 增加 uncertainty coverage 和按 `metric/relative/ambiguous/unobservable` 分层的结果；
- 如果无法获得物理 ground truth，将论文主张限制为 segmentation 与 evidence/physics self-consistency。

### R4. Step 4 尚未执行正式的 scale observability test

**现状**

当前实现主要是基于 bbox center、depth median 和近似焦距得到 camera-frame pseudo-3D。尚未形成相机位姿、地面、尺度候选及 covariance，也没有把输出分类为 `metric | relative | ambiguous | unobservable`。

**影响**

- 单目不可辨识情况下可能输出虚假的精确速度；
- 后续物理约束可能在错误尺度上产生看似合理的轨迹；
- uncertainty 无法反映 scale 与 camera-pose 的耦合。

**建议**

- 实现 condition/posterior-spread based observability gate；
- 保留多个 scale-conditioned hypotheses，而不是提前选择单值；
- uncertainty 必须联合传播到 Step 5 的速度、加速度和 object trajectories；
- 不可观测时输出 relative state、宽区间或 `unobservable`。

## P0：会阻止正常使用或影响实验有效性的问题

### 1. Step 8 将推理与人工标注评估强制绑定

**现状**

`temporal_video_segmentation()` 完成预测后会无条件查找人工标注目录，并立即执行 test split 评估。如果 `annotations/video_segmentation` 不存在，即使用户只希望生成预测，流水线也会抛出 `FileNotFoundError`。

**影响**

- 未携带标注的数据集无法运行到 Step 8 以后。
- 生产推理、无标签实验和评估被绑定在同一执行路径中。
- `--max-step 8` 的含义不再只是“完成时间分段”，还隐含“必须完成有标签评估”。
- 标注或报告生成失败会让已经成功得到的预测看起来像整个阶段失败。

**证据位置**

- `src/exp_august/modules.py::temporal_video_segmentation`
- `src/exp_august/evaluation.py`

**建议**

- 将评估改为显式选项，例如 `--evaluate`。
- 预测阶段只负责生成 `temporal_segments`。
- 评估和 PDF 报告作为独立、可恢复的后处理阶段运行。
- 标注不存在时应跳过评估并记录状态，而不是使预测失败。

### 2. Step 10 的“重要对象选择”尚未真正实现

**现状**

配置中明确写有：

```yaml
important_objects:
  selection_strategy: not_implemented
  passthrough_selected_objects: true
```

普通对象目前主要采用透传行为。候选对象虽然具备评分、数量限制和过滤逻辑，但这不等价于完整的重要对象选择算法。

**影响**

- Step 10 的名称容易使实验使用者误认为系统已经进行了语义重要性筛选。
- Step 11 生成的逻辑原子可能包含大量与场景决策无关的对象。
- 下游符号表示规模、噪声水平和评估结果可能受到影响。
- 论文描述若将其称为完整选择模块，可能与实现不一致。

**证据位置**

- `configs/exp_driving/default.yaml`
- `src/exp_driving_videos/modules/important_objects_driving_mini.py`
- `src/exp_august/modules.py::important_object_selection`

**建议**

- 在模块输出和实验报告中明确标记当前策略为 pass-through baseline。
- 定义可验证的重要性目标与 ground truth，或明确采用可解释的规则型选择标准。
- 为普通对象和 candidate 对象统一评分语义。
- 在实现完整策略前，可将公开步骤名改为 `important_object_candidate_handoff`，避免过度表达能力。

## P1：高维护风险或容易导致隐性错误的问题

### 3. 遗留线性流水线依赖大型、非类型化的 `state: dict`

**现状**

每一步接收并返回一个不断扩展的字典。字段包括 `videos`、`detections`、`tracks`、`positions_3d`、`ego_motion`、`relative_object_motion`、`temporal_segments` 等，但缺少统一 schema、静态类型和运行时契约验证。

截至 2026-08-13，目标 runner 的 Steps 1-3 已使用版本化 Pydantic
contracts、阶段配置哈希和带内容哈希的 `ArtifactRef`，因此这一问题已在
新路径的前三个边界得到局部解决；遗留 `pipeline.py` 及目标 Steps 4-11
仍未迁移，不能将该风险标记为整体关闭。

**影响**

- 字段拼写或结构变化通常只能在较晚阶段暴露。
- `{**state, **result}` 可能静默覆盖已有字段。
- 不同实验代际对同名字段可能有不同假设。
- 单元测试需要大量 mock 才能覆盖跨阶段契约。
- 很难判断哪些字段是必需输入、派生数据或仅用于诊断。

**证据位置**

- `src/exp_august/modules.py`
- `src/exp_july/perception/pipeline.py`

**建议**

- 为每一步定义 `TypedDict`、dataclass 或 Pydantic schema。
- 在阶段入口检查必需字段、版本和基本不变量。
- 避免无边界的字典展开，显式声明新增和允许覆盖的字段。
- 为缓存 JSON 添加 `schema_version` 和迁移策略。

### 4. 公开步骤边界与真实计算边界不一致

**现状**

August Step 6 为了执行轨迹修复，内部已经计算 July Step 8A 的相对运动；August Step 7 只是检查并公开该结果，写一个 handoff 文件，并未真正重新计算相对运动。

类似地，August Step 5 聚合了多个 July 内部步骤，底层编号与公开编号并不一致。

**影响**

- `--max-step 6` 的结果实际已经包含名义上属于 Step 7 的数据。
- 单步性能计时无法准确反映真实算法成本。
- 缓存失效和故障定位容易被错误归因。
- 用户根据公开步骤推断数据依赖时可能得到错误结论。

**证据位置**

- `src/exp_august/modules.py::trajectory_refinement`
- `src/exp_august/modules.py::relative_motion_representation`

**建议**

- 将“修复所需的预修复相对运动”和“修复后的最终相对运动”拆成不同字段与阶段。
- 或将 Step 7 明确定义为 `relative_motion_handoff`，并在文档中说明它是接口边界。
- 性能统计同时记录公开步骤和内部子步骤。

### 5. 多代实验实现被运行时交叉拼装

**现状**

August 并非独立算法实现，而是组合：

- `exp_july/perception` 的检测、跟踪、三维轨迹、自车运动和轨迹修复；
- `exp_driving_videos/modules` 的时间分段、分段运动、对象筛选和逻辑原子；
- `exp_august` 自己的数据切分、阶段映射、评估与追踪适配。

**影响**

- 修改旧实验代码可能意外改变 August 的复现结果。
- 同一函数的 July 编号、August 编号和 driving 概念名可能不同。
- 很难冻结一套真正独立、可发布的 Paper-1 实现。
- provenance 虽记录来源，但不能阻止底层实现漂移。

**证据位置**

- `src/exp_august/modules.py::_july`
- `src/exp_august/modules.py::_driving_config`
- `src/exp_july/perception/__init__.py`

**建议**

- 提取稳定的共享核心包，实验目录只保留配置与编排。
- 给被 August 使用的核心函数建立版本化公共接口。
- 对 Paper-1 发布版本冻结代码提交、配置哈希和模型版本。
- 建立真实输入的小型端到端 golden test。

### 6. 缓存契约分散且强依赖文件路径

**现状**

多个阶段支持逐视频缓存，并包含跨机器路径重定位逻辑。缓存有效性检查、指纹和 schema 规则分散在不同模块中。

**影响**

- 配置、模型或算法变化后可能错误复用旧缓存。
- JSON 内嵌绝对路径导致缓存不可自然移植。
- 不同阶段对“有效缓存”的定义不一致。
- 部分缓存可能被重定位成功，但其上游语义已经过期。

**证据位置**

- `src/exp_july/perception/pipeline.py::relocate_cached_payload`
- `src/exp_july/perception/pipeline.py::relocate_json_cache_file`
- 各阶段的 manifest 和 fingerprint 逻辑

**建议**

- 使用统一的缓存元数据结构：代码版本、schema 版本、配置哈希、模型哈希、输入指纹。
- JSON 中优先存相对 artifact URI，而不是宿主机绝对路径。
- 把缓存验证和迁移放入统一基础设施。

## P2：实验可解释性、运行体验和代码质量问题

### 7. 顶层文档与当前 August 执行链存在代际混杂

**现状**

根目录 README 主要描述旧的 18/25 步 driving/July 流水线；August 实际只有 11 个公开步骤，并明确不运行规则学习尾部。

**影响**

- 新使用者难以确定应该运行 `docker.sh`、`d2.sh` 还是 `d3.sh`。
- 大量规则学习配置容易被误认为属于 August。
- 论文实现边界不够直观。

**证据位置**

- `README.md`
- [`PIPELINE_STEPS.md`](../pipeline/PIPELINE_STEPS.md)
- `src/exp_august/README.md`
- `d2.sh`、`d3.sh`、`docker.sh`

**建议**

- 在根 README 增加“实验版本矩阵”和推荐入口。
- 明确标注 active、legacy、scaffold 和 archived 模块。
- 为 August 提供一张唯一的步骤—实现—输出目录映射表。

### 8. 大型模块承担过多职责

**现状**

`src/exp_july/perception/pipeline.py` 同时包含：

- 数据编排；
- 几何和轨迹算法；
- 缓存处理；
- 可视化和 PDF 生成；
- LLM/策略闭环；
- manifest 与审计输出。

**影响**

- 修改局部算法时回归范围过大。
- 模块加载和理解成本高。
- 可视化依赖与核心推理耦合。
- 很难进行细粒度单元测试。

**建议**

- 按 domain、I/O、cache、visualization、evaluation、orchestration 拆分。
- 将纯函数计算和有副作用的 artifact 写入分离。
- 把阶段编排限制在较薄的 adapter 层。

### 9. 遗留 Step 2 的日志处理存在特殊运行时规避逻辑

**现状**

公共 runner 为抑制底层输出而重定向 stdout/stderr，但目标检测阶段因 Ultralytics logging 可能递归进入 emergency handler，被单独排除。

**影响**

- 日志行为依赖第三方库初始化顺序。
- 真实警告可能被吞掉。
- 进度条转发依靠正则匹配终端文本，较脆弱。

**证据位置**

- `src/exp_august/pipeline.py::_SelectedTqdmStream`
- `src/exp_august/pipeline.py::_tracked`

**建议**

- 使用结构化阶段事件或 callback，而不是解析 tqdm 文本。
- 将第三方 logger 配置集中管理。
- 区分普通噪声、警告和错误，不要整体重定向到空设备。

### 10. 可选 LLM 后端可能造成实验语义差异

**现状**

有 API key 时使用模型服务；没有 API key 时使用确定性的空/default cohort 与 repair 计划。两者都能让流水线完成，但实际轨迹修复能力不同。

**影响**

- “运行成功”并不代表运行了同一种方法。
- 不同机器的结果可能因环境凭据而发生显著变化。
- 仅比较输出目录时容易忽略 backend 差异。

**证据位置**

- `src/exp_august/modules.py::_offline_refinement_generator`
- `src/exp_august/modules.py::trajectory_refinement`
- `06_trajectory_refinement/refinement_backend.json`

**建议**

- 将 backend 设为显式 CLI 参数，而不是仅根据 API key 隐式选择。
- 正式实验缺少所需后端时应 fail fast。
- 离线模式应明确命名为 baseline，并在汇总结果中突出显示。

### 11. 部分配置与当前 August 范围无关

**现状**

`configs/exp_driving/default.yaml` 同时包含时间分段、逻辑原子、规则生成、因果筛选、OD calibration、神经符号 baseline 等大量配置，而 August 只使用其中一部分。

**影响**

- 很难审计一次 August 运行究竟读取了哪些配置。
- 修改无关配置可能给使用者造成错误预期。
- 配置快照体积大且语义不清晰。

**建议**

- 新建 `configs/exp_august/default.yaml`，只包含实际使用项。
- 启动时输出 resolved config 与未使用字段报告。
- 将共享配置通过显式 include/继承组织，而不是整份复用。

## 建议的处理顺序

1. 解耦 Step 8 预测与评估，保证无标签推理可运行。
2. 明确 Step 10 是 baseline，随后实现和评估真正的重要对象选择。
3. 沿用目标 Steps 1-3 的 Pydantic 模式，为 Steps 4-11 建立版本化输入/输出 schema 和适配器。
4. 修正 Step 6/7 的真实计算边界和公开语义。
5. 提取稳定共享核心，减少 August 对实验目录实现的直接依赖。
6. 统一缓存元数据、指纹和相对路径规则。
7. 拆分 August 专用配置并整理根目录文档。
8. 最后重构日志、可视化和超大型 July pipeline 模块。

## 推荐的验收条件

- 无标注目录时，可以成功运行至 Step 11，只跳过评估。
- 每个阶段在输入字段缺失或 schema 不兼容时给出明确错误。
- 相同数据、seed、配置、模型和 backend 在不同机器上得到一致的 manifest 与核心结果。
- test 视频不会进入任何拟合、统计更新、阈值选择或策略校准。
- Step 10 输出包含明确的选择策略版本、评分依据和选择/拒绝原因。
- `--max-step N` 的运行成本和输出严格符合公开步骤定义。
- August resolved config 不包含未参与执行的规则学习配置。
- 至少存在一个真实小视频的端到端 golden test，覆盖 Step 1 至 Step 11。
