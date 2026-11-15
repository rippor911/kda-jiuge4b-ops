# 九格4B KDA 算子与通信优化实验摘要

## 研究目标

在国产 BI-V150 上针对 Megatron 九格4B真实训练负载优化算子、通信和 host 调用，以完整参数更新延迟及吞吐评价收益，而非仅展示单 kernel 加速。

## 实验环境

8×BI-V150（32 GiB/卡），CoreX 5.0，定制 PyTorch 2.10.0；FM9G-4B（62层），TP8/PP1/DP1、BF16、seq512、MBS1、GBS16；Adam/AdamW、distributed optimizer、activation recompute。Local Spec 基线含组内实际 eager Attention；未启用 IXTE、CoreX Apex 或厂商 Flash Attention。Megatron 固定为 `5be9626709af2722333bf54797c954c09edeada3`。

## 实验方法

真实训练 profiling → 识别瓶颈 → Codex/KDA 自动优化 → 独立正确性验证 → Megatron 集成 → 端到端配对实验 → KEEP/NO_KEEP。

| B6 组件 | 实际修改 |
|---|---|
| GQA subgroup | 按真实 QKV 分片将 TP8 gather 缩小到两 rank 子组；对应 backward reduce-scatter 等价 |
| RoPE | 自定义 Triton 前反向与受控 cos/sin 缓存，保持舍入；仅 TP8+GQA 组合启用 |
| SwiGLU host | 复用原生编译算术 kernel，缩短 host 调用；不宣称新算术 kernel |
| AdamW fused | 融合 FP32 master、动量及参数更新，保留 bias correction、step 和数值语义 |
| 残差 host | 零 dropout、无 bias 时用原生 add/autograd 绕过编译/AOT 包装；不减少训练计算 |

各组件保留开关、guard/fallback。RMSNorm、Attention cast/mask 未取得可接受综合收益，维持 NO_KEEP。

## 性能结果

同一组最终 B6 配对实验，前10次参数更新预热，第11–15次完整更新同步计时取均值；通过两轮交错运行复测，profiler 关闭。

| 版本 | 平均 step time |
|---|---:|
| 原生 B0 | 8.48338 s |
| B5 | 7.96850 s |
| B6 | 7.15443 s |

B6 相比 B0：**延迟降低15.67%，吞吐提升约18.58%，处理吞吐1145.02 tokens/s**。未拼接历史不同轮次数据。两次复测属于描述性波动验证，不作为严格置信区间。

B5→B6 边际降低约814 ms。匹配 trace 中总 kernel、collective、recompute 数量一致；每步去掉3968次 compiled forward 和1984次 compiled backward 包装。残差 GPU 时间没有改善，支持收益主要来自 host dispatch 缩短；插桩约带来两倍开销，不能用 profiler 时间逐毫秒解释正式收益。

## 正确性与可靠性

- B0/B6各50步、八卡400对逐步 loss 和 grad norm 差异均为0。
- 第25、50步模型参数、FP32主参数、Adam状态全量哈希一致；其他步检查关键梯度与采样状态，不宣称全部层全部梯度逐步全量证明。
- 完整保存、退出、加载、续训 PASS：B6→B6、B0→B6、B6→B0，补充 B5→B6。必须使用 checkpoint 兼容补丁和 load 后安装入口。
- 计时完整性审计 PASS：同步窗口包含 forward/backward、recompute、通信、optimizer 与必要 host 开销；所有 rank 更新次数一致。
- TP4PP2、TP2PP4仅完成短训原生 fallback；LoRA尚未验证。

## 结论与局限

**B6 KEEP，达到当前 TP8 CPT 合同下的10%阶段目标；不代表最终128卡验收通过。** 依赖 CoreX 5.0、特定并行布局、Megatron接口及编译 ABI。未知合同须关闭未验证组件并重新检查，不能推广15.67%至PP、LoRA或多机。

本摘要依据已验收 `KDA_B6_RELEASE_VALIDATION`；`RESULTS.json` 保存同轮数字与原始验收证据哈希。原始数据、权重、checkpoint与trace经授权单独保管，不随仓库提交。发布许可仍待确认。
