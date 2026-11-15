# 九格 4B KDA 算子与通信优化

面向 8×BI-V150 / CoreX 5.0 的 Megatron 九格 4B 真实 CPT 训练。衡量完整参数更新的延迟与吞吐，保留原生路径和严格数值验证。

**当前状态：B6 在固定 TP8 CPT 合同下 KEEP；上传许可待人工确认。不是 128 卡最终验收，也不是通用 Megatron 加速包。**

## 实际结果

同一组两轮交错 10+5 实验：B0 **8.48338 s**，B5 **7.96850 s**，B6 **7.15443 s/step**。B6 延迟降低 **15.67%**，吞吐提升 **18.58%**，达到 **1145.02 tokens/s**。

50 步 loss/grad norm 一致，第 25/50 步全量模型、FP32 主参数和 Adam 状态哈希一致；完整 checkpoint 恢复通过，**需要兼容补丁**。详见 [实验摘要](reports/EXPERIMENT_SUMMARY.md)。

## 五项优化

| 组件 | 修改范围 |
|---|---|
| GQA subgroup | TP8 全组 QKV gather 改为已推导并验证的两 rank 子组通信；反向使用相应 reduce-scatter |
| RoPE | 自定义 Triton F/B 与受控 cos/sin 缓存；仅在 TP8+GQA 组合启用 |
| SwiGLU | 保留原生编译器生成的算术 kernel，减少 host 包装；不是新算术 kernel |
| AdamW | FP32 主参数及 moments 等价融合更新，保留 step、参数组和回退 |
| 残差加法 | 零 dropout、无 bias 条件下减少编译/AOT host 包装，继续原生 add/autograd |

## 目录

- `src/`：独立实现及独立 backend adapter，保留现有模块名，避免无关重构。
- `adapters/jiuge_training/`：Attention/GQA/RoPE 前端接入模块。
- `configs/`：冻结训练合同、实际前端快照哈希和固定依赖版本。
- `patches/`：前端 ModuleSpec 接入与 Megatron checkpoint 兼容补丁。
- `tests/`、`scripts/`：静态审查、外部运行目录构建、导入、轻量正确性、训练及对照。
- `docs/`：复现、来源和发布限制；`reports/`：脱敏摘要与证据哈希。

不包含 Megatron/组内训练前端完整仓库、权重、数据、checkpoint、trace 或历史日志。

## 快速开始

先取得获授权的外部依赖，详见 [REPRODUCING.md](docs/REPRODUCING.md)。使用平台定制环境，不通过 pip 安装官方 NVIDIA Torch 替换它。

```bash
python -m unittest discover -s tests -p test_repository.py
python scripts/prepare.py --jiuge "$JIUGE_SNAPSHOT" --megatron "$MEGATRON" --work "$WORK"
python scripts/check_environment.py --work "$WORK"
python scripts/run.py --work "$WORK" --model "$MODEL_DIR" --data "$DATA_DIR" \
  --out "$OUTPUT_ROOT/B6" --backend B6 --dry-run
```

`--backend B0` 为原生数学路径，`B5` 关闭新增残差优化，`B6` 全组合开启。去掉 `--dry-run` 才启动真实八卡训练；启动前强制检查占用。所有输出和临时编译文件位于仓库外。

```bash
python tests/test_operators.py --work "$WORK"  # 先确认八卡空闲；单卡小规模测试
python scripts/run.py --work "$WORK" --model "$MODEL_DIR" --data "$DATA_DIR" \
  --out "$OUTPUT_ROOT/B0" --backend B0
```

新包装入口完成导入/配置/补丁及轻量测试后，不等同重新完成完整训练性能验收。原始性能来源和覆盖范围以报告为准。

## 边界

BF16、seq512、MBS1、GBS16、TP8/PP1/DP1、AdamW distributed optimizer、recompute=true。Local Spec 包含组内已有 eager Attention，不因 CLI 为 local 就称全部未经修改。RMSNorm、Attention cast/mask 不启用。PP2/PP4 仅历史两步 fallback；LoRA 未验证。

源码 guard、shape/dtype、编译 ABI 与固定版本都有约束；不满足 guard 时回退，未知训练合同必须先显式关闭组件。组内快照及厂商编译器输出的上传权利尚待确认，Private 仓库不豁免发布限制。见 [来源与许可](docs/SOURCE_AND_PERMISSIONS.md)。
