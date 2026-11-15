# 复现与 checkpoint 恢复

## 1. 外部依赖与版本

1. 平台提供的 8×BI-V150、CoreX 5.0.0、PyTorch `2.10.0+corex.5.0.0` 及匹配 Triton/驱动/通信库；不要替换为 NVIDIA wheel，也不要清空 `LD_LIBRARY_PATH`。
2. Megatron commit `5be9626709af2722333bf54797c954c09edeada3` 的干净个人 checkout。完整仓库不随本项目提交。
3. jiuge-training 的公开基准 commit 为 `9fabd71df4897f08267df5431aa480ab9ef70a75`；**实际使用获授权归档的组内 native 快照，存在工作树修改，不能仅 checkout 此 commit 代替它**。联系项目维护者通过获授权渠道取得快照；`configs/dependencies.lock.json` 对全部 Python 文件逐一校验。没有匹配快照则构建失败，不偷偷降级。
4. 获授权的 FM9G-4B HF 目录、与原实验相同的已处理 CPT 数据及 tokenizer/config；权重与数据均不上传。数据目录格式由快照中的 `data.py` 定义，使用其已审计预处理流程，不用 synthetic data 代替正式数据。每次运行记录数据文件哈希；不同数据不得复用报告加速比。

设置 `JIUGE_SNAPSHOT`、`MEGATRON`、`WORK`、`MODEL_DIR`、`DATA_DIR`、`OUTPUT_ROOT` 为自己的个人持久化路径。`WORK` 和输出目录必须在本 Git 仓库外，且每次使用新的运行名。

将 `XDG_CACHE_HOME/HF_HOME/TORCH_HOME/TRITON_CACHE_DIR/TORCHINDUCTOR_CACHE_DIR/TMPDIR` 指向个人缓存；如果平台需要 `TRITON_LIBCUDA_PATH`，使用平台已验证的路径。环境动态库设置沿用平台启动脚本，不写入仓库凭据。

## 2. 准备与预检

```bash
python scripts/prepare.py --jiuge "$JIUGE_SNAPSHOT" --megatron "$MEGATRON" --work "$WORK"
python scripts/check_environment.py --work "$WORK"
python -m unittest discover -s tests -p test_repository.py
python tests/test_operators.py --work "$WORK"
```

prepare 仅在新外部目录复制前端源码、建立个人 Megatron clone，应用 `jiuge-b6.patch` 与 `megatron-checkpoint-compat.patch`，验证前端与已验收版本逐字节一致。生成的 SwiGLU 文件只移除服务器路径注释，Python AST 不变，对应 dispatch 哈希同步更新并记录原始/导出哈希。模型算术和 guard 不重写。

小规模测试检查 synthetic BDA/SwiGLU F/B/recompute 和多步 AdamW 参数/moments/step 字节一致。GQA 八 rank、RoPE 真实 workload、完整训练及跨后端恢复已有历史验收；本次整理不重复全部 GPU 测试。轻量测试不代替完整训练验收。

## 3. 正式 10+5 与数值对照

```bash
python scripts/run.py --work "$WORK" --model "$MODEL_DIR" --data "$DATA_DIR" --out "$OUTPUT_ROOT/B0-r1" --backend B0
python scripts/run.py --work "$WORK" --model "$MODEL_DIR" --data "$DATA_DIR" --out "$OUTPUT_ROOT/B6-r1" --backend B6
python scripts/run.py --work "$WORK" --model "$MODEL_DIR" --data "$DATA_DIR" --out "$OUTPUT_ROOT/B6-r2" --backend B6
python scripts/run.py --work "$WORK" --model "$MODEL_DIR" --data "$DATA_DIR" --out "$OUTPUT_ROOT/B0-r2" --backend B0
python scripts/compare_runs.py "$OUTPUT_ROOT/B0-r1" "$OUTPUT_ROOT/B6-r1"
```

同一模型初始权重、数据顺序、seed=1234、参数组和优化器。前10步预热，11–15步同步 wall window 取最慢 rank /5；包含计算、通信、optimizer 与必要 host 开销。profiler 关闭。比较 `steps-rank*.json` 中 synchronized_window；不要把单个 CPU enqueue 时间或诊断快照耗时当正式性能。

formal 首步保存运行时 RNG，供后续诊断使用。RNG/checkpoint 属于可信本地输入，勿加载不可信 pickle。

## 4. 完整恢复与50步诊断

```bash
python scripts/run.py --work "$WORK" --model "$MODEL_DIR" --data "$DATA_DIR" --out "$OUTPUT_ROOT/B6-cont4" --backend B6 --mode diagnostic --steps 4 --rng-reference "$OUTPUT_ROOT/B0-r1"
python scripts/run.py --work "$WORK" --model "$MODEL_DIR" --data "$DATA_DIR" --out "$OUTPUT_ROOT/B6-save2" --backend B6 --mode diagnostic --steps 2 --save --rng-reference "$OUTPUT_ROOT/B0-r1"
python scripts/run.py --work "$WORK" --model "$MODEL_DIR" --data "$DATA_DIR" --out "$OUTPUT_ROOT/B6-resume4" --backend B6 --mode diagnostic --steps 4 --load "$OUTPUT_ROOT/B6-save2/checkpoints"
python scripts/compare_runs.py "$OUTPUT_ROOT/B6-cont4" "$OUTPUT_ROOT/B6-resume4"
```

同样用 B0 保存/B6 恢复、B6 保存/B0 恢复作交叉对照。加载后继续命中 fused backend 的次数由 runner 检查；不能只检查 checkpoint 文件存在。新完整 checkpoint 约54 GiB，预留额外空间。

`--mode diagnostic --steps 50 --rng-reference ...` 分别运行 B0/B6，再 `compare_runs.py`。保留原恒定 LR/WD 的240样本调度元数据，仅延长循环；每步保存关键梯度及采样参数/状态，25/50步全量模型/master/moments。哈希拷贝有显著开销，不用于 benchmark。

checkpoint 补丁修复：逐参数 CPU step 独立 clone；分布式参数 shard 不保存 step 时保留 metadata 恢复值；zero-worker single DataLoader iterator 重建不消耗已恢复 CPU RNG。前端 patch 保留正常受控退出；observer 在 load 后安装优化器。缺少补丁、改为多 worker 或其他格式均不继承恢复 PASS。

## 5. 提交与发布

```bash
python scripts/review_export.py
python scripts/review_export.py --history
```

扫描只是技术审查，不是发布许可。先取得代码/报告/编译器输出的上传批准，再确认 GitHub owner/repo/Private 与写权限。仅通过获授权 `gh` 或专用 SSH 登录，remote 不含凭据；不 force push。当前发布状态见 `docs/SOURCE_AND_PERMISSIONS.md`。
