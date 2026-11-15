# 来源、许可与上传门禁

**当前状态：LOCAL_REVIEW_ONLY。未找到覆盖本次源码与报告的 GitHub 上传许可，不执行 push。**

| 材料 | 来源 | 本仓库处理 |
|---|---|---|
| GQA/Attention、MLP adapter | KDA 路线修改，部分派生自固定 Megatron 方法 | 标明来源，附 Megatron 原许可；项目成果发布仍需确认 |
| RoPE、AdamW、残差 host 实现 | 已验收 B6 候选 | 记录原始/导出 SHA256，不改数学；成果归属与上传许可待确认 |
| SwiGLU native_forward/backward | 定制 CoreX/PyTorch 编译器输出 | 必需 ABI 依赖，不宣称自研算术；需确认厂商输出可再分发 |
| 组内 jiuge-training native 快照 | 获授权本地实验副本，非干净公开 commit | 不整库上传；仅接入 delta patch，快照通过授权渠道外部提供 |
| 实验报告与指标 | B6 验收记录 | 仅脱敏摘要和证据哈希；指标是否可发布仍需项目方确认 |

`LICENSES/` 保留相关第三方许可证，但不把它解释为整个研究项目的上传批准。不自行给所有成果授予开源许可。

需要维护者确认：这些候选/派生 adapter/接入与恢复补丁/编译器输出/脱敏指标允许上传指定的 GitHub Private 仓库，且不违反课题或单位限制。之后仍需核验目标仓库访问、Private 可见性及写权限。

已排除：凭据、个人账号/手机号/学号、服务器访问地址、权重、数据、checkpoint、梯度/Adam 原始张量、trace、缓存、虚拟环境、二进制和原始运行日志。脚本扫描是启发式；提交者还需按文件白名单人工审阅暂存区及完整历史。
