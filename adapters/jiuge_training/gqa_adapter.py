"""Explicit adapter generated from the pinned SelfAttention method; only assembly differs."""
import os
import hashlib
from pathlib import Path
import megatron.core.transformer.attention as pinned_attention
SOURCE_COMPATIBLE = hashlib.sha256(Path(pinned_attention.__file__).read_bytes()).hexdigest() == 'a019cd00a708a22237383456a7ba85bf522201c14a5b7cb0c1e4f0f12eac9837'
import torch
from torch import Tensor
import torch.distributed as dist
from megatron.core.transformer.attention import SelfAttention, apply_module, SplitAlongDim, get_pg_rank
from .gqa_communication import pair_group, assemble

class GQASubgroupSelfAttention(SelfAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gqa_pair = None
        supported = (SOURCE_COMPATIBLE and os.environ.get('KDA_GQA_BACKEND', 'full_tp') == 'kv_pair'
                     and self.world_size == 8 and dist.get_world_size() == 8
                     and self.config.num_query_groups == 4
                     and self.config.num_attention_heads == 40
                     and self.hidden_size_per_attention_head == 64
                     and not self.config.sequence_parallel
                     and self.linear_qkv.weight.shape == (384, 2560)
                     and getattr(self.linear_qkv.weight, 'partition_stride', 1) == 1
                     and not self.linear_qkv.gather_output
                     and self.config.params_dtype == torch.bfloat16
                     and not getattr(self.config, 'attention_output_gate', False))
        if supported:
            self.gqa_pair = pair_group(self.pg_collection.tp)

    def get_query_key_value_tensors(
        self,
        hidden_states: Tensor,
        key_value_states: Tensor | None = None,
        output_gate: bool = False,
        split_qkv: bool = True,
    ) -> (
        tuple[Tensor, Tensor, Tensor, Tensor]
        | tuple[Tensor, Tensor, Tensor]
        | tuple[Tensor, list[int]]
    ):
        """
        Derives `query`, `key` and `value` tensors from `hidden_states`.
        If `output_gate` is True, then also derives `gate` tensor.
        If `split_qkv=False`, then the unsplit mixed_qkv tensor is returned.
        """
        # If no output gate: Attention heads [sq, b, h] --> [sq, b, ng * (np/ng + 2) * hn)]
        # If have output gate: Attention heads [sq, b, h] --> [sq, b, ng * (2 * np/ng + 2) * hn)]
        if self.gqa_pair is None or output_gate:
            return super().get_query_key_value_tensors(hidden_states, key_value_states, output_gate, split_qkv)
        mixed_qkv, _ = apply_module(self.linear_qkv)(hidden_states)
        num_query_heads_per_group = (
            self.num_attention_heads_per_partition // self.num_query_groups_per_partition
        )
        num_qkv_heads_per_group = num_query_heads_per_group + 2
        if output_gate:
            num_qkv_heads_per_group += num_query_heads_per_group

        assert self.config.num_query_groups is not None
        if self.config.num_query_groups < self.world_size:
            # Note that weights are interleaved in the following manner:
            # q1 q2 k1 v1 | q3 q4 k2 v2 | q5 q6 k3 v3 | ...
            # When tp_size > num_kv_heads, we split "q1 q2 k1 v1" over multiple
            # ranks, so a rank does not have a clean partitioning of just the q_heads
            # it needs. Instead, we perform the following steps:
            # 1. Assemble the full "q1 q2 k1 v1 | q3 q4 k2 v2 | q5 q6 k3 v3 | ..."
            #    through an AG.
            # 2. Pull out the right slice (e.g., "q1 q2 k1 v1" or "q3 q4 k2 v2").
            # 3. Split q_heads (e.g., q1, q2), k_heads (e.g., k1), v_heads (e.g., v1).
            # 4. Further index into query to get only the q_heads that this rank is
            #    responsible for (e.g., q1).
            # The block of code below performs steps 1 and 2.
            mixed_qkv = assemble(mixed_qkv, self.gqa_pair)

        # If no output gate: [sq, b, hp] --> [sq, b, ng, (np/ng + 2) * hn]
        # If have output gate: [sq, b, hp] --> [sq, b, ng, (2 * np/ng + 2) * hn]
        new_tensor_shape = mixed_qkv.size()[:-1] + (
            self.num_query_groups_per_partition,
            num_qkv_heads_per_group * self.hidden_size_per_attention_head,
        )
        mixed_qkv = mixed_qkv.view(*new_tensor_shape)

        # Split the tensor into query, gate, key, and value.
        if output_gate:
            if not split_qkv:
                raise ValueError("split_qkv not supported for gated attention yet.")
            # If have output gate: [sq, b, ng, (2 * np/ng + 2) * hn]
            # --> [sq, b, ng, np/ng * hn], [sq, b, ng, np/ng * hn],
            # [sq, b, ng, hn], [sq, b, ng, hn]
            split_arg_list = [
                num_query_heads_per_group * self.hidden_size_per_attention_head,
                num_query_heads_per_group * self.hidden_size_per_attention_head,
                self.hidden_size_per_attention_head,
                self.hidden_size_per_attention_head,
            ]

            if SplitAlongDim is not None:
                (query, gate, key, value) = SplitAlongDim(mixed_qkv, 3, split_arg_list)
            else:
                (query, gate, key, value) = torch.split(mixed_qkv, split_arg_list, dim=3)
        else:
            # If no output gate: [sq, b, ng, (np/ng + 2) * hn]
            # --> [sq, b, ng, np/ng * hn], None, [sq, b, ng, hn], [sq, b, ng, hn]
            split_arg_list = [
                num_query_heads_per_group * self.hidden_size_per_attention_head,
                self.hidden_size_per_attention_head,
                self.hidden_size_per_attention_head,
            ]

            # Return unsplit mixed_qkv and split_arg_list
            if not split_qkv:
                return mixed_qkv, split_arg_list

            if SplitAlongDim is not None:
                (query, key, value) = SplitAlongDim(mixed_qkv, 3, split_arg_list)
            else:
                (query, key, value) = torch.split(mixed_qkv, split_arg_list, dim=3)

        # Query [sq, b, ng, np/ng * hn] -> [sq, b, np, hn]
        query = query.reshape(query.size(0), query.size(1), -1, self.hidden_size_per_attention_head)

        if self.config.num_query_groups < self.world_size:
            # query above corresponds to (num_q_heads / num_kv_heads) q_heads.
            # Index appropriately into query to get (num_q_heads / tp_size) q_heads.
            # This is step 4 in the list of steps above.
            idx = get_pg_rank(self.pg_collection.tp) % (
                self.world_size // self.config.num_query_groups
            )
            size = self.num_attention_heads_per_partition // (
                self.world_size // self.config.num_query_groups
            )
            query = query[:, :, idx * size : (idx + 1) * size, :]

        if self.q_layernorm is not None:
            query = apply_module(self.q_layernorm)(query)

        if self.k_layernorm is not None:
            key = apply_module(self.k_layernorm)(key)

        if self.config.test_mode:
            self.run_realtime_tests()

        if output_gate:
            # Gate [sq, b, ng, np/ng * hn] -> [sq, b, np, hn]
            gate = gate.reshape(*gate.shape[:2], -1, self.hidden_size_per_attention_head)
            if self.config.num_query_groups < self.world_size:
                idx = get_tensor_model_parallel_rank() % (
                    self.world_size // self.config.num_query_groups
                )
                size = self.num_attention_heads_per_partition // (
                    self.world_size // self.config.num_query_groups
                )
                gate = gate[:, :, idx * size : (idx + 1) * size, :]
            return query, key, value, gate

        return query, key, value
