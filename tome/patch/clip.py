# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------
# import sys
# print(sys.path)
# sys.path.append('/root/data1/zmm/CLIP/clip/')
# print(sys.path)

from typing import Tuple

import torch
from torch import nn
from typing import Callable, List, Optional, Tuple


# from timm.models.vision_transformer import Attention, Block, VisionTransformer
from clip.model import  vitResidualAttentionBlock

from tome.merge import bipartite_soft_matching, merge_source, merge_wavg
from tome.utils import parse_r


class ToMeBlock(vitResidualAttentionBlock):
    """
    Modifications:
     - Apply ToMe between the attention and mlp blocks
     - Compute and propogate token size and potentially the token sources.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Note: this is copied from clip.models.vision_transformer.ResidualAttentionBlock with modifications.
        x = self.ln_1(x)
        attn_size = (
                self._tome_info["size"] if self._tome_info["prop_attn"] else None
            )
        x_attn, metric = self.attn(x, size=attn_size)
        x = x + x_attn

        r = self._tome_info["r"].pop(0)
        if r > 0:
            # Apply ToMe here
            merge, _ = bipartite_soft_matching(
                metric, #[N, C]
                r,
                self._tome_info["class_token"],
                self._tome_info["distill_token"],
            )
            if self._tome_info["trace_source"]:
                self._tome_info["source"] = merge_source(
                    merge, x.permute(1,0,2), self._tome_info["source"]
                )
            x, self._tome_info["size"] = merge_wavg(merge, x.permute(1,0,2), self._tome_info["size"])

        x = x.permute(1,0,2) + self.mlp(self.ln_2(x.permute(1,0,2)))
        return x

def linear(input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    r"""
    Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.

    This operator supports :ref:`TensorFloat32<tf32_on_ampere>`.

    Shape:

        - Input: :math:`(N, *, in\_features)` N is the batch size, `*` means any number of
          additional dimensions
        - Weight: :math:`(out\_features, in\_features)`
        - Bias: :math:`(out\_features)`
        - Output: :math:`(N, *, out\_features)`
    """
    # if has_torch_function_variadic(input, weight):
    #     return handle_torch_function(linear, (input, weight), input, weight, bias=bias)
    return torch._C._nn.linear(input, weight, bias)


def _in_projection_packed(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    b: Optional[torch.Tensor] = None,
) -> List[torch.Tensor]:
    r"""
    Performs the in-projection step of the attention operation, using packed weights.
    Output is a triple containing projection tensors for query, key and value.

    Args:
        q, k, v: query, key and value tensors to be projected. For self-attention,
            these are typically the same tensor; for encoder-decoder attention,
            k and v are typically the same tensor. (We take advantage of these
            identities for performance if they are present.) Regardless, q, k and v
            must share a common embedding dimension; otherwise their shapes may vary.
        w: projection weights for q, k and v, packed into a single tensor. Weights
            are packed along dimension 0, in q, k, v order.
        b: optional projection biases for q, k and v, packed into a single tensor
            in q, k, v order.

    Shape:
        Inputs:
        - q: :math:`(..., E)` where E is the embedding dimension
        - k: :math:`(..., E)` where E is the embedding dimension
        - v: :math:`(..., E)` where E is the embedding dimension
        - w: :math:`(E * 3, E)` where E is the embedding dimension
        - b: :math:`E * 3` where E is the embedding dimension

        Output:
        - in output list :math:`[q', k', v']`, each output tensor will have the
            same shape as the corresponding input tensor.
    """
    E = q.size(-1)
    if k is v:
        if q is k:
            # self-attention
            return linear(q, w, b).chunk(3, dim=-1)
        else:
            # encoder-decoder attention
            w_q, w_kv = w.split([E, E * 2])
            if b is None:
                b_q = b_kv = None
            else:
                b_q, b_kv = b.split([E, E * 2])
            return (linear(q, w_q, b_q),) + linear(k, w_kv, b_kv).chunk(2, dim=-1)
    else:
        w_q, w_k, w_v = w.chunk(3)
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = b.chunk(3)
        return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)

class ToMeAttention(nn.MultiheadAttention):
    """
    Modifications:
     - Apply proportional attention
     - Return the mean of k over heads from attention
    """

    def forward(
        self, x: torch.Tensor, size: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Note: this is copied from timm.models.vision_transformer.Attention with modifications.
        N, B, C = x.shape
        
        if self._qkv_same_embed_dim:
            q, k, v = _in_projection_packed(x, x, x, self.in_proj_weight, self.in_proj_bias) #[N, B, C]
                
        scale = self.head_dim**-0.5
        attn = (q.permute(1,0,2) * scale @ k.permute(1,2,0)) #[N, B, C]

        # Apply proportional attention
        if size is not None:
            attn = attn + size.log()[:, None, :, 0]

        attn = attn.softmax(dim=-1)


        x = (attn @ v.permute(1,0,2)).permute(1,0,2)

        # Return k as well here
        return x, k.mean(1)


def make_tome_class(transformer_class):
    class ToMeVisionTransformer(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """

        def forward(self, *args, **kwdargs) -> torch.Tensor:
            self._tome_info["r"] = parse_r(12, self.r)
            self._tome_info["size"] = None
            self._tome_info["source"] = None

            return super().forward(*args, **kwdargs)

    return ToMeVisionTransformer


def apply_patch(model, trace_source: bool = False, prop_attn: bool = True):
    """
    Applies ToMe to this transformer. Afterward, set r using model.r.

    If you want to know the source of each token (e.g., for visualization), set trace_source = true.
    The sources will be available at model._tome_info["source"] afterward.

    For proportional attention, set prop_attn to True. This is only necessary when evaluating models off
    the shelf. For trianing and for evaluating MAE models off the self set this to be False.
    """

    for module in model.visual.modules():
        if module.__class__.__name__ == "VisionTransformer":
            vit_class = module.__class__
    
            ToMeVisionTransformer = make_tome_class(vit_class)

            module.__class__ = ToMeVisionTransformer
            module.r = 2
            module._tome_info = {
                "r": module.r,
                "size": None,
                "source": None,
                "trace_source": trace_source,
                "prop_attn": prop_attn,
                "class_token": True,
                "distill_token": False,
            }

            if hasattr(module, "dist_token") and module.dist_token is not None:
                module._tome_info["distill_token"] = True

            for module_ in module.modules():
                if isinstance(module_, vitResidualAttentionBlock):
                    module_.__class__ = ToMeBlock
                    module_._tome_info = module._tome_info
                elif isinstance(module_, nn.MultiheadAttention):
                    module_.__class__ = ToMeAttention