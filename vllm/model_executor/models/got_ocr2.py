from functools import partial
from typing import Iterable, List, Mapping, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import Qwen2Config

from vllm.attention.backends.abstract import AttentionMetadata
from vllm.config import VllmConfig
from vllm.inputs import INPUT_REGISTRY, DecoderOnlyInputs, token_inputs
from vllm.inputs.registry import DummyData, InputContext
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from vllm.model_executor.models import SupportsMultiModal
from vllm.model_executor.models.qwen2 import Qwen2Model
from vllm.model_executor.models.utils import (is_pp_missing_parameter,
                                              maybe_prefix)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import (MULTIMODAL_REGISTRY, MultiModalData,
                             MultiModalKwargs)
from vllm.multimodal.utils import cached_get_tokenizer
from vllm.sequence import IntermediateTensors, SequenceData

DEFAULT_IMAGE_PATCH_TOKEN = "<imgpad>"
DEFAULT_IMAGE_PATCH_TOKEN_ID = 151859
DEFAULT_IM_START_TOKEN = "<img>"
DEFAULT_IM_START_TOKEN_ID = 151857
DEFAULT_IM_END_TOKEN = "</img>"
DEFAULT_IM_END_TOKEN_ID = 151858
IMAGE_PLACEHOLDER_LENGTH = 256
IMAGE_SIZE = 1024
IMAGE_PLACEHOLDER_TOKENS = [DEFAULT_IMAGE_PATCH_TOKEN_ID
                            ] * IMAGE_PLACEHOLDER_LENGTH


class MLPBlock(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


# This class and its supporting functions below lightly adapted from the ViTDet backbone available at: https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py # noqa
class ImageEncoderViT(nn.Module):

    def __init__(
            self,
            img_size: int = 1024,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.0,
            out_chans: int = 256,
            qkv_bias: bool = True,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            act_layer: Type[nn.Module] = nn.GELU,
            use_abs_pos: bool = True,
            use_rel_pos: bool = False,
            rel_pos_zero_init: bool = True,
            window_size: int = 0,
            global_attn_indexes: Tuple[int, ...] = (),
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
        """ # noqa: E501
        super().__init__()
        self.img_size = img_size

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size,
                            embed_dim))

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )

        self.net_2 = nn.Conv2d(256,
                               512,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               bias=False)
        self.net_3 = nn.Conv2d(512,
                               1024,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        x = self.neck(x.permute(0, 3, 1, 2))
        x = self.net_2(x)
        x = self.net_3(x)

        return x


class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""  # noqa: E501

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """ # noqa: E501
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else
            (window_size, window_size),
        )

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim,
                            mlp_dim=int(dim * mlp_ratio),
                            act=act_layer)

        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x


class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """ # noqa: E501
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                input_size is not None
            ), "Input size must be provided if using relative positional encoding."  # noqa: E501
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(
                torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(
                torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = (self.qkv(x).reshape(B, H * W, 3, self.num_heads,
                                   -1).permute(2, 0, 3, 1, 4))
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h,
                                          self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = ((attn @ v).view(B, self.num_heads, H, W,
                             -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1))
        x = self.proj(x)

        return x


def window_partition(x: torch.Tensor,
                     window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """ # noqa: E501
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size,
               window_size, C)
    windows = (x.permute(0, 1, 3, 2, 4,
                         5).contiguous().view(-1, window_size, window_size, C))
    return windows, (Hp, Wp)


def window_unpartition(
    windows: torch.Tensor,
    window_size: int,
    pad_hw: Tuple[int, int],
    hw: Tuple[int, int],
) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """ # noqa: E501
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size,
                     window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int,
                rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1,
                                                  max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords -
                       k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """ # noqa: E501
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] +
            rel_w[:, :, :, None, :]).view(B, q_h * q_w, k_h * k_w)

    return attn


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x


def build_GOT_vit_b():
    return _build_vary(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
    )


def _build_vary(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
):
    prompt_embed_dim = 256
    # image_size = 1024
    vit_patch_size = 16
    image_encoder = ImageEncoderViT(
        depth=encoder_depth,
        embed_dim=encoder_embed_dim,
        img_size=IMAGE_SIZE,
        mlp_ratio=4,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=encoder_num_heads,
        patch_size=vit_patch_size,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=encoder_global_attn_indexes,
        window_size=14,
        out_chans=prompt_embed_dim,
    )
    return image_encoder


class GOTImageEvalProcessor:

    def __init__(self, image_size=384, mean=None, std=None):
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)

        self.normalize = transforms.Normalize(mean, std)

        self.transform = transforms.Compose([
            transforms.Resize(
                (image_size, image_size),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            self.normalize,
        ])

    def __call__(self, item):
        return self.transform(item)


class GOTConfig(Qwen2Config):
    model_type = "GOT"


class GOTOCR2Model(Qwen2Model):
    config_class = GOTConfig

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        self.vision_tower_high = build_GOT_vit_b()
        self.mm_projector_vary = nn.Linear(1024, 1024, bias=True)

    def merge_embeddings(self, input_ids, inputs_embeds, images):
        image_features = []
        for image in images:
            image = image.unsqueeze(0)
            cnn_feature = self.vision_tower_high(image)
            cnn_feature = cnn_feature.flatten(2).permute(0, 2, 1)  # 256*1024
            image_feature = self.mm_projector_vary(cnn_feature)
            image_features.append(image_feature)

        # 修改By YiJiang
        if inputs_embeds.dim() == 2:
            batch_size = len(image_features)
            input_ids = input_ids.view(batch_size, -1)
            NB, D = inputs_embeds.shape
            inputs_embeds = inputs_embeds.view(batch_size, NB // batch_size, D)

        dummy_image_features = torch.zeros(256,
                                           1024,
                                           device=inputs_embeds.device,
                                           dtype=inputs_embeds.dtype)
        use_im_start_end = True
        new_input_embeds = []

        for cur_input_ids, cur_input_embeds, cur_image_features in zip(
                input_ids, inputs_embeds, image_features):
            if (cur_input_ids == DEFAULT_IMAGE_PATCH_TOKEN_ID).sum() == 0:
                cur_input_embeds = (cur_input_embeds +
                                    (0.0 * dummy_image_features).sum())
                new_input_embeds.append(cur_input_embeds)
                continue

            if use_im_start_end:
                if (cur_input_ids == DEFAULT_IM_START_TOKEN_ID).sum() != (
                        cur_input_ids == DEFAULT_IM_END_TOKEN_ID).sum():
                    raise ValueError(
                        "The number of image start tokens and image end tokens should be the same."  # noqa: E501
                    )

                image_start_tokens = torch.where(
                    cur_input_ids == DEFAULT_IM_START_TOKEN_ID)[0]
                for image_start_token_pos, per_cur_image_features in zip(
                        image_start_tokens, cur_image_features):
                    per_cur_image_features = per_cur_image_features.to(
                        device=cur_input_embeds.device)
                    num_patches = per_cur_image_features.shape[0]

                    if (cur_input_ids[image_start_token_pos + num_patches + 1]
                            != DEFAULT_IM_END_TOKEN_ID):
                        raise ValueError(
                            "The image end token should follow the image start token."  # noqa: E501
                        )

                    cur_input_embeds = torch.cat(
                        (
                            cur_input_embeds[:image_start_token_pos + 1],
                            per_cur_image_features,
                            cur_input_embeds[image_start_token_pos +
                                             num_patches + 1:],
                        ),
                        dim=0,
                    )

                new_input_embeds.append(cur_input_embeds)
            else:
                raise NotImplementedError

        inputs_embeds = torch.stack(new_input_embeds, dim=0)

        # 修改 By YiJiang
        if inputs_embeds.dim() == 3:
            B, N, D = inputs_embeds.shape
            inputs_embeds = inputs_embeds.reshape(N * B, D)

        return inputs_embeds

    def forward(
        self,
        input_ids: Tensor,
        positions: Tensor,
        kv_caches: List[Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        images = kwargs.pop("images", None)
        inputs_embeds = self.embed_tokens(input_ids)

        vision_tower_high = getattr(self, "vision_tower_high", None)
        if vision_tower_high is not None and images is not None:
            images = images.to(dtype=inputs_embeds.dtype)
            inputs_embeds = self.merge_embeddings(input_ids, inputs_embeds,
                                                  images)

        return super().forward(
            input_ids=input_ids,
            positions=positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )


got_qwen_processor = None


def mm_input_mapper_for_got_qwen(
    ctx: InputContext,
    data: MultiModalData[object],
) -> MultiModalKwargs:
    global got_qwen_processor
    if got_qwen_processor is None:
        got_qwen_processor = GOTImageEvalProcessor(image_size=IMAGE_SIZE)
    batch_data = {"images": got_qwen_processor(data)}
    return MultiModalKwargs(batch_data)


def dummy_data_for_got_qwen(ctx: InputContext, seq_len: int,
                            mm_counts: Mapping[str, int]) -> DummyData:
    num_images = mm_counts["image"]
    pad_token_length = seq_len - IMAGE_PLACEHOLDER_LENGTH * num_images - 2
    if pad_token_length < 0:
        raise RuntimeError(
            f"GOT-OCR2 cannot process {num_images} images in a prompt, "
            "please increase max_model_len or reduce image limit by "
            "--limit-mm-per-prompt.")

    seq_data = SequenceData.from_prompt_token_counts(
        (DEFAULT_IM_START_TOKEN_ID, 1),
        (DEFAULT_IMAGE_PATCH_TOKEN_ID, IMAGE_PLACEHOLDER_LENGTH * num_images),
        (DEFAULT_IM_END_TOKEN_ID, 1),
        (0, pad_token_length),
    )

    image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=0)
    mm_data = {"image": image if num_images == 1 else [image] * num_images}

    return DummyData(seq_data, mm_data)


def input_processor_for_got_qwen(ctx: InputContext, inputs: DecoderOnlyInputs):
    # insert DEFAULT_IM_START_TOKEN/DEFAULT_IM_END_TOKEN and expand pad tokens
    input_prompt_tokens_ids = inputs["prompt_token_ids"]
    assert input_prompt_tokens_ids.count(DEFAULT_IMAGE_PATCH_TOKEN_ID) == 1
    pad_token_index = input_prompt_tokens_ids.index(
        DEFAULT_IMAGE_PATCH_TOKEN_ID)

    model_config = ctx.model_config
    tokenizer = cached_get_tokenizer(
        model_config.tokenizer,
        trust_remote_code=model_config.trust_remote_code)

    new_input_prompt_tokens_ids = [
        *input_prompt_tokens_ids[:pad_token_index],
        DEFAULT_IM_START_TOKEN_ID,
        *IMAGE_PLACEHOLDER_TOKENS,
        DEFAULT_IM_END_TOKEN_ID,
        *input_prompt_tokens_ids[pad_token_index + 1:],
    ]

    prompt = inputs.get("prompt")
    if prompt is None:
        prompt = tokenizer.decode(new_input_prompt_tokens_ids)
    return token_inputs(
        prompt_token_ids=new_input_prompt_tokens_ids,
        prompt=prompt,
        multi_modal_data=inputs["multi_modal_data"],
    )


def get_got_qwen_max_image_tokens(ctx: InputContext) -> int:
    assert ctx.model_config.max_model_len > 258
    return ctx.model_config.max_model_len


@MULTIMODAL_REGISTRY.register_image_input_mapper(mm_input_mapper_for_got_qwen)
@MULTIMODAL_REGISTRY.register_max_image_tokens(get_got_qwen_max_image_tokens)
@INPUT_REGISTRY.register_dummy_data(dummy_data_for_got_qwen)
@INPUT_REGISTRY.register_input_processor(input_processor_for_got_qwen)
class GOTQwenForCausalLM(nn.Module, SupportsMultiModal):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        cache_config = vllm_config.cache_config
        multimodal_config = vllm_config.model_config.multimodal_config
        assert (not cache_config.enable_prefix_caching
                ), "GOTQwen currently does not support prefix caching"

        self.config = config
        self.quant_config = quant_config
        self.multimodal_config = multimodal_config

        self.model = GOTOCR2Model(vllm_config=vllm_config,
                                  prefix=maybe_prefix(prefix, "model"))
        if config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = get_sampler()
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        # print(f"input_ids in model forward: {input_ids}")
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
            intermediate_tensors=intermediate_tensors,
            **kwargs,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
