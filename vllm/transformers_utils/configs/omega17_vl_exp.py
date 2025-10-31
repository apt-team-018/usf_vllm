# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Omega17 VL Exp model configuration"""

from typing import Optional

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation


class Omega17VLExpTextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Omega17VLExpTextModel`]. It is used to instantiate a
    Omega17-VL-EXP model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 151936):
            Vocabulary size of the Omega17Exp model.
        hidden_size (`int`, *optional*, defaults to 2048):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 5632):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 16):
            This is the number of key_value heads that should be used to implement Grouped Query Attention.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 128000):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        decoder_sparse_step (`int`, *optional*, defaults to 1):
            The frequency of the MoE layer.
        moe_intermediate_size (`int`, *optional*, defaults to 1408):
            Intermediate size of the routed expert.
        num_experts_per_tok (`int`, *optional*, defaults to 4):
            Number of selected experts.
        num_experts (`int`, *optional*, defaults to 60):
            Number of routed experts.
        mlp_only_layers (`List[int]`, *optional*, defaults to `[]`):
            Indicate which layers use Omega17VLExpMLP rather than Omega17VLExpSparseMoeBlock
        rope_theta (`float`, *optional*, defaults to 5000000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings.
        head_dim (`int`, *optional*):
            The dimension of the head. If not specified, will default to `hidden_size // num_attention_heads`.
    """

    model_type = "omega17_vl_exp_text"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: Optional[int] = 151936,
        hidden_size: Optional[int] = 2048,
        intermediate_size: Optional[int] = 5632,
        num_hidden_layers: Optional[int] = 24,
        num_attention_heads: Optional[int] = 16,
        num_key_value_heads: Optional[int] = 16,
        hidden_act: Optional[str] = "silu",
        max_position_embeddings: Optional[int] = 128000,
        initializer_range: Optional[float] = 0.02,
        rms_norm_eps: Optional[float] = 1e-6,
        use_cache: Optional[bool] = True,
        tie_word_embeddings: Optional[bool] = False,
        attention_bias: Optional[bool] = False,
        attention_dropout: Optional[float] = 0.0,
        decoder_sparse_step: Optional[int] = 1,
        moe_intermediate_size: Optional[int] = 1408,
        num_experts_per_tok: Optional[int] = 4,
        num_experts: Optional[int] = 60,
        mlp_only_layers: Optional[list[int]] = None,
        rope_theta: Optional[float] = 5000000.0,
        rope_scaling: Optional[dict] = None,
        head_dim: Optional[int] = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.head_dim = head_dim or hidden_size // num_attention_heads
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling

        # Validate the correctness of rotary position embeddings parameters
        rope_config_validation(self)

        # MoE arguments
        self.decoder_sparse_step = decoder_sparse_step
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.mlp_only_layers = [] if mlp_only_layers is None else mlp_only_layers

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


class Omega17VLExpVisionConfig(PretrainedConfig):
    """Omega17 VL Exp Vision Configuration"""

    model_type = "omega17_vl_exp"
    base_config_key = "vision_config"

    def __init__(
        self,
        depth=27,
        hidden_size=1152,
        hidden_act="gelu_pytorch_tanh",
        intermediate_size=4304,
        num_heads=16,
        in_channels=3,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=3584,
        num_position_embeddings=2304,
        deepstack_visual_indexes=None,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.depth = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.out_hidden_size = out_hidden_size
        self.num_position_embeddings = num_position_embeddings
        self.initializer_range = initializer_range
        self.deepstack_visual_indexes = (
            deepstack_visual_indexes
            if deepstack_visual_indexes is not None
            else [8, 16, 24]
        )


class Omega17VLExpConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Omega17VLExpModel`]. It is used to instantiate a
    Omega17-VL-EXP model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`Union[PretrainedConfig, dict]`, *optional*, defaults to `Omega17VLExpTextConfig`):
            The config object or dictionary of the text backbone.
        vision_config (`Union[PretrainedConfig, dict]`,  *optional*, defaults to `Omega17VLExpVisionConfig`):
            The config object or dictionary of the vision backbone.
        image_token_id (`int`, *optional*, defaults to 151655):
            The image token index to encode the image prompt.
        video_token_id (`int`, *optional*, defaults to 151656):
            The video token index to encode the image prompt.
        vision_start_token_id (`int`, *optional*, defaults to 151652):
            The start token index to encode the image prompt.
        vision_end_token_id (`int`, *optional*, defaults to 151653):
            The end token index to encode the image prompt.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie the word embeddings.
    """

    model_type = "omega17_vl_exp"
    sub_configs = {
        "vision_config": Omega17VLExpVisionConfig,
        "text_config": Omega17VLExpTextConfig,
    }
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id=151655,
        video_token_id=151656,
        vision_start_token_id=151652,
        vision_end_token_id=151653,
        tie_word_embeddings=False,
        **kwargs,
    ):
        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()
        else:
            self.vision_config = vision_config

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            self.text_config = self.sub_configs["text_config"]()
        else:
            self.text_config = text_config

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        super().__init__(**kwargs, tie_word_embeddings=tie_word_embeddings)


__all__ = ["Omega17VLExpConfig", "Omega17VLExpTextConfig", "Omega17VLExpVisionConfig"]