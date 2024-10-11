# adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/embeddings.py
import torch

from diffusers.models.embeddings import CogVideoXPatchEmbed, get_2d_sincos_pos_embed
from .base_module import BaseModule
from .utils import DistriConfig
from .logger import init_logger

logger = init_logger(__name__)


class DistriPatchEmbed(BaseModule):
    def __init__(self, module: CogVideoXPatchEmbed, distri_config: DistriConfig):
        super(DistriPatchEmbed, self).__init__(module, distri_config)

    def forward(self, text_embeds: torch.Tensor, image_embeds: torch.Tensor):
        r"""
        Args:
            text_embeds (`torch.Tensor`):
                Input text embeddings. Expected shape: (batch_size, seq_length, embedding_dim).
            image_embeds (`torch.Tensor`):
                Input image embeddings. Expected shape: (batch_size, num_frames, channels, height, width).
        """
        # Replace the original patch_embedding with the distri_patch_embedding
        module = self.module
        distri_config = self.distri_config

        text_embeds = module.text_proj(text_embeds)

        batch, num_frames, channels, height, width = image_embeds.shape
        image_embeds = image_embeds.reshape(-1, channels, height, width)
        image_embeds = module.proj(image_embeds)
        image_embeds = image_embeds.view(batch, num_frames, *image_embeds.shape[1:])
        image_embeds = image_embeds.flatten(3).transpose(2, 3)  # [batch, num_frames, height x width, channels]
        image_embeds = image_embeds.flatten(1, 2)  # [batch, num_frames x height x width, channels]

        print("image_embeds.shape:", image_embeds.shape)
        print("text_embeds.shape:", text_embeds.shape)
        embeds = torch.cat(
            [text_embeds, image_embeds], dim=1
        ).contiguous()  # [batch, seq_length + num_frames x height x width, channels]

        if module.use_positional_embeddings:
            print("use positional embedding...")
            print("module.pos_embedding.shape:", module.pos_embedding.shape)
            pre_time_compression_frames = (num_frames - 1) * module.temporal_compression_ratio + 1
            if (
                module.sample_height != height
                or module.sample_width != width
                or module.sample_frames != pre_time_compression_frames
            ):
                print("need to update positional embedding...")
                pos_embedding = module._get_positional_embeddings(height, width, pre_time_compression_frames)
                pos_embedding = pos_embedding.to(embeds.device, dtype=embeds.dtype)
            else:
                pos_embedding = module.pos_embedding

            print("pos_embedding.shape:", pos_embedding.shape)

            seq_length = height * width * num_frames // (module.patch_size**2)
            print("seq_length:", seq_length)
            text_pos_embeds = pos_embedding[:, :module.max_text_seq_length]
            pos_embeds = pos_embedding[:, module.max_text_seq_length:module.max_text_seq_length + seq_length]
            print("pos_embedding.shape:", pos_embedding.shape)
            print("text_pos_embeds.shape:", text_pos_embeds.shape)

            print("distri_config.split_idx():", distri_config.split_idx())
            print("distri_config.n_device_per_batch:", distri_config.n_device_per_batch)
            b, c, h = pos_embeds.shape
            pos_embeds = pos_embeds.view(b, distri_config.n_device_per_batch, -1, h)[
                :, distri_config.split_idx()
            ]
            print(f"device: {torch.distributed.get_rank()}: pos_embeds: {pos_embeds.shape}")
            pos_embedding = torch.cat([text_pos_embeds, pos_embeds], dim=1)

            embeds = embeds + pos_embedding

        return embeds
 