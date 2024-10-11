import copy

from diffusers import CogVideoXPipeline
from diffusers.pipelines.cogvideo.pipeline_cogvideox import *
from diffusers.models.embeddings import PatchEmbed, CogVideoXPatchEmbed
from nexfort.utils.cost import time_cost
from diffusers.models.attention import Attention

from torch import distributed as dist
from torch import nn

from patch_parallel import (
    DistriConv2dPP,
    DistriSelfAttentionPP,
    DistriPatchEmbed,
    DistriCogVideoXTransformer3DModel,
)
from patch_parallel.base_model import BaseModule, BaseModel
from patch_parallel.utils import (
    DistriConfig,
    PatchParallelismCommManager,
)

distri_config = DistriConfig(height=480, width=720, warmup_steps=4, mode="corrected_async_gn")
print("distri_config.world_size:", distri_config.world_size)
print("distri_config.mode", distri_config.mode)


class CogVideoXPipelineDist(CogVideoXPipeline):
    def __init__(
        self,
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        vae: AutoencoderKLCogVideoX,
        transformer: CogVideoXTransformer3DModel,
        scheduler: Union[CogVideoXDDIMScheduler, CogVideoXDPMScheduler],
    ):
        super().__init__(tokenizer, text_encoder, vae, transformer, scheduler)

        if distri_config.world_size > 1 and distri_config.n_device_per_batch > 1:
            model = transformer
            print(transformer)
            for name, module in model.named_modules():
                if isinstance(module, BaseModule):
                    continue
                for subname, submodule in module.named_children():
                    # print(subname, submodule)
                    if isinstance(submodule, nn.Conv2d):
                        # pass
                        print("Replace conv2d...")
                        kernel_size = submodule.kernel_size
                        if kernel_size == (1, 1) or kernel_size == 1:
                            continue
                        wrapped_submodule = DistriConv2dPP(
                            submodule, distri_config, is_first_layer=True
                        )
                        setattr(module, subname, wrapped_submodule)

                    elif isinstance(submodule, CogVideoXPatchEmbed):
                        print("Replace PatchEmbed...")
                        wrapped_submodule = DistriPatchEmbed(submodule, distri_config)
                        # print(subname, wrapped_submodule)
                        setattr(module, subname, wrapped_submodule)
                        
                    elif isinstance(submodule, Attention):
                        print("Replace self attention...")
                        if subname == "attn1":  # self attention
                            wrapped_submodule = DistriSelfAttentionPP(
                                submodule, distri_config
                            )
                            setattr(module, subname, wrapped_submodule)
            self.transformer = DistriCogVideoXTransformer3DModel(model, distri_config)
            logger.info(
                f"Using parallelism for DiT, world_size: {distri_config.world_size} and n_device_per_batch: {distri_config.n_device_per_batch}"
            )
            device = distri_config.device
            print("distri_config.device.init:", distri_config.device)
            self.buffer_list = None
            self.output_buffer = None
            self.transformer = self.transformer.to(device)
            self.text_encoder = self.text_encoder.to(device)
            self.vae = self.vae.to(device)

        else:
            logger.info("Not using parallelism for DiT")

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    @time_cost(debug_level=1, on_gpu=True)
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 480,
        width: int = 720,
        num_frames: int = 48,
        fps: int = 8,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 6,
        use_dynamic_cfg: bool = False,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 226,
    ) -> Union[CogVideoXPipelineOutput, Tuple]:
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_frames (`int`, defaults to `48`):
                Number of frames to generate. Must be divisible by self.vae_scale_factor_temporal. Generated video will
                contain 1 extra frame because CogVideoX is conditioned with (num_seconds * fps + 1) frames where
                num_seconds is 6 and fps is 4. However, since videos can be saved at any fps, the only condition that
                needs to be satisfied is that of divisibility mentioned above.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of videos to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] instead
                of a plain tuple.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int`, defaults to `226`):
                Maximum sequence length in encoded prompt. Must be consistent with
                `self.transformer.config.max_text_seq_length` otherwise may lead to poor results.

        Examples:

        Returns:
            [`~pipelines.cogvideo.pipeline_cogvideox.CogVideoXPipelineOutput`] or `tuple`:
            [`~pipelines.cogvideo.pipeline_cogvideox.CogVideoXPipelineOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is a list with the generated images.
        """
        latents = self.stage0(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            fps=fps,
            num_inference_steps=num_inference_steps,
            timesteps=timesteps,
            guidance_scale=guidance_scale,
            use_dynamic_cfg=use_dynamic_cfg,
            num_videos_per_prompt=num_videos_per_prompt,
            eta=eta,
            generator=generator,
            latents=latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            return_dict=return_dict,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )
        output = self.stage1(
            num_frames=num_frames, fps=fps, latents=latents, output_type=output_type, return_dict=return_dict
        )

        return output

    @torch.no_grad()
    @time_cost(debug_level=1, on_gpu=True)
    def stage0(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 480,
        width: int = 720,
        num_frames: int = 48,
        fps: int = 8,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 6,
        use_dynamic_cfg: bool = False,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        return_dict: bool = True,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 226,
    ) -> torch.FloatTensor:
        if num_frames > 49:
            raise ValueError(
                "The number of frames must be less than 49 for now due to static positional embeddings. This will be updated in the future to remove this limitation."
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        height = height or self.transformer.config.sample_size * self.vae_scale_factor_spatial
        width = width or self.transformer.config.sample_size * self.vae_scale_factor_spatial
        num_videos_per_prompt = 1

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            callback_on_step_end_tensor_inputs,
            prompt_embeds,
            negative_prompt_embeds,
        )
        self._guidance_scale = guidance_scale
        self._interrupt = False

        # 2. Default call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        
        
        device = distri_config.device
        print("distri_config.device:", distri_config.device)

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        print("do_classifier_free_guidance:", do_classifier_free_guidance)

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            negative_prompt,
            do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )


        prompt_embeds = prompt_embeds.to(device)
        negative_prompt_embeds = negative_prompt_embeds.to(device)

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        self._num_timesteps = len(timesteps)

        # 5. Prepare latents.
        latent_channels = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            latent_channels,
            num_frames,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # Used to create communication buffer
        comm_manager = None
        if distri_config.n_device_per_batch > 1:
            comm_manager = PatchParallelismCommManager(distri_config)
            self.transformer.set_comm_manager(comm_manager)
            for i, cog_video_x_block in enumerate(self.transformer.module.transformer_blocks):
                attn1 = cog_video_x_block.attn1
                attn1.set_comm_manager(comm_manager)

        b, n, c, h, w = latents.shape
        if distri_config.split_scheme == 'row':
            split_dim = 3
        elif distri_config.split_scheme == 'col':
            split_dim = 4

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Create rotary embeds if required
        image_rotary_emb = (
            self._prepare_rotary_positional_embeddings(height, width, latents.size(1), device)
            if self.transformer.config.use_rotary_positional_embeddings
            else None
        )

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)





        latent_model_input_pre = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input_pre = self.scheduler.scale_model_input(latent_model_input_pre, timesteps[0])
        t_pre = copy.deepcopy(timesteps[0])

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timestep_pre = t_pre.expand(latent_model_input_pre.shape[0])

        self.transformer.set_counter(0)
        self.transformer(
            hidden_states=latent_model_input_pre,
            encoder_hidden_states=prompt_embeds,
            timestep=timestep_pre,
            return_dict=False,
        )[0]

        # TODO(lixiang)
        # Only used for creating the communication buffer
        # self.transformer.set_counter(0)
        # pipeline.transformer(**static_inputs, return_dict=False, record=True)
        print("comm_manager.numel", comm_manager.numel)
        if comm_manager.numel > 0:
            print("comm_manager.numel...")
            comm_manager.create_buffer()

        self.transformer.set_counter(0)
        self.transformer(
            hidden_states=latent_model_input_pre,
            encoder_hidden_states=prompt_embeds,
            timestep=timestep_pre,
            return_dict=False,
        )[0]







        with self.progress_bar(total=num_inference_steps) as progress_bar:
            # for DPM-solver++
            old_pred_original_sample = None
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                # predict noise model_output
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep,
                    image_rotary_emb=image_rotary_emb,
                    return_dict=False,
                )[0]
                noise_pred = noise_pred.float()

                # print("comm_manager.numel", comm_manager.numel)
                # if comm_manager.numel > 0:
                #     comm_manager.create_buffer()

                # perform guidance
                if use_dynamic_cfg:
                    self._guidance_scale = 1 + guidance_scale * (
                        (1 - math.cos(math.pi * ((num_inference_steps - t.item()) / num_inference_steps) ** 5.0)) / 2
                    )
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                print("all_gather start...")
                if self.output_buffer is None:
                    self.output_buffer = torch.empty((b, n, c, h, w), device=noise_pred.device, dtype=noise_pred.dtype)
                if self.buffer_list is None:
                    self.buffer_list = [torch.empty_like(noise_pred.view(-1)) for _ in range(distri_config.world_size)]
                dist.all_gather(self.buffer_list, noise_pred.contiguous().view(-1), async_op=False)
                buffer_list = [buffer.view(noise_pred.shape) for buffer in self.buffer_list]
                torch.cat(buffer_list, dim=split_dim, out=self.output_buffer)
                # torch.cat(self.buffer_list[: distri_config.n_device_per_batch], dim=0, out=self.output_buffer[0:1])
                # torch.cat(self.buffer_list[distri_config.n_device_per_batch :], dim=0, out=self.output_buffer[1:2])
                noise_pred = self.output_buffer
                print("all_gather done...")

                # compute the previous noisy sample x_t -> x_t-1
                if not isinstance(self.scheduler, CogVideoXDPMScheduler):
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                else:
                    latents, old_pred_original_sample = self.scheduler.step(
                        noise_pred,
                        old_pred_original_sample,
                        t,
                        timesteps[i - 1] if i > 0 else None,
                        latents,
                        **extra_step_kwargs,
                        return_dict=False,
                    )
                latents = latents.to(prompt_embeds.dtype)

                # call the callback, if provided
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        # self.counter += 1
        return latents

    @torch.no_grad()
    @time_cost(debug_level=1, on_gpu=True)
    def stage1(
        self,
        num_frames: int = 48,
        fps: int = 8,
        latents: Optional[torch.FloatTensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
    ) -> Union[CogVideoXPipelineOutput, Tuple]:
        if not output_type == "latent":
            print(f"latents device {latents.device} shape {latents.shape} dtype {latents.dtype}")
            video = self.decode_latents(latents)
            video = self.video_processor.postprocess_video(video=video, output_type=output_type)
        else:
            video = latents

        # ----------------- changed below -----------------#
        # Offload all models
        # self.maybe_free_model_hooks()
        # ----------------- changed above-----------------#

        if not return_dict:
            return (video,)

        return CogVideoXPipelineOutput(frames=video)
