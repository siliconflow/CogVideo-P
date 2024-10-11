"""

This script demonstrates how to generate a video from a text prompt using CogVideoX with 🤗Huggingface Diffusers Pipeline.

Note:
    This script requires the `diffusers>=0.30.0` library to be installed.
    If the video exported using OpenCV appears “completely green” and cannot be viewed, lease switch to a different player to watch it. This is a normal phenomenon.

Base run:
    $ python inference/cog_dist.py --prompt "A girl ridding a bike." --num_inference_steps 50

PP run:
    $ python inference/cog_dist.py --prompt "A girl ridding a bike." --model_path /data0/hf_models/CogVideoX-2b --num_inference_steps 50 --pp --pp_async

PP run with full async mode:
    $ python inference/cog_dist.py --prompt "A girl ridding a bike." --model_path /data0/hf_models/CogVideoX-2b --num_inference_steps 50 --pp --full_async

PP run with sync mode:
    $ python inference/cog_dist.py --prompt "A girl ridding a bike." --model_path /data0/hf_models/CogVideoX-2b --num_inference_steps 50 --pp --sync_run

"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "6, 7"

import argparse
import tempfile
from typing import Union, List

import queue
import PIL
import imageio
import numpy as np
import time
import torch
import torch.distributed as dist
from nexfort.utils.logging import logger

from cog_pipeline import CogVideoXPipelineDist as CogVideoXPipeline

from nexfort.distributed.group import PipelineGroupCoordinator as PipelineGroup


class Runner:
    def __init__(self, rank, pp_size, backend, args):
        self.args = args
        self.rank = rank
        self.pp = self.args.pp
        logger.debug(f"Cuda support: {torch.cuda.is_available()}, devices num {torch.cuda.device_count()}")
        torch.cuda.set_device(rank)
        self.device = torch.device(self.args.device)
        self.dtype = torch.float16

        if self.pp:
            """ Initialize the distributed environment. """
            self.pp_size = pp_size
            os.environ["MASTER_ADDR"] = "127.0.0.1"
            os.environ["MASTER_PORT"] = "29501"
            dist.init_process_group(backend, rank=rank, world_size=self.pp_size)
            self.pp_group = PipelineGroup(
                group_ranks=[[0, 1]],
                local_rank=self.rank,
                torch_distributed_backend=backend,
            )
            self.pp_group.set_recv_buffer(
                num_pipefusion_patches=1,
                patches_shape_list=[],
                feature_map_shape=[1, 13, 16, 60, 90],
                dtype=self.dtype,
            )

        # Load the pre-trained CogVideoX pipeline with the specified precision (float16) and move it to the specified device
        pipe = CogVideoXPipeline.from_pretrained(self.args.model_path, torch_dtype=self.args.dtype)
        if self.pp:
            # Pipeline model partition to save CUDA memory
            if self.rank == 0:
                pipe.text_encoder.to(self.device)
                pipe.transformer.to(self.device)
                pipe.vae = None
            elif self.rank == 1:
                pipe.text_encoder = None
                pipe.transformer = None
                pipe.vae.to(self.device)
        else:
            pass
            # pipe.to(self.device)
        self.model = pipe

        torch.cuda.synchronize()

    def finalize(self):
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    def __call__(self, task, args):
        with torch.no_grad():
            start_t = time.time()
            if not self.pp or (self.pp and self.rank == 0):
                print(f"pp stage {self.rank} run")
                print("stage 0 run")
                # prompt_embeds, _ = self.model.encode_prompt(
                #     prompt=args.prompt,  # The textual description for video generation
                #     negative_prompt=None,  # The negative prompt to guide the video generation
                #     do_classifier_free_guidance=True,  # Whether to use classifier-free guidance
                #     num_videos_per_prompt=args.num_videos_per_prompt,  # Number of videos to generate per prompt
                #     max_sequence_length=226,  # Maximum length of the sequence, must be 226
                #     device=self.device,  # Device to use for computation
                #     dtype=self.args.dtype,  # Data type for computation
                # )

                # Generate the video frames using the pipeline
                latents = self.model.stage0(
                    num_frames=self.args.num_frames,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    prompt=args.prompt,
                    generator=torch.Generator(device="cuda").manual_seed(1),
                    height = 512,
                    width = 512,
                    # negative_prompt_embeds=torch.zeros_like(prompt_embeds),  # Not Supported negative prompt
                )

            # stage 0 to stage 1 communation
            if self.pp and self.rank == 0:
                comm_start = time.time()
                self.pp_group.pipeline_send(latents)
                print(f"pp stage {self.rank} nccl comm cost {time.time() - comm_start}")

            if self.pp and self.rank == 1:
                print(f"pp stage {self.rank} run")
                comm_start = time.time()
                latents = self.pp_group.pipeline_recv()
                print(f"pp stage {self.rank} nccl comm cost {time.time() - comm_start}")

            if not self.pp or (self.pp and self.rank == 1):
                print("stage 1 run")
                video = self.model.stage1(
                    num_frames=self.args.num_frames,
                    latents=latents,
                ).frames[0]

                # Export the generated frames to a video file. fps must be 8
                if not self.args.pp:
                    export_to_video_imageio(video, f"{self.args.output_path}/output.mp4", fps=8)
                if self.args.pp and self.rank == 1:
                    # last stage of pipeline do file save
                    export_to_video_imageio(video, f"{self.args.output_path}/{task.task_id}.mp4", fps=8)

            end_t = time.time()
            logger.debug(f"rank {self.rank} run e2e elapsed: {end_t - start_t:4f} s")
        return f"rank {self.rank} finished"


def get_mem_cost(device=0):
    cuda_mem_max_used = torch.cuda.max_memory_allocated(device) / (1024**3)
    cuda_mem_max_reserved = torch.cuda.max_memory_reserved(device) / (1024**3)
    print(f"Device {device} Max used CUDA memory : {cuda_mem_max_used:.3f}GiB")
    print(f"Device {device} Max reserved CUDA memory : {cuda_mem_max_reserved:.3f}GiB")


def export_to_video_imageio(
    video_frames: Union[List[np.ndarray], List[PIL.Image.Image]], output_video_path: str = None, fps: int = 8
) -> str:
    """
    Export the video frames to a video file using imageio lib to Avoid "green screen" issue (for example CogVideoX)
    """
    if output_video_path is None:
        output_video_path = tempfile.NamedTemporaryFile(suffix=".mp4").name
    if isinstance(video_frames[0], PIL.Image.Image):
        video_frames = [np.array(frame) for frame in video_frames]
    with imageio.get_writer(output_video_path, fps=fps) as writer:
        for frame in video_frames:
            writer.append_data(frame)
    return output_video_path


def finish_callback(task_id):
    print(f"Hi, main process, task {task_id} is done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video from a text prompt using CogVideoX")
    parser.add_argument("--prompt", type=str, required=True, help="The description of the video to be generated")
    parser.add_argument(
        "--model_path", type=str, default="THUDM/CogVideoX-2b", help="The path of the pre-trained model to be used"
    )
    parser.add_argument(
        "--output_path", type=str, default="./output", help="The path where the generated video will be saved"
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=50, help="Number of steps for the inference process"
    )
    parser.add_argument("--num_frames", type=int, default=48, help="Number of video frames, max 48")
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="The scale for classifier-free guidance")
    parser.add_argument("--num_videos_per_prompt", type=int, default=1, help="Number of videos to generate per prompt")
    parser.add_argument(
        "--device", type=str, default="cuda", help="The device to use for computation (e.g., 'cuda' or 'cpu')"
    )

    parser.add_argument(
        "--dtype", type=str, default="float16", help="The data type for computation (e.g., 'float16' or 'float32')"
    )
    parser.add_argument("--pp", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--pp_async", action="store_true")
    parser.add_argument("--full_async", action="store_true")
    args = parser.parse_args()

    args.dtype = torch.float16 if args.dtype == "float16" else torch.float32

    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    print(f"CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")

    if not args.pp:
        runner = Runner(0, 1, None, args)

        torch.cuda.synchronize()
        start = time.time()
        runner(None, args)
        # runner(None, args)
        # runner(None, args)
        torch.cuda.synchronize()
        end = time.time()
        logger.warning(f"1 times run total time: {end - start:4f} s")
    else:
        from drunner import DRunner

        # Initialize the distributed runner with the Runner class and command-line arguments
        drunner = DRunner(Runner, args)

        if args.pp_async:
            # pipeline async run
            task_num = 3
            sent = 0
            got = 0
            torch.cuda.synchronize()
            start = time.time()
            while True:
                if sent < task_num and drunner.can_put():
                    task = drunner.sync_put(args, callback=finish_callback)
                    print(f"add task {task}")
                    print(f"waiting size{drunner.waiting_size()}")
                    sent += 1
                if got < task_num and drunner.can_get():
                    print(f"waiting size{drunner.done_size()}")
                    print(f"sync get {drunner.sync_get()}")
                    got += 1
                if sent == task_num and got == task_num:
                    print("all finish!")
                    break
            torch.cuda.synchronize()
            end = time.time()
            logger.warning(f"{task_num} times run total time: {end - start:4f} s")
        elif args.full_async:
            # async run
            task0 = drunner.async_put(args, callback=finish_callback)
            print(f"add task {task0}")
            print(f"waiting size{drunner.waiting_size()}")
            task1 = drunner.async_put(args, callback=finish_callback)
            print(f"add task {task1}")
            print(f"waiting size{drunner.waiting_size()}")
            task2 = drunner.async_put(args, callback=finish_callback)
            print(f"add task {task2}")
            print(f"waiting size{drunner.waiting_size()}")

            print(f"sync get {drunner.sync_get()}")
            print(f"waiting size{drunner.waiting_size()}")

            print(f"sync get {drunner.sync_get()}")
            print(f"waiting size{drunner.waiting_size()}")

            print(f"sync get {drunner.sync_get()}")
            print(f"waiting size{drunner.waiting_size()}")
        else:
            # sync run
            # warmup
            drunner(args)
            drunner.synchronize()
            # run
            torch.cuda.synchronize()
            start = time.time()
            for i in range(1):
                out = drunner(args)
                print(f"drunner out {out}")
                del out

            # sync and finalize
            drunner.synchronize()
            torch.cuda.synchronize()
            end = time.time()
            logger.warning(f"1 times run total time: {end - start:4f} s")

        drunner.synchronize()
        drunner.finalize()
