from __future__ import annotations

import base64
import logging
import math
import os
import sys
import time
import warnings
from functools import lru_cache
from io import BytesIO

import requests
import torch
import torchvision
from packaging import version
from PIL import Image
from torchvision import io, transforms
from torchvision.transforms import InterpolationMode
from typing import Optional
import torchvision.transforms.functional as F
import numpy as np


logger = logging.getLogger(__name__)

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = int(os.getenv("MAX_PIXELS", 256 * 28 * 28)) # 16384*28*28
MAX_RATIO = 200
print(f"MAX_PIXELS: {MAX_PIXELS}")
print("using qwen-vl-utils in /mnt/wxd/wangxd/VideoReasoner/src")

# VIDEO_MIN_PIXELS = 128 * 28 * 28
# VIDEO_MAX_PIXELS = 768 * 28 * 28
VIDEO_MIN_PIXELS = 128 * 28 * 28
# VIDEO_MAX_PIXELS = 128 * 28 * 28
# VIDEO_MAX_PIXELS = 196 * 28 * 28
VIDEO_MAX_PIXELS = 256 * 28 * 28
# VIDEO_MAX_PIXELS = int(os.getenv("VIDEO_MAX_PIXELS", 196*28*28)) # train-196, eval-256
print(f"VIDEO_MAX_PIXELS: {VIDEO_MAX_PIXELS}")
FRAME_FACTOR = 2
FPS = 2.0

ZOOM_FPS = int(os.getenv("ZOOM_FPS", 2))

FPS_MIN_FRAMES = 4
# FPS_MAX_FRAMES = 32
FPS_MAX_FRAMES = int(os.getenv("FPS_MAX_FRAMES", 32))

# Set the maximum number of video token inputs.
# Here, 128K represents the maximum number of input tokens for the VLLM model.
# Remember to adjust it according to your own configuration.
VIDEO_TOTAL_PIXELS = int(float(os.environ.get('VIDEO_MAX_PIXELS', 128000 * 28 * 28 * 0.9)))
logger.info(f"set VIDEO_TOTAL_PIXELS: {VIDEO_TOTAL_PIXELS}")


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def to_rgb(pil_image: Image.Image) -> Image.Image:
      if pil_image.mode == 'RGBA':
          white_background = Image.new("RGB", pil_image.size, (255, 255, 255))
          white_background.paste(pil_image, mask=pil_image.split()[3])  # Use alpha channel as mask
          return white_background
      else:
          return pil_image.convert("RGB")


def fetch_image(ele: dict[str, str | Image.Image], size_factor: int = IMAGE_FACTOR) -> Image.Image:
    if "image" in ele:
        image = ele["image"]
    else:
        image = ele["image_url"]
    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif image.startswith("http://") or image.startswith("https://"):
        response = requests.get(image, stream=True)
        image_obj = Image.open(BytesIO(response.content))
    elif image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif image.startswith("data:image"):
        if "base64," in image:
            _, base64_data = image.split("base64,", 1)
            data = base64.b64decode(base64_data)
            image_obj = Image.open(BytesIO(data))
    else:
        image_obj = Image.open(image)
    if image_obj is None:
        raise ValueError(f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")
    image = to_rgb(image_obj)
    ## resize
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=size_factor,
        )
    else:
        width, height = image.size
        min_pixels = ele.get("min_pixels", MIN_PIXELS)
        max_pixels = ele.get("max_pixels", MAX_PIXELS)
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=size_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    image = image.resize((resized_width, resized_height))

    return image


def smart_nframes(
    ele: dict,
    total_frames: int,
    video_fps: int | float,
) -> int:
    """calculate the number of frames for video used for model inputs.

    Args:
        ele (dict): a dict contains the configuration of video.
            support either `fps` or `nframes`:
                - nframes: the number of frames to extract for model inputs.
                - fps: the fps to extract frames for model inputs.
                    - min_frames: the minimum number of frames of the video, only used when fps is provided.
                    - max_frames: the maximum number of frames of the video, only used when fps is provided.
        total_frames (int): the original total number of frames of the video.
        video_fps (int | float): the original fps of the video.

    Raises:
        ValueError: nframes should in interval [FRAME_FACTOR, total_frames].

    Returns:
        int: the number of frames for video used for model inputs.
    """
    assert not ("fps" in ele and "nframes" in ele), "Only accept either `fps` or `nframes`"
    if "nframes" in ele:
        nframes = round_by_factor(ele["nframes"], FRAME_FACTOR)
    else:
        fps = ele.get("fps", FPS)
        min_frames = ceil_by_factor(ele.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR)
        max_frames = floor_by_factor(ele.get("max_frames", min(FPS_MAX_FRAMES, total_frames)), FRAME_FACTOR)
        nframes = total_frames / video_fps * fps
        if nframes > total_frames:
            logger.warning(f"smart_nframes: nframes[{nframes}] > total_frames[{total_frames}]")
        nframes = min(min(max(nframes, min_frames), max_frames), total_frames)
        nframes = floor_by_factor(nframes, FRAME_FACTOR)
    if not (FRAME_FACTOR <= nframes and nframes <= total_frames):
        raise ValueError(f"nframes should in interval [{FRAME_FACTOR}, {total_frames}], but got {nframes}.")
    return nframes


def smart_nframes_manual(
    ele: dict,
    total_frames: int,
    video_fps: int | float,
    fps_max_frames=768
) -> int:
    """calculate the number of frames for video used for model inputs.

    Args:
        ele (dict): a dict contains the configuration of video.
            support either `fps` or `nframes`:
                - nframes: the number of frames to extract for model inputs.
                - fps: the fps to extract frames for model inputs.
                    - min_frames: the minimum number of frames of the video, only used when fps is provided.
                    - max_frames: the maximum number of frames of the video, only used when fps is provided.
        total_frames (int): the original total number of frames of the video.
        video_fps (int | float): the original fps of the video.

    Raises:
        ValueError: nframes should in interval [FRAME_FACTOR, total_frames].

    Returns:
        int: the number of frames for video used for model inputs.
    """
    assert not ("fps" in ele and "nframes" in ele), "Only accept either `fps` or `nframes`"
    if "nframes" in ele:
        nframes = round_by_factor(ele["nframes"], FRAME_FACTOR)
    else:
        fps = ele.get("fps", FPS)
        min_frames = ceil_by_factor(ele.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR)
        max_frames = floor_by_factor(ele.get("max_frames", min(fps_max_frames, total_frames)), FRAME_FACTOR)
        nframes = total_frames / video_fps * fps
        if nframes > total_frames:
            logger.warning(f"smart_nframes: nframes[{nframes}] > total_frames[{total_frames}]")
        nframes = min(min(max(nframes, min_frames), max_frames), total_frames)
        nframes = floor_by_factor(nframes, FRAME_FACTOR)
    if not (FRAME_FACTOR <= nframes and nframes <= total_frames):
        raise ValueError(f"nframes should in interval [{FRAME_FACTOR}, {total_frames}], but got {nframes}.")
    return nframes

def _read_video_torchvision(
    ele: dict,
) -> (torch.Tensor, float):
    """read video using torchvision.io.read_video

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    video_path = ele["video"]
    if version.parse(torchvision.__version__) < version.parse("0.19.0"):
        if "http://" in video_path or "https://" in video_path:
            warnings.warn("torchvision < 0.19.0 does not support http/https video path, please upgrade to 0.19.0.")
        if "file://" in video_path:
            video_path = video_path[7:]
    st = time.time()
    video, audio, info = io.read_video(
        video_path,
        start_pts=ele.get("video_start", 0.0),
        end_pts=ele.get("video_end", None),
        pts_unit="sec",
        output_format="TCHW",
    )
    total_frames, video_fps = video.size(0), info["video_fps"]
    logger.info(f"torchvision:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    idx = torch.linspace(0, total_frames - 1, nframes).round().long()
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps
    print(f"smart_nframes process: {nframes=}, {total_frames=}, {video_fps=}, {sample_fps=}")
    video = video[idx]
    return video, sample_fps


def is_decord_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("decord") is not None


def _read_video_decord(
    ele: dict,
) -> (torch.Tensor, float):
    """read video using decord.VideoReader

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    import decord
    video_path = ele["video"]
    st = time.time()
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0), num_threads=1)
    # TODO: support start_pts and end_pts
    if 'video_start' in ele or 'video_end' in ele:
        raise NotImplementedError("not support start_pts and end_pts in decord for now.")
    total_frames, video_fps = len(vr), vr.get_avg_fps()
    logger.info(f"decord:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()
    if ele.get("skip", False):
        video = torch.zeros((16, 3, 280, 280))
    else:
        video = vr.get_batch(idx).asnumpy()
        video = torch.tensor(video).permute(0, 3, 1, 2)  # Convert to TCHW format
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps
    print(f"smart_nframes process: {nframes=}, {total_frames=}, {video_fps=}, {sample_fps=}")
    # if ele.get("return_selected_indices", False):
    default_frames = smart_nframes_manual(ele, total_frames=total_frames, video_fps=video_fps)
    return video, sample_fps, idx, default_frames
    # return video, sample_fps


VIDEO_READER_BACKENDS = {
    "decord": _read_video_decord,
    "torchvision": _read_video_torchvision,
}

FORCE_QWENVL_VIDEO_READER = os.getenv("FORCE_QWENVL_VIDEO_READER", None)


@lru_cache(maxsize=1)
def get_video_reader_backend() -> str:
    if FORCE_QWENVL_VIDEO_READER is not None:
        video_reader_backend = FORCE_QWENVL_VIDEO_READER
    elif is_decord_available():
        video_reader_backend = "decord"
    else:
        video_reader_backend = "torchvision"
    print(f"qwen-vl-utils using {video_reader_backend} to read video.", file=sys.stderr)
    return video_reader_backend


def fetch_video(ele: dict, image_factor: int = IMAGE_FACTOR, return_video_sample_fps: bool = False) -> torch.Tensor | list[Image.Image]:
    if isinstance(ele["video"], str):
        video_reader_backend = get_video_reader_backend()
        try:
            outputs = VIDEO_READER_BACKENDS[video_reader_backend](ele)
            # if ele.get("return_selected_indices", False):
            #     video, sample_fps, selected_indices = outputs
            # else:
            #     video, sample_fps = outputs
            video, sample_fps, selected_indices, default_frames = outputs

        except Exception as e:
            logger.warning(f"video_reader_backend {video_reader_backend} error, use torchvision as default, msg: {e}")
            video, sample_fps = VIDEO_READER_BACKENDS["torchvision"](ele)

        nframes, _, height, width = video.shape
        min_pixels = ele.get("min_pixels", VIDEO_MIN_PIXELS)
        total_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
        max_pixels = max(min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR), int(min_pixels * 1.05))
        max_pixels_supposed = ele.get("max_pixels", max_pixels)
        if max_pixels_supposed > max_pixels:
            logger.warning(f"The given max_pixels[{max_pixels_supposed}] exceeds limit[{max_pixels}].")
        max_pixels = min(max_pixels_supposed, max_pixels)

        print(f"max_pixels: {max_pixels/28/28} * 28 * 28")
        # import pdb; pdb.set_trace()
        # print(ele["resized_height"], ele["resized_width"])

        if "resized_height" in ele and "resized_width" in ele:
            resized_height, resized_width = smart_resize(
                ele["resized_height"],
                ele["resized_width"],
                factor=image_factor,
            )
        else:
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=image_factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
        video = transforms.functional.resize(
            video,
            [resized_height, resized_width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ).float()

        # import pdb; pdb.set_trace()
        print(f'video shape {video.shape}')
        
        if return_video_sample_fps:
            return video, sample_fps, selected_indices, default_frames
        
        # if ele.get("return_selected_indices", False):
        #     return video, sample_fps, selected_indices

        return video, selected_indices, default_frames
    else:
        assert isinstance(ele["video"], (list, tuple))
        process_info = ele.copy()
        process_info.pop("type", None)
        process_info.pop("video", None)
        images = [
            fetch_image({"image": video_element, **process_info}, size_factor=image_factor)
            for video_element in ele["video"]
        ]
        nframes = ceil_by_factor(len(images), FRAME_FACTOR)
        if len(images) < nframes:
            images.extend([images[-1]] * (nframes - len(images)))
        if return_video_sample_fps:
            return images, process_info.pop("fps", 2.0)
        return images

def fetch_video_given_keyframes(ele: dict, image_factor: int = IMAGE_FACTOR, return_video_sample_fps: bool = False, keyframes: list[int] = None, max_frames: int = 768) -> torch.Tensor | list[Image.Image]:
    if isinstance(ele["video"], str):
        import decord
        video_path = ele["video"]
        try:
            st = time.time()
            vr = decord.VideoReader(video_path, ctx=decord.cpu(0), num_threads=1)
            # TODO: support start_pts and end_pts
            if 'video_start' in ele or 'video_end' in ele:
                raise NotImplementedError("not support start_pts and end_pts in decord for now.")
            total_frames, video_fps = len(vr), vr.get_avg_fps()
        except:
            print("**********using torchvision instead********")
            if version.parse(torchvision.__version__) < version.parse("0.19.0"):
                if "http://" in video_path or "https://" in video_path:
                    warnings.warn("torchvision < 0.19.0 does not support http/https video path, please upgrade to 0.19.0.")
                if "file://" in video_path:
                    video_path = video_path[7:]
            st = time.time()
            video, audio, info = io.read_video(
                video_path,
                start_pts=ele.get("video_start", 0.0),
                end_pts=ele.get("video_end", None),
                pts_unit="sec",
                output_format="TCHW",
            )
            total_frames, video_fps = video.size(0), info["video_fps"]
        video_duration = total_frames / video_fps
        # logger.info(f"decord:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")
        nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
        fps = ele.get("fps", ZOOM_FPS)
        # nframes = int(total_frames / video_fps * fps)
        print(f"nframes: {nframes}=total_frames: {total_frames}/video_fps: {video_fps}*fps: {fps}")

        # NOTE find keyframes in 128 frames
        indices = np.linspace(0, total_frames - 1, num=128, dtype=int)
        selected_keyframe_indices = indices[keyframes].tolist()

        # keyframes
        frames_keyframes = vr.get_batch(selected_keyframe_indices).asnumpy()

        idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()
        video = vr.get_batch(idx).asnumpy()

        # save to image
        # for i in range(video.shape[0]):
        #     image = Image.fromarray(video[i])
        #     image.save(f'/root/Open-R1-Video-V1/tmp_multi/{i}.jpg')

        video = torch.tensor(video).permute(0, 3, 1, 2)  # Convert to TCHW format
        sample_fps = nframes / max(total_frames, 1e-6) * video_fps

        # keyframes
        video_keyframes = torch.tensor(frames_keyframes).permute(0, 3, 1, 2)  # Convert to TCHW format

        nframes, _, height, width = video.shape

        min_pixels = ele.get("min_pixels", VIDEO_MIN_PIXELS)
        total_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
        max_pixels = max(min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR), int(min_pixels * 1.05))
        max_pixels_supposed = ele.get("max_pixels", max_pixels)
        if max_pixels_supposed > max_pixels:
            logger.warning(f"The given max_pixels[{max_pixels_supposed}] exceeds limit[{max_pixels}].")
        max_pixels = min(max_pixels_supposed, max_pixels)
        if "resized_height" in ele and "resized_width" in ele:
            resized_height, resized_width = smart_resize(
                ele["resized_height"],
                ele["resized_width"],
                factor=image_factor,
            )
        else:
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=image_factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
        video = transforms.functional.resize(
            video,
            [resized_height, resized_width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ).float()

        # NOTE MIN_PIXELS, MAX_PIXELS
        min_pixels = ele.get("min_pixels", MIN_PIXELS)
        # max_pixels = ele.get("max_pixels", MAX_PIXELS)
        max_pixels = 300*28*28
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=IMAGE_FACTOR,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

        video_keyframes = transforms.functional.resize(
            video_keyframes,
            [resized_height, resized_width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ).float()

        # import pdb; pdb.set_trace()
        # print(f'video shape {video.shape}')
        print(f"video_keyframes shape {video_keyframes.shape}")

        if return_video_sample_fps:
            return video, video_keyframes, sample_fps
        return video, video_keyframes




def fetch_video_given_multi_durations(ele: dict, image_factor: int = IMAGE_FACTOR, return_video_sample_fps: bool = False, key_durations: list[float] = None, least_sample_keyframes: int = 16) -> torch.Tensor | list[Image.Image]:
    if isinstance(ele["video"], str):
        import decord
        video_path = ele["video"]
        try:
            st = time.time()
            vr = decord.VideoReader(video_path, ctx=decord.cpu(0), num_threads=1)
            # TODO: support start_pts and end_pts
            if 'video_start' in ele or 'video_end' in ele:
                raise NotImplementedError("not support start_pts and end_pts in decord for now.")
            total_frames, video_fps = len(vr), vr.get_avg_fps()
        except:
            print("**********using torchvision instead********")
            if version.parse(torchvision.__version__) < version.parse("0.19.0"):
                if "http://" in video_path or "https://" in video_path:
                    warnings.warn("torchvision < 0.19.0 does not support http/https video path, please upgrade to 0.19.0.")
                if "file://" in video_path:
                    video_path = video_path[7:]
            st = time.time()
            video, audio, info = io.read_video(
                video_path,
                start_pts=ele.get("video_start", 0.0),
                end_pts=ele.get("video_end", None),
                pts_unit="sec",
                output_format="TCHW",
            )
            total_frames, video_fps = video.size(0), info["video_fps"]
        video_duration = total_frames / video_fps
        # logger.info(f"decord:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")
        # nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
        fps = ele.get("fps", ZOOM_FPS)
        nframes = int(total_frames / video_fps * fps)
        print(f"nframes: {nframes}=total_frames: {total_frames}/video_fps: {video_fps}*fps: {fps}")
        # if nframes > 768:
        #     nframes = 768

        idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist() # a list

        indices = np.linspace(0, total_frames - 1, num=nframes, dtype=int) # smart_nframes

        print(f'final video indices len: {len(indices)}')

        timestamps = np.array([vr.get_frame_timestamp(idx) for idx in indices]) # timestamps with smart_nframes

        # import pdb; pdb.set_trace()

        window = least_sample_keyframes
        n = timestamps.shape[0]
        half = window // 2
        mask = np.zeros(n, dtype=bool)

        timestamps = timestamps.mean(axis=1)

        if key_durations is None:
            key_durations = [0, video_duration]
            mask = np.ones(n, dtype=bool)
        else:
            if len(key_durations) <=2:
                start_key_times = [key_durations[0]]
                end_key_times = [key_durations[1]]
            else:
                start_key_times = key_durations[::2]
                end_key_times = key_durations[1::2]

            # TODO if key_times have large error, the duration is not correct

            for s, e in zip(start_key_times, end_key_times):
                # closest index
                start_idx = int(np.argmin(np.abs(timestamps - s)))
                end_idx = int(np.argmin(np.abs(timestamps - e)))

                if end_idx - start_idx < window:
                    end_idx = min(n, start_idx + window)

                mask[start_idx:end_idx] = True
        
        # map to indices
        selected_indices = indices[mask]

        # if only get the key continous frames
        print(f'selected_indices: {selected_indices}')
        # print(f'len selected_indices: {len(selected_indices)}')

        # video = vr.get_batch(idx).asnumpy()
        video = vr.get_batch(selected_indices).asnumpy()

        # save to image
        # for i in range(video.shape[0]):
        #     image = Image.fromarray(video[i])
        #     image.save(f'/root/Open-R1-Video-V1/tmp_multi/{i}.jpg')

        video = torch.tensor(video).permute(0, 3, 1, 2)  # Convert to TCHW format
        sample_fps = nframes / max(total_frames, 1e-6) * video_fps

        nframes, _, height, width = video.shape

        min_pixels = ele.get("min_pixels", VIDEO_MIN_PIXELS)
        total_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
        max_pixels = max(min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR), int(min_pixels * 1.05))
        max_pixels_supposed = ele.get("max_pixels", max_pixels)
        if max_pixels_supposed > max_pixels:
            logger.warning(f"The given max_pixels[{max_pixels_supposed}] exceeds limit[{max_pixels}].")
        max_pixels = min(max_pixels_supposed, max_pixels)
        if "resized_height" in ele and "resized_width" in ele:
            resized_height, resized_width = smart_resize(
                ele["resized_height"],
                ele["resized_width"],
                factor=image_factor,
            )
        else:
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=image_factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
        video = transforms.functional.resize(
            video,
            [resized_height, resized_width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ).float()

        # import pdb; pdb.set_trace()
        print(f'video shape {video.shape}')

        if return_video_sample_fps:
            return video, sample_fps
        return video

def fetch_video_given_multi_durations_fast_old(ele: dict, image_factor: int = IMAGE_FACTOR, return_video_sample_fps: bool = False, key_durations: list[float] = None, least_sample_keyframes: int = 16, max_frames: int = 768, pad_frame: bool = True, use_ratio: bool = False) -> torch.Tensor | list[Image.Image]:
    if isinstance(ele["video"], str):
        import decord
        video_path = ele["video"]
        try:
            st = time.time()
            vr = decord.VideoReader(video_path, ctx=decord.cpu(0), num_threads=1)
            # TODO: support start_pts and end_pts
            if 'video_start' in ele or 'video_end' in ele:
                raise NotImplementedError("not support start_pts and end_pts in decord for now.")
            total_frames, video_fps = len(vr), vr.get_avg_fps()
        except:
            print("**********using torchvision instead********")
            if version.parse(torchvision.__version__) < version.parse("0.19.0"):
                if "http://" in video_path or "https://" in video_path:
                    warnings.warn("torchvision < 0.19.0 does not support http/https video path, please upgrade to 0.19.0.")
                if "file://" in video_path:
                    video_path = video_path[7:]
            st = time.time()
            video, audio, info = io.read_video(
                video_path,
                start_pts=ele.get("video_start", 0.0),
                end_pts=ele.get("video_end", None),
                pts_unit="sec",
                output_format="TCHW",
            )
            total_frames, video_fps = video.size(0), info["video_fps"]
        video_duration = total_frames / video_fps
        # logger.info(f"decord:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")
        # nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
        fps = ele.get("fps", ZOOM_FPS)
        nframes = int(total_frames / video_fps * fps)
        print(f"nframes: {nframes}=total_frames: {total_frames}/video_fps: {video_fps}*fps: {fps}")

        selected_indices = []
        if key_durations is None:
            # no fps, total 768 frames
            key_durations = [0, video_duration]
            indices = np.linspace(0, total_frames - 1, num=min(max_frames, total_frames), dtype=int)
            selected_indices.extend(indices)
        else:
            if len(key_durations) <=2:
                start_key_times = [key_durations[0]]
                end_key_times = [key_durations[1]]
            else:
                start_key_times = key_durations[::2]
                end_key_times = key_durations[1::2]
            for s, e in zip(start_key_times, end_key_times):
                # if use ratio, [0, 1] * duration
                if use_ratio:
                    s = s * video_duration
                    # e = e * video_duration
                    e = s + 2 # add 10s
                    # e = s + 10 # add 10s
                print(f"start_time: {s}, end_time: {e}")
                start_frame = int(s * video_fps)
                end_frame = int(e * video_fps)

                start_frame = max(0, start_frame)
                end_frame = min(total_frames, end_frame)

                if end_frame >= total_frames - 1:
                    end_frame = total_frames - 1

                frame_interval = max(1, int(video_fps / fps)) # fps根据总时长来修改 fps=min(2, 64/duration)
                indices = np.arange(start_frame, end_frame, frame_interval)
                selected_indices.extend(indices)

        # if only get the key continous frames
        selected_indices = np.unique(np.array(selected_indices, dtype=int)) # unique will sort the indices
        selected_indices = selected_indices[selected_indices < total_frames]

        # if 768, would skip this
        missing = least_sample_keyframes - len(selected_indices)

        if pad_frame:
            if missing > 0:
                min_idx = selected_indices[0] if len(selected_indices) > 0 else total_frames // 2
                max_idx = selected_indices[-1] if len(selected_indices) > 0 else total_frames // 2

                # half
                n_before = missing // 2
                n_after = missing - n_before

                before = np.linspace(
                    max(0, min_idx - n_before * 2),
                    min_idx - 1,
                    num=n_before,
                    dtype=int
                ) if n_before > 0 else np.array([], dtype=int)

                after = np.linspace(
                    max_idx + 1,
                    min(total_frames - 1, max_idx + n_after * 2),
                    num=n_after,
                    dtype=int
                ) if n_after > 0 else np.array([], dtype=int)

                extra_indices = np.concatenate([before, after])
                selected_indices = np.concatenate([selected_indices, extra_indices])

                # filter
                selected_indices = np.unique(selected_indices)
                if len(selected_indices) > least_sample_keyframes:
                    selected_indices = np.linspace(
                        selected_indices[0],
                        selected_indices[-1],
                        num=least_sample_keyframes,
                        dtype=int
                    )
        


        if len(selected_indices) > max_frames:
            selected_indices = np.linspace(
                selected_indices[0], # num
                selected_indices[-1], # num
                num=max_frames,
                dtype=int
            )
        # 4段区间，100s，fps=64/100,
        
        selected_indices = selected_indices[selected_indices < total_frames]

        print(f'selected_indices: {selected_indices}')
        print(f'len selected_indices: {len(selected_indices)}')

        # TODO sort
        selected_indices.sort()

        video = vr.get_batch(selected_indices).asnumpy()

        video = torch.tensor(video).permute(0, 3, 1, 2)  # Convert to TCHW format
        sample_fps = nframes / max(total_frames, 1e-6) * video_fps

        nframes, _, height, width = video.shape

        min_pixels = ele.get("min_pixels", VIDEO_MIN_PIXELS)
        total_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
        max_pixels = max(min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR), int(min_pixels * 1.05))
        max_pixels_supposed = ele.get("max_pixels", max_pixels)
        if max_pixels_supposed > max_pixels:
            logger.warning(f"The given max_pixels[{max_pixels_supposed}] exceeds limit[{max_pixels}].")
        max_pixels = min(max_pixels_supposed, max_pixels)
        if "resized_height" in ele and "resized_width" in ele:
            resized_height, resized_width = smart_resize(
                ele["resized_height"],
                ele["resized_width"],
                factor=image_factor,
            )
        else:
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=image_factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
        video = transforms.functional.resize(
            video,
            [resized_height, resized_width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ).float()

        # import pdb; pdb.set_trace()
        print(f'video shape {video.shape}')

        if return_video_sample_fps:
            return video, sample_fps
        
        return video, selected_indices
        # return video


def fetch_video_given_multi_durations_fast_only_key_duration(ele: dict, image_factor: int = IMAGE_FACTOR, return_video_sample_fps: bool = False, key_durations: list[float] = None, least_sample_keyframes: int = 16, max_frames: int = 768, pad_frame: bool = True, use_ratio: bool = False) -> torch.Tensor | list[Image.Image]:
    if isinstance(ele["video"], str):
        import decord
        video_path = ele["video"]
        try:
            st = time.time()
            vr = decord.VideoReader(video_path, ctx=decord.cpu(0), num_threads=1)
            # TODO: support start_pts and end_pts
            if 'video_start' in ele or 'video_end' in ele:
                raise NotImplementedError("not support start_pts and end_pts in decord for now.")
            total_frames, video_fps = len(vr), vr.get_avg_fps()
        except:
            print("**********using torchvision instead********")
            if version.parse(torchvision.__version__) < version.parse("0.19.0"):
                if "http://" in video_path or "https://" in video_path:
                    warnings.warn("torchvision < 0.19.0 does not support http/https video path, please upgrade to 0.19.0.")
                if "file://" in video_path:
                    video_path = video_path[7:]
            st = time.time()
            video, audio, info = io.read_video(
                video_path,
                start_pts=ele.get("video_start", 0.0),
                end_pts=ele.get("video_end", None),
                pts_unit="sec",
                output_format="TCHW",
            )
            total_frames, video_fps = video.size(0), info["video_fps"]
        video_duration = total_frames / video_fps
        # logger.info(f"decord:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")
        # nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
        fps = ele.get("fps", ZOOM_FPS)
        nframes = int(total_frames / video_fps * fps)
        print(f"nframes: {nframes}=total_frames: {total_frames}/video_fps: {video_fps}*fps: {fps}")

        selected_indices = []
        if key_durations is None:
            # no fps, total 768 frames
            key_durations = [0, video_duration]
            indices = np.linspace(0, total_frames - 1, num=min(max_frames, total_frames), dtype=int)
            selected_indices.extend(indices)
        else:
            if len(key_durations) <=2:
                start_key_times = [key_durations[0]]
                end_key_times = [key_durations[1]]
            else:
                start_key_times = key_durations[::2]
                end_key_times = key_durations[1::2]
            intervals = []
            for s, e in zip(start_key_times, end_key_times):
                # if use ratio, [0, 1] * duration
                if use_ratio:
                    s = s * video_duration
                    # e = e * video_duration
                    e = s + 2 # add 10s
                    # e = s + 10 # add 10s
                # print(f"start_time: {s}, end_time: {e}")
                intervals.append([s, e])
            
            # 先去除完全相同的重复区间
            unique_intervals = []
            seen = set()
            for interval in intervals:
                # 将区间转换为元组以便放入集合
                interval_tuple = tuple(interval)
                if interval_tuple not in seen:
                    seen.add(interval_tuple)
                    unique_intervals.append(interval)
            
            # 按起始位置排序
            unique_intervals.sort(key=lambda x: x[0])
            
            merged = []
            current_start, current_end = unique_intervals[0]
            
            for interval in unique_intervals[1:]:
                start, end = interval
                # 如果当前区间与下一个区间有重叠
                if start <= current_end:
                    # 合并区间，取最大的结束位置
                    current_end = max(current_end, end)
                else:
                    # 没有重叠，将当前区间加入结果
                    merged.append([current_start, current_end])
                    current_start, current_end = start, end
            
            # 添加最后一个区间
            merged.append([current_start, current_end])
            print(f"merged intervals: {merged}")
            select_time_duration = 0
            for interval in merged:
                s, e = interval
                time_duration = e - s # s
                select_time_duration += time_duration
                # start_frame = int(s * video_fps)
                # end_frame = int(e * video_fps)
                # start_frame = max(0, start_frame)
                # end_frame = min(total_frames, end_frame)
                # if end_frame >= total_frames - 1:
                #     end_frame = total_frames - 1
            custom_fps = min(2, max_frames / select_time_duration) # fps根据总时长来修改 fps=min(2, 64/duration)
            for interval in merged:
                s, e = interval
                start_frame = int(s * video_fps)
                end_frame = int(e * video_fps)
                start_frame = max(0, start_frame)
                end_frame = min(total_frames, end_frame)
                frame_interval = max(1, int(video_fps / custom_fps))
                indices = np.arange(start_frame, end_frame, frame_interval)
                selected_indices.extend(indices)

        # if only get the key continous frames
        # selected_indices = np.unique(np.array(selected_indices, dtype=int)) # unique will sort the indices
        # selected_indices = selected_indices[selected_indices < total_frames]

        # if 768, would skip this
        missing = least_sample_keyframes - len(selected_indices)

        if pad_frame:
            if missing > 0:
                min_idx = selected_indices[0] if len(selected_indices) > 0 else total_frames // 2
                max_idx = selected_indices[-1] if len(selected_indices) > 0 else total_frames // 2

                # half
                n_before = missing // 2
                n_after = missing - n_before

                before = np.linspace(
                    max(0, min_idx - n_before * 2),
                    min_idx - 1,
                    num=n_before,
                    dtype=int
                ) if n_before > 0 else np.array([], dtype=int)

                after = np.linspace(
                    max_idx + 1,
                    min(total_frames - 1, max_idx + n_after * 2),
                    num=n_after,
                    dtype=int
                ) if n_after > 0 else np.array([], dtype=int)

                extra_indices = np.concatenate([before, after])
                selected_indices = np.concatenate([selected_indices, extra_indices])

                # filter
                # selected_indices = np.unique(selected_indices)
                # if len(selected_indices) > least_sample_keyframes:
                #     selected_indices = np.linspace(
                #         selected_indices[0],
                #         selected_indices[-1],
                #         num=least_sample_keyframes,
                #         dtype=int
                #     )
        
        # if len(selected_indices) > max_frames:
        #     selected_indices = np.linspace(
        #         selected_indices[0], # num
        #         selected_indices[-1], # num
        #         num=max_frames,
        #         dtype=int
        #     )
        # 4段区间，100s，fps=64/100,
        
        # selected_indices = selected_indices[selected_indices < total_frames]

        print(f'selected_indices: {selected_indices}')
        print(f'len selected_indices: {len(selected_indices)}')

        # TODO sort
        selected_indices.sort()

        video = vr.get_batch(selected_indices).asnumpy()

        video = torch.tensor(video).permute(0, 3, 1, 2)  # Convert to TCHW format
        sample_fps = nframes / max(total_frames, 1e-6) * video_fps

        nframes, _, height, width = video.shape

        min_pixels = ele.get("min_pixels", VIDEO_MIN_PIXELS)
        total_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
        max_pixels = max(min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR), int(min_pixels * 1.05))
        max_pixels_supposed = ele.get("max_pixels", max_pixels)
        if max_pixels_supposed > max_pixels:
            logger.warning(f"The given max_pixels[{max_pixels_supposed}] exceeds limit[{max_pixels}].")
        max_pixels = min(max_pixels_supposed, max_pixels)
        if "resized_height" in ele and "resized_width" in ele:
            resized_height, resized_width = smart_resize(
                ele["resized_height"],
                ele["resized_width"],
                factor=image_factor,
            )
        else:
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=image_factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
        video = transforms.functional.resize(
            video,
            [resized_height, resized_width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ).float()

        # import pdb; pdb.set_trace()
        print(f'video shape {video.shape}')

        if return_video_sample_fps:
            return video, sample_fps
        
        return video, selected_indices
        # return video


def sample_frames_with_key_regions(duration, key_intervals, target_frames=64, 
                                  base_fps=30, key_region_fps=None):
    """
    结合关键区间和均匀采样的视频帧采样
    
    参数:
    duration: 视频总时长(秒)
    key_intervals: 关键区间列表，格式为[[s1,e1], [s2,e2],...]
    target_frames: 目标帧数，默认为64
    base_fps: 基础帧率，用于计算总帧数和均匀采样，默认为30
    key_region_fps: 关键区间采样帧率，如果为None则使用base_fps
    
    返回:
    选择的帧索引列表
    """
    total_frames = int(duration * base_fps)
    
    # 设置关键区间采样率
    if key_region_fps is None:
        key_region_fps = base_fps
    
    # 1. 标记关键帧（使用关键区间采样率）
    key_frames = set()
    print(key_intervals)
    for start, end in key_intervals:
        # 计算关键区间内的帧（使用关键区间采样率）
        start_frame_key = int(start * key_region_fps)
        end_frame_key = int(end * key_region_fps)
        print(start_frame_key, end_frame_key)
        
        # 转换为基础帧率坐标系
        start_frame_base = int(start * base_fps)
        end_frame_base = int(end * base_fps)
        
        # 在关键区间内均匀采样
        if end_frame_key > start_frame_key:  # 确保区间有效
            key_interval_frames = np.linspace(start_frame_key, end_frame_key, 
                                            end_frame_key - start_frame_key + 1, dtype=int)
            print(key_interval_frames)
            # 映射回基础帧率坐标系
            for frame_key in key_interval_frames:
                # 计算在基础帧率中的对应帧
                time_pos = frame_key / key_region_fps  # 时间位置（秒）
                frame_base = int(time_pos * base_fps)  # 基础帧率中的帧索引
                frame_base = min(frame_base, total_frames - 1)  # 确保不越界
                # import pdb; pdb.set_trace()
                key_frames.add(frame_base)
    # print(f"key_frames: {key_frames}") # 其实是因为key_frames是无序的
    key_frames = sorted(key_frames)
    print(f"len: {len(key_frames)}, key_frames: {key_frames}")
    
    # 2. 计算整体均匀采样间隔
    uniform_indices = np.linspace(0, total_frames-1, target_frames, dtype=int)
    # print(f"均匀采样点，{len(uniform_indices)}: {uniform_indices}")
    
    # 3. 记录哪些均匀采样点被舍弃了
    discarded_uniform_frames = set(uniform_indices)  # 先记录所有均匀采样点
    all_frames = set(key_frames)  # 先加入所有关键帧
    # print(f"len: {len(all_frames)}")

    wait_uniform_frames = set()
    # 4. 检查每对均匀采样点
    for i in range(len(uniform_indices)-1):
        start = uniform_indices[i]
        end = uniform_indices[i+1]
        
        # 检查这段区间内是否有关键帧
        interval_has_key_frames = any(start <= frame <= end for frame in key_frames)
        
        # 如果没有关键帧，则保留这两个均匀采样点
        if not interval_has_key_frames:
            wait_uniform_frames.add(start)
            wait_uniform_frames.add(end)
            # 从被舍弃的集合中移除（因为被保留了）
            discarded_uniform_frames.discard(start)
            discarded_uniform_frames.discard(end)
        # 如果有关键帧，则舍弃这两个均匀采样点（不添加）
        # 这两个帧会保留在discarded_uniform_frames中
    
    # 5. 转换为排序后的列表
    # selected_frames = sorted(all_frames)
    # 排序wait_uniform_frames
    wait_uniform_frames = sorted(wait_uniform_frames)
    # print(f"wait_uniform_frames: {wait_uniform_frames}")
    # 如果选择的帧数量小于目标帧，均匀补充wait_uniform_frames中的帧
    if len(all_frames) < target_frames:
        # 补充wait_uniform_frames中的帧
        additional_needed = target_frames - len(all_frames)
        print(f"Need supply {additional_needed} frame using wait_uniform_frames")
        if wait_uniform_frames:
            additional_indices = np.linspace(0, len(wait_uniform_frames)-1, additional_needed, dtype=int)
            for idx in additional_indices:
                all_frames.add(wait_uniform_frames[idx])
    
    selected_frames = sorted(all_frames)
    print(f"after add wait_uniform_frames, len: {len(selected_frames)}, selected_frames are: {selected_frames}")
    # print(f"舍弃的均匀采样点: {discarded_uniform_frames}")
    
    # 6. 如果帧数小于目标帧，优先加入被舍弃的均匀采样帧
    if len(selected_frames) < target_frames:
        print(f"Need supply {target_frames - len(selected_frames)} 帧")
        additional_needed = target_frames - len(selected_frames)
        
        # 将舍弃的均匀采样帧排序
        discarded_sorted = sorted(discarded_uniform_frames)
        
        # 优先加入被舍弃的均匀采样帧
        if discarded_sorted:
            # 取前additional_needed个被舍弃的均匀采样帧，从这些舍弃帧中均匀选择additional_needed个
            additional_indices = np.linspace(0, len(discarded_sorted)-1, additional_needed, dtype=int)
            frames_to_add = [discarded_sorted[idx] for idx in additional_indices]
            selected_frames.extend(frames_to_add)
            selected_frames.sort()
            additional_needed -= len(frames_to_add)
        
        # 如果还不够，再从其他帧中补充
        if additional_needed > 0:
            all_possible = set(range(total_frames))
            unselected = sorted(list(all_possible - set(selected_frames)))
            
            if unselected:
                # 在未选择的帧中均匀采样
                additional_indices = np.linspace(0, len(unselected)-1, additional_needed, dtype=int)
                for idx in additional_indices:
                    selected_frames.append(unselected[idx])
                selected_frames.sort()
    print(f"after add other frames, len: {len(selected_frames)}, selected_frames are: {selected_frames}")
    
    return selected_frames


def fetch_video_given_multi_durations_fast(ele: dict, image_factor: int = IMAGE_FACTOR, return_video_sample_fps: bool = False, key_durations: list[float] = None, least_sample_keyframes: int = 16, max_frames: int = 768, pad_frame: bool = True, use_ratio: bool = False) -> torch.Tensor | list[Image.Image]:
    if isinstance(ele["video"], str):
        import decord
        video_path = ele["video"]
        try:
            st = time.time()
            vr = decord.VideoReader(video_path, ctx=decord.cpu(0), num_threads=1)
            # TODO: support start_pts and end_pts
            if 'video_start' in ele or 'video_end' in ele:
                raise NotImplementedError("not support start_pts and end_pts in decord for now.")
            total_frames, video_fps = len(vr), vr.get_avg_fps()
        except:
            print("**********using torchvision instead********")
            if version.parse(torchvision.__version__) < version.parse("0.19.0"):
                if "http://" in video_path or "https://" in video_path:
                    warnings.warn("torchvision < 0.19.0 does not support http/https video path, please upgrade to 0.19.0.")
                if "file://" in video_path:
                    video_path = video_path[7:]
            st = time.time()
            video, audio, info = io.read_video(
                video_path,
                start_pts=ele.get("video_start", 0.0),
                end_pts=ele.get("video_end", None),
                pts_unit="sec",
                output_format="TCHW",
            )
            total_frames, video_fps = video.size(0), info["video_fps"]
        video_duration = total_frames / video_fps
        # logger.info(f"decord:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")
        # nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
        fps = ele.get("fps", ZOOM_FPS)
        nframes = int(total_frames / video_fps * fps)
        print(f"nframes: {nframes}=total_frames: {total_frames}/video_fps: {video_fps}*fps: {fps}")

        selected_indices = []
        if key_durations is None:
            # no fps, total 768 frames
            key_durations = [0, video_duration]
            indices = np.linspace(0, total_frames - 1, num=min(max_frames, total_frames), dtype=int)
            selected_indices.extend(indices)
        else:
            if len(key_durations) <=2:
                start_key_times = [key_durations[0]]
                end_key_times = [key_durations[1]]
            else:
                start_key_times = key_durations[::2]
                end_key_times = key_durations[1::2]
            intervals = []
            for s, e in zip(start_key_times, end_key_times):
                # if use ratio, [0, 1] * duration
                if use_ratio:
                    s = s * video_duration
                    # e = e * video_duration
                    e = s + 2 # add 10s
                    # e = s + 10 # add 10s
                # print(f"start_time: {s}, end_time: {e}")
                intervals.append([s, e])
            
            # 先去除完全相同的重复区间
            unique_intervals = []
            seen = set()
            for interval in intervals:
                # 将区间转换为元组以便放入集合
                interval_tuple = tuple(interval)
                if interval_tuple not in seen:
                    seen.add(interval_tuple)
                    unique_intervals.append(interval)
            
            # 按起始位置排序
            unique_intervals.sort(key=lambda x: x[0])
            
            merged = []
            current_start, current_end = unique_intervals[0]
            
            for interval in unique_intervals[1:]:
                start, end = interval
                # 如果当前区间与下一个区间有重叠
                if start <= current_end:
                    # 合并区间，取最大的结束位置
                    current_end = max(current_end, end)
                else:
                    # 没有重叠，将当前区间加入结果
                    merged.append([current_start, current_end])
                    current_start, current_end = start, end
            
            # 添加最后一个区间
            merged.append([current_start, current_end])
            print(f"merged intervals: {merged}")

            # 赋值
            duration = video_duration
            key_intervals = merged
            base_fps = video_fps

            key_frame_upper_ratio = 0.5 # TODO 这个参数作为超参数
            key_frame_upper_ratio = float(os.getenv("KEY_FRAME_UPPER_RATIO", key_frame_upper_ratio))
            target_frames = max_frames
            key_upper_frames = int(target_frames * key_frame_upper_ratio)
            print(f"key_upper_frames: {key_upper_frames}")

            key_intervals_length = sum(end - start for start, end in key_intervals)
            print(f"key_intervals length: {key_intervals_length}")
            key_upper_fps = key_upper_frames / key_intervals_length
            MAX_ZOOM_FPS = 6 # TODO 这个参数作为超参数
            MAX_ZOOM_FPS = float(os.getenv("MAX_ZOOM_FPS", MAX_ZOOM_FPS))
            key_region_fps = min(key_upper_fps, MAX_ZOOM_FPS) # 实在不够，让它继续后面的加帧操作
            print(f"key_upper_fps: {key_upper_fps}") # 如果关键区间时间很少的话，会导致fps很大

            selected_indices = sample_frames_with_key_regions(
                duration, key_intervals, target_frames=target_frames, base_fps=base_fps, key_region_fps=key_region_fps
            )

        

        print(f'selected_indices: {selected_indices}')
        print(f'len selected_indices: {len(selected_indices)}')

        # TODO sort
        selected_indices.sort()

        video = vr.get_batch(selected_indices).asnumpy()

        video = torch.tensor(video).permute(0, 3, 1, 2)  # Convert to TCHW format
        sample_fps = nframes / max(total_frames, 1e-6) * video_fps

        nframes, _, height, width = video.shape

        min_pixels = ele.get("min_pixels", VIDEO_MIN_PIXELS)
        total_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
        max_pixels = max(min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR), int(min_pixels * 1.05))
        max_pixels_supposed = ele.get("max_pixels", max_pixels)
        if max_pixels_supposed > max_pixels:
            logger.warning(f"The given max_pixels[{max_pixels_supposed}] exceeds limit[{max_pixels}].")
        max_pixels = min(max_pixels_supposed, max_pixels)
        if "resized_height" in ele and "resized_width" in ele:
            resized_height, resized_width = smart_resize(
                ele["resized_height"],
                ele["resized_width"],
                factor=image_factor,
            )
        else:
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=image_factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
        video = transforms.functional.resize(
            video,
            [resized_height, resized_width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ).float()

        # import pdb; pdb.set_trace()
        print(f'video shape {video.shape}')

        if return_video_sample_fps:
            return video, sample_fps
        
        return video, selected_indices
        # return video


def extract_vision_info(conversations: list[dict] | list[list[dict]]) -> list[dict]:
    vision_infos = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if (
                        "image" in ele
                        or "image_url" in ele
                        or "video" in ele
                        or ele["type"] in ("image", "image_url", "video")
                    ):
                        vision_infos.append(ele)
    return vision_infos


def process_vision_segment_info(
    conversations: list[dict] | list[list[dict]],
    return_video_kwargs: bool = False,
) -> tuple[list[Image.Image] | None, list[torch.Tensor | list[Image.Image]] | None, Optional[dict]]:

    vision_infos = extract_vision_info(conversations)
    ## Read images or videos
    image_inputs = []
    video_inputs = []
    segment_inputs = []
    video_sample_fps_list = []
    default_frames_list = []
    video_index_list = []
    for vision_info in vision_infos:
        if vision_info["type"] == "segment":
            segment_input, segment_sample_fps, segment_index, _ = fetch_video(vision_info, return_video_sample_fps=True)
            segment_inputs.append(segment_input)
        elif "image" in vision_info or "image_url" in vision_info:
            image_inputs.append(fetch_image(vision_info))
        elif "video" in vision_info:
            video_input, video_sample_fps, video_index, default_frames = fetch_video(vision_info, return_video_sample_fps=True)
            video_sample_fps_list.append(video_sample_fps)
            video_inputs.append(video_input)
            video_index_list.append(video_index)
            default_frames_list.append(default_frames)
        else:
            raise ValueError("image, image_url or video should in content.")
    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None
    if len(segment_inputs) == 0:
        segment_inputs = None
    if return_video_kwargs:
        return image_inputs, video_inputs, segment_inputs, video_index_list, {'fps': video_sample_fps_list}
    return image_inputs, video_inputs, segment_inputs, video_index_list


def process_vision_keyframes_info(
    conversations: list[dict] | list[list[dict]],
    keyframes,
    return_video_kwargs: bool = False,
) -> tuple[list[Image.Image] | None, list[torch.Tensor | list[Image.Image]] | None, Optional[dict]]:

    vision_infos = extract_vision_info(conversations)
    ## Read images or videos
    image_inputs = []
    video_inputs = []
    segment_inputs = []
    video_keyframes_inputs = []
    video_sample_fps_list = []
    for vision_info in vision_infos:
        if vision_info["type"] == "segment":
            segment_input, segment_sample_fps = fetch_video(vision_info, return_video_sample_fps=True)
            segment_inputs.append(segment_input)
        # NOTE leave for video_keyframes
        # elif "image" in vision_info or "image_url" in vision_info:
        #     image_inputs.append(fetch_image(vision_info))
        elif "video" in vision_info:
            video_input, video_keyframes, video_sample_fps = fetch_video_given_keyframes(vision_info, return_video_sample_fps=True, keyframes=keyframes)
            video_sample_fps_list.append(video_sample_fps)
            video_inputs.append(video_input)
            for idx, frame in enumerate(video_keyframes):
                frame = frame / 255
                img = F.to_pil_image(frame)
                video_keyframes_inputs.append(img) # fetch_image
                # save images
                # img.save(f'/mnt/bn/wxd/wangxd/VideoReasoner/select_keyframe_{idx}.png')
        else:
            raise ValueError("image, image_url or video should in content.")
    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None
    if len(segment_inputs) == 0:
        segment_inputs = None
    if len(video_keyframes_inputs) ==0:
        video_keyframes_inputs = None
    if return_video_kwargs:
        return image_inputs, video_inputs, video_keyframes_inputs, {'fps': video_sample_fps_list}
    return image_inputs, video_inputs, video_keyframes_inputs



def process_vision_info(
    conversations: list[dict] | list[list[dict]],
    return_video_kwargs: bool = False,
) -> tuple[list[Image.Image] | None, list[torch.Tensor | list[Image.Image]] | None, Optional[dict]]:

    vision_infos = extract_vision_info(conversations)
    ## Read images or videos
    image_inputs = []
    video_inputs = []
    video_index_list = []
    video_sample_fps_list = []
    default_frames_list = []
    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            image_inputs.append(fetch_image(vision_info))
        elif "video" in vision_info:
            video_input, video_sample_fps, video_index, default_frames = fetch_video(vision_info, return_video_sample_fps=True)
            video_sample_fps_list.append(video_sample_fps)
            video_inputs.append(video_input)
            video_index_list.append(video_index)
            default_frames_list.append(default_frames)
        else:
            raise ValueError("image, image_url or video should in content.")
    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None
    if return_video_kwargs:
        return image_inputs, video_inputs, video_index_list, default_frames_list, {'fps': video_sample_fps_list}
    return image_inputs, video_inputs, video_index_list, default_frames_list

def process_vision_given_multi_durations(
    conversations: list[dict] | list[list[dict]],
    return_video_kwargs: bool = False,
    key_durations: list[float] = None,
    least_sample_keyframes: int = 16,
    max_frames: int=768,
    pad_frame: bool = True,
    use_ratio: bool = False,
) -> tuple[list[Image.Image] | None, list[torch.Tensor | list[Image.Image]] | None, Optional[dict]]:

    vision_infos = extract_vision_info(conversations)
    
    # import pdb; pdb.set_trace()
    ## Read images or videos
    image_inputs = []
    video_inputs = []
    video_index_list = []
    video_sample_fps_list = []
    for vision_info in vision_infos:
        # video_inputs.append(fetch_video_given_multi_durations_fast(ele=vision_info, key_durations=key_durations, least_sample_keyframes=least_sample_keyframes, max_frames=max_frames, pad_frame=pad_frame, use_ratio=use_ratio)) # only get frame
        video_input, video_index = fetch_video_given_multi_durations_fast(ele=vision_info, key_durations=key_durations, least_sample_keyframes=least_sample_keyframes, max_frames=max_frames, pad_frame=pad_frame, use_ratio=use_ratio)
        video_inputs.append(video_input)
        video_index_list.append(video_index)

    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None
    if return_video_kwargs:
        return image_inputs, video_inputs, {'fps': video_sample_fps_list}

    return image_inputs, video_inputs, video_index_list