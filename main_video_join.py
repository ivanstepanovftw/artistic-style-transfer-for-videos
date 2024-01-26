import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


def tensor_to_image(tensor: torch.Tensor, means: torch.Tensor, stds: torch.Tensor) -> Image.Image:
    transform = transforms.Compose([
        transforms.Normalize((-means / stds).tolist(), (1.0 / stds).tolist()),
        transforms.Lambda(lambda x: x[[2, 1, 0], ...]),  # RGB to BGR
        transforms.ToPILImage()
    ])
    return transform(tensor)


def image_to_frame(image: Image.Image) -> np.ndarray:
    return np.array(image)


def main():
    video_filepath = "examples/video/sintel.mp4"
    root_path = "root/sintel.mp4"
    output_filepath, fourcc = "output/sintel.mp4/out_mp4v.mp4", "mp4v"
    # output_filepath, fourcc = "output/sintel.mp4/out_vp90.webm", "VP90"
    # output_filepath, fourcc = "output/sintel.mp4/out_xvid.avi", "xvid"
    # output_filepath, fourcc = "output/sintel.mp4/out_mjpg.avi", "MJPG"
    # output_filepath, fourcc = "output/sintel.mp4/out_ffv1.avi", "FFV1"

    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

    mean = torch.tensor((0.485, 0.456, 0.406))
    std = torch.tensor((0.229, 0.224, 0.225))

    # Clamping range for normalized image
    min_vals = (0 - mean) / std
    max_vals = (1 - mean) / std

    frames = sorted(Path(root_path).glob(f"*"), key=lambda x: int(x.name))
    print(f"frames: {frames}")

    vidcap = cv2.VideoCapture(video_filepath)
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Make video from input frames
    fourcc = cv2.VideoWriter_fourcc(*fourcc)

    out = None

    for frame_filepath in frames:
        print(f"frame_filepath: {frame_filepath}")
        styled_filepath = f"{frame_filepath}/styled.pt"
        styled = torch.load(styled_filepath)
        styled.data = styled.data.clamp_(min_vals[:, None, None], max_vals[:, None, None])
        styled = image_to_frame(tensor_to_image(styled, mean, std))
        if not out:
            frame_shape = styled.shape[1], styled.shape[0]
            print(f"Frame shape: {frame_shape}")
            out = cv2.VideoWriter(output_filepath, fourcc, vidcap.get(cv2.CAP_PROP_FPS), frame_shape)
        out.write(styled)

    if out is not None:
        out.release()


if __name__ == '__main__':
    main()
