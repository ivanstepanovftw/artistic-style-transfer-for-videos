import logging
import os

import cv2
import numpy as np
import torch
import torchvision
import torchvision.models.optical_flow
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


class DivisibleBy8:
    def __call__(self, tensor):
        """Preprocesses a tensor to be divisible by 8. This is required by the RAFT model."""
        h, w = tensor.shape[-2:]
        h = h - h % 8
        w = w - w % 8
        tensor = tensor[..., :h, :w]
        return tensor


def frame_to_tensor(frame: np.array, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.ToTensor(),
        DivisibleBy8(),
        torchvision.transforms.Lambda(lambda x: x[[2, 1, 0], ...]),  # BGR to RGB
        transforms.Normalize(mean, std)
    ])
    return transform(frame)


def main():
    video_filepath = 'video/sintel.mp4'
    root_filepath = "root/sintel.mp4"

    # logger = logging.getLogger(__name__)
    # logger.setLevel(logging.DEBUG)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = torchvision.models.optical_flow.raft_large(weights=torchvision.models.optical_flow.Raft_Large_Weights.C_T_SKHT_V1).to(device)
    model = torchvision.models.optical_flow.raft_small(weights=torchvision.models.optical_flow.Raft_Small_Weights.C_T_V2).to(device)
    model.eval()

    mean = torch.tensor((0.485, 0.456, 0.406)).to(device)
    std = torch.tensor((0.229, 0.224, 0.225)).to(device)

    raft_mean = torch.tensor((0.5, 0.5, 0.5)).to(device)
    raft_std = torch.tensor((0.5, 0.5, 0.5)).to(device)

    vidcap = cv2.VideoCapture(video_filepath)
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    content_prev = None

    batch_size = 1
    batch = []

    for frame_idx in tqdm(range(frame_count), desc="Splitting video and computing optical flow"):
        frame_filepath = f"{root_filepath}/{frame_idx}"
        content_filepath = f"{frame_filepath}/content.pt"
        forward_flow_filepath = f"{frame_filepath}/forward_flow.pt"
        backward_flow_filepath = f"{frame_filepath}/backward_flow.pt"

        success, frame_bgr = vidcap.read()
        if not success:
            raise Exception(f"Failed to read frame {frame_idx} from video of {frame_count} frames")

        os.makedirs(frame_filepath, exist_ok=True)

        content = frame_to_tensor(frame_bgr, mean, std).to(device)
        # if not os.path.exists(content_filepath):
        torch.save(content, content_filepath)

        content_cur = frame_to_tensor(frame_bgr, raft_mean, raft_std).to(device)

        if content_prev is not None:
            # if not os.path.exists(forward_flow_filepath):
            #     batch.append((forward_flow_filepath, content_prev, content_cur))
            if not os.path.exists(backward_flow_filepath):
                batch.append((backward_flow_filepath, content_cur, content_prev))

            if len(batch) >= batch_size:
                image1 = torch.stack([x[1] for x in batch], dim=0)
                image2 = torch.stack([x[2] for x in batch], dim=0)
                print(image1.shape, image2.shape)
                flow: list[torch.tensor] = model(image1, image2)
                for i, (flow_filepath, _, _) in enumerate(batch):
                    # print(flow[i].shape)
                    torch.save(flow[i][0], flow_filepath)
                batch = []
        content_prev = content_cur

    if len(batch) > 0:
        image1 = torch.stack([x[1] for x in batch])
        image2 = torch.stack([x[2] for x in batch])
        flow = model(image1, image2)
        for i, (flow_filepath, _, _) in enumerate(batch):
            torch.save(flow[i], flow_filepath)


if __name__ == '__main__':
    main()
