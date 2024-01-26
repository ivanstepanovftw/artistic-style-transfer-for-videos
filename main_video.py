import os
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Any, Iterable, Mapping, Callable

import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from efficientnet_pytorch import EfficientNet
from scipy.ndimage import map_coordinates
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
from tqdm import tqdm


def load_image(image_path) -> Image.Image:
    image = Image.open(image_path).convert('RGB')
    return image


def image_to_tensor(image: Image.Image, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return transform(image)


def tensor_to_image(tensor: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> Image.Image:
    transform = transforms.Compose([
        transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist()),
        transforms.ToPILImage()
    ])
    return transform(tensor)


def flatten_values(a: Iterable | Mapping | Any):
    if isinstance(a, (dict, )):
        for x in a.values():
            yield from flatten_values(x)
    elif isinstance(a, (list, tuple, set)):
        for x in a:
            yield from flatten_values(x)
    else:
        yield a


def total_variation2d(x: torch.Tensor):
    return torch.sum(torch.abs(x[:, :, :-1] - x[:, :, 1:])) + torch.sum(torch.abs(x[:, :-1, :] - x[:, 1:, :]))


@contextmanager
def register_hooks(
        model: torch.nn.Module,
        hook: Callable,
        **kwargs
):
    handles = []
    try:
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                hook: Callable = partial(hook, name=name, **kwargs)
                handle = module.register_forward_hook(hook)
                handles.append(handle)
        yield handles
    finally:
        for handle in handles:
            handle.remove()


def stat_recorder_hook(
        module: torch.nn.Module,
        input: torch.Tensor,
        output: torch.Tensor,
        name: str,
        *,
        storage: dict[str, dict[str, torch.Tensor]]
):
    if output.size(0) * output.size(2) * output.size(3) == 1:
        return

    mean = output.mean(dim=[0, 2, 3])
    std = output.std(dim=[0, 2, 3])
    skewness = (output - mean[None, :, None, None]).pow(3).mean(dim=[0, 2, 3]) / std.pow(3)
    # kurtosis = (output - mean[None, :, None, None]).pow(4).mean(dim=[0, 2, 3]) / std.pow(4)
    # mean_abs = output.abs().mean(dim=[0, 2, 3])
    # std_abs = output.abs().std(dim=[0, 2, 3])
    # for stat in (mean, std, skewness, kurtosis, mean_abs, std_abs):
    #     if not torch.isfinite(stat).all():
    #         with torch.no_grad():
    #             stat[~torch.isfinite(stat)] = 0
    storage[name] = {
        "mean": mean,
        "std": std,
        "skewness": skewness,
        # "kurtosis": kurtosis,
    }


def get_stats(model: torch.nn.Module, image: torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
    stats = {}
    with register_hooks(model, stat_recorder_hook, storage=stats):
        _ = model(image[None])
    return stats


def alpha_composite(im1, im2, opacity1=1.0, opacity2=1.0):
    # Validate the opacity values
    if not 0 <= opacity1 <= 1 or not 0 <= opacity2 <= 1:
        raise ValueError('Opacity must be between 0 and 1')

    # Scale the alpha channels by the provided opacity values
    im1[..., 3] = im1[..., 3] * opacity1
    im2[..., 3] = im2[..., 3] * opacity2

    # Normalize the alpha channels to be between 0 and 1
    im1_alpha = im1[..., 3] / 255.0
    im2_alpha = im2[..., 3] / 255.0

    # Compute the composite alpha channel
    composite_alpha = im1_alpha + im2_alpha * (1 - im1_alpha)

    # Handle case where composite_alpha is 0 to avoid divide by zero error
    mask = composite_alpha > 0
    composite_alpha = np.where(mask, composite_alpha, 1)

    # Compute the composite image
    composite_image = np.empty_like(im1)
    for channel in range(3):
        composite_image[..., channel] = (
            im1[..., channel] * im1_alpha
            + im2[..., channel] * im2_alpha * (1 - im1_alpha)
        ) / composite_alpha

    # Add the composite alpha channel to the image
    composite_image[..., 3] = composite_alpha * 255

    return composite_image.astype(np.uint8)


def warp(image: np.ndarray, backward_flow: np.ndarray, order=3) -> np.ndarray:
    channels, height, width = image.shape
    index_grid = np.mgrid[0:height, 0:width].astype(float)
    # Widely, first channel is horizontal x-axis flow, the second channel is vertical y-axis flow.
    coordinates = index_grid + backward_flow[::-1]
    remapped = np.empty(image.shape, dtype=image.dtype)
    for i in range(channels):
        remapped[i] = map_coordinates(image[i], coordinates, order=order, mode='constant', cval=0)
    return remapped


def main():
    root_path = "root/sintel.mp4"
    style_filepath = "examples/style/doodle.png"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = torchvision.models.vgg16(pretrained=True).features
    # model = torchvision.models.resnet18(pretrained=True)
    model = EfficientNet.from_pretrained('efficientnet-b0')
    # model = EfficientNet.from_pretrained('efficientnet-b4')

    named_layers = [name for name, module in model.named_modules() if isinstance(module, torch.nn.Conv2d)]
    style_content_weights_per_layer = {
        name: (
            # style
            # 1,
            1 - i / (len(named_layers) - 1),

            # content
            # 1
            # i / (len(named_layers) - 1),
            # (i / (len(named_layers) - 1)) ** 2,
            0,
        )
        for i, name in enumerate(named_layers)
    }
    print(f"style_content_weights_per_layer: {style_content_weights_per_layer}")

    style_weight = 1e+2
    content_weight = 1e+0
    temporal_weight = 1e+0
    total_variation_weight = 0

    # Disable grad
    for param in model.parameters():
        param.requires_grad_(False)
    # Disable running stats
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.track_running_stats = False
    # Unset "inplace"
    for module in model.modules():
        if hasattr(module, "inplace"):
            module.inplace = False

    model = model.to(device)
    model.eval()
    print(f"model: {model}")

    mean = torch.tensor((0.485, 0.456, 0.406)).to(device)
    std = torch.tensor((0.229, 0.224, 0.225)).to(device)

    # Clamping range for normalized image
    min_vals = (0 - mean) / std
    max_vals = (1 - mean) / std

    style_image = load_image(style_filepath)
    style_image = image_to_tensor(style_image, mean, std).to(device)
    style_stats = get_stats(model, style_image)
    print(f"style_stats.keys(): {style_stats.keys()}")
    assert torch.isfinite(torch.cat(list(flatten_values(style_stats)))).all()

    frames = sorted(Path(root_path).glob(f"*"), key=lambda x: int(x.name))
    print(f"frames: {frames}")

    styled_prev = None

    for frame_filepath in frames:
        # Remove all f"{frame_filepath}/styled_{epoch}_{iteration}.png"
        for filepath in Path(frame_filepath).glob("styled_*.png"):
            os.remove(filepath)

    for frame_filepath in frames:
        print(f"frame_filepath: {frame_filepath}")

        styled_filepath = f"{frame_filepath}/styled.pt"
        content_filepath = f"{frame_filepath}/content.pt"
        # forward_flow_filepath = f"{frame_filepath}/forward_flow.pt"
        backward_flow_filepath = f"{frame_filepath}/backward_flow.pt"

        content = torch.load(content_filepath).to(device)

        if styled_prev is None:
            styled = content.clone().to(device)
        else:
            backward_flow = torch.load(backward_flow_filepath).to(device)
            styled = styled_prev.clone().to(device)
            styled = torch.from_numpy(warp(styled.detach().numpy(), backward_flow.detach().numpy())).to(device)
        styled.requires_grad_(True)

        # optimizer = torch.optim.LBFGS([styled], lr=1, max_iter=40)
        optimizer = torch.optim.Adam([styled], lr=0.1)
        # optimizer = torch.optim.SGD([styled], lr=1, momentum=0.9, nesterov=True)

        epochs = 1 if isinstance(optimizer, torch.optim.LBFGS) else 40

        content_stats = get_stats(model, content)
        assert torch.isfinite(torch.cat(list(flatten_values(content_stats)))).all()

        for epoch in range(epochs):
            iteration = 0

            def closure() -> float:
                nonlocal iteration
                print(f"closure(): frame_filepath: {frame_filepath}, epoch: {epoch}, iteration: {iteration}")
                optimizer.zero_grad(set_to_none=True)

                with torch.no_grad():
                    styled.data = styled.data.clamp_(min_vals[:, None, None], max_vals[:, None, None])
                tensor_to_image(styled, mean, std).save(f"{frame_filepath}/styled_{epoch}_{iteration}.png")
                styled_stats = get_stats(model, styled)

                loss = torch.zeros(1, device=device)
                for name, (style_w, content_w) in style_content_weights_per_layer.items():
                    # If requested layer weight not found in style or content stats, skip it
                    if name not in styled_stats or name not in style_stats or name not in content_stats:
                        continue
                    loss += style_weight * style_w * torch.nn.functional.mse_loss(
                        torch.cat(list(flatten_values(styled_stats[name]))),
                        torch.cat(list(flatten_values(style_stats[name]))))
                    loss += content_weight * content_w * torch.nn.functional.mse_loss(
                        torch.cat(list(flatten_values(styled_stats[name]))),
                        torch.cat(list(flatten_values(content_stats[name]))))
                loss += total_variation_weight * total_variation2d(styled)
                if styled_prev is not None:
                    loss += temporal_weight * torch.nn.functional.mse_loss(styled, styled_prev)
                assert not torch.isnan(loss).any()

                print(loss)
                iteration += 1
                loss.backward()
                return loss.item()

            optimizer.step(closure)

        torch.save(styled, styled_filepath)
        styled_prev = styled.clone().to(device)


if __name__ == '__main__':
    main()
