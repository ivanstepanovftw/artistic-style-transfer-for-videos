# Artistic Style Transfer for Videos

Artistic Style Transfer for Videos is a Python application that applies the style transfer technique to both images and videos. This technique reimagines your photos or videos in the style of another image, such as a famous artwork, using neural network algorithms.

Unofficial implementation in PyTorch of an "Artistic style transfer for videos" [[arXiv]](https://arxiv.org/abs/1604.08610).

## Features

- Model-agnostic. Fast EfficientNet-B0 is used as a feature extractor. RAFT is used as an optical flow estimator.
- High-order statistics of feature grid are used to estimate the style, thus eliminating dependency from content/style image shape.

## Limitations

- Not a real-time application.

## TODO

- Remove hardcoded paths.
- Add long-term temporal consistency.
- Implement multi-pass algorithm.
- Copy audio from the original video.

[//]: # (- Masking of target objects. Can be done with multiplying gradients by a mask.)

## Installation

### Prerequisites
- Python 3.7+
- pip
- [Git LFS](https://git-lfs.github.com/) (optional)

### Install

1. Clone the repository with [Git LFS](https://git-lfs.github.com/) support.
2. Verify that example files are downloaded with:
   ```shell
   git lfs pull
   ```
3. Install dependencies with:
   ```shell
   python -m pip install -U -r requirements.txt
   ```

## Video Style Transfer

Apply style transfer to video files, transforming each frame to carry the artistic style of your choice.

### Usage

First, prepare your video for style transfer:

```shell
python main_video_prepare.py
```

Next, apply the style transfer to the video:

```shell
python main_video.py
```

Finally, join the processed frames back into a video:

```shell
python main_video_join.py
```

Arguments and filepaths need to be hardcoded in [`main_video_prepare.py`](main_video_prepare.py), [`main_video.py`](main_video.py), and [`main_video_join.py`](main_video_join.py).

## Alternatives

* [Original implementation in Lua Torch](https://github.com/manuelruder/artistic-videos)
* [Scaling up GANs for Text-to-Image Synthesis](https://mingukkang.github.io/GigaGAN/)

## License

Licensed under either of

* Apache License, Version 2.0, ([LICENSE-APACHE-2.0](LICENSE-APACHE-2.0) or http://www.apache.org/licenses/LICENSE-2.0)
* MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.

## Acknowledgements

This project is inspired by the pioneering work in neural style transfer. Special thanks to the authors of the original research papers and the open-source community for making their code available.

* [Artistic style transfer for videos](https://arxiv.org/abs/1604.08610)
* [Artistic style transfer for videos and spherical images](https://arxiv.org/abs/1708.04538)
* [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)
* [Demystifying Neural Style Transfer](https://arxiv.org/abs/1701.01036)
* [RAFT: Recurrent All-Pairs Field Transforms for Optical Flow](https://arxiv.org/abs/2003.12039)
