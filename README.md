# Pytorch version of 'How far are we from solving the 2D \& 3D Face Alignment problem? (and a dataset of 230,000 3D facial landmarks)'

For official torch7 version please refer to [github](https://github.com/1adrianb/face-alginment-training)

This is the reinplement of training code for 2D-FAN and 3D-FAN decribed in "How far" paper. Please visit [author's](https://www.adrianbulat.com) webpage or [arxiv](https://arxiv.org/abs/) for tech details.

Thanks for bear's prior work on pose estimation work, [pytorch-pose](https://github.com/bearpaw/pytorch-pose). And in this project, I reused branch of utils function from pytorch-pose.

Pretrained models are available soon.

## Requirments

- Install the latest [PyTorch](http://pytorch.org) version.

### Packages

- [scipy](https://github.com/torch/cutorch)
- [torchvision](https://github.com/torch/nn)
- [progress](https://link.com)

## Setup

1. Clone the github repository and install all the dependencies mentiones above.

```bash

git  clone https://github.com/hzh8311/pyhowfar
cd pyhowfar
```

2. Download the 300W-LP dataset from the authors webpage. In order to train on your own data the dataloader.lua file needs to be adapted.

3. Download the 300W-LP annotations converted to t7 format by paper author from [here](https://www.adrianbulat.com/downloads/FaceAlignment/landmarks.zip), extract it and move the ```landmarks``` folder to the root of the 300W-LP dataset.

## Usage

In order to run the demo please download the required models available bellow and the associated data.

```bash
python main.py
```

In order to see all the available options please run:

```bash
python main.py --help
```

## What's different?

- Add 300-W-LP test set for validation.
- Followed the excatly training procedur describle in the paper (except binary network part).
- Add experiments evaluation in terms of **Mean error**, **AUC@0.07**
- TODO: add evaluation on test sets(300W, 300VW, AFLW2000-3D)

## Citation

```
@inproceedings{bulat2017far,
  title={How far are we from solving the 2D \& 3D Face Alignment problem? (and a dataset of 230,000 3D facial landmarks)},
  author={Bulat, Adrian and Tzimiropoulos, Georgios},
  booktitle={International Conference on Computer Vision},
  year={2017}
}
```
