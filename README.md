# Pytorch version of 'How far are we from solving the 2D \& 3D Face Alignment problem? (and a dataset of 230,000 3D facial landmarks)'

For official torch7 version please refer to [face-alignment-training](https://github.com/1adrianb/face-alignment-training)

This is a reinplement of training code for 2D-FAN and 3D-FAN decribed in "How far" paper. Please visit [author's webpage](https://www.adrianbulat.com) or [arxiv](https://arxiv.org/abs/1703.07332) for technical details.

Thanks for bearpaw's excellent work on human pose estimation [pytorch-pose](https://github.com/bearpaw/pytorch-pose). And in this project, I reused a branch of helper function from pytorch-pose.

Pretrained models are available soon.

## Requirments

- Install the latest [PyTorch](http://pytorch.org), version 0.2.1 is fully supported and there is no further test on older version.

### Packages

- [scipy](https://www.scipy.org/)
- [torchvision](https://pytorch.org)
- [progress](https://pypi.python.org/pypi/progress)(optional) for better visualization.

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

- Pythoner friendly and there is no need for `.t7` format annotations
- Add 300-W-LP test set for validation.
- Followed the excatly same training procedure described in the paper (except binary network part).
- Add model evaluation in terms of **Mean error**, **AUC@0.07**
- TODO: add evaluation on test sets (300W, 300VW, AFLW2000-3D etc.).

## Citation

```
@inproceedings{bulat2017far,
  title={How far are we from solving the 2D \& 3D Face Alignment problem? (and a dataset of 230,000 3D facial landmarks)},
  author={Bulat, Adrian and Tzimiropoulos, Georgios},
  booktitle={International Conference on Computer Vision},
  year={2017}
}
```
