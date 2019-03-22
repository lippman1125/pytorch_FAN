## Pytorch version of 'How far are we from solving the 2D \& 3D Face Alignment problem? (and a dataset of 230,000 3D facial landmarks)'

  For official torch7 version please refer to face-alignment-training [https://github.com/1adrianb/face-alignment-training] 

  This is a reinplement of training code for 2D-FAN and 3D-FAN decribed in "How far" paper. Please visit author's webpage [https://www.adrianbulat.com] or arxiv [https://arxiv.org/abs/1703.07332] for technical details.

  Thanks for bearpaw's excellent work on human pose estimation [https://github.com/bearpaw/pytorch-pose] . And in this project, I reused a branch of helper function from pytorch-pose.

  Pretrained models are available soon.

## Requirments

   - Install the latest Pytorch [http://pytorch.org], version 1.0.0.

## Packages

   - [https://www.scipy.org/][scipy]<br>
   - [https://pytorch.org][torchvision]<br>
   - [https://pypi.python.org/pypi/progress] progress (optional) for better visualization.

## Train

   1. Clone the github repository and install all the dependencies mentiones above.
   
            git clone https://github.com/lippman1125/pytorch_FAN


   2. Download the LS3D-W dataset from the authors webpage (https://www.adrianbulat.com/face-alignment). 

   3. Download the 300W-LP annotations converted to t7 format by paper author from (https://www.adrianbulat.com/downloads/FaceAlignment/landmarks.zip).
   
   4. We merge LS3D-W dataset and 300W-LP dataset together to train our model.
   
      cd data/LS3D-W                    <br>
      tree -d -L 1                      <br>
      
            |-- 300VW-3D                
            |-- 300W-Testset-3D         
            |-- 300W_LP                 
            |-- AFLW2000-3D-Reannotated 
            `-- Menpo-3D
       
      Validation set is testset of 300W-LP

   5. Start to train:

            ./exp/train.sh
   
   
## Test

   1. Run the demo.

            python demo_video.py gpu


## What's different?

   - Train FAN only includes 2 Hourglass
   - Total Params: 11.55M   

## Performance

![GIF](landmarks.gif)

## Citation


      {
       @inproceedings{bulat2017far,
         title={How far are we from solving the 2D \& 3D Face Alignment problem? (and a dataset of 230,000 3D facial landmarks)},
         author={Bulat, Adrian and Tzimiropoulos, Georgios},
         booktitle={International Conference on Computer Vision},
         year={2017}
       }

## Refenerce

- https://github.com/hzh8311/pyhowfar