#!/bin/bash

set -e

cd ..

C="checkpoint/fan3d_wo_norm_att"
Btr=32
Bte=32
N=128
Dtr="data/LS3D-W"
Dte="data/LS3D-W-Test"
R=$C"/model_best.pth.tar"
L=$C"/checkpoint.pth.tar"
T='3D'

LOG=$C"/all.txt"

# train
python3 main.py -c $C --nFeats $N --data $Dtr --pointType $T \
       --val-batch $Bte --train-batch $Btr \
       --use-attention --resume $C/checkpoint.pth.tar

# test
python3 main.py -e -c $C --nFeat $N --data $Dte --resume $R \
       --pointType $T --val-batch $Bte \
       --use-attention

# calculate final result
# python demo.py -c $C --data $Dte --pointType $T

