#!/bin/bash

set -e

cd ..

C="checkpoint/fan3d_wo_norm_att"
Btr=16
Bte=8
N=128
Dtr="data/300W_LP"
Dte="data/LS3D-W"
R=$C"/model_best.pth.tar"
L=$C"/checkpoint.pth.tar"
T='3D'

LOG=$C"/all.txt"

# test
python main.py -e -c $C --nFeat $N --data $Dte --resume $R \
       --pointType $T --val-batch $Bte \
       --use-attention

# calculate final result
python demo.py -c $C --data $Dte --pointType $T


# C='checkpoint/fan3d_s4_b1_c128'
# R=$C"/model_best.pth.tar"
# # test
# python main.py -e -c $C --nFeat $N --data $D --resume $R \
    #        --pointType $T --val-batch $Bte

# # calculate final result
# python demo.py -c $C --data $D --pointType $T
