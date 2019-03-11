#!/bin/bash

set -e

cd ..

C="checkpoint/recurrent"
Btr=6
Bte=2
N=128
Dtr="data/300W_LP"
Dte="data/LS3D-W"
R=$C"/model_best.pth.tar"
L=$C"/checkpoint.pth.tar"
T='3D'

LOG=$C"/all.txt"

# test
python recurrent_main.py -e -c $C --nFeat $N --data $Dte --resume $R \
       --pointType $T --val-batch $Bte \
       --use-attention --netType recurrent_fan

# calculate final result
python demo.py -c $C --data $Dte --pointType $T
