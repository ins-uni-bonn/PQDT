#!/usr/bin/env sh
HOME=`pwd`

# Chamfer Distance
cd /$HOME/extensions/chamfer_dist
python setup.py install

# utils from pointnet2 pip install
cd /$HOME/extensions/Pointnet2_PyTorch/pointnet2_ops_lib
pip install -e . --no-build-isolation

# emd
cd /$HOME/extensions/emd
python setup.py install
