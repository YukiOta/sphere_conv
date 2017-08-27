# coding: utf-8
"""
球面配列モデルに対するコンボリューションを行うプログラム．
input: image
spherical convolution network
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
from collections import OrderedDict



##########
# テストデータの読み込み
DATA_dir = "../1data/"
print("loading 2016 data")
img_tmp = np.load(DATA_dir+"solardata.npz")
img_1117 = img_tmp['x_1117']
target_1117 = img_tmp['y_1117']
img_1130 = img_tmp['x_1130']
target_1130 = img_tmp['y_1130']
img_1218 = img_tmp['x_1218']
target_1218 = img_tmp['y_1218']
img_1221 = img_tmp['x_1221']
target_1221 = img_tmp['y_1221']
print("done")

img_1117 = ndimage.median_filter(img_1117, 3)
img_1117.shape
tmp = img_1117[100]
tmp = tmp.transpose(1, 2, 0)

##########




















# end
