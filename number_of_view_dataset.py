import numpy as np
import glob
import os
from collections import defaultdict

# market1501
# duke
# cuhk03
results = {}
dd = 'cuhk01'
#for dd in ("market1501", "duke", "cuhk03", "viper", 'cuhk01'):
datapath = '/home/paul/datasets/{}/bounding_box_train'.format(dd)
imglist = glob.glob(os.path.join(datapath, '*.png'))
res = [0, 0, 0, 0, 0, 0, 0, 0, 0]
for imgfile in imglist:
    img_name = os.path.basename(imgfile)
    cam_view = int(img_name.split("c")[1][0])

    res[cam_view] = res[cam_view] + 1
results[dd] = res

print(results)
