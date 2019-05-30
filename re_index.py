import glob
import os
import shutil

from shutil import copyfile
#copy folder tree from source to destination
def copyfolder(src,dst):
    files=os.listdir(src)
    if not os.path.isdir(dst):
        os.mkdir(dst)
    for view in files:
        if not os.path.isdir(dst + '/' +view):
            os.mkdir(dst + '/' + view)
        views = os.listdir(src + '/' + view)
        for tt in views:
            copyfile(src+'/'+view+'/'+tt,dst+'/'+view+'/'+tt)

root = '/home/fstu1/datasets/market1501/bounding_box_train'
data_path = '/home/fstu1/datasets/market1501/multiviews'

reid_index=0
folders=os.listdir(data_path)
folders = sorted(folders)

for foldernames in folders:
    copyfolder(data_path+'/'+foldernames,
               '/home/fstu1/datasets/market1501/reindexed'+
               '/'+str(reid_index).zfill(4))
    reid_index=reid_index+1
