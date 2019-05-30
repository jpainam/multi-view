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
               '/home/fstu1/datasets/market1501/ordered'+
               '/'+str(reid_index).zfill(4))
    reid_index=reid_index+1

'''imglist = glob.glob(os.path.join(root, '*.jpg'))
for imgpath in imglist:
    img_name = os.path.basename(imgpath)
    camId = img_name.split('c')[1][0]
    new_path = os.path.join('/home/fstu1/datasets/market1501/multiviews', img_name.split('_')[0])
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    per_cam_view = os.path.join(new_path, "cam_{}".format(camId))
    if not os.path.exists(per_cam_view):
        os.makedirs(per_cam_view)
    shutil.copy(imgpath, os.path.join(per_cam_view, img_name))

print('Creating folder per view finished')
new_path = '/home/fstu1/datasets/market1501/multiviews'
for classes in os.listdir(new_path):
    path_view = os.path.join(new_path, classes)
    for camId in os.listdir(path_view):
        v = 0
        path_cam = os.path.join(path_view, camId)
        imglist = glob.glob(os.path.join(path_cam, '*.jpg'))
        imglist = sorted(imglist)
        for img in imglist:
            img_name = os.path.basename(img)
            dst = os.path.join(path_view, 'view_{}'.format(v))
            if not os.path.exists(dst):
                os.makedirs(dst)
            shutil.move(img, os.path.join(dst, img_name))
            v = v + 1

print('Removing empty directory')
new_path = '/home/fstu1/datasets/market1501/multiviews'

def removeEmptyFolders(path, removeRoot=True):
    if not os.path.isdir(path):
        return
    files = os.listdir(path)
    if len(files):
        for f in files:
            fullpath = os.path.join(path, f)
            if os.path.isdir(fullpath):
                removeEmptyFolders(fullpath)
    files = os.listdir(path)
    if len(files) == 0 and removeRoot:
        os.rmdir(path)

removeEmptyFolders(new_path)

#multi-query
query_path = os.path.join(root, 'gt_bbox')
# for dukemtmc-reid, we do not need multi-query
if os.path.isdir(query_path):
    query_save_path = os.path.join(root, 'pytorch', 'multi-query')
    if not os.path.isdir(query_save_path):
        os.mkdir(query_save_path)

    for root, dirs, files in os.walk(query_path, topdown=True):
        for name in files:
            if not name[-3:] == 'jpg':
                continue
            ID = name.split('_')
            src_path = query_path + '/' + name
            dst_path = query_save_path + '/' + ID[0]
            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)
            shutil.copyfile(src_path, dst_path + '/' + name)

'''
