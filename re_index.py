import glob
import os
import shutil

root = '/home/paul/datasets/market1501/bounding_box_train'
'''
imglist = glob.glob(os.path.join(root, '*.jpg'))
for imgpath in imglist:
    img_name = os.path.basename(imgpath)
    camId = img_name.split('c')[1][0]
    new_path = os.path.join('/home/paul/datasets/market1501/multiviews', img_name.split('_')[0])
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    per_cam_view = os.path.join(new_path, "cam_{}".format(camId))
    if not os.path.exists(per_cam_view):
        os.makedirs(per_cam_view)
    shutil.copy(imgpath, os.path.join(per_cam_view, img_name))

print('Creating folder per view finished')
new_path = '/home/paul/datasets/market1501/multiviews'
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
'''
print('Removing empty directory')
new_path = '/home/paul/datasets/market1501/multiviews'

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