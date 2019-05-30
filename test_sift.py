import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
from LocalitySensitiveHashing import LocalitySensitiveHashing

img = cv2.imread('img.jpg')
cv_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(cv_img,None)
img = cv2.drawKeypoints(cv_img,kp,img)
cv2.imwrite('sift_keypoints.jpg',img)
kp, des = sift.detectAndCompute(cv_img,None)
des = des[:80, :]
print(des.shape)

first_column = pd.Series(['row_{}'.format(row) for row in range(des.shape[0])])
des = pd.DataFrame(des)
des = pd.concat([first_column, des], axis=1)

des.to_csv('csv_data.csv', header=False, index=False)

# N keypoint
# 128 X N descriptor
# csv_file = open('csv_data.csv', 'w')
# writer = csv.writer(csv_file)
# writer.writerows(des)
# LSH Algorithm
n_bucket = 100
lsh = LocalitySensitiveHashing(datafile="data_for_lsh.csv",
                               dim=10,
                               r=50,
                               b=n_bucket)

lsh.get_data_from_csv()
lsh.show_data_for_lsh()
lsh.initialize_hash_store()
lsh.hash_all_data()
exit(0)
similarity_neighborhoods = lsh.lsh_basic_for_nearest_neighbors()
print(similarity_neighborhoods)
