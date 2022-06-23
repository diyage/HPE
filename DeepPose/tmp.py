from DeepPose.tools.cv2_ import CV2
import h5py
import numpy as np
import scipy
from scipy.io import loadmat
from DeepPose.tools.config import DataSetConfig
import os


img = CV2.imread(os.path.join(DataSetConfig.root_path, 'images', 'im0005.jpg'))  # 175*102*3
print(img.shape)

img = CV2.resize(img, (100, 200))
CV2.circle(img, (10.0, 50.0), 2, (255, 255, 0), -1)
print(img.shape)
#
# CV2.circle(img, (10, 10), 2, (255, 255, 0), -1)
CV2.imshow('hehe', img)
CV2.waitKey(0)
# f = loadmat(os.path.join(DataSetConfig.root_path, 'joints.mat'))
# joints = f.get('joints')  # type: np.ndarray
# joint = joints[0:2, :, 4]
#
#
# for index in range(14):
#     dim_0 = joint[0][index]
#     dim_1 = joint[1][index]
#
#     CV2.circle(img, (int(dim_0), int(dim_1)), 2, (255, 255, 0), -1)
#
# for connection in DataSetConfig.connections:
#     pos_0 = (0, 0)
#     if isinstance(connection[0], tuple):
#         k_a, k_b = connection[0]
#         pos_0 = (int((joint[0][k_a] + joint[0][k_b]) / 2), int((joint[1][k_a] + joint[1][k_b]) / 2))
#     else:
#         pos_0 = (int(joint[0][connection[0]]), int(joint[1][connection[0]]))
#
#     pos_1 = (int(joint[0][connection[1]]), int(joint[1][connection[1]]))
#     CV2.line(img, pos_0, pos_1, (0, 255, 0), thickness=2)
#
# CV2.imshow('hehe', img)
# CV2.waitKey(0)




