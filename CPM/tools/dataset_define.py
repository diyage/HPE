from torch.utils.data import Dataset
from torchvision.transforms import transforms
import os
from scipy.io import loadmat
import numpy as np
from DeepPose.tools.cv2_ import CV2
from CPM.tools.config import DataSetConfig
import torch


class LspDataSet(Dataset):
    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform=None,
                 data_set_opt: DataSetConfig = None,
                 use_visibility: bool = False):
        super().__init__()
        self.__root = root
        self.__train = train
        if transform is None:
            self.__transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            self.__transform = transform
        if data_set_opt is None:
            self.__dataset_opt = DataSetConfig()
        else:
            self.__dataset_opt = data_set_opt

        self.__use_visibility = use_visibility

        self.__images_abs_path, self.__key_points = self.get_images_path_and_key_points()

    def get_images_path_and_key_points(self):
        images_name = ['im' + ('000' + str(i + 1))[-4:] + '.jpg' for i in range(2000)]
        images_abs_path = [os.path.join(self.__root, 'images', val) for val in images_name]

        f = loadmat(os.path.join(self.__root, 'joints.mat'))
        joints = f.get('joints')  # type: np.ndarray  # 3*14*20000

        if not self.__use_visibility:
            joints = joints[0:2, :, :]  # type: np.ndarray # 2*14*20000

        joints = np.transpose(joints, axes=(2, 1, 0))  # type: np.ndarray # 20000*14*(2 or 3)
        number = int(0.7 * len(images_abs_path))
        if self.__train:
            return images_abs_path[0:number], joints[0:number]
        else:
            return images_abs_path[number:], joints[number:]

    def __len__(self):
        return len(self.__images_abs_path)

    @staticmethod
    def resize(image: np.ndarray, key_point: np.ndarray, new_size: tuple):
        h_size = image.shape[0]  # h
        w_size = image.shape[1]  # w

        key_point[:, 0] = key_point[:, 0] / w_size  # w
        key_point[:, 1] = key_point[:, 1] / h_size  # h

        image = CV2.resize(image, new_size=new_size)
        return image, key_point

    def __getitem__(self, index):
        # for image
        image_path = self.__images_abs_path[index]
        image = CV2.imread(image_path)
        image = CV2.cvtColorToRGB(image)
        # for key point
        key_point = self.__key_points[index]  # 14*(2 or 3)

        image, key_point = self.resize(image.copy(), key_point.copy(),
                                       new_size=(self.__dataset_opt.image_w, self.__dataset_opt.image_h))

        # scaled on heat_map
        k_p_x_on_map = key_point[:, 0] * self.__dataset_opt.heat_map_w
        k_p_y_on_map = key_point[:, 1] * self.__dataset_opt.heat_map_h

        # get gt_heat_map
        # # for each key_point
        gt_heat_map = []
        for now_key_point in zip(k_p_x_on_map, k_p_y_on_map):
            tmp = guassian_kernel(self.__dataset_opt.heat_map_w,
                                  self.__dataset_opt.heat_map_h,
                                  now_key_point[0],
                                  now_key_point[1],
                                  self.__dataset_opt.sigma)
            gt_heat_map.append(tmp)
        # # for background
        gt_map = np.array(gt_heat_map)
        gt_backg = np.ones([self.__dataset_opt.heat_map_h,
                            self.__dataset_opt.heat_map_w]) - np.max(gt_map, 0)
        gt_map = np.append(gt_map, gt_backg[np.newaxis, :, :], axis=0)

        # get center_map
        # # scaled on new image size
        k_p_x_on_new_ = key_point[:, 0] * self.__dataset_opt.image_w
        k_p_y_on_new_ = key_point[:, 1] * self.__dataset_opt.image_h
        # compute center position for all key_points
        center_x = (k_p_x_on_new_.max() +
                    k_p_x_on_new_.min()) / 2
        center_y = (k_p_y_on_new_.max() +
                    k_p_y_on_new_.min()) / 2

        center_map = guassian_kernel(self.__dataset_opt.image_w,
                                     self.__dataset_opt.image_h,
                                     center_x,
                                     center_y,
                                     self.__dataset_opt.sigma)

        res = {
            'image': self.__transform(image),
            'gt_map': torch.tensor(gt_map, dtype=torch.float32),
            'center_map': torch.tensor(center_map, dtype=torch.float32),
        }

        if self.__use_visibility:
            res.update({
                'visibility': torch.tensor(key_point[:, 2], dtype=torch.float32)
            })

        return res


def guassian_kernel(size_w, size_h, center_x, center_y, sigma):
    sigma = sigma * np.sqrt(size_w * size_h)
    gridy, gridx = np.mgrid[0:size_h, 0:size_w]
    D2 = (gridx - center_x) ** 2 + (gridy - center_y) ** 2
    return np.exp(-D2 / 2.0 / sigma / sigma)
