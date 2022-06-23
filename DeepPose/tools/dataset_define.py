from torch.utils.data import Dataset
from torchvision.transforms import transforms
import os
from scipy.io import loadmat
import numpy as np
from DeepPose.tools.cv2_ import CV2
from DeepPose.tools.config import DataSetConfig
import torch


class LspDataSet(Dataset):
    def __init__(self,
                 root: str,
                 train: bool=True,
                 transform=None,
                 data_set_opt: DataSetConfig = None):
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

        self.__images_abs_path, self.__key_points = self.get_images_path_and_key_points()

    def get_images_path_and_key_points(self):
        images_name = ['im' + ('000' + str(i+1))[-4:] + '.jpg' for i in range(2000)]
        images_abs_path = [os.path.join(self.__root, 'images', val) for val in images_name]

        f = loadmat(os.path.join(self.__root, 'joints.mat'))
        joints = f.get('joints')  # type: np.ndarray  # 3*14*20000
        joints = joints[0:2, :, :]  # type: np.ndarray # 2*14*20000

        joints = np.transpose(joints, axes=(2, 1, 0))  # type: np.ndarray # 20000*14*2
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
        key_point = self.__key_points[index]  # 14*2

        image, key_point = self.resize(image.copy(), key_point.copy(), self.__dataset_opt.image_size)
        return self.__transform(image), torch.tensor(key_point, dtype=torch.float32)
