from torch.utils.data import Dataset, DataLoader
import os
from scipy.io import loadmat
import numpy as np
from DeepPose.tools.cv2_ import CV2
import torch
import albumentations as alb


class StrongLspDataSet(Dataset):
    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: alb.Compose = None,
    ):
        super().__init__()
        self.__root = root
        self.__train = train
        if transform is None:
            self.__transform = alb.Compose([
                alb.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]
                )
            ], keypoint_params=alb.KeypointParams(format='xy'))
        else:
            self.__transform = transform

        self.__images_abs_path, self.__key_points = self.get_images_path_and_key_points()

    def get_images_path_and_key_points(self):
        images_name = ['im' + ('000' + str(i+1))[-4:] + '.jpg' for i in range(2000)]
        images_abs_path = [os.path.join(self.__root, 'images', val) for val in images_name]

        f = loadmat(os.path.join(self.__root, 'joints.mat'))
        joints = f.get('joints')  # type: np.ndarray  # 3*14*2000
        joints = joints[0:2, :, :]  # type: np.ndarray # 2*14*2000

        joints = np.transpose(joints, axes=(2, 1, 0))  # type: np.ndarray # 2000*14*2
        number = int(0.7 * len(images_abs_path))
        if self.__train:
            return images_abs_path[0:number], joints[0:number]
        else:
            return images_abs_path[number:], joints[number:]

    def pull_image(
            self,
            index: int
    ):
        # for image
        image_path = self.__images_abs_path[index]
        image = CV2.imread(image_path)
        # for key point
        key_point = self.__key_points[index]  # 14*2
        key_point[:, 0] = np.clip(key_point[:, 0], 0, image.shape[1] - 1)
        key_point[:, 1] = np.clip(key_point[:, 1], 0, image.shape[0] - 1)
        return image, key_point

    def __len__(self):
        return len(self.__images_abs_path)

    def __getitem__(
            self,
            index: int
    ):
        old_image, old_keypoints = self.pull_image(index)

        res = self.__transform(image=old_image, keypoints=old_keypoints)
        new_image, new_keypoints = res.get('image'), res.get('keypoints')
        new_image = np.transpose(new_image, axes=(2, 0, 1))

        return torch.tensor(new_image, dtype=torch.float32),  \
            torch.tensor(new_keypoints, dtype=torch.float32) / new_image.shape[1]


def get_strong_lsp_loader(
        root: str,
        train: bool,
        transform: alb.Compose,
        batch_size: int = 32,
        num_workers: int = 4,
):
    data_set = StrongLspDataSet(
        root,
        train,
        transform,
    )
    data_loader = DataLoader(
        data_set,
        batch_size,
        shuffle=True if train else False,
        num_workers=num_workers
    )
    return data_loader


def debug_strong_lsp():
    from DeepPose.tools.config import DataSetConfig
    from DeepPose.tools.visualize import Vis
    trans = alb.Compose([
        alb.HueSaturationValue(),
        alb.RandomBrightnessContrast(),
        alb.HorizontalFlip(p=1.0),
        alb.GaussNoise(),
        alb.Resize(DataSetConfig.image_size[0], DataSetConfig.image_size[1])
    ], keypoint_params=alb.KeypointParams(format='xy'))
    d = StrongLspDataSet(
        '/home/dell/data/LSP/lsp_dataset',
        train=True,
        transform=trans
    )
    image, key_points = d.pull_image(np.random.randint(len(d)))

    Vis.plot_key_points(
        image,
        key_points,
        connections=DataSetConfig.connections,
        title='old'
    )
    res = trans(image=image, keypoints=key_points)

    new_image, new_key_points = res.get('image'), res.get('keypoints')
    Vis.plot_key_points(
        new_image,
        new_key_points,
        connections=DataSetConfig.connections,
        title='old'
    )


def debug_lsp_data_loader():
    from DeepPose.tools.config import DataSetConfig
    trans = alb.Compose([
        alb.HueSaturationValue(),
        alb.RandomBrightnessContrast(),
        alb.HorizontalFlip(p=1.0),
        alb.GaussNoise(),
        alb.Resize(DataSetConfig.image_size[0], DataSetConfig.image_size[1]),
        alb.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ], keypoint_params=alb.KeypointParams(format='xy'))

    loader = get_strong_lsp_loader(
        '/home/dell/data/LSP/lsp_dataset',
        train=True,
        transform=trans,
    )
    for _, (img, key_point) in enumerate(loader):
        print(img)
        print(key_point)
        break


if __name__ == '__main__':
    # debug_strong_lsp()
    debug_lsp_data_loader()
