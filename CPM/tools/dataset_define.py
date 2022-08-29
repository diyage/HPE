
from torch.utils.data import Dataset, DataLoader
import os
from scipy.io import loadmat
import numpy as np
from DeepPose.tools.cv2_ import CV2
import torch
import albumentations as alb


def guassian_kernel(size_w, size_h, center_x, center_y, sigma):
    sigma = sigma * np.sqrt(size_w * size_h)
    gridy, gridx = np.mgrid[0:size_h, 0:size_w]
    D2 = (gridx - center_x) ** 2 + (gridy - center_y) ** 2
    ans = np.exp(-D2 / 2.0 / sigma / sigma)
    return ans


class StrongLspDataSet(Dataset):
    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: alb.Compose = None,
            heat_map_size: tuple = (45, 45),
            heat_map_sigma: float = 0.15,
            image_size: tuple = (368, 368),
            use_visibility: bool = False
    ):
        super().__init__()
        self.__root = root
        self.__train = train
        self.heat_map_size = heat_map_size
        self.heat_map_sigma = heat_map_sigma
        self.image_size = image_size
        self.use_visibility = use_visibility

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
        images_name = ['im' + ('000' + str(i + 1))[-4:] + '.jpg' for i in range(2000)]
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

    def create_heatmap(
            self,
            key_point: np.ndarray
    ):
        # scaled on heat_map
        k_p_x_on_map = key_point[:, 0] * self.heat_map_size[0]
        k_p_y_on_map = key_point[:, 1] * self.heat_map_size[1]

        # get gt_heat_map
        # # for each key_point
        gt_heat_map = []
        for now_key_point in zip(k_p_x_on_map, k_p_y_on_map):
            tmp = guassian_kernel(self.heat_map_size[0],
                                  self.heat_map_size[1],
                                  now_key_point[0],
                                  now_key_point[1],
                                  self.heat_map_sigma)
            gt_heat_map.append(tmp)
        # # for background
        gt_map = np.array(gt_heat_map)
        gt_backg = np.ones([self.heat_map_size[1],
                            self.heat_map_size[0]]) - np.max(gt_map, 0)
        gt_map = np.append(gt_map, gt_backg[np.newaxis, :, :], axis=0)

        # get center_map
        # # scaled on new image size
        k_p_x_on_new_ = key_point[:, 0] * self.image_size[0]
        k_p_y_on_new_ = key_point[:, 1] * self.image_size[1]
        # compute center position for all key_points
        center_x = (k_p_x_on_new_.max() +
                    k_p_x_on_new_.min()) / 2
        center_y = (k_p_y_on_new_.max() +
                    k_p_y_on_new_.min()) / 2

        center_map = guassian_kernel(self.image_size[0],
                                     self.image_size[1],
                                     center_x,
                                     center_y,
                                     self.heat_map_sigma)
        return gt_map, center_map

    def __len__(self):
        return len(self.__images_abs_path)

    def __getitem__(
            self,
            index: int
    ):
        old_image, old_keypoints = self.pull_image(index)

        res = self.__transform(image=old_image, keypoints=old_keypoints)
        new_image, new_keypoints = res.get('image'), res.get('keypoints')
        new_key_points_scaled = np.array(new_keypoints) / self.image_size[0]
        heat_map, center_map = self.create_heatmap(new_key_points_scaled)

        new_image_transe = np.transpose(new_image, axes=(2, 0, 1))

        res = {
            'image': torch.tensor(new_image_transe, dtype=torch.float32),
            'gt_map': torch.tensor(heat_map, dtype=torch.float32),
            'center_map': torch.tensor(center_map, dtype=torch.float32),
        }

        if self.use_visibility:
            res.update({
                'visibility': torch.tensor(new_key_points_scaled[:, 2], dtype=torch.float32)
            })

        return res


def get_strong_lsp_loader(
        root: str,
        train: bool,
        transform: alb.Compose,
        heat_map_size: tuple = (45, 45),
        heat_map_sigma: float = 0.15,
        image_size: tuple = (368, 368),
        use_visibility: bool = False,
        batch_size: int = 32,
        num_workers: int = 4,
):
    data_set = StrongLspDataSet(
        root,
        train,
        transform,
        heat_map_size=heat_map_size,
        heat_map_sigma=heat_map_sigma,
        image_size=image_size,
        use_visibility=use_visibility
    )
    data_loader = DataLoader(
        data_set,
        batch_size,
        shuffle=True if train else False,
        num_workers=num_workers
    )
    return data_loader


def debug_strong_lsp():
    from CPM.tools.config import DataSetConfig
    from CPM.tools.visualize import Vis

    trans = alb.Compose([
        alb.HueSaturationValue(),
        alb.RandomBrightnessContrast(),
        alb.HorizontalFlip(p=1.0),
        alb.GaussNoise(),
        alb.Resize(DataSetConfig.image_w, DataSetConfig.image_h)
    ], keypoint_params=alb.KeypointParams(format='xy'))

    d = StrongLspDataSet(
        '/home/dell/data/LSP/lsp_dataset',
        train=True,
        transform=trans,
        heat_map_size=(DataSetConfig.heat_map_w, DataSetConfig.heat_map_h),
        heat_map_sigma=0.1,
        image_size=(DataSetConfig.image_w, DataSetConfig.image_h)
    )

    image, key_points = d.pull_image(np.random.randint(len(d)))
    res = trans(image=image, keypoints=key_points)

    new_image, new_key_points = res.get('image'), res.get('keypoints')
    new_key_points_scaled = np.array(new_key_points) / DataSetConfig.image_h

    heat_map, center_map = d.create_heatmap(new_key_points_scaled)
    Vis.plot_key_point_using_heat_map(
        new_image,
        heat_map,
        DataSetConfig.connections
    )
    # CV2.imshow('center_map', center_map)
    # CV2.waitKey(0)
    for ind in range(heat_map.shape[0]):
        CV2.imshow('{}'.format(ind), heat_map[ind])
        CV2.waitKey(0)


def debug_lsp_data_loader():
    from CPM.tools.config import DataSetConfig
    trans = alb.Compose([
        alb.HueSaturationValue(),
        alb.RandomBrightnessContrast(),
        alb.HorizontalFlip(p=1.0),
        alb.GaussNoise(),
        alb.Resize(DataSetConfig.image_w, DataSetConfig.image_h),
        alb.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ], keypoint_params=alb.KeypointParams(format='xy'))

    loader = get_strong_lsp_loader(
        '/home/dell/data/LSP/lsp_dataset',
        train=True,
        transform=trans,
        heat_map_size=(DataSetConfig.heat_map_w, DataSetConfig.heat_map_h),
        heat_map_sigma=DataSetConfig.heat_map_sigma,
        image_size=(DataSetConfig.image_w, DataSetConfig.image_h)
    )
    for _, res in enumerate(loader):
        print(res.keys())
        image, gt_map, center_map = res.get('image'), res.get('gt_map'), res.get('center_map')
        print(image.shape)
        # print(image)
        print(gt_map.shape)
        # print(gt_map)
        print(center_map.shape)
        # print(center_map)
        break


if __name__ == '__main__':
    debug_strong_lsp()
    # debug_lsp_data_loader()
