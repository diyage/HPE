from DeepPose.tools.cv2_ import CV2
import torch
import numpy as np
from typing import Union


class Vis:
    def __init__(self):
        pass

    @staticmethod
    def get_matrix_max_pos(matrix: Union[torch.Tensor, np.ndarray]) -> tuple:
        if isinstance(matrix, np.ndarray):
            matrix = torch.tensor(matrix)
        max_index = matrix.argmax()
        tmp = torch.zeros(*matrix.shape).view(-1)
        tmp[max_index] = 1.0
        tmp = tmp.view(*matrix.shape)
        # get r/c
        index_r = tmp.sum(dim=1).argmax()  # r
        index_c = tmp.sum(dim=0).argmax()  # c

        return int(index_c.item()), int(index_r.item())

    @staticmethod
    def plot_key_point_using_heat_map(
            image,
            heat_map,
            connections,
            title: str = '',
            saved_abs_path: str = ''
    ):
        '''

        :param image: np.ndarray(h,w,c),  np.uint8, BGR
        :param heat_map: np.ndarray (key_points_num + 1,h,w),
        :param connections:
        :param title:
        :param saved_abs_path:
        :return:
        '''
        if isinstance(image, torch.Tensor):
            image = image.cpu().detach().numpy().copy()
            image = np.transpose(image, axes=(1, 2, 0))
            image = 255.0 * (image * np.array([0.5, 0.5, 0.5]) + np.array([0.5, 0.5, 0.5]))

            image = np.array(image, dtype=np.uint8)

            heat_map = heat_map.cpu().detach().numpy().copy()

        image = image.copy()
        heat_map = heat_map[:-1]  # do not use background
        image_h, image_w = image.shape[0], image.shape[1]
        heat_map_h, heat_map_w = heat_map.shape[1], heat_map.shape[2]

        # get_key_points
        key_points = []
        for kp_map in heat_map:
            # pos on heat map
            x, y = Vis.get_matrix_max_pos(kp_map)
            # pos on image
            x = 1.0 * x / heat_map_w * image_w
            y = 1.0 * y / heat_map_h * image_h
            key_points.append([x, y])

        key_points = np.array(key_points)
        Vis.plot_key_points(image, key_points, connections, title, saved_abs_path)

    @staticmethod
    def plot_key_points(image,
                        key_point,
                        connections,
                        title: str = '',
                        saved_abs_path: str = ''
                        ):
        '''

        :param image:  np.ndarray(h,w,c),  np.uint8, BGR
        :param key_point: np.ndarray(n_joints, 2), abs position
        :param connections:
        :param title:
        :param saved_abs_path:
        :return:
        '''
        if isinstance(image, torch.Tensor):
            image = image.cpu().detach().numpy().copy()
            image = np.transpose(image, axes=(1, 2, 0))
            image = 255.0 * (image * np.array([0.5, 0.5, 0.5]) + np.array([0.5, 0.5, 0.5]))

            image = np.array(image, dtype=np.uint8)

            key_point = key_point.cpu().detach().numpy().copy()
            key_point[:, 0] = key_point[:, 0] * image.shape[1]  # scaled to abs
            key_point[:, 1] = key_point[:, 1] * image.shape[0]  # scaled to abs

        image = image.copy()
        for val in key_point:
            key_pos = (int(val[0]), int(val[1]))
            CV2.circle(image, key_pos, 2, (255, 255, 0), -1)

        for connection in connections:

            if isinstance(connection[0], tuple):
                k_a, k_b = connection[0]
                pos_0 = (int((key_point[k_a][0] + key_point[k_b][0]) / 2),
                         int((key_point[k_a][1] + key_point[k_b][1]) / 2))
            else:
                k = connection[0]
                pos_0 = (int(key_point[k][0]), int(key_point[k][1]))

            pos_1 = (int(key_point[connection[1]][0]), int(key_point[connection[1]][1]))
            CV2.line(image, pos_0, pos_1, (0, 255, 0), thickness=2)

        if saved_abs_path != '':
            CV2.imwrite(saved_abs_path, image)
        else:
            CV2.imshow(title, image)
            CV2.waitKey(0)





