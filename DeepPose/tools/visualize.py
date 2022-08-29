from DeepPose.tools.cv2_ import CV2
import torch
import numpy as np


class Vis:
    def __init__(self):
        pass

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





