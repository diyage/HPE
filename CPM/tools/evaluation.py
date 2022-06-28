from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
from CPM.tools.visualize import Vis
from DeepPose.tools.cv2_ import CV2
from CPM.tools.model_define import CPMNet


class OKS:
    def __init__(self,
                 model: CPMNet,
                 data_loader_for_compute_sigma: DataLoader,
                 image_h: int,
                 image_w: int,
                 ):
        self.model = model
        self.image_h = image_h
        self.image_w = image_w
        self.oks_sigma = self.__compute_sigma(data_loader_for_compute_sigma)

    @staticmethod
    def compute_std_for_batch(out_map: torch.Tensor,
                              gt_map: torch.Tensor,
                              image_h: int,
                              image_w: int):
        assert out_map.shape == gt_map.shape

        image_num, joints_num = gt_map.shape[0], gt_map.shape[1] - 1

        s_square = 1.0 * image_h * image_w
        joints_d2_vec = [[] for _ in range(joints_num)]

        for i in range(image_num):
            for j in range(joints_num):
                out_map_reverse = out_map[i][j].cpu().detach().numpy().copy()
                out_map_reverse = CV2.resize(out_map_reverse, new_size=(image_h, image_w))

                gt_map_reverse = gt_map[i][j].cpu().detach().numpy().copy()
                gt_map_reverse = CV2.resize(gt_map_reverse, new_size=(image_h, image_w))

                out_x, out_y = Vis.get_matrix_max_pos(out_map_reverse)
                gt_x, gt_y = Vis.get_matrix_max_pos(gt_map_reverse)
                d2 = np.sqrt(((out_x - gt_x) ** 2 + (out_y - gt_y) ** 2) / s_square)
                joints_d2_vec[j].append(d2)

        return np.std(joints_d2_vec, axis=1)

    def __compute_sigma(self,
                        data_loader: DataLoader):

        self.model.eval()
        res = []
        device = next(self.model.parameters()).device
        for _, info in enumerate(data_loader):
            image = info['image'].to(device)
            gt_map = info['gt_map'].to(device)
            center_map = info['center_map'].to(device)

            out = self.model(image, center_map)
            # out = out[:, -1]
            out = self.model.get_best_out(out)
            assert out.shape == gt_map.shape

            sig = self.compute_std_for_batch(out, gt_map, self.image_h, self.image_w)
            res.append(sig)

        res = np.array(res)
        return np.mean(res, axis=0)

    @staticmethod
    def compute_oks_for_batch(out_map: torch.Tensor,
                              gt_map: torch.Tensor,
                              sigma: np.ndarray,
                              image_h: int,
                              image_w: int,
                              average: bool = True):
        assert out_map.shape == gt_map.shape
        variances = (2 * sigma) ** 2
        image_num, joints_num = gt_map.shape[0], gt_map.shape[1] - 1
        res = [[] for _ in range(joints_num)]
        s_square = 1.0 * image_h * image_w

        for i in range(image_num):
            for j in range(joints_num):
                out_map_reverse = out_map[i][j].cpu().detach().numpy().copy()
                out_map_reverse = CV2.resize(out_map_reverse, new_size=(image_h, image_w))

                gt_map_reverse = gt_map[i][j].cpu().detach().numpy().copy()
                gt_map_reverse = CV2.resize(gt_map_reverse, new_size=(image_h, image_w))

                out_x, out_y = Vis.get_matrix_max_pos(out_map_reverse)
                gt_x, gt_y = Vis.get_matrix_max_pos(gt_map_reverse)

                e = ((out_x - gt_x) ** 2 + (out_y - gt_y) ** 2) / 2 / s_square / variances[j]
                res[j].append(np.exp(-e))

        if average:
            return np.mean(res)
        else:
            return np.array(res)

    def eval_oks(self,
                 data_loader: DataLoader,
                 heat_map_sigma: float):
        self.model.eval()
        device = next(self.model.parameters()).device
        res = []
        for _, info in enumerate(data_loader):
            image = info['image'].to(device)
            gt_map = info['gt_map'].to(device)

            out = self.model(image, heat_map_sigma=heat_map_sigma)
            # out = out[:, -1]
            out = self.model.get_best_out(out)
            assert out.shape == gt_map.shape

            oks = self.compute_oks_for_batch(out,
                                             gt_map,
                                             self.oks_sigma,
                                             self.image_h,
                                             self.image_w,
                                             average=True)
            res.append(oks)
        res = np.array(res)
        return np.mean(res)

    def eval_map(self,
                 data_loader: DataLoader,
                 heat_map_sigma: float,
                 oks_threshold: list):

        self.model.eval()
        device = next(self.model.parameters()).device
        res = []
        for _, info in enumerate(data_loader):
            image = info['image'].to(device)
            gt_map = info['gt_map'].to(device)

            out = self.model(image, heat_map_sigma=heat_map_sigma)
            # out = out[:, -1]
            out = self.model.get_best_out(out)
            assert out.shape == gt_map.shape

            oks = self.compute_oks_for_batch(out,
                                             gt_map,
                                             self.oks_sigma,
                                             self.image_h,
                                             self.image_w,
                                             average=False)
            # joints_num * batch_num
            oks = np.mean(oks, axis=0)  # batch_num

            for threshold in oks_threshold:
                a = np.size(oks[oks > threshold])
                b = np.size(oks)
                res.append(1.0*a/b)

        res = np.array(res)
        return np.mean(res)

