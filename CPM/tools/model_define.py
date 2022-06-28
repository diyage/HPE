import torch.nn as nn
import torch.nn.functional as F
import torch
from CPM.tools.config import DataSetConfig
from CPM.tools.dataset_define import guassian_kernel


class CPMNet(nn.Module):
    def __init__(self, data_set_config: DataSetConfig):
        super().__init__()
        self.img_h = data_set_config.image_h
        self.img_w = data_set_config.image_w
        self.out_c = len(data_set_config.key_points)

        self.pool_center_lower = nn.AvgPool2d(kernel_size=9, stride=8)
        ###########################################################################
        self.conv1_stage1 = nn.Conv2d(3, 128, kernel_size=9, padding=4)
        self.bn1_stage1 = nn.BatchNorm2d(128)
        self.pool1_stage1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2_stage1 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        self.bn2_stage1 = nn.BatchNorm2d(128)
        self.pool2_stage1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3_stage1 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        self.bn3_stage1 = nn.BatchNorm2d(128)
        self.pool3_stage1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv4_stage1 = nn.Conv2d(128, 32, kernel_size=5, padding=2)
        self.bn4_stage1 = nn.BatchNorm2d(32)

        self.conv5_stage1 = nn.Conv2d(32, 512, kernel_size=9, padding=4)
        self.bn5_stage1 = nn.BatchNorm2d(512)

        self.conv6_stage1 = nn.Conv2d(512, 512, kernel_size=1)
        self.bn6_stage1 = nn.BatchNorm2d(512)

        self.conv7_stage1 = nn.Conv2d(512, self.out_c + 1, kernel_size=1)
        ###########################################################################

        ###########################################################################
        self.conv1_stage2 = nn.Conv2d(3, 128, kernel_size=9, padding=4)
        self.bn1_stage2 = nn.BatchNorm2d(128)
        self.pool1_stage2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2_stage2 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        self.bn2_stage2 = nn.BatchNorm2d(128)
        self.pool2_stage2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3_stage2 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        self.bn3_stage2 = nn.BatchNorm2d(128)
        self.pool3_stage2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv4_stage2 = nn.Conv2d(128, 32, kernel_size=5, padding=2)
        self.bn4_stage2 = nn.BatchNorm2d(32)

        self.Mconv1_stage2 = nn.Conv2d(32 + self.out_c + 2, 128, kernel_size=11, padding=5)
        self.Mbn1_stage2 = nn.BatchNorm2d(128)

        self.Mconv2_stage2 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mbn2_stage2 = nn.BatchNorm2d(128)

        self.Mconv3_stage2 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mbn3_stage2 = nn.BatchNorm2d(128)

        self.Mconv4_stage2 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.Mbn4_stage2 = nn.BatchNorm2d(128)

        self.Mconv5_stage2 = nn.Conv2d(128, self.out_c + 1, kernel_size=1, padding=0)

        ###########################################################################

        ###########################################################################
        self.conv1_stage3 = nn.Conv2d(128, 32, kernel_size=5, padding=2)
        self.bn1_stage3 = nn.BatchNorm2d(32)

        self.Mconv1_stage3 = nn.Conv2d(32 + self.out_c + 2, 128, kernel_size=11, padding=5)
        self.Mbn1_stage3 = nn.BatchNorm2d(128)

        self.Mconv2_stage3 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mbn2_stage3 = nn.BatchNorm2d(128)

        self.Mconv3_stage3 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mbn3_stage3 = nn.BatchNorm2d(128)

        self.Mconv4_stage3 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.Mbn4_stage3 = nn.BatchNorm2d(128)

        self.Mconv5_stage3 = nn.Conv2d(128, self.out_c + 1, kernel_size=1, padding=0)
        ###########################################################################
        ###########################################################################
        self.conv1_stage4 = nn.Conv2d(128, 32, kernel_size=5, padding=2)
        self.bn1_stage4 = nn.BatchNorm2d(32)

        self.Mconv1_stage4 = nn.Conv2d(32 + self.out_c + 2, 128, kernel_size=11, padding=5)
        self.Mbn1_stage4 = nn.BatchNorm2d(128)

        self.Mconv2_stage4 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mbn2_stage4 = nn.BatchNorm2d(128)

        self.Mconv3_stage4 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mbn3_stage4 = nn.BatchNorm2d(128)

        self.Mconv4_stage4 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.Mbn4_stage4 = nn.BatchNorm2d(128)

        self.Mconv5_stage4 = nn.Conv2d(128, self.out_c + 1, kernel_size=1, padding=0)
        ###########################################################################
        ###########################################################################
        self.conv1_stage5 = nn.Conv2d(128, 32, kernel_size=5, padding=2)
        self.bn1_stage5 = nn.BatchNorm2d(32)

        self.Mconv1_stage5 = nn.Conv2d(32 + self.out_c + 2, 128, kernel_size=11, padding=5)
        self.Mbn1_stage5 = nn.BatchNorm2d(128)

        self.Mconv2_stage5 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mbn2_stage5 = nn.BatchNorm2d(128)

        self.Mconv3_stage5 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mbn3_stage5 = nn.BatchNorm2d(128)

        self.Mconv4_stage5 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.Mbn4_stage5 = nn.BatchNorm2d(128)

        self.Mconv5_stage5 = nn.Conv2d(128, self.out_c + 1, kernel_size=1, padding=0)
        ###########################################################################
        ###########################################################################
        self.conv1_stage6 = nn.Conv2d(128, 32, kernel_size=5, padding=2)
        self.bn1_stage6 = nn.BatchNorm2d(32)

        self.Mconv1_stage6 = nn.Conv2d(32 + self.out_c + 2, 128, kernel_size=11, padding=5)
        self.Mbn1_stage6 = nn.BatchNorm2d(128)

        self.Mconv2_stage6 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mbn2_stage6 = nn.BatchNorm2d(128)

        self.Mconv3_stage6 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mbn3_stage6 = nn.BatchNorm2d(128)

        self.Mconv4_stage6 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.Mbn4_stage6 = nn.BatchNorm2d(128)

        self.Mconv5_stage6 = nn.Conv2d(128, self.out_c + 1, kernel_size=1, padding=0)

    def _stage1(self, image):
        """
        Output result of stage 1
        :param image: source image with (368, 368)
        :return: conv7_stage1_map
        """
        x = self.pool1_stage1(F.relu(self.bn1_stage1(self.conv1_stage1(image))))
        x = self.pool2_stage1(F.relu(self.bn2_stage1(self.conv2_stage1(x))))
        x = self.pool3_stage1(F.relu(self.bn3_stage1(self.conv3_stage1(x))))
        x = F.relu(self.bn4_stage1(self.conv4_stage1(x)))
        x = F.relu(self.bn5_stage1(self.conv5_stage1(x)))
        x = F.relu(self.bn6_stage1(self.conv6_stage1(x)))
        x = self.conv7_stage1(x)

        return x

    def _middle(self, image):
        """
        Compute shared pool3_stage_map for the following stage
        :param image: source image with (368, 368)
        :return: pool3_stage2_map
        """
        x = self.pool1_stage2(F.relu(self.bn1_stage2(self.conv1_stage2(image))))
        x = self.pool2_stage2(F.relu(self.bn2_stage2(self.conv2_stage2(x))))
        x = self.pool3_stage2(F.relu(self.bn3_stage2(self.conv3_stage2(x))))

        return x

    def _stage2(self, pool3_stage2_map, conv7_stage1_map, pool_center_lower_map):
        """
        Output result of stage 2
        :param pool3_stage2_map
        :param conv7_stage1_map
        :param pool_center_lower_map:
        :return: Mconv5_stage2_map
        """
        x = F.relu(self.bn4_stage2(self.conv4_stage2(pool3_stage2_map)))
        x = torch.cat([x, conv7_stage1_map, pool_center_lower_map], dim=1)
        x = F.relu(self.Mbn1_stage2(self.Mconv1_stage2(x)))
        x = F.relu(self.Mbn2_stage2(self.Mconv2_stage2(x)))
        x = F.relu(self.Mbn3_stage2(self.Mconv3_stage2(x)))
        x = F.relu(self.Mbn4_stage2(self.Mconv4_stage2(x)))
        x = self.Mconv5_stage2(x)

        return x

    def _stage3(self, pool3_stage2_map, Mconv5_stage2_map, pool_center_lower_map):
        """
        Output result of stage 3
        :param pool3_stage2_map:
        :param Mconv5_stage2_map:
        :param pool_center_lower_map:
        :return: Mconv5_stage3_map
        """
        x = F.relu(self.bn1_stage3(self.conv1_stage3(pool3_stage2_map)))
        x = torch.cat([x, Mconv5_stage2_map, pool_center_lower_map], dim=1)
        x = F.relu(self.Mbn1_stage3(self.Mconv1_stage3(x)))
        x = F.relu(self.Mbn2_stage3(self.Mconv2_stage3(x)))
        x = F.relu(self.Mbn3_stage3(self.Mconv3_stage3(x)))
        x = F.relu(self.Mbn4_stage3(self.Mconv4_stage3(x)))
        x = self.Mconv5_stage3(x)

        return x

    def _stage4(self, pool3_stage2_map, Mconv5_stage3_map, pool_center_lower_map):
        """
        Output result of stage 4
        :param pool3_stage2_map:
        :param Mconv5_stage3_map:
        :param pool_center_lower_map:
        :return:Mconv5_stage4_map
        """
        x = F.relu(self.bn1_stage4(self.conv1_stage4(pool3_stage2_map)))
        x = torch.cat([x, Mconv5_stage3_map, pool_center_lower_map], dim=1)
        x = F.relu(self.Mbn1_stage4(self.Mconv1_stage4(x)))
        x = F.relu(self.Mbn2_stage4(self.Mconv2_stage4(x)))
        x = F.relu(self.Mbn3_stage4(self.Mconv3_stage4(x)))
        x = F.relu(self.Mbn4_stage4(self.Mconv4_stage4(x)))
        x = self.Mconv5_stage4(x)

        return x

    def _stage5(self, pool3_stage2_map, Mconv5_stage4_map, pool_center_lower_map):
        """
        Output result of stage 5
        :param pool3_stage2_map:
        :param Mconv5_stage4_map:
        :param pool_center_lower_map:
        :return:Mconv5_stage5_map
        """
        x = F.relu(self.bn1_stage5(self.conv1_stage5(pool3_stage2_map)))
        x = torch.cat([x, Mconv5_stage4_map, pool_center_lower_map], dim=1)
        x = F.relu(self.Mbn1_stage5(self.Mconv1_stage5(x)))
        x = F.relu(self.Mbn2_stage5(self.Mconv2_stage5(x)))
        x = F.relu(self.Mbn3_stage5(self.Mconv3_stage5(x)))
        x = F.relu(self.Mbn4_stage5(self.Mconv4_stage5(x)))
        x = self.Mconv5_stage5(x)

        return x

    def _stage6(self, pool3_stage2_map, Mconv5_stage5_map, pool_center_lower_map):
        """
        Output result of stage 6
        :param pool3_stage2_map:
        :param Mconv5_stage6_map:
        :param pool_center_lower_map:
        :return:Mconv5_stage6_map
        """
        x = F.relu(self.bn1_stage6(self.conv1_stage6(pool3_stage2_map)))
        x = torch.cat([x, Mconv5_stage5_map, pool_center_lower_map], dim=1)
        x = F.relu(self.Mbn1_stage6(self.Mconv1_stage6(x)))
        x = F.relu(self.Mbn2_stage6(self.Mconv2_stage6(x)))
        x = F.relu(self.Mbn3_stage6(self.Mconv3_stage6(x)))
        x = F.relu(self.Mbn4_stage6(self.Mconv4_stage6(x)))
        x = self.Mconv5_stage6(x)

        return x

    def forward(self, image, center_map=None, heat_map_sigma=None):
        assert tuple(image.data.shape[-2:]) == (self.img_h, self.img_w)

        if center_map is None:
            # eval phase
            center_map = guassian_kernel(self.img_w, self.img_h, self.img_w/2, self.img_h/2, heat_map_sigma)
            center_map = torch.tensor(center_map, dtype=torch.float32).to(image.device)
            center_map = center_map.unsqueeze(0).expand(image.shape[0], image.shape[2], image.shape[3])

        pool_center_lower_map = self.pool_center_lower(center_map)
        pool_center_lower_map = pool_center_lower_map.unsqueeze(1)

        conv7_stage1_map = self._stage1(image)  # result of stage 1

        pool3_stage2_map = self._middle(image)

        Mconv5_stage2_map = self._stage2(pool3_stage2_map, conv7_stage1_map,
                                         pool_center_lower_map)  # result of stage 2
        Mconv5_stage3_map = self._stage3(pool3_stage2_map, Mconv5_stage2_map,
                                         pool_center_lower_map)  # result of stage 3
        Mconv5_stage4_map = self._stage4(pool3_stage2_map, Mconv5_stage3_map,
                                         pool_center_lower_map)  # result of stage 4
        Mconv5_stage5_map = self._stage5(pool3_stage2_map, Mconv5_stage4_map,
                                         pool_center_lower_map)  # result of stage 5
        Mconv5_stage6_map = self._stage6(pool3_stage2_map, Mconv5_stage5_map,
                                         pool_center_lower_map)  # result of stage 6

        return torch.stack([conv7_stage1_map, Mconv5_stage2_map, Mconv5_stage3_map,
                            Mconv5_stage4_map, Mconv5_stage5_map, Mconv5_stage6_map], dim=1)

