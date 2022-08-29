from CPM.tools.dataset_define import get_strong_lsp_loader
from CPM.tools.config import Config
from CPM.tools.model_define import CPMNet
from CPM.tools.helper import Helper
import albumentations as alb
""""
This project has a little bug(s)!!
"""

config = Config()
config.trainer_config.device = 'cuda:1'
config.data_config.heat_map_sigma = 0.1

m = CPMNet(
    config.data_config
).to(config.trainer_config.device)

helper = Helper(
    m,
    config,
    restore_epoch=-1
)

trans_train = alb.Compose([
        alb.HueSaturationValue(),
        alb.RandomBrightnessContrast(),
        alb.RandomRotate90(),
        alb.ColorJitter(),
        alb.Blur(),
        alb.HorizontalFlip(),
        alb.VerticalFlip(),
        alb.GaussNoise(),
        alb.Resize(config.data_config.image_w, config.data_config.image_h),
        alb.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ], keypoint_params=alb.KeypointParams(format='xy'))

train_loader = get_strong_lsp_loader(
    config.data_config.root_path,
    train=True,
    transform=trans_train,
    heat_map_size=(config.data_config.heat_map_w, config.data_config.heat_map_h),
    heat_map_sigma=config.data_config.heat_map_sigma,
    image_size=(config.data_config.image_w, config.data_config.image_h),
    batch_size=config.trainer_config.BATCH_SIZE,
    num_workers=4
)
trans_test = alb.Compose([
        alb.Resize(config.data_config.image_w, config.data_config.image_h),
        alb.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ], keypoint_params=alb.KeypointParams(format='xy'))

test_loader = get_strong_lsp_loader(
    config.data_config.root_path,
    train=False,
    transform=trans_test,
    heat_map_size=(config.data_config.heat_map_w, config.data_config.heat_map_h),
    heat_map_sigma=config.data_config.heat_map_sigma,
    image_size=(config.data_config.image_w, config.data_config.image_h),
    batch_size=config.trainer_config.BATCH_SIZE,
    num_workers=4
)

helper.go(
    train_loader,
    test_loader
)
