from DeepPose.tools.dataset_define import get_strong_lsp_loader
from DeepPose.tools.config import Config
from DeepPose.tools.model_define import DeepPose
from DeepPose.tools.helper import Helper
import albumentations as alb


config = Config()
config.trainer_config.device = 'cuda:1'

m = DeepPose(
    nJoints=len(config.data_config.key_points)
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
        alb.Resize(config.data_config.image_size[0], config.data_config.image_size[1]),
        alb.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ], keypoint_params=alb.KeypointParams(format='xy'))

train_loader = get_strong_lsp_loader(
    config.data_config.root_path,
    train=True,
    transform=trans_train,
    batch_size=config.trainer_config.BATCH_SIZE,
    num_workers=4
)
trans_test = alb.Compose([
        alb.Resize(config.data_config.image_size[0], config.data_config.image_size[1]),
        alb.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ], keypoint_params=alb.KeypointParams(format='xy'))

test_loader = get_strong_lsp_loader(
    config.data_config.root_path,
    train=False,
    transform=trans_test,
    batch_size=config.trainer_config.BATCH_SIZE,
    num_workers=4
)

helper.go(
    train_loader,
    test_loader
)