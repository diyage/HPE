class DataSetConfig:
    image_h: int = 368
    image_w: int = 368
    heat_map_h: int = 45
    heat_map_w: int = 45
    image_c: int = 3
    key_points: list = ['Right ankle', 'Right knee', 'Right hip', 'Left hip', 'Left knee', 'Left ankle',
                        'Right wrist', 'Right elbow', 'Right shoulder', 'Left shoulder', 'Left elbow', 'Left wrist',
                        'Neck', 'Head top']
    heat_map_sigma: float = 0.15
    root_path = '/home/dell/data/LSP/lsp_dataset'
    connections: list = [(0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8), (9, 10), (10, 11), (8, 12), (9, 12),
                         (12, 13), ((2, 3), 12), ((2, 3), 2), ((2, 3), 3)]


class TrainerConfig:
    lr: float = 1e-3
    device: str = 'cuda:1'
    MAX_EPOCH: int = 2000
    WARM_UP_END_EPOCH: int = 5
    MAP_Threshold: list = [0.5, 0.6, 0.7]
    BATCH_SIZE: int = 32


class Config:
    ABS_PATH: str = '/home/dell/data2/models/'
    eval_frequency: int = 10
    data_config = DataSetConfig()
    trainer_config = TrainerConfig()


