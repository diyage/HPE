class DataSetConfig:
    root_path = '/home/dell/data/LSP/lsp_dataset'
    key_points: list = ['Right ankle', 'Right knee', 'Right hip', 'Left hip', 'Left knee', 'Left ankle',
                        'Right wrist', 'Right elbow', 'Right shoulder', 'Left shoulder', 'Left elbow', 'Left wrist',
                        'Neck', 'Head top']
    connections: list = [(0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8), (9, 10), (10, 11), (8, 12), (9, 12),
                         (12, 13), ((2, 3), 12), ((2, 3), 2), ((2, 3), 3)]

    image_size: tuple = (220, 220)
    BATCH_SIZE: int = 2


class TrainerConfig:
    lr: float = 2e-4
    device: str = 'cpu'
    MAX_EPOCH: int = 200
    ABS_PATH: str = '/home/dell/data2/models/'


