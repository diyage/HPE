from CPM.tools.model_define import CPMNet
from CPM.tools.config import DataSetConfig,TrainerConfig
import torch
import torch.nn as nn
from CPM.tools.dataset_define import LspDataSet
from CPM.tools.trainer import CPMTrainer
from torch.utils.data import DataLoader

data_set_config = DataSetConfig()
trainer_config = TrainerConfig()

m = CPMNet(data_set_config).to(trainer_config.device)

d = LspDataSet(root='E:\PyCharm\DataSet\lsp', train=True)
dl = DataLoader(d, batch_size=2, shuffle=True)

trainer = CPMTrainer(m, opt_data_set=data_set_config, opt_trainer=trainer_config)
trainer.train(data_loader_train=dl, data_loader_test=None)



