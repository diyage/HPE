from CPM.tools.model_define import CPMNet
from CPM.tools.config import DataSetConfig, TrainerConfig
from CPM.tools.dataset_define import LspDataSet
from CPM.tools.trainer import CPMTrainer
from torch.utils.data import DataLoader

data_set_config = DataSetConfig()
trainer_config = TrainerConfig()

model = CPMNet(data_set_config).to(trainer_config.device)

d_train = LspDataSet(root=data_set_config.root_path, train=True)
d_l_train = DataLoader(d_train, batch_size=data_set_config.BATCH_SIZE, shuffle=True)

d_test = LspDataSet(root=data_set_config.root_path, train=False)
d_l_test = DataLoader(d_test, batch_size=data_set_config.BATCH_SIZE, shuffle=False)

trainer = CPMTrainer(model,
                     opt_data_set=data_set_config,
                     opt_trainer=trainer_config)

trainer.train(data_loader_train=d_l_train,
              data_loader_test=d_l_test)



