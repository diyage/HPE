from DeepPose.tools.dataset_define import LspDataSet
from torch.utils.data import DataLoader
from DeepPose.tools.config import DataSetConfig, TrainerConfig
from DeepPose.tools.model_define import DeepPose
from DeepPose.tools.trainer import DeepPoseTrainer

opt_data_set = DataSetConfig()
opt_data_set.BATCH_SIZE = 1
opt_data_set.root_path = 'E:\PyCharm\DataSet\lsp/'
opt_trainer = TrainerConfig()
opt_trainer.device = 'cpu'

d_train = LspDataSet(opt_data_set.root_path, data_set_opt=opt_data_set, train=True)
d_train_l = DataLoader(d_train, batch_size=opt_data_set.BATCH_SIZE, shuffle=True)

d_test = LspDataSet(opt_data_set.root_path, data_set_opt=opt_data_set, train=False)
d_test_l = DataLoader(d_test, batch_size=opt_data_set.BATCH_SIZE, shuffle=False)

m = DeepPose(nJoints=len(opt_data_set.key_points)).to(opt_trainer.device)

trainer = DeepPoseTrainer(m, opt_trainer=opt_trainer, opt_data_set=opt_data_set)

from DeepPose.tools.visualize import Vis
for batch_index, (x, y) in enumerate(d_test_l):
    x = x.to(opt_trainer.device)
    y = y.to(opt_trainer.device)
    out = trainer.model(x)

    for index in range(x.shape[0]):

        Vis.plot_key_points(x[index], y[index],
                            connections=opt_data_set.connections,
                            )
        Vis.plot_key_points(x[index], out[index],
                            connections=opt_data_set.connections,
                            )

