from DeepPose.tools.dataset_define import LspDataSet
from torch.utils.data import DataLoader
from DeepPose.tools.config import DataSetConfig
from DeepPose.tools.model_define import DeepPose
from DeepPose.tools.trainer import DeepPoseTrainer

opt_data_set = DataSetConfig()

d_train = LspDataSet(opt_data_set.root_path, data_set_opt=opt_data_set, train=True)
d_train_l = DataLoader(d_train, batch_size=opt_data_set.BATCH_SIZE, shuffle=True)

d_test = LspDataSet(opt_data_set.root_path, data_set_opt=opt_data_set, train=False)
d_test_l = DataLoader(d_test, batch_size=opt_data_set.BATCH_SIZE, shuffle=False)

m = DeepPose(nJoints=len(opt_data_set.key_points))

trainer = DeepPoseTrainer(m)
trainer.train(d_train_l, d_test_l)