import sys

sys.path.insert(0, '/home/willch/Proj/waymo_open_challenage')
import torch
from torch.utils.data import DataLoader
from waymo_open_data_model_play.data_load.pickle_data_loader import WaymoMotionPickleDataset
from waymo_open_data_model_play.model.navie_model import MlpSdcNet


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    train_id_to_file_map_path = '/home/willch/Proj/waymo_open_challenage/' + \
        'pickle_files/training/example_id_to_file_name.pickle'
    train_file_metadata_path = '/home/willch/Proj/waymo_open_challenage/' + \
        'pickle_files/training/file_name_to_metadata.pickle'

    waymo_train_data_set = WaymoMotionPickleDataset(
        train_id_to_file_map_path, train_file_metadata_path)

    batch_size = 32
    train_dataloader = DataLoader(waymo_train_data_set,
                                batch_size,
                                shuffle=True)

    for one_data_instance in train_dataloader:
        print(one_data_instance['sdc_history_feature'].shape)
        break