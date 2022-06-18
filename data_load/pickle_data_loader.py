""" Class to load the pickle files in the training/testing process.
"""

from asyncio import DatagramProtocol
import os
import torch
import pickle

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class WaymoMotionPickleDataset(Dataset):
    def __init__(self, id_to_file_map_path, file_metadata_path):
        with open(id_to_file_map_path, 'rb') as handle:
            self._example_id_to_file_name = pickle.load(handle)
        with open(file_metadata_path, 'rb') as handle:
            self._file_name_to_metadata = pickle.load(handle)
        self._num = 0
        for _, metadata in self._file_name_to_metadata.items():
            self._num += metadata['data_size']
        self._current_data_file = ''
        self._converted_data_list = []
        self._current_file_metadata = None

    def __len__(self):
        return self._num

    def __getitem__(self, idx):
        src_file_name = self._example_id_to_file_name[idx]
        if src_file_name == self._current_data_file:
            return self._converted_data_list[
                idx - self._current_file_metadata['start_idx']]

        with open(src_file_name, 'rb') as handle:
            self._converted_data_list = pickle.load(handle)
        self._current_data_file = src_file_name
        self._current_file_metadata = self._file_name_to_metadata[src_file_name]
        return self._converted_data_list[
            idx - self._current_file_metadata['start_idx']]


if __name__ == '__main__':
    id_to_file_map_path = '/home/willch/Proj/waymo_open_challenage/pickle_files/training/example_id_to_file_name.pickle'
    file_metadata_path = '/home/willch/Proj/waymo_open_challenage/pickle_files/training/file_name_to_metadata.pickle'
    pickle_dataset = WaymoMotionPickleDataset(id_to_file_map_path, file_metadata_path)
    pickle_dataloader = DataLoader(pickle_dataset, batch_size =2, shuffle=True)
    one_batch_data = next(iter(pickle_dataloader))
    print(one_batch_data['sdc_history_feature'].size())
    print(one_batch_data['map_feature'].size())
    print(one_batch_data['sdc_future_feature'].size())
    print(one_batch_data['agent_history_feature'].size())