"""Module to read TFRecord files' Scenario messages and generate two files:
idx_to_scenario file and scenario_metadata file.

"""
import argparse
import os
import pickle
import tensorflow as tf
import numpy as np
from waymo_open_dataset.protos import scenario_pb2
from re import search

class MetadataExtractor:
    def __init__(self, origin_data_root, dst_root, file_head):
        """Constructor for the class.
        
        Args:
        origin_data: str, pointing to the raw data folder.
        dst_root: str, pointing to the destination folder, where we plan
            to write the metadata.
        """
        self._origin_data_root = origin_data_root
        self._dst_root = dst_root
        self._num_of_files = 0
        self._num_of_data = 0
        self._file_regex = "^{}".format(file_head)
        print("the origin root: {}, dst root: {}, match regex: {}".format(
            self._origin_data_root, self._dst_root, self._file_regex))
        # create the dst folder if needed.
        if not os.path.exists(dst_root):
            os.makedirs(dst_root)
            print("Create the dst folder at " + dst_root)
        self._idx_to_scenario = {}
        self._scenario_metadata = {}

    def extract_metadata(self):
        """Extract the idx_to_scenario dict and scenario_metadata dict.

        _idx_to_scenario dict: whose key is the index of one training instance (int),
        and value is the scenario tfrecord file path.
        _scenario_metadata dict: whose key is the tfrecord file path, value is a list,
        which are start instance idx, instance number in this file.
        """
        
        file_names = [os.path.join(self._origin_data_root, f) 
                      for f in os.listdir(self._origin_data_root)
                      if os.path.isfile(os.path.join(self._origin_data_root, f)) and
                      search(self._file_regex, f) is not None]
        print("Read valid files: {}".format(len(file_names)))
        cumulative_count = 0
        processed_file_count = 0
        file_names.sort()
        total_file_count = len(file_names)
        for file in file_names:
            print("processing: {}".format(file))
            raw_dataset = tf.data.TFRecordDataset(file, buffer_size=1e7)
            num_in_file = 0
            start_idx = cumulative_count
            for _ in raw_dataset:
                self._idx_to_scenario[cumulative_count] = file
                num_in_file += 1
                cumulative_count += 1
                print(num_in_file)
            del raw_dataset
            self._scenario_metadata[file] = [start_idx, num_in_file]
            processed_file_count += 1
            print("instances in file: {}".format(num_in_file))
            if processed_file_count == int(total_file_count * 0.2):
                print("finished 20 percent files")
            elif processed_file_count == int(total_file_count * 0.4):
                print("finished 40 percent files")
            elif processed_file_count == int(total_file_count * 0.6):
                print("finished 60 percent files")
            elif processed_file_count == int(total_file_count * 0.8):
                print("finished 80 percent files")


    def dump_pickle(self):
        idx_to_scenario_file_path = self._dst_root + "/idx_to_scenario.pickle"
        scenario_metadata_file_path = self._dst_root + "/scenario_metadata.pickle"
        print("target idx to scenario path: {}".format(idx_to_scenario_file_path))
        print("target scenario metadata path: {}".format(scenario_metadata_file_path))
        with open(idx_to_scenario_file_path, 'wb') as f:
            pickle.dump(self._idx_to_scenario, f)
        print("finish write idx to scenario at {}".format(
            idx_to_scenario_file_path))
        with open(scenario_metadata_file_path, 'wb') as f:
            pickle.dump(self._scenario_metadata, f)
        print("finish write scenario metadata at {}".format(
            scenario_metadata_file_path))

    def debug_load_pickle(self):
        idx_to_scenario_file_path = self._dst_root + "/idx_to_scenario.pickle"
        scenario_metadata_file_path = self._dst_root + "/scenario_metadata.pickle"
        print("target idx to scenario path: {}".format(idx_to_scenario_file_path))
        print("target scenario metadata path: {}".format(scenario_metadata_file_path))
        with open(idx_to_scenario_file_path, 'r') as f:
            idx_to_scenario = pickle.load(f)
            count = 0
            for key, value in idx_to_scenario.items():
                print(key, '->', value)
                count += 1
                if count > 5: break
        print("finish write idx to scenario at {}".format(
            idx_to_scenario_file_path))
        with open(scenario_metadata_file_path, 'r') as f:
            scenario_metadata = pickle.load(f)
            count = 0
            for key, value in scenario_metadata.items():
                print(key, '->', value)
                count += 1
                if count > 5: break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract metadata for tfrecords.')
    parser.add_argument('--origin_data_root', type=str,
                        help='path to the origin data folder')
    parser.add_argument('--dst_data_root', type=str,
                        help='path to the destination data folder')
    parser.add_argument('--file_head', type=str,
                        help='head substring for the file')
    args = parser.parse_args()
    print(args)
    metadata_extractor = MetadataExtractor(
        args.origin_data_root, args.dst_data_root, args.file_head)
    metadata_extractor.extract_metadata()
    metadata_extractor.dump_pickle()
    metadata_extractor.debug_load_pickle()
