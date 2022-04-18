"""Module to read TFRecord files' Scenario proto message

"""
import tensorflow as tf

import numpy as np
from waymo_open_dataset.protos import scenario_pb2

class DataConverter:
    """Class to read the TFRecord and convert the proto training
    data into dict, whose key is a string, value is a np array. This
    class provides API to dump the list of above dicts into pickle files.
    """
    def __init__(self):
        """Default constructor for the converter class.
        """
        self.__converted_data_maps = []

    def process_one_tfrecord(self, scenario):
        """Converts the Scenario proto to the key-val dict. Inserts such
        dict into the __converted_data_maps array.

        Args:
            scenario: scenario_pb2.Scenario object.

        Returns:
            None
        """

    def process_one_tfrecord_file(self, src_file_path, dst_file_path):
        """Read one TFRecord file based on src_file_path, and dump it as a pickle file.

        Args:
            src_file_path: str pointing to a TFRecord file.
            dst_file_path: str pointing to a destination pickle file.

        Returns:
            None
        """

    def process_all_tfrecord_file(self, src_folder_path, dst_folder_path):
        """Read all TFRecord files based on source folder path, and dump all of them into
        destination file path.

        Args:
            src_folder_path: str pointing to a folder which holds TFRecord files.
            dst_folder_path: str pointing to the destination folder to write pickle files.

        Returns:
            None
        """

if __name__ == '__main__':
    filenames = [
        '/home/ryan/Documents/waymo_motion/data/scenario/testing/testing.tfrecord-00000-of-00150']
    raw_dataset = tf.data.TFRecordDataset(filenames)

    for raw_record in raw_dataset.take(1):
        example = scenario_pb2.Scenario()
        example.ParseFromString(raw_record.numpy())
        print(example)
        break
