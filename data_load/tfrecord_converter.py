"""Module to read TFRecord files' Scenario messages to pickle files so that we can build the loader.

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
        # The list which contains the dicts. Each dict represents one training
        # instance. The key of dict is the name of feature, the val is the np
        # arrays.
        self._converted_data_list = []
        self._data_size = 0

    def process_one_tfrecord(self, scenario):
        """Converts the Scenario proto to the key-val dict. Inserts such
        dict into the _converted_data_list array.

        Args:
            scenario: scenario_pb2.Scenario object.

        Returns:
            None
        """
        # Process the sdc_history feature.
        # The feature dimensions are: [attributions]. The attributions
        # include x, y, z, heading, v_x, v_y, length, width, height, valid],
        # which are centralized based on the sdc's current pose (i.e. x, y, heading).

        # Process the other agent history feature.
        # The feature dimensions are: [# other agents, attributions]. The
        # attributions include x, y, z, heading, v_x, v_y, length, width, height, valid],
        # which are centralized based on the sdc's current pose (i.e. x, y, heading).

        # Process the other agent future groundtruth.
        # The feature dimensions are: [# other agents, x, y, heading, valid],
        # which are centralized based on the sdc's current pose (i.e. x, y, heading).

        # Process the sdc future groundtruth.
        # The feature dimensions are: [4], the elements are x, y, heading, valid.
        # which are centralized based on the sdc's current pose (i.e. x, y, heading).

        # Process the roadmap polyline features.
        # The feautre dimensions are: [num of polylines, attributions]. The attributions
        # are: [x_s, y_s, x_e, y_e, p_x, p_y, length, |p|]

        # process the metadata features.

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
        '/home/willch/Proj/waymo_open_challenage/data/training/uncompressed_scenario_training_training.tfrecord-00000-of-01000']
    raw_dataset = tf.data.TFRecordDataset(filenames)
    num = 0
    for raw_record in raw_dataset:
        example = scenario_pb2.Scenario()
        example.ParseFromString(raw_record.numpy())
        print(example)
        break
        num += 1
    print(num)
