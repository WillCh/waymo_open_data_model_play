"""Module to read TFRecord files' Scenario messages to pickle files so that we can build the loader.

"""
from ossaudiodev import SNDCTL_DSP_GETBLKSIZE
import tensorflow as tf
import math

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
        one_data_instance = {}
        # Process the sdc_history feature.
        # The feature dimensions are: [history_timestamps, attributions]. The attributions
        # include x, y, z, heading, v_x, v_y, length, width, height, valid],
        # which are centralized based on the sdc's current pose (i.e. x, y, heading).
        sdc_track = scenario.tracks[scenario.sdc_track_index]
        current_sdc_state = sdc_track.states[scenario.current_time_index]
        sdc_history_feature = []
        for i in range(scenario.current_time_index):
            sdc_history_feature.append(self.normalize_object_state(
                current_sdc_state.center_x,current_sdc_state.center_y, 
                current_sdc_state.center_z, current_sdc_state.heading, sdc_track.states[i]))
        one_data_instance['sdc_history_feature'] = sdc_history_feature

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

    def normalize_object_state(self, sdc_x, sdc_y, sdc_z, sdc_heading, object_state):
        """Normalizes the object_state's x, y, z, heading, v_x, v_y by the sdc_x, sdc_y
        and sdc_heading.
        
        Args:
            sdc_x: float, the pose of sdc at x axis (global coordination).
            sdc_y: float, the pose of sdc at y axis (global coordination).
            sdc_z: float, the pose of sdc at z axis (global coordination).
            sdc_heading: float (rad), the heading of sdc (the angle between
                sdc heading and global x axis).
            object_state: proto of ObjectState.

        Returns: a list of normalized x, y, z, heading, v_x, v_y,
            length, width, height, valid
        """
        if not object_state.valid:
            return [0] * 10
        normalized_x = object_state.center_x - sdc_x
        normalized_y = object_state.center_y - sdc_y
        normalized_z = object_state.center_z - sdc_z
        normalized_heading = self.normalize_heading_to_range(
            object_state.heading - sdc_heading)
        # rotate the velocity vector by the sdc heading clockwise.
        normalized_v_x = (object_state.velocity_x * math.cos(sdc_heading) +
            object_state.velocity_y * math.sin(sdc_heading))
        normalized_v_y = (-object_state.velocity_x * math.sin(sdc_heading) +
            object_state.velocity_y * math.cos(sdc_heading))

        return [normalized_x, normalized_y, normalized_z, normalized_heading, normalized_v_x, normalized_v_y,
                object_state.length, object_state.width, object_state.height, object_state.valid]
    def normalize_heading_to_range(self, heading):
        """Converts the heading (rads) to range of [-pi, pi)
        
        Args:
            heading: float, the rad of the heading.
        
        Returns: the normalized heading (float in rad)
        """
        updated_h = heading % (2.0 * math.pi)
        updated_h = updated_h - 2 * math.pi if updated_h > math.pi else updated_h
        return updated_h

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
