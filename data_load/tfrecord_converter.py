"""Module to read TFRecord files' Scenario messages to pickle files so that we can build the loader.

"""
from ossaudiodev import SNDCTL_DSP_GETBLKSIZE
import tensorflow as tf
import math
import pickle

import numpy as np
from waymo_open_dataset.protos import scenario_pb2

NEAR_ZERO = 1e-5

class DataConverter:
    """Class to read the TFRecord and convert the proto training
    data into dict, whose key is a string, value is a np array. This
    class provides API to dump the list of above dicts into pickle files.
    """
    def __init__(self, max_agent_num, max_polyline_num):
        """Default constructor for the converter class.
        """
        # The list which contains the dicts. Each dict represents one training
        # instance. The key of dict is the name of feature, the val is the np
        # arrays.
        self._converted_data_list = []
        self._data_size = 0
        self._max_agent_num = max_agent_num
        self._max_polyline_num = max_polyline_num

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
        # The feature dimensions are: [# other agents, history_timestamps, attributions]. The
        # attributions include x, y, z, heading, v_x, v_y, length, width, height, valid],
        # which are centralized based on the sdc's current pose (i.e. x, y, heading).
        agent_history_feature = []
        for agent_idx in range(len(scenario.tracks)):
            if agent_idx == scenario.sdc_track_index: continue
            one_agent_feature = []
            for i in range(scenario.current_time_index):            
                one_agent_feature.append(self.normalize_object_state(
                    current_sdc_state.center_x,current_sdc_state.center_y, 
                    current_sdc_state.center_z, current_sdc_state.heading, scenario.tracks[agent_idx].states[i]))
            agent_history_feature.append(one_agent_feature)
        def agent_dist_to_sdc(agent_feature_list):
            for agent_at_time in agent_feature_list:
                if agent_at_time[-1]:
                    return agent_at_time[0] ** 2.0 + agent_at_time[1] ** 2.0
            return 1e6
        agent_history_feature.sort(key=agent_dist_to_sdc)
        # Cut the agent by max_agent_num or append it with zeros.
        if len(agent_history_feature) > self._max_agent_num:
            agent_history_feature = agent_history_feature[0 : self._max_agent_num]
        elif len(agent_history_feature) < self._max_agent_num:
            zero_feature = [0] * 10
            zero_agent_feature = [zero_feature] * scenario.current_time_index
            for _ in range(self._max_agent_num - len(agent_history_feature)):
                agent_history_feature.append(zero_agent_feature)
        one_data_instance['agent_history_feature'] = agent_history_feature
        
        # TODO (haoyu): process the other agent future groundtruth.
        # The feature dimensions are: [# other agents, future_timestamps, attributions],
        # which are centralized based on the sdc's current pose (i.e. x, y, heading).

        # Process the sdc future groundtruth.
        # The feature dimensions are: [future_timestamps, 4], the elements are x, y,
        # heading, valid, which are centralized based on the sdc's current pose 
        # (i.e. x, y, heading).
        sdc_future_feature = []
        for i in range(scenario.current_time_index + 1, len(sdc_track.states)):
            sdc_future_feature.append(self.normalize_object_state(
                current_sdc_state.center_x,current_sdc_state.center_y, 
                current_sdc_state.center_z, current_sdc_state.heading, sdc_track.states[i],
                is_label=True))
        one_data_instance['sdc_future_feature'] = sdc_future_feature

        # Process the roadmap polyline features.
        # The feautre dimensions are: [num of polylines, attributions]. The attributions
        # are: [x_s, y_s, x_e, y_e, p_x, p_y, length, |p|]
        map_feature = []
        for m_feature in scenario.map_features:
            if m_feature.HasField("lane"):

            elif m_feature.HasField("road_line"):
            elif m_feature.HasField("road_edge"):
            elif m_feature.HasField("stop_sign"):

        # process the metadata features.
        self._converted_data_list.append(one_data_instance)

    def normalize_object_state(self, sdc_x, sdc_y, sdc_z, sdc_heading, object_state,
                               is_label=False):
        """Normalizes the object_state's x, y, z, heading, v_x, v_y by the sdc_x, sdc_y
        and sdc_heading.
        
        Args:
            sdc_x: float, the pose of sdc at x axis (global coordination).
            sdc_y: float, the pose of sdc at y axis (global coordination).
            sdc_z: float, the pose of sdc at z axis (global coordination).
            sdc_heading: float (rad), the heading of sdc (the angle between
                sdc heading and global x axis).
            object_state: proto of ObjectState.
            is_label: if True generate the states for ground truth.

        Returns: a list of normalized x, y, z, heading, v_x, v_y,
            length, width, height, valid if is_label is False, otherwise, a list of
            normalized x, y, and heading.
        """
        if not object_state.valid:
            return [0] * 10
        delta_x = object_state.center_x - sdc_x
        delta_y = object_state.center_y - sdc_y
        normalized_z = object_state.center_z - sdc_z
        normalized_x = (delta_x * math.cos(sdc_heading) + delta_y * math.sin(sdc_heading))
        normalized_y = (-delta_x * math.sin(sdc_heading) + delta_y * math.cos(sdc_heading))
        normalized_heading = self.normalize_heading_to_range(
            object_state.heading - sdc_heading)
        if is_label:
            return [normalized_x, normalized_y, normalized_heading]
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

    def normalize_one_polyline_section(
        self, start_x, start_y, end_x, end_y, sdc_x, sdc_y, sdc_heading):
        """Convert the one polyline segment to a coordinate whose center is at {sdc_x, sdc_y} and
        x axis is sdc_heading. Extract the polyline segment features attributions in the 
        new coordination. The math can be found at page:
        https://diego.assencio.com/?index=ec3d5dfdfc0b6a0d147a656f0af332bd.

        Args:
            start_x:
            start_y:
            end_x:
            end_y:
            sdc_x:
            sdc_y:
            sdc_heading:

        Returns: the normalized polyline segment attributions: [s_x, s_y, e_x, e_y, p_x, p_y,
            distance to ego, and length of polyline]
        """
        delta_s_x = start_x - sdc_x
        delta_s_y = start_y - sdc_y
        s_x = (delta_s_x * math.cos(sdc_heading) + delta_s_y * math.sin(sdc_heading))
        s_y = (-delta_s_x * math.sin(sdc_heading) + delta_s_y * math.cos(sdc_heading))
        delta_e_x = end_x - sdc_x
        delta_e_y = end_y - sdc_y
        e_x = (delta_e_x * math.cos(sdc_heading) + delta_e_y * math.sin(sdc_heading))
        e_y = (-delta_e_x * math.sin(sdc_heading) + delta_e_y * math.cos(sdc_heading))
        m_x = e_x - s_x
        m_y = e_y - s_y
        m_norm = (m_x ** 2.0 + m_y ** 2.0) ** 0.5
        if m_norm < NEAR_ZERO:
            return [s_x, s_y, e_x, e_y, s_x, s_y, (s_x ** 2.0 + s_y ** 2.0) ** 0.5, 0]
        # Note: we are at the normalized coordination, thus, the sdc pose is 0, 0.
        sdc_s_x = 0 - s_x
        sdc_s_y = 0 - s_y
        lambda_s = (sdc_s_x * m_x + sdc_s_y * m_y) / m_norm / m_norm
        if lambda_s < 0:
            p_x = s_x
            p_y = s_y
        elif lambda_s > 1:
            p_x = e_x
            p_y = e_y
        else:
            p_x = s_x + lambda_s * m_x
            p_y = s_y + lambda_s * m_y
        return [s_x, s_y, e_x, e_y, p_x, p_y, (p_x ** 2.0 + p_y ** 2.0) ** 0.5, m_norm]


    def process_one_tfrecord_file(self, src_file_path, dst_file_path):
        """Read one TFRecord file based on src_file_path, and dump it as a pickle file.

        Args:
            src_file_path: str pointing to a TFRecord file.
            dst_file_path: str pointing to a destination pickle file.

        Returns:
            None
        """
        raw_dataset = tf.data.TFRecordDataset(src_file_path)
        num = 0
        for raw_record in raw_dataset:
            example = scenario_pb2.Scenario()
            example.ParseFromString(raw_record.numpy())
            self.process_one_tfrecord(example)
            num += 1
            break
        with open(dst_file_path, 'wb') as handle:
            pickle.dump(self._converted_data_list, handle)

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
    data_converter = DataConverter(64, 512)
    data_converter.process_one_tfrecord_file(filenames[0], '/home/willch/Proj/waymo_open_challenage/pickle_files/test/test.pickle')

    """
    raw_dataset = tf.data.TFRecordDataset(filenames)
    num = 0
    for raw_record in raw_dataset:
        example = scenario_pb2.Scenario()
        example.ParseFromString(raw_record.numpy())
        print(example)
        break
        num += 1
    print(num)
    """
