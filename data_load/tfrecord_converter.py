"""Module to read TFRecord files' Scenario messages to pickle files so that we can build the loader.

"""
from ossaudiodev import SNDCTL_DSP_GETBLKSIZE
import tensorflow as tf
import math
import pickle
import os
import re

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
        # Number of training instance.
        self._data_size = 0
        self._max_agent_num = max_agent_num
        self._max_polyline_num = max_polyline_num
        # The key is example id (an global int which is the idx of example instance).
        # The value is saved pickle file name. The torch will use this dict to
        # load specific pickle file and find the training instance.
        self._example_id_to_file_name = {}
        # The dict whose key is the pickle file path, and value is the metadata
        # of such pickle file.
        self._file_name_to_metadata = {}

    def process_one_tfrecord(self, scenario):
        """Converts the Scenario proto to the key-val dict. Inserts such
        dict into the _converted_data_list array. There are 6 steps for the data
        processing (all the coordinations are sdc centralized):
            1) process sdc history features which are [history_timestamps, sdc attributions];
            2) process sdc future features which are [future_timestamps, 4], the elements are x, y,
                heading, valid;
            3) process other agent history features which are: [# other agents, 
                history_timestamps, other agent's attributions]; 
            4) process other agent future features which are [# other agents, future_timestamps, 4];
            5) process map features which are [num of polylines, attributions];
            6) process metadata features.

        Args:
            scenario: scenario_pb2.Scenario object.

        Returns:
            None
        """
        one_data_instance = {}
        # 1) Process the sdc_history feature.
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
        one_data_instance['sdc_history_feature'] = np.array(sdc_history_feature)

        # 2) Process the sdc future groundtruth.
        # The feature dimensions are: [future_timestamps, 4], the elements are x, y,
        # heading, valid, which are centralized based on the sdc's current pose 
        # (i.e. x, y, heading).
        sdc_future_feature = []
        for i in range(scenario.current_time_index + 1, len(sdc_track.states)):
            sdc_future_feature.append(self.normalize_object_state(
                current_sdc_state.center_x,current_sdc_state.center_y, 
                current_sdc_state.center_z, current_sdc_state.heading, sdc_track.states[i],
                is_label=True))
        one_data_instance['sdc_future_feature'] = np.array(sdc_future_feature)
        all_timestamps_count = len(sdc_track.states)

        # 3 & 4) Process the other agent history and future feature.
        # The feature dimensions are: [# other agents, history_timestamps, attributions]. The
        # attributions include x, y, z, heading, v_x, v_y, length, width, height, valid],
        # which are centralized based on the sdc's current pose (i.e. x, y, heading).
        agent_all_timestamps_feature = []
        for agent_idx in range(len(scenario.tracks)):
            if agent_idx == scenario.sdc_track_index: continue
            one_agent_feature = []
            for ts in range(all_timestamps_count):            
                one_agent_feature.append(self.normalize_object_state(
                    current_sdc_state.center_x,current_sdc_state.center_y, 
                    current_sdc_state.center_z, current_sdc_state.heading, scenario.tracks[agent_idx].states[ts]))
            agent_all_timestamps_feature.append(one_agent_feature)
        def agent_dist_to_sdc(agent_feature_list):
            for agent_at_time in agent_feature_list:
                if agent_at_time[-1]:
                    return agent_at_time[0] ** 2.0 + agent_at_time[1] ** 2.0
            return 1e6
        agent_all_timestamps_feature.sort(key=agent_dist_to_sdc)
        # Cut the agent by max_agent_num or append it with zeros.
        if len(agent_all_timestamps_feature) > self._max_agent_num:
            agent_all_timestamps_feature = agent_all_timestamps_feature[0 : self._max_agent_num]
        elif len(agent_all_timestamps_feature) < self._max_agent_num:
            zero_feature = [0] * 10
            zero_agent_feature = [zero_feature] * all_timestamps_count
            for _ in range(self._max_agent_num - len(agent_all_timestamps_feature)):
                agent_all_timestamps_feature.append(zero_agent_feature)

        # Separates the history agent features and future agent features.
        agent_history_feature = []
        agent_future_feature = []
        # The idx corresponds to x, y, heading, valid.
        future_attribution_idx = [0, 1, 3, 9]
        for agent_idx in range(self._max_agent_num):
            one_agent_all_timestamps_feature = agent_all_timestamps_feature[agent_idx]
            one_agent_history_feature = []
            one_agent_future_feature = []
            for history_ts in range(scenario.current_time_index):
                one_agent_history_feature.append(one_agent_all_timestamps_feature[history_ts])
            agent_history_feature.append(one_agent_history_feature)
            for future_ts in range(scenario.current_time_index + 1, all_timestamps_count):
                one_agent_one_timestamp_feature = [one_agent_all_timestamps_feature[future_ts][idx] for idx in future_attribution_idx]
                one_agent_future_feature.append(one_agent_one_timestamp_feature)
            agent_future_feature.append(one_agent_future_feature)
        one_data_instance['agent_history_feature'] = np.array(agent_history_feature)
        # The feature dimensions are: [# other agents, future_timestamps, attributions],
        # which are centralized based on the sdc's current pose (i.e. x, y, heading).
        one_data_instance['agent_future_feature'] = np.array(agent_future_feature)

        # 5) Process the roadmap polyline features.
        # The feautre dimensions are: [num of polylines, attributions]. The attributions
        # are: [x_s, y_s, x_e, y_e, p_x, p_y, |P|, |e-s|, lane_type, lane_id, valid].
        map_feature = []
        for m_feature in scenario.map_features:
            if m_feature.HasField("lane"):
                lane_features = self.generate_polyline_feature(
                    current_sdc_state.center_x,current_sdc_state.center_y, 
                    current_sdc_state.heading, m_feature.lane.polyline, m_feature.lane.type, m_feature.id)
                map_feature.extend(lane_features)
            elif m_feature.HasField("road_line"):
                road_line_feature = self.generate_polyline_feature(
                    current_sdc_state.center_x,current_sdc_state.center_y, 
                    current_sdc_state.heading, m_feature.road_line.polyline, m_feature.road_line.type + 4,
                    m_feature.id)
                map_feature.extend(road_line_feature)
            elif m_feature.HasField("road_edge"):
                road_edge_feature = self.generate_polyline_feature(
                    current_sdc_state.center_x,current_sdc_state.center_y, 
                    current_sdc_state.heading, m_feature.road_edge.polyline, m_feature.road_edge.type + 4 + 9,
                    m_feature.id)
                map_feature.extend(road_edge_feature)
            elif m_feature.HasField("stop_sign"):
                stop_sign_feature = self.generate_polyline_feature(
                    current_sdc_state.center_x,current_sdc_state.center_y, 
                    current_sdc_state.heading, [m_feature.stop_sign.position, m_feature.stop_sign.position],
                    4 + 9 + +3 + 1, m_feature.id)
                map_feature.extend(stop_sign_feature)
            # TODO (haoyu): add the features for crosswalk.
        # Sort and filter the polyline elements.
        def polyline_dist_to_sdc(polyline_feature_list):
            return polyline_feature_list[6]
        map_feature.sort(key=polyline_dist_to_sdc)
        # Cut the polyline features by max_polyline_num or append it with zeros.
        if len(map_feature) > self._max_polyline_num:
            map_feature = map_feature[0 : self._max_polyline_num]
        elif len(agent_history_feature) < self._max_polyline_num:
            # 11 means the total dimension here.
            zero_polyline_feature = [0] * 11
            for _ in range(self._max_polyline_num - len(map_feature)):
                map_feature.append(zero_polyline_feature)
        one_data_instance['map_feature'] = np.array(map_feature)

        # 6) process the metadata features.
        one_data_instance['scenario_id'] = scenario.scenario_id
        self._converted_data_list.append(one_data_instance)

    def generate_polyline_feature(self, sdc_x, sdc_y, sdc_heading, 
                                  points, lane_type, lane_id):
        """Generate the polyline feature lists for the points.

        Args:
            sdc_x: The x pose of sdc at the current timestamp.
            sdc_y: The y pose of sdc at the current timestamp.
            sdc_heading: The heading of sdc at the current timestamp.
            points: list of MapPoint proto.
            lane_type: int representing the type of lane.
            lane_id: int representing a unique id of the lane.

        Returns: list of list of extracted features. Each list element is a list 
            of feature attributions: [s_x, s_y, e_x, e_y, p_x, p_y,
            distance to ego, and length of polyline, lane_type, lane_id, valid]
        """
        results = []
        for point_idx in range(len(points)- 1):
            one_polyline_feature = self.normalize_one_polyline_section(
                points[point_idx].x, points[point_idx].y, 
                points[point_idx + 1].x, points[point_idx + 1].y,
                sdc_x, sdc_y, sdc_heading)
            one_polyline_feature.append(lane_type)
            one_polyline_feature.append(lane_id)
            # Appends valid which is always true here.
            one_polyline_feature.append(1.0)
            results.append(one_polyline_feature)
        return results

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
            start_x: global x coordination of the start point in the vector.
            start_y: global y coordination of the start point in the vector.
            end_x: global x coordination of the end point in the vector.
            end_y: global y coordination of the end point in the vector.
            sdc_x: global x coordination of the sdc pose.
            sdc_y: global y coordination of the sdc pose.
            sdc_heading: global heading of the sdc.

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
        example_id = self._data_size
        for raw_record in raw_dataset:
            example = scenario_pb2.Scenario()
            example.ParseFromString(raw_record.numpy())
            self.process_one_tfrecord(example)
            self._example_id_to_file_name[example_id] = dst_file_path
            num += 1
            example_id += 1
        with open(dst_file_path, 'wb') as handle:
            pickle.dump(self._converted_data_list, handle)
        file_metadata = {}
        file_metadata['start_idx'] = self._data_size
        file_metadata['data_size'] = num
        self._file_name_to_metadata[dst_file_path] = file_metadata
        self._data_size += num
        self._converted_data_list = []

    def process_all_tfrecord_file(self, src_folder_path, dst_folder_path, 
                                  src_file_prefix, dst_file_prefix):
        """Read all TFRecord files based on source folder path, and dump all of them into
        destination file path. This function also constructs the data instance index
        dict which will be used for pytorch data loader.

        Args:
            src_folder_path: str pointing to a folder which holds TFRecord files.
            dst_folder_path: str pointing to the destination folder to write pickle files.
            src_file_prefix: prefix of source file. We use this to check whether a file
                should be read in the src folder.
            dst_file_prefix: prefix of dst file, which we plan to write to.

        Returns:
            None
        """
        for file in os.listdir(src_folder_path):
            if file.startswith(src_file_prefix):
                src_file_name = os.path.join(src_folder_path, file)
                print("processing src: {}".format(src_file_name))
                start_idx = file.find('.tfrecord') + 9
                matched_tail = file[start_idx:]
                dst_file_name = dst_file_prefix + matched_tail + '.pickle'
                dst_file_name = os.path.join(dst_folder_path, dst_file_name)
                print('dst is {}'.format(dst_file_name))
                self.process_one_tfrecord_file(src_file_name, dst_file_name)
        # Writes the dict to the disk.
        dst_metadata_name = os.path.join(dst_folder_path,
                                         'example_id_to_file_name.pickle')
        with open(dst_metadata_name, 'wb') as handle:
            pickle.dump(self._example_id_to_file_name, handle)
        
        dst_metadata_name = os.path.join(dst_folder_path,
                                         'file_name_to_metadata.pickle')
        with open(dst_metadata_name, 'wb') as handle:
            pickle.dump(self._file_name_to_metadata, handle)


if __name__ == '__main__':
    filenames = [
        '/home/willch/Proj/waymo_open_challenage/data/training/uncompressed_scenario_training_training.tfrecord-00000-of-01000']
    data_converter = DataConverter(64, 4096)
    #data_converter.process_one_tfrecord_file(filenames[0], '/home/willch/Proj/waymo_open_challenage/pickle_files/test/test.pickle')
    data_converter.process_all_tfrecord_file(
        '/home/willch/Proj/waymo_open_challenage/data/validating/',
        '/home/willch/Proj/waymo_open_challenage/pickle_files/validating/',
        'uncompressed_scenario_validation_validation', 'converted_pickle_data')
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
