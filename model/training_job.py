import sys

sys.path.insert(0, '/home/willch/Proj/waymo_open_challenage')

from torch.utils.data import DataLoader
from waymo_open_data_model_play.data_load.pickle_data_loader import WaymoMotionPickleDataset