import sys

sys.path.insert(0, '/home/willch/Proj/waymo_open_challenage')
import torch
from torch.utils.data import DataLoader
from waymo_open_data_model_play.data_load.pickle_data_loader import WaymoMotionPickleDataset
from waymo_open_data_model_play.model.navie_model import BaselineSimplyMp
from waymo_open_data_model_play.model.loss_util import CombinedRegressionClassificationLoss


def train_loop(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, one_data_instance in enumerate(dataloader):
        # Compute prediction and loss
        sdc_history_feature = one_data_instance['sdc_history_feature'].float()
        agent_history_feature = one_data_instance['agent_history_feature'].float()
        map_feature = one_data_instance['map_feature'].float()
        sdc_future_feature = one_data_instance['sdc_future_feature'][:, :, 0:2].float()
        sdc_history_feature = sdc_history_feature.to(device)
        agent_history_feature = agent_history_feature.to(device)
        sdc_future_feature = sdc_future_feature.to(device)
        map_feature = map_feature.to(device)

        pred, logit = model(sdc_history_feature,
                     agent_history_feature,
                     map_feature)
        loss = loss_fn(sdc_future_feature, pred, logit)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(sdc_history_feature)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


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
    # Pinned memory is used as a staging area for data transfers between the CPU
    # and the GPU. By setting pin_memory=True when we initialise the data loader
    # we are directly allocating space in pinned memory. This avoids the time cost
    # of transfering data from the host to the pinned (non-pageable) staging area
    # every time we move the data onto the GPU later in the code. You can read more
    # about pinned memory on the nvidia blog.
    kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}
    train_dataloader = DataLoader(waymo_train_data_set,
                                  batch_size,
                                  shuffle=True, **kwargs)

    model = BaselineSimplyMp(
        num_future_states=80, num_trajs=3,
        history_timestamps=10, sdc_attribution_dim=10,
        agent_attribution_dim=10, map_attribution_dim=11).to(device)

    loss_fn = CombinedRegressionClassificationLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, device)
        print("Done!")
        """
    for one_data_instance in train_dataloader:
        print(one_data_instance['sdc_history_feature'].shape)
        break
    """