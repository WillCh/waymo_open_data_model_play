import os
import torch

from typing import Tuple
from torch import nn
from waymo_open_data_model_play.model.encoder_util import MlpNet, McgNet, LearnableQuery



class BaselineSimplyMp(nn.Module):
    def __init__(self, num_future_states, num_trajs,
                 history_timestamps, sdc_attribution_dim,
                 agent_attribution_dim, map_attribution_dim):
        super(BaselineSimplyMp, self).__init__()
        self.sdc_history_encoder = MlpNet(
            history_timestamps * sdc_attribution_dim, output_dimension=32)
        self.agent_history_mlp_pre_cg_encoder = MlpNet(
            history_timestamps * agent_attribution_dim, output_dimension=32)
        self.agent_history_encoder = McgNet(
            element_attribution_dim=32,
            context_attribution_dim=32,
            internal_embed_size=64,
            num_cg=2)
        self.map_encoder = McgNet(
            element_attribution_dim=map_attribution_dim,
            context_attribution_dim=32,
            internal_embed_size=64,
            num_cg=3)
        self.learnable_embed_dim = 128
        self.learnable_decoder = LearnableQuery(
            num_trajs, query_dim=32, context_dim=64+64+32, 
            internal_embed_size=self.learnable_embed_dim)
        self.traj_regression_mlp_decoder = MlpNet(
            input_dimension=128, output_dimension=num_future_states*2)
        self.traj_likelihood_mlp_decoder = MlpNet(
            input_dimension=128, output_dimension=num_trajs)
        self.num_trajs = num_trajs
        self.num_future_states = num_future_states
        self.likelihood_softmax = nn.Softmax(dim=1)

    def forward(self, sdc_history: torch.Tensor,
                agent_history: torch.Tensor, 
                map: torch.Tensor) -> Tuple[torch.Tensor]:
        """Forward function for the baseline simplified MP++ model.

        Args:
            sdc_history: SDC history feature tensor, whose dimensions are
                [B, history_timestamps, attributions].
            agent_history: agent history feature tensor, whose dimensions are
                [B, num_agents, history_timestamps, attributions].
            map: map feature tensor, whose dimensions are: [B, num_polylines,
                attributions].

        Returns:
            Two tensors. The first one represents the decoded trajectories,
            whose dimensions are [B, num_traj, 2 * num_future_states]. The
            second one represents the chosen likelihood whose dimensions are
            [B, num_traj].
        """
        batch_size = sdc_history.shape[0]
        num_agent = agent_history.shape[1]
        sdc_embedding = self.sdc_history_encoder(sdc_history)
        agent_embedding = self.agent_history_mlp_pre_cg_encoder(
            agent_history.view(batch_size * num_agent, -1))
        agent_embedding = agent_embedding.view(batch_size, num_agent, -1)
        _, agent_embedding = self.agent_history_encoder(
            agent_embedding, sdc_embedding)
        _, map_embedding = self.map_encoder(map, sdc_embedding)
        ctx_embed = torch.cat((sdc_embedding, agent_embedding, map_embedding), dim=1)
        # Decoded traj should be [B, num_trajs, internal_embed], decoded_ctx should be
        # [B, internal_embed].
        decoded_traj, decoded_ctx = self.learnable_decoder(ctx_embed)
        decoded_traj = decoded_traj.reshape(batch_size * self.num_trajs, 
                                            self.learnable_embed_dim)
        decoded_traj = self.traj_regression_mlp_decoder(decoded_traj)
        decoded_traj = decoded_traj.reshape(
            batch_size, self.num_trajs, self.num_future_states * 2)
        traj_likelihood = self.likelihood_softmax(
            self.traj_likelihood_mlp_decoder(decoded_ctx))
        return decoded_traj, traj_likelihood