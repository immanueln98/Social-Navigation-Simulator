import numpy as np
import sys

import torch
import torch.nn as nn


def make_mlp(dim_list, activation='relu', batch_norm=True, dropout=0):
    layers = torch.nn.ModuleList()
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)


def get_noise(shape, noise_type, device):
    if noise_type == 'gaussian':
        return torch.randn(*shape, device=device)
    elif noise_type == 'uniform':
        return torch.rand(*shape).sub_(0.5).mul_(2.0, device=device)
    raise ValueError('Unrecognized noise type "%s"' % noise_type)


class Encoder(nn.Module):
    """Encoder is part of both TrajectoryGenerator and
    TrajectoryDiscriminator"""

    def __init__(
            self, embedding_dim=64, h_dim=64, mlp_dim=64, num_layers=1,
            dropout=0.0, return_c=False
    ):
        super(Encoder, self).__init__()
        self.device = torch.device("cpu")
        self.mlp_dim = 1024
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        # Need cell state of encoder for Intention Force Generator
        self.return_c = return_c
        self.encoder = nn.LSTM(
            embedding_dim, h_dim, num_layers, dropout=dropout
        )

        self.spatial_embedding = nn.Linear(2, embedding_dim).to(self.device)

    def init_hidden(self, batch):
        return (
            torch.zeros(self.num_layers, batch, self.h_dim, device=self.device),
            torch.zeros(self.num_layers, batch, self.h_dim, device=self.device))

    def forward(self, obs_traj):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        Output:
        - final_h: Tensor of shape (self.num_layers, batch, self.h_dim)
        """
        # Encode observed Trajectory
        batch = obs_traj.size(1)
        # obs_traj_embedding = self.spatial_embedding(obs_traj.view(-1, 2))
        # obs_traj_embedding = obs_traj_embedding.view(
        #     -1, batch, self.embedding_dim
        # )
        obs_traj_embedding = self.spatial_embedding(obs_traj.reshape(-1, 2))
        obs_traj_embedding = obs_traj_embedding.reshape(
            -1, batch, self.embedding_dim
        )
        state_tuple = self.init_hidden(batch)
        output, state = self.encoder(obs_traj_embedding, state_tuple)
        # Return h and c values if ret c else just h value
        # print(f'return c: {self.return_c}')
        return state[0], state[1] if self.return_c else state[0]


class Decoder(nn.Module):
    """Decoder is part of TrajectoryGenerator"""

    def __init__(
            self, seq_len, embedding_dim=64, h_dim=128, mlp_dim=1024, num_layers=1,
            pool_every_timestep=True, dropout=0.0, bottleneck_dim=1024,
            activation='relu', batch_norm=True, pooling_type='pool_net',
            neighborhood_size=2.0, grid_size=8
    ):
        super(Decoder, self).__init__()

        self.seq_len = seq_len
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.pool_every_timestep = pool_every_timestep

        self.decoder = nn.LSTM(
            embedding_dim, self.h_dim, num_layers, dropout=dropout
        )

        if pool_every_timestep:
            if pooling_type == 'pool_net':
                self.pool_net = PoolHiddenNet(
                    embedding_dim=self.embedding_dim,
                    h_dim=self.h_dim,
                    mlp_dim=mlp_dim,
                    bottleneck_dim=bottleneck_dim,
                    activation=activation,
                    batch_norm=batch_norm,
                    dropout=dropout
                )
            elif pooling_type == 'spool':
                self.pool_net = SocialPooling(
                    h_dim=self.h_dim,
                    activation=activation,
                    batch_norm=batch_norm,
                    dropout=dropout,
                    neighborhood_size=neighborhood_size,
                    grid_size=grid_size
                )

            mlp_dims = [self.h_dim + bottleneck_dim, mlp_dim, self.h_dim]
            self.mlp = make_mlp(
                mlp_dims,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout
            )

        self.spatial_embedding = nn.Linear(2, embedding_dim)
        self.hidden2pos = nn.Linear(self.h_dim, 2)

    def forward(self, last_pos, last_pos_rel, state_tuple, seq_start_end):
        """
        Inputs:
        - last_pos: Tensor of shape (batch, 2)
        - last_pos_rel: Tensor of shape (batch, 2)
        - state_tuple: (hh, ch) each tensor of shape (num_layers, batch, h_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch
        Output:
        - pred_traj: tensor of shape (self.seq_len, batch, 2)
        """
        batch = last_pos.size(0)
        pred_traj_fake_rel = []
        decoder_input = self.spatial_embedding(last_pos_rel)
        # decoder_input = decoder_input.view(1, batch, self.embedding_dim)
        decoder_input = decoder_input.reshape(1, batch, self.embedding_dim)
        for _ in range(self.seq_len):
            # print(f'Decoder input shape:{decoder_input.shape},')
            # print(f'cell shape:{state_tuple[1].shape}, hidden shape {state_tuple[0].shape}')
            output, state_tuple = self.decoder(decoder_input, state_tuple)

            rel_pos = self.hidden2pos(output.reshape(-1, self.h_dim))
            curr_pos = rel_pos + last_pos

            if self.pool_every_timestep:
                decoder_h = state_tuple[0]
                pool_h = self.pool_net(decoder_h, seq_start_end, curr_pos)
                decoder_h = torch.cat(
                    [decoder_h.reshape(-1, self.h_dim), pool_h], dim=1)
                decoder_h = self.mlp(decoder_h)
                decoder_h = torch.unsqueeze(decoder_h, 0)
                state_tuple = (decoder_h, state_tuple[1])

            embedding_input = rel_pos

            decoder_input = self.spatial_embedding(embedding_input)
            decoder_input = decoder_input.reshape(1, batch, self.embedding_dim)
            pred_traj_fake_rel.append(rel_pos.reshape(batch, -1))
            last_pos = curr_pos

        pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0)
        return pred_traj_fake_rel, state_tuple[0]


class PoolHiddenNet(nn.Module):
    """Pooling module as proposed in our paper"""

    def __init__(
            self, embedding_dim=64, h_dim=64, mlp_dim=64, bottleneck_dim=32,
            activation='relu', batch_norm=True, dropout=0.0
    ):
        super(PoolHiddenNet, self).__init__()

        self.mlp_dim = 1024
        self.h_dim = h_dim
        self.bottleneck_dim = bottleneck_dim
        self.embedding_dim = embedding_dim

        mlp_pre_dim = embedding_dim + h_dim
        mlp_pre_pool_dims = [mlp_pre_dim, 512, bottleneck_dim]

        self.spatial_embedding = nn.Linear(2, embedding_dim)
        self.mlp_pre_pool = make_mlp(
            mlp_pre_pool_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout)

    def repeat(self, tensor, num_reps):
        """
        Inputs:
        -tensor: 2D tensor of any shape
        -num_reps: Number of times to repeat each row
        Outpus:
        -repeat_tensor: Repeat each row such that: R1, R1, R2, R2
        """
        col_len = tensor.size(1)
        tensor = tensor.unsqueeze(dim=1).repeat(1, num_reps, 1)
        tensor = tensor.reshape(-1, col_len)
        return tensor

    def forward(self, h_states, seq_start_end, end_pos):
        """
        Inputs:
        - h_states: Tensor of shape (num_layers, batch, h_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch
        - end_pos: Tensor of shape (batch, 2)
        Output:
        - pool_h: Tensor of shape (batch, bottleneck_dim)
        """
        pool_h = []
        for start, end in seq_start_end:
            start = start.item()
            end = end.item()
            num_ped = end - start
            curr_hidden = h_states.reshape(-1, self.h_dim)[start:end]
            curr_end_pos = end_pos[start:end]
            curr_hidden_1 = curr_hidden.repeat(num_ped, 1)
            curr_end_pos_1 = curr_end_pos.repeat(num_ped, 1)
            curr_end_pos_2 = self.repeat(curr_end_pos, num_ped)
            curr_rel_pos = curr_end_pos_1 - curr_end_pos_2
            curr_rel_embedding = self.spatial_embedding(curr_rel_pos)
            mlp_h_input = torch.cat([curr_rel_embedding, curr_hidden_1], dim=1)
            curr_pool_h = self.mlp_pre_pool(mlp_h_input)
            curr_pool_h = curr_pool_h.reshape(num_ped, num_ped, -1).max(1)[0]
            pool_h.append(curr_pool_h)
        pool_h = torch.cat(pool_h, dim=0)
        return pool_h


class SocialPooling(nn.Module):
    """Current state of the art pooling mechanism:
    http://cvgl.stanford.edu/papers/CVPR16_Social_LSTM.pdf"""

    def __init__(
            self, h_dim=64, activation='relu', batch_norm=True, dropout=0.0,
            neighborhood_size=2.0, grid_size=8, pool_dim=None
    ):
        super(SocialPooling, self).__init__()
        self.h_dim = h_dim
        self.grid_size = grid_size
        self.neighborhood_size = neighborhood_size
        if pool_dim:
            mlp_pool_dims = [grid_size * grid_size * h_dim, pool_dim]
        else:
            mlp_pool_dims = [grid_size * grid_size * h_dim, h_dim]

        self.mlp_pool = make_mlp(
            mlp_pool_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout
        )

    def get_bounds(self, ped_pos):
        top_left_x = ped_pos[:, 0] - self.neighborhood_size / 2
        top_left_y = ped_pos[:, 1] + self.neighborhood_size / 2
        bottom_right_x = ped_pos[:, 0] + self.neighborhood_size / 2
        bottom_right_y = ped_pos[:, 1] - self.neighborhood_size / 2
        top_left = torch.stack([top_left_x, top_left_y], dim=1)
        bottom_right = torch.stack([bottom_right_x, bottom_right_y], dim=1)
        return top_left, bottom_right

    def get_grid_locations(self, top_left, other_pos):
        cell_x = torch.floor(
            ((other_pos[:, 0] - top_left[:, 0]) / self.neighborhood_size) *
            self.grid_size)
        cell_y = torch.floor(
            ((top_left[:, 1] - other_pos[:, 1]) / self.neighborhood_size) *
            self.grid_size)
        grid_pos = cell_x + cell_y * self.grid_size
        return grid_pos

    def repeat(self, tensor, num_reps):
        """
        Inputs:
        -tensor: 2D tensor of any shape
        -num_reps: Number of times to repeat each row
        Outpus:
        -repeat_tensor: Repeat each row such that: R1, R1, R2, R2
        """
        col_len = tensor.size(1)
        tensor = tensor.unsqueeze(dim=1).repeat(1, num_reps, 1)
        tensor = tensor.reshape(-1, col_len)
        return tensor

    def forward(self, h_states, seq_start_end, end_pos):
        """
        Inputs:
        - h_states: Tesnsor of shape (num_layers, batch, h_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - end_pos: Absolute end position of obs_traj (batch, 2)
        Output:
        - pool_h: Tensor of shape (batch, h_dim)
        """
        pool_h = []
        for start, end in seq_start_end:
            start = start.item()
            end = end.item()
            num_ped = end - start
            grid_size = self.grid_size * self.grid_size
            curr_hidden = h_states.reshape(-1, self.h_dim)[start:end]
            curr_hidden_repeat = curr_hidden.repeat(num_ped, 1)
            curr_end_pos = end_pos[start:end]
            curr_pool_h_size = num_ped * grid_size + 1
            curr_pool_h = curr_hidden.new_zeros((curr_pool_h_size, self.h_dim))
            top_left, bottom_right = self.get_bounds(curr_end_pos)
            curr_end_pos = curr_end_pos.repeat(num_ped, 1)
            top_left = self.repeat(top_left, num_ped)
            bottom_right = self.repeat(bottom_right, num_ped)
            grid_pos = self.get_grid_locations(top_left, curr_end_pos).type_as(seq_start_end)

            x_bound = (curr_end_pos[:, 0] >= bottom_right[:, 0]) + (curr_end_pos[:, 0] <= top_left[:, 0])

            y_bound = (curr_end_pos[:, 1] >= top_left[:, 1]) + (curr_end_pos[:, 1] <= bottom_right[:, 1])

            within_bound = x_bound + y_bound
            within_bound[0::num_ped + 1] = 1
            within_bound = within_bound.reshape(-1)
            grid_pos += 1
            total_grid_size = self.grid_size * self.grid_size
            offset = torch.arange(0, total_grid_size * num_ped, total_grid_size).type_as(seq_start_end)

            offset = self.repeat(offset.reshape(-1, 1), num_ped).reshape(-1)
            grid_pos += offset
            grid_pos[within_bound != 0] = 0
            grid_pos = grid_pos.reshape(-1, 1).expand_as(curr_hidden_repeat)
            curr_pool_h = curr_pool_h.scatter_add(0, grid_pos, curr_hidden_repeat)
            curr_pool_h = curr_pool_h[1:]
            pool_h.append(curr_pool_h.reshape(num_ped, -1))
        pool_h = torch.cat(pool_h, dim=0)
        pool_h = self.mlp_pool(pool_h)
        return pool_h


class TrajectoryGenerator(nn.Module):
    def __init__(
            self, obs_len, pred_len, embedding_dim=16, encoder_h_dim=32,
            decoder_h_dim=32, mlp_dim=64, num_layers=1, noise_dim=(0,),
            noise_type='gaussian', noise_mix_type='ped', pooling_type=None,
            pool_every_timestep=True, dropout=0.0, bottleneck_dim=64,
            activation='relu', batch_norm=True, neighborhood_size=2.0, grid_size=8
    ):
        super(TrajectoryGenerator, self).__init__()

        if pooling_type and pooling_type.lower() == 'none':
            pooling_type = None
        self.device = torch.device("cpu")
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.mlp_dim = mlp_dim
        self.encoder_h_dim = encoder_h_dim
        self.decoder_h_dim = decoder_h_dim
        self.embedding_dim = embedding_dim
        self.noise_dim = noise_dim
        self.num_layers = num_layers
        self.noise_type = noise_type
        self.noise_mix_type = noise_mix_type
        self.noise_shape = None
        self.pooling_type = pooling_type
        self.noise_first_dim = noise_dim[0]
        self.pool_every_timestep = pool_every_timestep
        self.bottleneck_dim = 1024

        self.encoder = Encoder(
            embedding_dim=embedding_dim,
            h_dim=encoder_h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        self.encoder.to(self.device)

        self.decoder = Decoder(
            pred_len,
            embedding_dim=embedding_dim,
            h_dim=decoder_h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            pool_every_timestep=pool_every_timestep,
            dropout=dropout,
            bottleneck_dim=bottleneck_dim,
            activation=activation,
            batch_norm=batch_norm,
            pooling_type=pooling_type,
            grid_size=grid_size,
            neighborhood_size=neighborhood_size
        )
        self.decoder.to(self.device)

        if pooling_type == 'pool_net':
            self.pool_net = PoolHiddenNet(
                embedding_dim=self.embedding_dim,
                h_dim=encoder_h_dim,
                mlp_dim=mlp_dim,
                bottleneck_dim=bottleneck_dim,
                activation=activation,
                batch_norm=batch_norm
            )
        elif pooling_type == 'spool':
            self.pool_net = SocialPooling(
                h_dim=encoder_h_dim,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout,
                neighborhood_size=neighborhood_size,
                grid_size=grid_size
            )

        if self.noise_dim[0] == 0:
            self.noise_dim = None
        else:
            self.noise_first_dim = noise_dim[0]

        # Decoder Hidden
        if pooling_type:
            input_dim = encoder_h_dim + bottleneck_dim
        else:
            input_dim = encoder_h_dim

        if self.mlp_decoder_needed():
            mlp_decoder_context_dims = [
                input_dim, mlp_dim, decoder_h_dim - self.noise_first_dim
            ]

            self.mlp_decoder_context = make_mlp(
                mlp_decoder_context_dims,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout
            )

    def add_noise(self, _input, seq_start_end, user_noise=None, injection_idx=None):
        """
        Inputs:
        - _input: Tensor of shape (_, decoder_h_dim - noise_first_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - user_noise: Generally used for inference when you want to see
        relation between different types of noise and outputs.
        Outputs:
        - decoder_h: Tensor of shape (_, decoder_h_dim)
        """
        if not self.noise_dim:
            return _input

        if self.noise_mix_type == 'global':
            self.noise_shape = (seq_start_end.size(0),) + self.noise_dim
        else:
            self.noise_shape = (_input.size(0),) + self.noise_dim

        if user_noise is not None and injection_idx is None:
            z_decoder = user_noise
            # print(f'User noise added. z_decoder shape: {z_decoder.shape}')

        else:
            z_decoder = get_noise(self.noise_shape, self.noise_type, self.device)
            if injection_idx is not None and user_noise is not None:
                z_decoder[injection_idx] = user_noise
            # print(f'z_decoder shape: {z_decoder.shape}')
            # sys.exit(0)

        if self.noise_mix_type == 'global':
            _list = []
            for idx, (start, end) in enumerate(seq_start_end):
                start = start.item()
                end = end.item()
                _vec = z_decoder[idx].reshape(1, -1)
                _to_cat = _vec.repeat(end - start, 1)
                _list.append(torch.cat([_input[start:end], _to_cat], dim=1))
            decoder_h = torch.cat(_list, dim=0)
            return decoder_h

        decoder_h = torch.cat([_input, z_decoder], dim=1)

        return decoder_h

    def mlp_decoder_needed(self):
        if (
                self.noise_dim or self.pooling_type or
                self.encoder_h_dim != self.decoder_h_dim
        ):
            return True
        else:
            return False

    def forward(self, obs_traj, obs_traj_rel, seq_start_end, user_noise=None, injection_idx=None):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        - obs_traj_rel: Tensor of shape (obs_len, batch, 2)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - user_noise: Generally used for inference when you want to see
        relation between different types of noise and outputs.
        Output:
        - pred_traj_rel: Tensor of shape (self.pred_len, batch, 2)
        """
        batch = obs_traj_rel.size(1)
        # Encode seq
        final_encoder_h, final_encoder_c = self.encoder(obs_traj_rel)
        # Pool States
        if self.pooling_type:
            end_pos = obs_traj[-1, :, :]
            pool_h = self.pool_net(final_encoder_h, seq_start_end, end_pos)
            # Construct input hidden states for decoder
            mlp_decoder_context_input = torch.cat(
                [final_encoder_h.reshape(-1, self.encoder_h_dim), pool_h], dim=1)
        else:
            mlp_decoder_context_input = final_encoder_h.reshape(
                -1, self.encoder_h_dim)

        # Add Noise
        if self.mlp_decoder_needed():
            noise_input = self.mlp_decoder_context(mlp_decoder_context_input)
        else:
            noise_input = mlp_decoder_context_input
        decoder_h = self.add_noise(
            noise_input, seq_start_end, user_noise=user_noise, injection_idx=injection_idx)
        decoder_h = torch.unsqueeze(decoder_h, 0)

        decoder_c = torch.zeros(self.num_layers, batch, self.decoder_h_dim, device=self.device)

        state_tuple = (decoder_h, decoder_c)
        last_pos = obs_traj[-1]
        last_pos_rel = obs_traj_rel[-1]
        # Predict Trajectory

        decoder_out = self.decoder(
            last_pos,
            last_pos_rel,
            state_tuple,
            seq_start_end,
        )
        pred_traj_fake_rel, final_decoder_h = decoder_out

        return pred_traj_fake_rel


class TrajectoryDiscriminator(nn.Module):
    def __init__(
            self, obs_len, pred_len, embedding_dim=16, h_dim=64, mlp_dim=64,
            num_layers=1, activation='relu', batch_norm=True, dropout=0.0,
            d_type='local'
    ):
        super(TrajectoryDiscriminator, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.d_type = d_type

        self.encoder = Encoder(
            embedding_dim=embedding_dim,
            h_dim=h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout,
            return_c=False
        )

        real_classifier_dims = [h_dim, mlp_dim, 1]
        self.real_classifier = make_mlp(
            real_classifier_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout
        )
        if d_type == 'global':
            mlp_pool_dims = [h_dim + embedding_dim, mlp_dim, h_dim]
            self.pool_net = PoolHiddenNet(
                embedding_dim=embedding_dim,
                h_dim=h_dim,
                mlp_dim=mlp_pool_dims,
                bottleneck_dim=h_dim,
                activation=activation,
                batch_norm=batch_norm
            )

    def forward(self, traj, traj_rel, seq_start_end=None):
        """
        Inputs:
        - traj: Tensor of shape (obs_len + pred_len, batch, 2)
        - traj_rel: Tensor of shape (obs_len + pred_len, batch, 2)
        - seq_start_end: A list of tuples which delimit sequences within batch
        Output:
        - scores: Tensor of shape (batch,) with real/fake scores
        """
        # TODO fix broken return_c flag in encoder
        final_h, _ = self.encoder(traj_rel)
        # Note: In case of 'global' option we are using start_pos as opposed to
        # end_pos. The intution being that hidden state has the whole
        # trajectory and relative postion at the start when combined with
        # trajectory information should help in discriminative behavior.
        # print(f'final_h shape: {final_h}')
        if self.d_type == 'local':
            classifier_input = final_h.squeeze()
        else:
            classifier_input = self.pool_net(
                final_h.squeeze(), seq_start_end, traj[0]
            )
        # Return scores
        return self.real_classifier(classifier_input)


class IntentionForceGenerator(nn.Module):
    def __init__(
            self, obs_len, pred_len, embedding_dim=16, encoder_h_dim=64,
            decoder_h_dim=34, mlp_dim=64, num_layers=1, dropout=0.0, bottleneck_dim=32,
            activation='relu', batch_norm=True):
        super(IntentionForceGenerator, self).__init__()
        self.device = torch.device("cpu")
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.mlp_dim = mlp_dim
        self.encoder_h_dim = encoder_h_dim
        self.decoder_h_dim = decoder_h_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.bottleneck_dim = 1024

        self.encoder = Encoder(
            embedding_dim=embedding_dim,
            h_dim=encoder_h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout,
            return_c=False
        )

        self.decoder = Decoder(
            pred_len,
            embedding_dim=embedding_dim,
            h_dim=decoder_h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout,
            bottleneck_dim=bottleneck_dim,
            activation=activation,
            batch_norm=batch_norm,
            pool_every_timestep=False)

    def forward(self, obs_traj, obs_traj_rel, seq_start_end, goal_point=None):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, len(seq_start_end), 2)
        - obs_traj_rel: Tensor of shape (obs_len, len(seq_start_end), 2)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - user_noise: Generally used for inference when you want to see
        relation between different types of noise and outputs.
        Output:
        - pred_traj_rel: Tensor of shape (self.pred_len, len(seq_start_end), 2)
        """
        # print(f'obs_traj_rel: {obs_traj_rel.shape}')
        # Encode seq
        final_encoder_h, final_encoder_c = self.encoder(obs_traj_rel, )
        # print(f'final_h shape: {final_encoder_h.size()}\n'
        #       f'final_c shape: {final_encoder_c.size()}')

        # Goal injection ala naviGAN
        torch.set_printoptions(precision=2)
        # print(f'End point sample: {obs_traj[-1, 0: 10]}')
        if goal_point is None:
            goal_point = torch.zeros((1, obs_traj.shape[1], 2), device=self.device)
            # obs_traj[-1].reshape(1, batch, 2)
        # print(final_encoder_h.shape)
        # print(goal_point.shape)
        final_encoder_h = torch.cat([final_encoder_h, goal_point], dim=2)
        # Pad cell tensor with zeros to make dims of h and c equal which is needed by decoder's input
        final_encoder_c = torch.cat([final_encoder_c, torch.zeros((1, obs_traj.shape[1], 2), device=self.device)],
                                    dim=2)
        # final_encoder_c = torch.cat([final_encoder_c, goal_point], dim=2)
        # print(f'final_h shape with goal: {final_encoder_h.size()}')
        state_tuple = (final_encoder_h, final_encoder_c)
        last_pos = obs_traj[-1]
        last_pos_rel = obs_traj_rel[-1]

        # Predict Trajectory
        decoder_out = self.decoder(
            last_pos,
            last_pos_rel,
            state_tuple,
            seq_start_end,
        )
        pred_traj_fake_rel, final_decoder_h = decoder_out

        return pred_traj_fake_rel


class CombinedGenerator(nn.Module):
    def __init__(
            self, obs_len, pred_len, embedding_dim=64, encoder_h_dim=64,
            decoder_h_dim=128, mlp_dim=1024, num_layers=1, noise_dim=(0,),
            noise_type='gaussian', noise_mix_type='ped', pooling_type=None,
            pool_every_timestep=True, dropout=0.0, bottleneck_dim=1024,
            activation='relu', batch_norm=True, neighborhood_size=2.0, grid_size=8
    ):
        super(CombinedGenerator, self).__init__()
        self.social = TrajectoryGenerator(obs_len, pred_len, embedding_dim, encoder_h_dim,
                                          decoder_h_dim, mlp_dim, num_layers, noise_dim,
                                          noise_type, noise_mix_type, pooling_type,
                                          pool_every_timestep, dropout, bottleneck_dim,
                                          activation, batch_norm, neighborhood_size, grid_size)

        self.goal = IntentionForceGenerator(obs_len, pred_len, embedding_dim, encoder_h_dim,
                                            decoder_h_dim, mlp_dim, num_layers, dropout, bottleneck_dim,
                                            activation, batch_norm)

    def forward(self, obs_traj, obs_traj_rel, seq_start_end, goal_point=None, goal_aggro=0.5, split=False):
        pred_traj_rel = self.social(obs_traj, obs_traj_rel, seq_start_end)
        goal_agent_indices = [index[0].item() for index in seq_start_end]
        goal_obs_traj = obs_traj[::, goal_agent_indices]
        goal_obs_traj_rel = obs_traj_rel[::, goal_agent_indices]
        goal_traj = self.goal(goal_obs_traj, goal_obs_traj_rel, seq_start_end, goal_point)
        # Mostly for plotting influence of each network
        if split:
            return pred_traj_rel, goal_traj
        # For now take average of the social and goal generators outputs as the final traj
        for i, goal_idx in enumerate(goal_agent_indices):
            # print(i)
            # print(goal_idx)
            pred_traj_rel[::, goal_idx] = (1 - goal_aggro) * pred_traj_rel[::, goal_idx] + goal_aggro * goal_traj[::, i]
        # print('*'*60)
        return pred_traj_rel