import torch
import numpy as np
from mpi_utils.mpi_utils import sync_networks
from rl_modules.replay_buffer import ReplayBuffer
from rl_modules.networks import QNetworkFlat, GaussianPolicyFlat
from mpi_utils.normalizer import normalizer
from her_modules.her import her_sampler
from updates import update_flat, update_deepsets
from vae_models.vae import ContextVAE


"""
SAC with HER (MPI-version)
"""

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class RLAgent:
    def __init__(self, args, compute_rew):

        self.args = args
        self.alpha = args.alpha
        self.env_params = args.env_params

        self.total_iter = 0

        self.freq_target_update = args.freq_target_update

        # create the network
        self.architecture = self.args.architecture

        if self.architecture == 'full_gn':
            from rl_modules.gn_models import GnSemantic
            self.model = GnSemantic(self.env_params, args)
        else:
            raise NotImplementedError

        # Define Goal Autoencoder
        self.goal_encoder = ContextVAE(args)

        # if use GPU
        if self.args.cuda:
            self.model.actor.cuda()
            self.model.critic.cuda()
            self.model.critic_target.cuda()
            self.goal_encoder.cuda()
        # sync the networks across the CPUs
        sync_networks(self.model.critic)
        sync_networks(self.model.actor)
        hard_update(self.model.critic_target, self.model.critic)
        sync_networks(self.model.critic_target)
        sync_networks(self.goal_encoder)

        # create the optimizer
        self.policy_optim = torch.optim.Adam(list(self.model.actor.parameters()), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(list(self.model.critic.parameters()), lr=self.args.lr_critic)
        
        # create the normalizer
        self.o_norm = normalizer(size=self.env_params['obs'], default_clip_range=self.args.clip_range)
        self.g_norm = normalizer(size=self.env_params['goal'], default_clip_range=self.args.clip_range)

        # Target Entropy
        if self.args.automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor(self.env_params['action'])).item()
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.args.lr_entropy)

        # her sampler
        if args.algo == 'continuous':
            self.continuous_goals = True
        else:
            self.continuous_goals = False
        self.her_module = her_sampler(self.args, compute_rew)

        # create the replay buffer
        self.buffer = ReplayBuffer(env_params=self.env_params,
                                  buffer_size=self.args.buffer_size,
                                  sample_func=self.her_module.sample_her_transitions
                                  )

    def act(self, obs, ag, g, no_noise):
        with torch.no_grad():
            # normalize policy inputs
            obs_norm = self.o_norm.normalize(obs)
            ag_norm = torch.tensor(self.g_norm.normalize(ag), dtype=torch.float32).unsqueeze(0)
            g_norm = torch.tensor(self.g_norm.normalize(g), dtype=torch.float32).unsqueeze(0)

            obs_tensor = torch.tensor(obs_norm, dtype=torch.float32).unsqueeze(0)
            if self.args.cuda:
                obs_tensor = obs_tensor.cuda()
                g_norm = g_norm.cuda()
                ag_norm = ag_norm.cuda()
            self.model.policy_forward_pass(obs_tensor, ag_norm, g_norm, no_noise=no_noise)
            if self.args.cuda:
                action = self.model.pi_tensor.cpu().numpy()[0]
            else:
                action = self.model.pi_tensor.numpy()[0]
                
        return action.copy()
    
    def store(self, episodes):
        self.buffer.store_episode(episode_batch=episodes)

    # pre_process the inputs
    def _preproc_inputs(self, obs, ag, g):
        obs_norm = self.o_norm.normalize(obs)
        delta_g = g - ag
        # concatenate the stuffs
        inputs = np.concatenate([obs_norm, delta_g])
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
        return inputs

    def train(self):
        # train the network
        self.total_iter += 1
        self._update_network()

        # soft update
        if self.total_iter % self.freq_target_update == 0:
            self._soft_update_target_network(self.model.critic_target, self.model.critic)
    
    def train_encoder(self):
        if self.args.vae_batch_sample_strategy == 'buffer':
            batch = self.goal_encoder.buffer.sample(self.args.vae_batch_size)
        elif self.args.vae_batch_sample_strategy == 'limits':
            batch = np.random.uniform(low=self.goal_encoder.lower_bounds, high=self.goal_encoder.upper_bounds, 
                                      size=(self.args.vae_batch_size, self.args.env_params['goal']))
        else:
            raise NotImplementedError
        # batch_norm = self.g_norm.normalize(batch)
        loss, loss_mse, loss_kld = self.goal_encoder.train(batch)

        return loss, loss_mse, loss_kld
                

    def _select_actions(self, state, no_noise=False):
        if not no_noise:
            action, _, _ = self.actor_network.sample(state)
        else:
            _, _, action = self.actor_network.sample(state)
        return action.detach().cpu().numpy()[0]

    # update the normalizer
    def _update_normalizer(self, episode):
        mb_obs = episode['obs']
        mb_ag = episode['ag']
        mb_g = episode['g']
        mb_actions = episode['act']
        mb_obs_next = mb_obs[1:, :]
        mb_ag_next = mb_ag[1:, :]
        # get the number of normalization transitions
        num_transitions = mb_actions.shape[0]
        # create the new buffer to store them
        buffer_temp = {'obs': np.expand_dims(mb_obs, 0),
                       'ag': np.expand_dims(mb_ag, 0),
                       'g': np.expand_dims(mb_g, 0),
                       'actions': np.expand_dims(mb_actions, 0),
                       'obs_next': np.expand_dims(mb_obs_next, 0),
                       'ag_next': np.expand_dims(mb_ag_next, 0),
                       }

        transitions = self.her_module.sample_her_transitions(buffer_temp, num_transitions)
        obs, g = transitions['obs'], transitions['g']
        # pre process the obs and g
        transitions['obs'], transitions['g'] = self._preproc_og(obs, g)
        # update
        self.o_norm.update(transitions['obs'])
        # recompute the stats
        self.o_norm.recompute_stats()

        self.g_norm.update(transitions['ag'])
        self.g_norm.recompute_stats()

    def _preproc_og(self, o, g):
        o = np.clip(o, -self.args.clip_obs, self.args.clip_obs)
        g = np.clip(g, -self.args.clip_obs, self.args.clip_obs)
        return o, g

    # soft update
    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

    # update the network
    def _update_network(self):
        # sample from buffer, this is done with LP is multi-head is true
        transitions = self.buffer.sample(self.args.batch_size)

        # pre-process the observation and goal
        o, o_next, g, ag, ag_next, actions, rewards = transitions['obs'], transitions['obs_next'], transitions['g'], transitions['ag'], \
                                                      transitions['ag_next'], transitions['actions'], transitions['r']
        transitions['obs'], transitions['g'] = self._preproc_og(o, g)
        transitions['obs_next'], transitions['g_next'] = self._preproc_og(o_next, g)
        _, transitions['ag'] = self._preproc_og(o, ag)
        _, transitions['ag_next'] = self._preproc_og(o, ag_next)

        # apply normalization
        obs_norm = self.o_norm.normalize(transitions['obs'])
        g_norm = self.g_norm.normalize(transitions['g'])
        ag_norm = self.g_norm.normalize(transitions['ag'])
        obs_next_norm = self.o_norm.normalize(transitions['obs_next'])
        ag_next_norm = self.g_norm.normalize(transitions['ag_next'])

        update_deepsets(self.model, self.policy_optim, self.critic_optim, self.alpha, self.log_alpha, self.target_entropy, self.alpha_optim,
            obs_norm, ag_norm, g_norm, obs_next_norm, ag_next_norm, actions, rewards, self.args)

    def save(self, model_path, epoch):
        torch.save([self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std,
                    self.model.actor.state_dict(), self.model.critic.state_dict()],
                    model_path + '/model_{}.pt'.format(epoch))

    def load(self, model_path, args):

        o_mean, o_std, g_mean, g_std, actor, critic = torch.load(model_path, map_location=lambda storage, loc: storage)
        self.model.actor.load_state_dict(actor)
        self.model.critic.load_state_dict(critic)
        self.model.actor.eval()
        self.o_norm.mean = o_mean
        self.o_norm.std = o_std
        self.g_norm.mean = g_mean
        self.g_norm.std = g_std
    
    def load_goal_encoder(self, vae_model_path):

        buffer, buffer_size, encoder, decoder = torch.load(vae_model_path, map_location=lambda storage, loc: storage)
        self.goal_encoder.encoder.load_state_dict(encoder)
        self.goal_encoder.decoder.load_state_dict(decoder)
        self.goal_encoder.buffer.buffer = buffer
        self.goal_encoder.buffer.current_size = buffer_size
