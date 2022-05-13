import numpy as np
from utils import get_idxs_per_relation
from mpi4py import MPI


START_GENERATE_GOALS = 1000

class GoalSampler:
    def __init__(self, args, policy):
        self.num_rollouts_per_mpi = args.num_rollouts_per_mpi
        self.rank = MPI.COMM_WORLD.Get_rank()

        self.goal_dim = 3 # distances
        self.relation_ids = get_idxs_per_relation(n=args.n_blocks)

        self.continuous = args.algo == 'continuous'

        self.policy = policy

        self.args = args

        self.init_stats()

    def sample_goal(self, n_goals, evaluation):
        """
        Sample n_goals goals to be targeted during rollouts
        evaluation controls whether or not to sample the goal uniformly or according to curriculum
        """
        if evaluation:
            # Evaluate on encountered goals
            if self.args.vae_batch_sample_strategy == 'buffer':
                goals = self.policy.goal_encoder.buffer.sample(n_goals)
            elif self.args.vae_batch_sample_strategy == 'limits':
                goals = np.random.uniform(low=self.policy.goal_encoder.lower_bounds, high=self.policy.goal_encoder.upper_bounds, 
                                           size = (n_goals, self.args.env_params['goal']))
                stop = 1
            else:
                raise NotImplementedError
        else:
            # Embeddings at this stage are kept zeros
            if self.policy.goal_encoder.buffer.current_size == 0:
                goals = np.random.uniform(low=0, high=0.1, size = (n_goals, self.args.env_params['goal']))
            elif self.policy.goal_encoder.buffer.current_size < START_GENERATE_GOALS:
                goals = self.policy.goal_encoder.buffer.sample(n_goals)
            else:
                embeddings = np.zeros((n_goals, 3))
                if self.args.cuda:
                    goals = self.policy.goal_encoder.inference(embeddings=embeddings, n=n_goals).detach().cpu().numpy()
                else:
                    goals = self.policy.goal_encoder.inference(embeddings=embeddings, n=n_goals).detach().numpy()
        return goals

    def update(self, episodes, t):
        """
        Update discovered goals list from episodes
        Update list of successes and failures for LP curriculum
        Label each episode with the last ag (for buffer storage)
        """
        all_episodes = MPI.COMM_WORLD.gather(episodes, root=0)

        discovered_goals = []
        if self.rank == 0:
            all_episode_list = [e for eps in all_episodes for e in eps]

            for e in all_episode_list:
                discovered_goals += [e for e in np.unique(np.around(e['ag'], decimals=3), axis=0)]

        discovered_goals = MPI.COMM_WORLD.bcast(discovered_goals, root=0)
        # self.sync()

        return episodes, discovered_goals

    def sync(self):
        self.encountered_distances = MPI.COMM_WORLD.bcast(self.encountered_distances, root=0)


    def build_batch(self, batch_size):
        goal_ids = np.random.choice(np.arange(len(self.discovered_goals)), size=batch_size)
        return goal_ids

    def init_stats(self):
        self.stats = dict()
        # Number of classes of eval
        if self.goal_dim == 30:
            n = 11
        else:
            n = 6
        self.stats['epoch'] = []
        self.stats['episodes'] = []
        self.stats['global_sr'] = []
        self.stats['avg_rew'] = []
        keys = ['goal_sampler', 'rollout', 'gs_update', 'store_episodes', 'store_goals', 
                'vae_train', 'norm_update', 'policy_train', 'eval', 'epoch', 'total']
        for k in keys:
            self.stats['t_{}'.format(k)] = []

    def save(self, epoch, episode_count, av_res, av_rew, global_sr, time_dict):
        self.stats['epoch'].append(epoch)
        self.stats['episodes'].append(episode_count)
        self.stats['global_sr'].append(global_sr)
        self.stats['avg_rew'].append(av_rew[0])
        for k in time_dict.keys():
            self.stats['t_{}'.format(k)].append(time_dict[k])
