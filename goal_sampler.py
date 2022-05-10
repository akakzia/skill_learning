import numpy as np
from utils import get_idxs_per_relation
from mpi4py import MPI


MAX_DATA_LEN = 10000
MIN_DATA_LEN = 1

class GoalSampler:
    def __init__(self, args):
        self.num_rollouts_per_mpi = args.num_rollouts_per_mpi
        self.rank = MPI.COMM_WORLD.Get_rank()

        self.goal_dim = 3 # distances
        self.relation_ids = get_idxs_per_relation(n=args.n_blocks)

        self.encountered_distances= []

        self.continuous = args.algo == 'continuous'

        self.init_stats()

    def sample_goal(self, n_goals, evaluation):
        """
        Sample n_goals goals to be targeted during rollouts
        evaluation controls whether or not to sample the goal uniformly or according to curriculum
        """
        # if evaluation and len(self.discovered_goals) > 0:
        #     goals = np.random.choice(self.discovered_goals, size=self.num_rollouts_per_mpi)
        # else:
        if len(self.encountered_distances) < MIN_DATA_LEN:
            # goals = np.random.choice([-1., 1.], size=(n_goals, self.goal_dim))
            goals = np.random.uniform(low=0., high=0.15, size=(n_goals, self.goal_dim))
        else:
            # sample uniformly from discovered goals
            # goal_ids = np.random.choice(range(len(self.discovered_goals)), size=n_goals)
            # goals = np.array(self.discovered_goals)[goal_ids]
            goal_ids = np.random.randint(len(self.encountered_distances), size=n_goals)
            goals = np.array([self.encountered_distances[i] for i in goal_ids])
        return goals

    def update(self, episodes, t):
        """
        Update discovered goals list from episodes
        Update list of successes and failures for LP curriculum
        Label each episode with the last ag (for buffer storage)
        """
        all_episodes = MPI.COMM_WORLD.gather(episodes, root=0)

        if self.rank == 0:
            all_episode_list = [e for eps in all_episodes for e in eps]

            for e in all_episode_list:
                self.encountered_distances.append(e['ag'][-1])

        self.encountered_distances = self.encountered_distances[-MAX_DATA_LEN:]
        self.sync()

        return episodes

    def sync(self):
        # self.discovered_goals = MPI.COMM_WORLD.bcast(self.discovered_goals, root=0)
        # self.discovered_goals_str = MPI.COMM_WORLD.bcast(self.discovered_goals_str, root=0)
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
        self.stats['nb_discovered'] = []
        keys = ['goal_sampler', 'rollout', 'gs_update', 'store', 'norm_update',
                'policy_train', 'eval', 'epoch', 'total']
        for k in keys:
            self.stats['t_{}'.format(k)] = []

    def save(self, epoch, episode_count, av_res, av_rew, global_sr, time_dict):
        self.stats['epoch'].append(epoch)
        self.stats['episodes'].append(episode_count)
        self.stats['global_sr'].append(global_sr)
        self.stats['avg_rew'].append(av_rew[0])
        for k in time_dict.keys():
            self.stats['t_{}'.format(k)].append(time_dict[k])
        self.stats['nb_discovered'].append(len(self.encountered_distances))
