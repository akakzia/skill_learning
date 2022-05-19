import numpy as np


def compute_incremental_reward(ag, g):
    return np.sum(np.abs(ag - g) < 0.01)
    
def is_success(ag, g):
    return (np.abs(ag - g) < 0.01).all()

def at_least_one_fallen(observation, n):
    """ Given a observation, returns true if at least one object has fallen """
    dim_body = 10
    dim_object = 15
    obs_objects = np.array([observation[dim_body + dim_object * i: dim_body + dim_object * (i + 1)] for i in range(n)])
    obs_z = obs_objects[:, 2]

    return (obs_z < 0.4).any()



class RolloutWorker:
    def __init__(self, env, policy, args):

        self.env = env
        self.policy = policy
        self.env_params = args.env_params
        self.continuous = args.algo == 'continuous'
        self.args = args

    def generate_rollout(self, goals, true_eval, biased_init=False, animated=False):

        episodes = []
        for i in range(goals.shape[0]):
            observation = self.env.unwrapped.reset_goal(goal=np.array(goals[i]), biased_init=biased_init)
            obs = observation['observation']
            g= goals[i]
            distance = obs[10:13] - obs[25:28]

            # ep_obs, ep_ag, ep_ag_bin, ep_g, ep_g_bin, ep_actions, ep_success, ep_rewards = [], [], [], [], [], [], [], []
            ep_obs, ep_ag, ep_g, ep_actions, ep_success, ep_rewards = [], [], [], [], [], []

            # Start to collect samples
            for t in range(self.env_params['max_timesteps']):
                # Run policy for one step
                no_noise = true_eval  # do not use exploration noise if running self-evaluations or offline evaluations
                # feed both the observation and mask to the policy module
                action = self.policy.act(obs.copy(), distance.copy(), g.copy(), no_noise)

                # feed the actions into the environment
                if animated:
                    self.env.render()

                observation_new, r, _, _ = self.env.step(action)
                obs_new = observation_new['observation']
                distance_new = obs_new[10:13] - obs_new[25:28]
                r = compute_incremental_reward(distance_new, g)

                # Append rollouts
                ep_obs.append(obs.copy())
                ep_ag.append(distance.copy())
                ep_g.append(g.copy())
                ep_actions.append(action.copy())
                ep_rewards.append(r)
                ep_success.append(is_success(distance_new, g))

                # Re-assign the observation
                obs = obs_new
                distance = distance_new

            ep_obs.append(obs.copy())
            ep_ag.append(distance.copy())
            # ep_ag.append(ag.copy())
            # ep_ag_bin.append(ag_bin.copy())

            # Gather everything
            episode = dict(obs=np.array(ep_obs).copy(),
                           act=np.array(ep_actions).copy(),
                           g=np.array(ep_g).copy(),
                           ag=np.array(ep_ag).copy(),
                           success=np.array(ep_success).copy(),
                           rewards=np.array(ep_rewards).copy())


            episodes.append(episode)

        return episodes

