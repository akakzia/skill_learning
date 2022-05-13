import torch
from rl_modules.rl_agent import RLAgent
import env
import gym
import numpy as np
from rollout import RolloutWorker
import json
from types import SimpleNamespace
from goal_sampler import GoalSampler
import  random
from mpi4py import MPI
from arguments import get_args
from utils import get_eval_goals

def get_env_params(env):
    obs = env.reset()
    # close the environment
    params = {'obs': obs['observation'].shape[0], 'goal': obs['desired_goal'].shape[0],
              'action': env.action_space.shape[0], 'action_max': env.action_space.high[0],
              'max_timesteps': env._max_episode_steps}
    return params

if __name__ == '__main__':
    num_eval = 1
    path = '/home/ahmed/Documents/Amaterasu/gobi/skill_learning/models/first_stage_05/models/'
    model_path = path + 'model_20.pt'
    vae_model_path = path + 'vae_model_20.pt'

    # with open(path + 'config.json', 'r') as f:
    #     params = json.load(f)
    # args = SimpleNamespace(**params)
    args = get_args()

    args.env_name = f'FetchManipulate{args.n_blocks:d}Objects-v0'

    # Make the environment
    env = gym.make(args.env_name)

    # set random seeds for reproduce
    args.seed = np.random.randint(1e6)
    env.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    np.random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    torch.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    if args.cuda:
        torch.cuda.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())

    args.env_params = get_env_params(env)

    # create the sac agent to interact with the environment
    if args.agent == "SAC":
        policy = RLAgent(args, env.compute_reward)
        policy.load(model_path, args)
        policy.load_goal_encoder(vae_model_path)
    else:
        raise NotImplementedError

    goal_sampler = GoalSampler(args, policy)

    # def rollout worker
    rollout_worker = RolloutWorker(env, policy,  args)

    eval_goals = goal_sampler.sample_goal(n_goals=50, evaluation=False)

    # eval_goals = np.array([[0.1, 0.1, 0.], [0.02, 0.02, 0.046]])

    all_results = []
    for i in range(num_eval):
        episodes = rollout_worker.generate_rollout(eval_goals, true_eval=True, animated=True)
        results = np.array([e['rewards'][-1] == 3. for e in episodes])
        all_results.append(results)

    results = np.array(all_results)
    print('Av Success Rate: {}'.format(results.mean()))