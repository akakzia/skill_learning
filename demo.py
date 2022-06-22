import torch
from rl_modules.rl_agent import RLAgent
import env
import gym
import numpy as np
from rollout import RolloutWorker
from goal_sampler import GoalSampler
import random
from mpi4py import MPI
from arguments import get_args
from utils import compute_reward_handreach

def get_env_params(env):
    obs = env.reset()
    # close the environment
    params = {'obs': obs['observation'].shape[0], 'goal': obs['desired_goal'].shape[0],
              'action': env.action_space.shape[0], 'action_max': env.action_space.high[0],
              'max_timesteps': env._max_episode_steps}
    return params

if __name__ == '__main__':
    num_eval = 1
    path = '/home/ahmed/Documents/Amaterasu/gobi/skill_learning/models/test_hand/models/'
    model_path = path + 'model_10.pt'
    # vae_model_path = path + 'vae_model_20.pt'

    args = get_args()

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
    policy = RLAgent(args, compute_reward_handreach)
    policy.load(model_path, args)
    # policy.load_goal_encoder(vae_model_path)

    goal_sampler = GoalSampler(args, policy)

    # def rollout worker
    rollout_worker = RolloutWorker(env, policy, compute_reward_handreach, args)

    # eval_goals = goal_sampler.sample_goal(n_goals=50, evaluation=False)
    eval_goals = np.expand_dims(np.array([0.99588355, 0.76734625, 0.1230811 , 1.00974046, 0.84833556,
       0.19989312, 1.03121636, 0.84795953, 0.20208773, 1.03865579,
       0.86772805, 0.20861712, 0.94991353, 0.84901251, 0.19089681]), axis=0)

    # eval_goals = np.array([[0.1, 0.1, 0.], [0.02, 0.02, 0.046]])

    all_results = []
    for i in range(num_eval):
        episodes = rollout_worker.generate_rollout(eval_goals, true_eval=True, animated=True)
        results = np.array([e['rewards'][-1] == 5. for e in episodes])
        all_results.append(results)

    results = np.array(all_results)
    print('Av Success Rate: {}'.format(results.mean()))

    # np.array([0.99588355, 0.76734625, 0.1230811 , 1.00974046, 0.84833556,
    #    0.19989312, 1.03121636, 0.84795953, 0.20208773, 1.03865579,
    #    0.86772805, 0.20861712, 0.94991353, 0.84901251, 0.19089681])
       
    # np.array([0.99594095, 0.76734045, 0.12073955, 1.01828249, 0.76654877,
    #    0.12024485, 1.03098339, 0.85214122, 0.20108755, 1.03997602,
    #    0.86995512, 0.20158985, 0.94994327, 0.84897383, 0.19085163])
       