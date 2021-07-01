import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    #just use CPU

from mylab.envs.tfenv import TfEnv
from garage.misc import logger
from garage.envs.normalized_env import normalize
from garage.envs.env_spec import EnvSpec

import os.path as osp
import argparse
# from example_save_trials import *
import tensorflow as tf
import joblib
import math
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default="TrafficAST-Random")
parser.add_argument('--inita', type=float, default=0.5)
parser.add_argument('--itr', type=int, default=1000)
parser.add_argument('--bs', type=int, default=2000)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--snapshot_mode', type=str, default="gap")
parser.add_argument('--snapshot_gap', type=int, default=500)
parser.add_argument('--log_dir', type=str, default='Random')
parser.add_argument('--args_data', type=str, default=None)
args = parser.parse_args()

# Create the logger
pre_dir = './Data/AST_inita{}/'.format(args.inita)
log_dir = pre_dir+args.log_dir\
            +('bs'+str(args.bs))\
            +('/'+'seed'+str(args.seed))
args.log_dir = log_dir

tabular_log_file = osp.join(log_dir, 'progress.csv')
text_log_file = osp.join(log_dir, 'text.txt')
params_log_file = osp.join(log_dir, 'args.txt')

logger.log_parameters_lite(params_log_file, args)
# logger.add_text_output(text_log_file)
logger.add_tabular_output(tabular_log_file)
logger.set_snapshot_dir(log_dir)
logger.set_snapshot_mode(args.snapshot_mode)
logger.set_snapshot_gap(args.snapshot_gap)
logger.set_log_tabular_only(False)
logger.push_prefix("[%s] " % args.exp_name)

seed = args.seed
top_k = 10

import mcts.BoundedPriorityQueues as BPQ
top_paths = BPQ.BoundedPriorityQueue(top_k)

np.random.seed(seed)
tf.set_random_seed(seed)
with tf.Session() as sess:
    # Create env
    from traffic.make_env import make_env
    env_inner = make_env(env_name='highway',
                        init_ast_action_scale=args.inita,)
    data = joblib.load("Data/Train/TRPO/seed0/itr_1000.pkl")
    policy_inner = data['policy']

    from mylab.rewards.ast_reward import ASTReward
    from mylab.envs.ast_env import ASTEnv
    from mylab.simulators.policy_simulator import PolicySimulator
    reward_function = ASTReward()
    simulator = PolicySimulator(env=env_inner,policy=policy_inner,max_path_length=100)
    env = TfEnv(ASTEnv(interactive=True,
                                 simulator=simulator,
                                 sample_init_state=False,
                                 s_0=0., # not used
                                 reward_function=reward_function,
                                 ))

    # Create policy
    from mylab.policies.random_policy import RandomPolicy
    policy = RandomPolicy(
        name='Policy',
        env_spec=env.spec,
    )

    from mylab.algos.random_search import RandomSearch
    algo = RandomSearch(
        env=env,
        policy=policy,
        batch_size=args.bs,
        n_itr=args.itr+1,
        max_path_length=100,
        top_paths = top_paths,
        plot=False,
        )

    algo.train(sess=sess)
    