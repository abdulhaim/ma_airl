#!/usr/bin/env python
import logging
import os
import itertools
import click
import gym

import make_env
from rl import bench
from rl import logger
from rl.common import set_global_seeds
from rl.common.vec_env.subproc_vec_env import SubprocVecEnv
from irl.dataset import MADataSet
from irl.mack.airl import learn
from sandbox.mack.policies import CategoricalPolicy
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from envs.box2d.gaze_two_agents import Maze_v1
from utils import *
from envs.box2d import *


def train(logdir, env_id, num_timesteps, lr, timesteps_per_batch, seed, num_cpu, expert_path,
          traj_limitation, ret_threshold, dis_lr, disc_type='decentralized', bc_iters=500, l2=0.1, d_iters=1,
          rew_scale=0.1):
    def create_env(rank):
        def _thunk():
            maze_sampler = MazeSampler()
            maze_sampler.gen_env_defs()

            env = Maze_v1(action_type='force',
                          maze_sampler= maze_sampler,
                          goals=[['LMA', 0, 0, 1], ['LMO', 0, 1, 1]],
                          strengths=[2,2],
                          sizes=[0,0,1,0],
                          densities=[0, 0, 0, 0],
                          init_positions= [7, 4, 1, 3],
                          action_space_types=[0, 0],
                          costs=[0,0],
                          temporal_decay=[0, 0],
                          visibility=[1, 1, 1, 1])

            env.seed()
            env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)),
                                allow_early_resets=True)
            gym.logger.setLevel(logging.WARN)
            return env
        return _thunk

    logger.configure(logdir, format_strs=['stdout', 'log', 'json', 'tensorboard'])

    set_global_seeds(seed)
    env = SubprocVecEnv([create_env(i) for i in range(num_cpu)], is_multi_agent=True)
    policy_fn = CategoricalPolicy
    expert = MADataSet(expert_path, ret_threshold=ret_threshold, traj_limitation=traj_limitation, nobs_flag=True)
    learn(policy_fn, expert, env, env_id, seed, total_timesteps=int(num_timesteps * 1.1), nprocs=num_cpu,
          nsteps=timesteps_per_batch // num_cpu, lr=lr, ent_coef=0.0, dis_lr=dis_lr,
          disc_type=disc_type, bc_iters=bc_iters, identical=make_env.get_identical(env_id), l2=l2, d_iters=d_iters,
          rew_scale=rew_scale)
    env.close()


@click.command()
@click.option('--logdir', type=click.STRING, default='/atlas/exp')
@click.option('--env', type=click.STRING, default='simple_spread')
@click.option('--expert_path', type=click.STRING,
              default="/Users/marwaabdulhai/Desktop/2020_2021/6.804/MA-AIRL/indep/D112420_053715_1_1_F8_E0_G['LMA', 0, 0, 1]_['LMO', 0, 1, 1]_ST3_3_SZ0_0_0_0_P15_3_6_5_A0_0_C0_0_D[0, 0]_M0.0_0.0_AN-0.55_-1.13_MCTS_L40/0/R0.0_0.0_PL1_EL1_0_0_s1000_r10_cI1.25_cB1000.0_e0.pik")
@click.option('--seed', type=click.INT, default=1)
@click.option('--traj_limitation', type=click.INT, default=200)
@click.option('--ret_threshold', type=click.FLOAT, default=-10)
@click.option('--dis_lr', type=click.FLOAT, default=0.1)
@click.option('--disc_type', type=click.Choice(['decentralized', 'decentralized-all']),
              default='decentralized')
@click.option('--bc_iters', type=click.INT, default=500)
@click.option('--l2', type=click.FLOAT, default=0.1)
@click.option('--d_iters', type=click.INT, default=1)
@click.option('--rew_scale', type=click.FLOAT, default=0)
def main(logdir, env, expert_path, seed, traj_limitation, ret_threshold, dis_lr, disc_type, bc_iters, l2, d_iters,
         rew_scale):
    env_ids = [env]
    lrs = [0.001]
    seeds = [seed]
    batch_sizes = [1000]

    for env_id, seed, lr, batch_size in itertools.product(env_ids, seeds, lrs, batch_sizes):
        train('atlas/exp' + '/airl_4/' + env_id + '/' + disc_type + '/s-{}/l-{}-b-{}-d-{}-c-{}-l2-{}-iter-{}-r-{}/seed-{}'.format(
              traj_limitation, lr, batch_size, dis_lr, bc_iters, l2, d_iters, rew_scale, seed),
              env_id, 5e7, lr, batch_size, seed, batch_size // 250, expert_path,
              traj_limitation, ret_threshold, dis_lr, disc_type=disc_type, bc_iters=bc_iters, l2=l2, d_iters=d_iters,
              rew_scale=rew_scale)


if __name__ == "__main__":
    main()