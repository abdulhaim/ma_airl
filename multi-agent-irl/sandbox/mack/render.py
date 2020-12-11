import gym
import click
import rl.common.tf_util as U
import multiagent
import tensorflow as tf
import make_env
import gym.spaces
import numpy as np
import time
from rl.common.misc_util import set_global_seeds
from sandbox.mack.acktr_disc import Model, onehot
from sandbox.mack.policies import CategoricalPolicy
from rl import bench
from rl.common.vec_env.subproc_vec_env import SubprocVecEnv
import imageio
import pickle as pkl


@click.command()
@click.option('--path', type=click.STRING,
              default='/Users/marwaabdulhai/Desktop/2020_2021/6.804/MA-AIRL/multi-agent-irl/atlas_2/exps/mack/simple_spread/l-0.1-b-1000/seed-1/checkpoint28000')

def render(path):
    tf.reset_default_graph()

    def create_env():
        env = make_env.make_env('simple_spread')
        env.seed(3)
        env = bench.Monitor(env, '/tmp/',  allow_early_resets=True)
        set_global_seeds(3)
        return env

    env = create_env()
    nenv = 1
    n_agents = len(env.observation_space)

    ob_space = env.observation_space
    ac_space = env.action_space
    n_actions = [action.n for action in ac_space]
    make_model = lambda: Model(
        CategoricalPolicy, ob_space, ac_space, 1, total_timesteps=1e7, nprocs=2, nsteps=500,
        nstack=1, ent_coef=0.01, vf_coef=0.5, vf_fisher_coef=1.0, lr=0.01, max_grad_norm=0.5, kfac_clip=0.001,
                               lrschedule='linear')
    model = make_model()
    model.load(path)

    images = []
    sample_trajs = []
    for i in range(200):
        all_ob, all_agent_ob, all_ac, all_rew, ep_ret = [], [], [], [], [0, 0, 0]
        for k in range(n_agents):
            all_ob.append([])
            all_ac.append([])
            all_rew.append([])
        obs = env.reset()
        obs = [ob[None, :] for ob in obs]
        action = [np.zeros([1]) for _ in range(n_agents)]
        step = 0
        done = False
        while not done:
            action, _, _ = model.step(obs, action)
            actions_list = [onehot(action[k][0], n_actions[k]) for k in range(n_agents)]
            for k in range(n_agents):
                all_ob[k].append(obs[k])
                all_ac[k].append(actions_list[k])
            all_agent_ob.append(np.concatenate(obs, axis=1))
            obs, rew, done, _ = env.step(actions_list)
            len(rew)
            for k in range(n_agents):
                all_rew[k].append(rew[k])
                ep_ret[k] += rew[k]
            obs = [ob[None, :] for ob in obs]
            step += 1
            img = env.render(mode='rgb_array')
            images.append(img[0])
            time.sleep(0.02)

            if step == 50 or True in done:
                done = True
                step = 0
            else:
                done = False
        for k in range(n_agents):
            all_ob[k] = np.squeeze(all_ob[k])
        all_agent_ob = np.squeeze(all_agent_ob)
        traj_data = {
            "ob": all_ob, "ac": all_ac, "rew": all_rew,
            "ep_ret": ep_ret, "all_ob": all_agent_ob
        }
        sample_trajs.append(traj_data)
        print(ep_ret[0], ep_ret[1])
    # pkl.dump(sample_trajs, open('atlas_2/checkpoint28000.pkl', 'wb'))
    imageio.mimsave("video_ground_truth" + '.mp4', images, fps=25)

if __name__ == '__main__':
    render()