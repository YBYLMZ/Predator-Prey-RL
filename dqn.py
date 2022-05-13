import os
import time
import numpy as np
from matplotlib import pyplot as plt

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from sympy import true

from predator_prey import QuadrotorFormation
from marl_predator_prey import QuadrotorFormationMARL


QuadrotorFormation = QuadrotorFormationMARL

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

TOTAL_TIMESTEPS = 1_000_000
CHECKPOINT_FREQ = 100_000
N_ENV = 8
DEMO = True
TEST = True


def make_env():
    def _init():
        env = QuadrotorFormation(visualization=False,
                                 moving_target=True, nenvs=N_ENV)
        return env.state
    return _init


if __name__ == '__main__':
    # callbacks
    if not TEST:
        checkpoint_callback = CheckpointCallback(
            save_freq=CHECKPOINT_FREQ, save_path='./model_checkpoints/', name_prefix='rl_model_dyno_')
    else:
        checkpoint_callback = None

    env = QuadrotorFormation(visualization=False, moving_target=True)

    model = DQN("MlpPolicy", env,
                tensorboard_log='./tensorboard_logs_dyno/', batch_size=32,

                verbose=2, policy_kwargs={"net_arch": [512, 512, 512]})  # exploration_fraction=0.99,

    if not TEST:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, log_interval=5,
                    tb_log_name="dqn_log_dyno", callback=checkpoint_callback)
        model.save("dqn_predator_dyno")
        del model

    model = model.load("rl_model_dyno__700000_steps.zip")
    obs = env.reset()
    total_rew = 0
    rews = []

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlim3d(0, env.x_lim)
    ax.set_ylim3d(0, env.y_lim)
    ax.set_zlim3d(0, env.z_lim)

    env = QuadrotorFormation(visualization=False, moving_target=True)
    env.reset()
    if DEMO:
        i = 0
        rewrrd = 0
        while i < 100:
            plt.cla()

            action, _states = model.predict(obs, deterministic=False)
            obs, rew, done, info = env.step(action)

            total_rew += rew
            rews.append(rew)

            if done:
                obs = env.reset()
                total_rew = 0
                rews = []

                i += 1
                time.sleep(0.5)
            print(total_rew)
            print(action)
            RESOLUTION = 5
            # obstacles

            # OpTiMiZe
            from itertools import product
            for obs_point in env.obstacle_points:
                x1, y1, z1, x2, y2, z2 = obs_point
                zex = np.linspace(z1, z2, RESOLUTION)
                iks = np.linspace(x1, x2, RESOLUTION)
                yeğ = np.linspace(y1, y2, RESOLUTION)

                for x, y in product(iks, yeğ):
                    ax.scatter(x, y, zex, color='black')

            for obs_point in env.static_obstacle_points:
                x1, y1, z1, x2, y2, z2 = obs_point
                zex = np.linspace(z1, z2, RESOLUTION)
                iks = np.linspace(x1, x2, RESOLUTION)
                yeğ = np.linspace(y1, y2, RESOLUTION)

                for x, y in product(iks, yeğ):
                    ax.scatter(x, y, zex, color='purple')

            # dıron
            x, y, z = obs[0:3]
            ax.scatter(x, y, z, color='pink')

            # tenk
            x, y, z = obs[30:33]
            ax.scatter(x, y, z, color='orange')

            for i in range(env.n_tank_bots):
                x, y, z = obs[90+(3*i):90+3*(i+1)]
                ax.scatter(x, y, z, color='red')

            for i in range(env.n_bots):
                x, y, z = obs[60+(3*i):60+3*(i+1)]
                ax.scatter(x, y, z, color='blue')

            ax.scatter(env.x_lim, env.y_lim, env.z_lim, color='white')
            ax.scatter(0, 0, 0, color='white')

            ax.scatter(env.x_lim/2, env.y_lim/2, env.z_lim, color='white')

            plt.pause(0.01)
