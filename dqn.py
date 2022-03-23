import os
import time
import numpy as np
from matplotlib import pyplot as plt

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env

from predator_prey import QuadrotorFormation

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

TOTAL_TIMESTEPS = 10000
CHECKPOINT_FREQ = 25000
N_ENV = 8


def make_env():
    def _init():
        env = QuadrotorFormation(
            n_tank_bots=1, visualization=False, moving_target=True, n_bots=4, nenvs=N_ENV)
        return env.state
    return _init


if __name__ == '__main__':
    # callbacks
    checkpoint_callback = CheckpointCallback(save_freq=CHECKPOINT_FREQ, save_path='./model_checkpoints/',
                                             name_prefix='rl_model')

    env = QuadrotorFormation(
        n_tank_bots=1, visualization=False, moving_target=True, n_bots=4)

    env = [make_env() for i in range(4)]

    model = PPO("MlpPolicy", env, verbose=1,
                tensorboard_log='./tensorboard_logs/', batch_size=256)
    model.learn(total_timesteps=25_000, log_interval=4,
                tb_log_name="dqn_second", callback=checkpoint_callback)
    model.save("dqn_predator2")

    obs = env.reset()
    total_rew = 0
    rews = []

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlim3d(0, 30)
    ax.set_ylim3d(0, 30)
    ax.set_zlim3d(0, 30)

    env = QuadrotorFormation(
        n_tank_bots=1, visualization=False, moving_target=True, n_bots=4)

    i = 0
    while i < 100:
        plt.cla()

        action, _states = model.predict(obs, deterministic=False)
        obs, rew, done, info = env.step(action)

        total_rew += rew
        rews.append(rew)

        # env.renqder()
        if done:
            obs = env.reset()
            total_rew = 0
            rews = []

            i += 1
            time.sleep(0.5)

        x, y, z = obs[0, ...]
        ax.scatter(x, y, z, color='black')
        x, y, z = obs[30, ...]
        ax.scatter(x, y, z, color='red')

        x, y, z = obs[20, ...]
        ax.scatter(x, y, z, color='blue')
        x, y, z = obs[21, ...]
        ax.scatter(x, y, z, color='blue')
        x, y, z = obs[22, ...]
        ax.scatter(x, y, z, color='blue')
        x, y, z = obs[23, ...]
        ax.scatter(x, y, z, color='blue')

        ax.scatter(40, 40, 12, color='white')
        ax.scatter(0, 0, 0, color='white')

        ax.scatter(20, 20, 12, color='white')
        ax.scatter(0, 0, 0, color='white')

        plt.pause(0.05)
