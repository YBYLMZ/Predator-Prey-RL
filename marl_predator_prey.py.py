from cv2 import exp
import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.classic_control import rendering
import numpy as np
from os import path
import itertools

from quadrotor_dynamics import Drone, Bot, Tank1, Tank_Bot
from collections import deque
from itertools import product
from matplotlib import pyplot as plt


font = {'family': 'sans-serif',
        'weight': 'bold',
        'size': 14}


class QuadrotorFormationMARL(gym.Env):

    def __init__(self, n_agents=1, n_bots=2,
                 n_tank_agents=1, n_tank_bots=2,
                 N_frame=5, visualization=False,
                 is_centralized=False, moving_target=True, exploration_learning=True):

        super(QuadrotorFormationMARL, self).__init__()
        self. exploration_learning = exploration_learning
        self.current_step = 0
        self.max_drone_agents = 10  # changes the observation matrix
        self.max_tank_agents = 10  # changes the observation matrix
        self.max_drone_bots = 10  # changes the observation matrix
        self.max_tank_bots = 10  # changes the observation matrix
        self.seed()
        self.n_action = 6
        self.observation_dim = 4
        self.dim_actions = 1
        self.n_agents = n_agents
        self.n_bots = n_bots
        self.tank_speed = 1
        self.drone_speed = 2
        self.tank_bot_speed = 1
        self.drone_bot_speed = 2
        #################editedlines###########################
        self.n_tank_agents = n_tank_agents
        self.n_tank_bots = n_tank_bots
        #####################obstacles####################
        self.dynamic_obstacles = True
        self.obstacle_movedirection = [1, 2, 4]
        self.obstacle_vel = 1
        ################editedlinesarefinished#################
        self.visualization = visualization
        self.is_centralized = is_centralized
        self.moving_target = moving_target
        self.action_dict = {0: "Xp", 1: "Xn", 2: "Yp", 3: "Yn"}

        self.quadrotors = []
        self.obstacle_point = []
        #################################Obstacles#########################

        self.obstacle_points = np.array([]
                                        )  # [[10, 5, 1, 12, 10, 2], [0, 10, 3, 1, 12, 5], [0, 11, 2, 2, 13, 3]])
        self.static_obstacle_points = np.array(
            [])  # [[1, 2, 3, 3, 5, 4], [1, 2, 0, 3, 5, 2], [1, 1, 3, 3, 3, 4]])  # [[0, 5, 1, 1, 8, 4], [10, 0, 0, 12, 3, 2], [6, 2, 1, 8, 5, 4]])

        self.obstacle_indices = None
        self.obstacle_pos_xy = None

        self.viewer = None
        self.dtau = 1e-3

        if self.n_tank_agents > 1:
            print("WARNING, CENTRALIZED TRAINING CANT HAVE MORE THAN 1 AGENT")
            self.n_tank_agents = 1
        if self.n_agents > 1:
            print("WARNING, CENTRALIZED TRAINING CANT HAVE MORE THAN 1 AGENT")
            self.n_agents = 1

        # 4*tank + 6*drone

        # 4*tank * 6*drone
        if self.n_tank_agents == 0:
            self.action_space = spaces.Discrete(6*self.n_agents)

        else:
            if True:
                self.action_space = spaces.Discrete(
                    (6*self.n_agents) * (4*self.n_tank_agents))

        # intitialize grid information
        self.x_lim = 10  # grid x limit
        self.y_lim = 10  # grid y limit
        self.z_lim = 5
        self.n_obstacle = 0
        self.uncertainty_grid = np.ones((self.x_lim, self.y_lim, self.z_lim))
        self.uncertainty_grid_visit = np.zeros(
            (self.x_lim, self.y_lim, self.z_lim))
        self.obs_shape = ((self.max_drone_agents + self.max_tank_agents +
                           self.max_drone_bots + self.max_tank_bots)*3+(self.x_lim*self.y_lim*self.z_lim)+6*self.n_obstacle,)
        """ self.obs_shape = self.x_lim * self.y_lim * self.z_lim + \
            (self.max_drone_agents + self.max_tank_agents +
             self.max_drone_bots + self.max_tank_bots)*3
        """
        self.observation_space = spaces.Box(low=-255, high=255,
                                            shape=self.obs_shape, dtype=np.float32)

        self.lim_values = [self.x_lim, self.y_lim, self.z_lim]

        self.grid_res = 1.0  # resolution for grids

        self.out_shape = 82  # width and height for uncertainty matrix

        self.neighbours = 1

        X, Y, Z = np.mgrid[0: self.x_lim: self.grid_res,
                           0: self.y_lim: self.grid_res,
                           0:self.z_lim: self.grid_res]
        self.uncertainty_grids = np.vstack(
            (X.flatten(), Y.flatten(), Z.flatten())).T

        self.N_frame = N_frame  # Number of frames to be stacked
        self.frame_update_iter = 2
        self.iteration = None

        self.action_list = []

        for p in itertools.product([0, 1, 2, 3, 4, 5], repeat=1):
            self.action_list.append(p)

        if self.exploration_learning == True:
            self.n_bots = 0
            self.n_tank_bots = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # // 6 -> tank move
    # divmod(scaler, )

    def step(self, action):
        if True:
            iteration = None
            is_centralized = False

        self.iteration = iteration
        done = False
        cubic_env = np.zeros((self.x_lim, self.y_lim, self.z_lim))

        # drone_total_explored_indices = []
        # tank_total_explored_indices = []

        for i in range(len(self.obstacle_points)):
            self.obstacle_move(i)

        # for i in range(self.n_agents):
          #  drone_total_explored_indices.append([])

        # for i in range(self.n_tank_agents):
            # tank_total_explored_indices.append([])

       # total_indices = np.arange(self.uncertainty_grids.shape[0])

        # self.no_obstacle_indices = np.setdiff1d(
            # total_indices, self.obstacle_indices)

        reward_list = np.ones(self.n_agents) * (-1)
        tank_reward_list = np.ones(self.n_tank_agents) * (-1)
        reward_list_uncertainty = np.zeros(self.n_agents)
        tank_reward_list_uncertainty = np.zeros(self.n_tank_agents)

        tank_action, action_1 = divmod(action, 6)

        if is_centralized:
            agents_actions = self.action_list[action_1]
            tank_agents_actions = self.action_list[tank_action]
        else:
            if self.n_agents:
                agents_actions = np.reshape(action_1, (self.n_agents,))
            if self.n_tank_agents:
                tank_agents_actions = np.reshape(
                    tank_action, (self.n_tank_agents))

        if self.n_agents:
            for agent_ind in range(self.n_agents):
                if self.quadrotors[agent_ind].is_alive:
                    current_action = agents_actions[agent_ind]
                    _, drone_agent_down = self.get_drone_des_grid(
                        agent_ind, current_action)
                    if drone_agent_down:
                        reward_list_uncertainty[agent_ind] -= 10000

        if self.n_tank_agents:
            for agent_ind in range(self.n_tank_agents):
                if self.tanks[agent_ind].is_alive:
                    tank_current_action = tank_agents_actions[agent_ind]
                    _, tank_agent_down = self.get_tank_des_grid(
                        agent_ind, tank_current_action)
                    if tank_agent_down:
                        tank_reward_list_uncertainty[agent_ind] -= 10000

        if self.moving_target:
            for bot_ind in range(self.n_bots):
                if self.bots[bot_ind].is_alive:
                    route_change = self.get_bot_des_grid(bot_ind)

                    if np.linalg.norm(self.bots[bot_ind].state - self.bots[bot_ind].target_state) < 3.5 or route_change:
                        target_pos = [self.np_random.randint(low=0, high=self.x_lim), self.np_random.randint(
                            low=0, high=self.y_lim), self.np_random.randint(low=0, high=self.z_lim)]
                        self.bots[bot_ind].target_state = target_pos

            for bot_ind in range(self.n_tank_bots):
                if self.tank_bots[bot_ind].is_alive:
                    route_change_tank = self.get_tank_bot_des_grid(bot_ind)

                    if np.linalg.norm(self.tank_bots[bot_ind].state-self.tank_bots[bot_ind].target_state) < 3.5 or route_change_tank:
                        target_pos = [self.np_random.randint(
                            low=0, high=self.x_lim), self.np_random.randint(low=0, high=self.y_lim), 0]
                        self.tank_bots[bot_ind].target_state = target_pos

        for agent_ind in range(self.n_agents):
            for other_agents_ind in range(self.n_agents):
                if agent_ind != other_agents_ind:
                    collision_distance = np.linalg.norm(
                        self.quadrotors[agent_ind].state-self.quadrotors[other_agents_ind].state)

                    if (collision_distance <= 3) and self.quadrotors[agent_ind].is_alive and self.quadrotors[other_agents_ind].is_alive:
                        self.quadrotors[agent_ind].is_alive = False
                        self.quadrotors[other_agents_ind].is_alive = False
                        self.quadrotors[agent_ind].state = np.array(
                            [-1, -1, -1])
                        self.quadrotors[other_agents_ind].state = np.array(
                            [-1, -1, -1])
                        reward_list[agent_ind] -= 299
                        reward_list[other_agents_ind] -= 299
                        reward_list_uncertainty[agent_ind] -= 100
                        reward_list_uncertainty[other_agents_ind] -= 100

            if not self.quadrotors[agent_ind].is_alive:
                reward_list[agent_ind] += 1

            if not done:
                for bot_ind in range(self.n_tank_bots):
                    drone_distance = np.linalg.norm(
                        self.quadrotors[agent_ind].state - self.tank_bots[bot_ind].state)
                    if drone_distance <= 3 and self.tank_bots[bot_ind].is_alive:
                        reward_list[agent_ind] += 51
                        self.tank_bots[bot_ind].is_alive = False
                        self.tank_bots[bot_ind].state = np.array([-1, -1, -1])
                        self.quadrotors[agent_ind].is_alive = False

            if not done:
                for bot_ind in range(self.n_bots):
                    drone_distance = np.linalg.norm(
                        self.quadrotors[agent_ind].state-self.bots[bot_ind].state)
                    if drone_distance <= 3 and self.bots[bot_ind].is_alive:
                        reward_list[agent_ind] += 51
                        self.bots[bot_ind].is_alive = False
                        self.bots[bot_ind].state = np.array([-1, -1, -1])

        for agent_ind in range(self.n_tank_agents):
            for other_agents_ind in range(self.n_tank_agents):
                if agent_ind != other_agents_ind:
                    collision_distance = np.linalg.norm(
                        self.tanks[agent_ind].state-self.tanks[other_agents_ind].state)

                    if (collision_distance <= 3) and self.tanks[agent_ind].is_alive and self.tanks[other_agents_ind].is_alive:
                        self.tanks[agent_ind].is_alive = False
                        self.tanks[other_agents_ind].is_alive = False
                        self.tanks[agent_ind].state = np.array([-1, -1, -1])
                        self.tanks[other_agents_ind].state = np.array(
                            [-1, -1, -1])
                        tank_reward_list[agent_ind] -= 300
                        tank_reward_list[other_agents_ind] -= 300
                        tank_reward_list_uncertainty[agent_ind] -= 10000
                        tank_reward_list_uncertainty[other_agents_ind] -= 10000

            if not self.tanks[agent_ind].is_alive:
                tank_reward_list[agent_ind] += 1

            if not done:
                for bot_ind in range(self.n_tank_bots):
                    drone_distance = np.linalg.norm(
                        self.tanks[agent_ind].state-self.tank_bots[bot_ind].state)

                    if drone_distance <= 3 and self.tank_bots[bot_ind].is_alive:
                        tank_reward_list[agent_ind] += 51
                        self.tank_bots[bot_ind].is_alive = False
                        self.tank_bots[bot_ind].state = np.array([-1, -1, -1])

        agent = any(map(lambda x: x.is_alive, self.quadrotors))
        tank_agent = any(map(lambda x: x.is_alive, self.tanks))

        bot_agent = any(map(lambda x: x.is_alive, self.bots))
        tank_bot_agent = any(map(lambda x: x.is_alive, self.tank_bots))

        """ 
        for agent_ind in range(self.n_agents):
            cubic_env[self.quadrotors[agent_ind].state[0]
                      ][self.quadrotors[agent_ind].state[1]][self.quadrotors[agent_ind].state[2]] = - 1
        for agent_ind in range(self.n_tank_agents):
            cubic_env[self.tanks[agent_ind].state[0]
                      ][self.tanks[agent_ind].state[1]][self.tanks[agent_ind].state[2]] = - 1
        for agent_ind in range(self.n_bots):
            cubic_env[self.bots[agent_ind].state[0]
                      ][self.bots[agent_ind].state[1]][self.bots[agent_ind].state[2]] = - 1
        for agent_ind in range(self.n_tank_bots):
            cubic_env[self.tank_bots[agent_ind].state[0]
                      ][self.tank_bots[agent_ind].state[1]][self.tank_bots[agent_ind].state[2]] = -1
         """

        for obstacle in self.obstacle_points:
            mox, moy, moz, ox, oy, oz = obstacle
            cubic_env[mox:ox][moy:oy][moz:oz] = 1

        # .####################### uncertainty_grids ##################################3
        # self.uncertainty_grid += 0.01

        for agent_ind in range(self.n_agents):

            x = self.quadrotors[agent_ind].state[0]
            y = self.quadrotors[agent_ind].state[1]
            z = self.quadrotors[agent_ind].state[2]

            r = self.neighbours

            #print(max(x-r, 0), min(x+r, self.x_lim))

            reward_list_uncertainty[agent_ind] = np.sum(self.uncertainty_grid[max(
                x-r, 0):min(x+r, self.x_lim), y, z]) + np.sum(self.uncertainty_grid[x, max(
                    y-r, 0):min(y+r, self.y_lim), z]) + np.sum(self.uncertainty_grid[x, y, max(
                        z-r, 0):min(z+r, self.z_lim)]) - 2 * self.uncertainty_grid[x, y, z] \
                - (np.sum(self.uncertainty_grid_visit[max(
                    x-r, 0):min(x+r, self.x_lim), y, z]) + np.sum(self.uncertainty_grid_visit[x, max(
                        y-r, 0):min(y+r, self.y_lim), z]) + np.sum(self.uncertainty_grid_visit[x, y, max(
                            z-r, 0):min(z+r, self.z_lim)]) - 2*self.uncertainty_grid_visit[x, y, z]) / 0.1

            self.uncertainty_grid[max(x-r, 0): min(x+r, self.x_lim), y, z] -= 0.3 * np.exp(self.uncertainty_grid_visit[max(
                x-r, 0):min(x+r, self.x_lim), y, z])

            self.uncertainty_grid[x, max(y-r, 0):min(y+r, self.y_lim), z] -= 0.3 * np.exp(self.uncertainty_grid_visit[x, max(
                y-r, 0):min(y+r, self.y_lim), z])

            self.uncertainty_grid[x, y, max(z-r, 0):min(z+r, self.z_lim)] -= 0.3 * np.exp(self.uncertainty_grid_visit[x, y, max(
                z-r, 0):min(z+r, self.z_lim)])

            self.uncertainty_grid_visit[max(
                x-r, 0):min(x+r, self.x_lim), y, z] += 1

            self.uncertainty_grid_visit[x, max(
                y-r, 0):min(y+r, self.y_lim), z] += 1

            self.uncertainty_grid_visit[x, y, z] += 1

            """ self.uncertainty_grid_visit[max(x-r, 0):min(x+r, self.x_lim)][max(y-r, 0):min(
            y+r, self.y_lim)][max(z-r, 0):min(z+r, self.z_lim)] += 1
            """
            self.uncertainty_grid = np.clip(self.uncertainty_grid, 0, 1)
        # print(self.uncertainty_grid.shape)
        # print(self.uncertainty_grid_visit.shape)
        """ print(action_1)
        print(self.quadrotors[0].state[0],
              self.quadrotors[0].state[1], self.quadrotors[0].state[2]) """
        for agent_ind in range(self.n_tank_agents):

            x = self.tanks[agent_ind].state[0]
            y = self.tanks[agent_ind].state[1]
            z = self.tanks[agent_ind].state[2]

            r = self.neighbours

            tank_reward_list_uncertainty[agent_ind] = np.sum(self.uncertainty_grid[max(
                x-r, 0):min(x+r, self.x_lim), y, z]) + np.sum(self.uncertainty_grid[x, max(
                    y-r, 0):min(y+r, self.y_lim), z]) + np.sum(self.uncertainty_grid[x, y, max(
                        z-r, 0):min(z+r, self.z_lim)]) - 2 * self.uncertainty_grid[x, y, z] \
                - (np.sum(self.uncertainty_grid_visit[max(
                    x-r, 0):min(x+r, self.x_lim), y, z]) + np.sum(self.uncertainty_grid_visit[x, max(
                        y-r, 0):min(y+r, self.y_lim), z]) + np.sum(self.uncertainty_grid_visit[x, y, max(
                            z-r, 0):min(z+r, self.z_lim)]) - 2*self.uncertainty_grid_visit[x, y, z]) / 0.1

            self.uncertainty_grid[max(x-r, 0): min(x+r, self.x_lim), y, z] -= 0.3 * np.exp(self.uncertainty_grid_visit[max(
                x-r, 0):min(x+r, self.x_lim), y, z])

            self.uncertainty_grid[x, max(y-r, 0):min(y+r, self.y_lim), z] -= 0.3 * np.exp(self.uncertainty_grid_visit[x, max(
                y-r, 0):min(y+r, self.y_lim), z])

            self.uncertainty_grid[x, y, max(z-r, 0):min(z+r, self.z_lim)] -= 0.3 * np.exp(self.uncertainty_grid_visit[x, y, max(
                z-r, 0):min(z+r, self.z_lim)])

            self.uncertainty_grid_visit[max(
                x-r, 0):min(x+r, self.x_lim), y, z] += 1

            self.uncertainty_grid_visit[x, max(
                y-r, 0):min(y+r, self.y_lim), z] += 1

            self.uncertainty_grid_visit[x, y, z] += 1

            """ self.uncertainty_grid_visit[max(x-r, 0):min(x+r, self.x_lim)][max(y-r, 0):min(
            y+r, self.y_lim)][max(z-r, 0):min(z+r, self.z_lim)] += 1
            """
            self.uncertainty_grid = np.clip(self.uncertainty_grid, 0, 1)
        # print(self.uncertainty_grid)

        # print(self.uncertainty_grid_visit)
        for obstacle_ind in range(self.obstacle_points):
            self.uncertainty_grid_visit[self.obstacle_points[obstacle_ind][0]:self.obstacle_points[obstacle_ind][3],
            self.obstacle_points[obstacle_ind][1]:self.obstacle_points[obstacle_ind][4],
            self.obstacle_points[obstacle_ind][2]:self.obstacle_points[obstacle_ind][5]] = - 1
            
        for obstacle_ind in range(self.static_obstacle_points):
            self.uncertainty_grid_visit[self.static_obstacle_points[obstacle_ind][0]:self.static_obstacle_points[obstacle_ind][3],
            self.static_obstacle_points[obstacle_ind][1]:self.static_obstacle_points[obstacle_ind][4],
            self.static_obstacle_points[obstacle_ind][2]:self.static_obstacle_points[obstacle_ind][5]] = - 1
        
        if (not agent and bot_agent) or (not bot_agent and not tank_bot_agent) or (not agent and not tank_agent):
            done = True

        if (self.exploration_learning) and (agent or tank_agent):
            done = False

        if done and bot_agent:
            pass
        
        if (np.sum(self.uncertainty_grid_visit == 0) < self.x_lim*self.y_lim*self.z_lim*0.3) & self.exploration_learning:
            print("Map succesfully explored in " +
                  str(self.current_step) + "steps")
            done = True

        if self.current_step > 200:
            done = True
            print("Map failed succesfully")
            reward_list -= 100
            reward_list_uncertainty -= 10
            reward_list_uncertainty -= 10

        for obstacle_ind in range(self.obstacle_points):
            self.uncertainty_grid_visit[self.obstacle_points[obstacle_ind][0]:self.obstacle_points[obstacle_ind][3],
            self.obstacle_points[obstacle_ind][1]:self.obstacle_points[obstacle_ind][4],
            self.obstacle_points[obstacle_ind][2]:self.obstacle_points[obstacle_ind][5]] = 0
        for obstacle_ind in range(self.static_obstacle_points):
            self.uncertainty_grid_visit[self.static_obstacle_points[obstacle_ind][0]:self.static_obstacle_points[obstacle_ind][3],
            self.static_obstacle_points[obstacle_ind][1]:self.static_obstacle_points[obstacle_ind][4],
            self.static_obstacle_points[obstacle_ind][2]:self.static_obstacle_points[obstacle_ind][5]] = 0

        if self.exploration_learning == False:

            reward = reward_list[0]+tank_reward_list[0]

        else:

            reward = np.sum(reward_list_uncertainty) + \
                np.sum(tank_reward_list_uncertainty)

        if self.visualization:

            fig = plt.figure()
            self.ax = fig.add_subplot(111, projection='3d')

            self.ax.set_xlim3d(0, self.x_lim)
            self.ax.set_ylim3d(0, self.y_lim)
            self.ax.set_zlim3d(0, self.z_lim)

            self.visualize()

            plt.clf()

        self.current_step += 1

        return (self.get_observation(),
                reward,
                done,
                {})

    def get_observation(self):
        state = -1*np.array(np.ones((self.max_drone_agents, 3)))
        tank_state = -1*np.array(np.ones((self.max_tank_agents, 3)))
        bot_state = -1*np.array(np.ones((self.max_drone_bots, 3)))
        Tank_bot_state = -1*np.array(np.ones((self.max_tank_bots, 3)))

        for agent_ind in range(self.n_agents):
            if self.quadrotors[agent_ind].is_alive:
                state[agent_ind] = np.array(self.quadrotors[agent_ind].state)

        for agent_ind in range(self.n_tank_agents):
            if self.tanks[agent_ind].is_alive:
                tank_state[agent_ind] = np.array(self.tanks[agent_ind].state)

        for agent_ind in range(self.n_bots):
            if self.bots[agent_ind].is_alive:
                bot_state[agent_ind] = np.array(self.bots[agent_ind].state)

        for agent_ind in range(self.n_tank_bots):
            if self.tank_bots[agent_ind].is_alive:
                Tank_bot_state[agent_ind] = np.array(
                    self.tank_bots[agent_ind].state)

        observationlist = np.concatenate(
            [state, tank_state, bot_state, Tank_bot_state]).flatten()
        observationlist = np.concatenate(
            [observationlist, self.uncertainty_grid.flatten(), self.obstacle_points.flatten(), self.static_obstacle_points.flatten()]).flatten()

        self.observation_space = observationlist
        return observationlist

    def generate_agent_position(self):
        self.quadrotors = []

        for _ in range(0, self.n_agents):
            while True:
                current_pos = [self.np_random.randint(low=0, high=self.x_lim), self.np_random.randint(
                    low=0, high=self.y_lim), self.np_random.randint(low=0, high=self.z_lim)]
                state0 = [current_pos[0], current_pos[1], current_pos[2]]
                if not self.is_inside(state0):
                    break

            self.quadrotors.append(Drone(state0))

    def generate_tank_agent_position(self):
        self.tanks = list()

        for _ in range(0, self.n_tank_agents):
            while True:
                current_pos = [self.np_random.randint(
                    low=0, high=self.x_lim-1), self.np_random.randint(low=0, high=self.y_lim-1), 0]
                state0 = [current_pos[0], current_pos[1], current_pos[2]]
                if not self.is_inside(state0):
                    break

            self.tanks.append(Tank1(state0))

    def generate_bot_position(self):
        self.bots = []

        for _ in range(0, self.n_bots):

            while True:
                current_pos = [self.np_random.randint(low=0, high=self.x_lim-1), self.np_random.randint(
                    low=0, high=self.y_lim-1), self.np_random.randint(low=0, high=self.z_lim-1)]
                target_pos = [self.np_random.randint(low=0, high=self.x_lim-1), self.np_random.randint(
                    low=0, high=self.y_lim-1), self.np_random.randint(low=0, high=self.z_lim-1)]

                state0 = [current_pos[0], current_pos[1], current_pos[2]]
                target_state0 = [target_pos[0], target_pos[1], target_pos[2]]

                if not self.is_inside(state0):
                    break

            self.bots.append(Bot(state0, target_state0))

    def generate_tank_bot_position(self):
        self.tank_bots = list()

        for _ in range(0, self.n_tank_bots):

            while True:
                current_pos = [self.np_random.randint(
                    low=0, high=self.x_lim-1), self.np_random.randint(low=0, high=self.y_lim-1), 0]
                target_pos = [self.np_random.randint(
                    low=0, high=self.x_lim-1), self.np_random.randint(low=0, high=self.y_lim-1), 0]

                state0 = [current_pos[0], current_pos[1], current_pos[2]]
                target_state0 = [target_pos[0], target_pos[1], target_pos[2]]

                if not self.is_inside(state0):
                    break

            self.tank_bots.append(Tank_Bot(state0, target_state0))

    def check_collision(self):
        collision = False

        for agent_ind in range(self.n_agents):

            for other_agents_ind in range(self.n_agents):

                if agent_ind != other_agents_ind:
                    dist = np.linalg.norm(
                        self.quadrotors[agent_ind].state-self.quadrotors[other_agents_ind].state)

                    if (dist <= 4):
                        collision = True

        for bot_ind in range(self.n_bots):

            for other_bots_ind in range(self.n_bots):

                if bot_ind != other_bots_ind:
                    dist = np.linalg.norm(
                        self.bots[bot_ind].state-self.bots[other_bots_ind].state)

                    if (dist <= 4):
                        collision = True

        for agent_ind in range(self.n_tank_agents):

            for other_agents_ind in range(self.n_tank_agents):

                if agent_ind != other_agents_ind:
                    dist = np.linalg.norm(
                        self.tanks[agent_ind].state-self.tanks[other_agents_ind].state)

                    if (dist <= 4):
                        collision = True

        for bot_ind in range(self.n_tank_bots):

            for other_bots_ind in range(self.n_tank_bots):

                if bot_ind != other_bots_ind:
                    dist = np.linalg.norm(
                        self.tank_bots[bot_ind].state-self.tank_bots[other_bots_ind].state)

                    if (dist <= 4):
                        collision = True

        return collision

    def reset(self):
        self.generate_agent_position()
        self.generate_bot_position()
        self.generate_tank_agent_position()
        self.generate_tank_bot_position()
        self.iteration = 1

        self.uncertainty_grid = np.ones((self.x_lim, self.y_lim, self.z_lim))
        self.uncertainty_grid_visit = np.zeros(
            (self.x_lim, self.y_lim, self.z_lim))

        self.current_step = 0
        collision = self.check_collision()

        if collision:
            return self.reset()
        else:
            pass

        return np.zeros(self.obs_shape)

    def get_drone_stack(self, agent_ind):
        drone_closest_grids = self.get_closest_n_grids(
            self.quadrotors[agent_ind].state[0:2], self.neighbour_grids)

        drone_stack = np.zeros(self.uncertainty_grids.shape[0])
        drone_stack[drone_closest_grids] = 1
        drone_stack = np.reshape(drone_stack, (self.out_shape, self.out_shape))

        return drone_stack

    def get_bot_stack(self, bot_ind):
        if self.bots[bot_ind].is_alive:
            bot_closest_grids = self.get_closest_n_grids(
                self.bots[bot_ind].state[0:2], self.neighbour_grids)

            bot_stack = np.zeros(self.uncertainty_grids.shape[0])
            bot_stack[bot_closest_grids] = 1
            bot_stack = np.reshape(bot_stack, (self.out_shape, self.out_shape))
        else:
            bot_stack = np.zeros(self.uncertainty_grids.shape[0])
            bot_stack = np.reshape(bot_stack, (self.out_shape, self.out_shape))

        return bot_stack

    def get_bot_des_grid(self, bot_index):
        self.prev_state = [self.bots[bot_index].state[0],
                           self.bots[bot_index].state[1], self.bots[bot_index].state[2]]
        change_route = False

        if self.bots[bot_index].state[0] - self.bots[bot_index].target_state[0] > 2:
            action = 1
            if not self.is_collided(self.bots[bot_index].state, action):
                self.bots[bot_index].state[0] -= self.drone_bot_speed
                self.bots[bot_index].state[0] = np.clip(
                    self.bots[bot_index].state[0], 0,  self.x_lim-1)

        elif self.bots[bot_index].state[0] - self.bots[bot_index].target_state[0] < -2:
            action = 0
            if not self.is_collided(self.bots[bot_index].state, action):
                self.bots[bot_index].state[0] += self.drone_bot_speed

                self.bots[bot_index].state[0] = np.clip(
                    self.bots[bot_index].state[0], 0,  self.x_lim-1)

        elif self.bots[bot_index].state[1] - self.bots[bot_index].target_state[1] > 2:
            action = 3
            if not self.is_collided(self.bots[bot_index].state, action):
                self.bots[bot_index].state[1] -= self.drone_bot_speed

                self.bots[bot_index].state[1] = np.clip(
                    self.bots[bot_index].state[1], 0,  self.y_lim-1)

        elif self.bots[bot_index].state[1] - self.bots[bot_index].target_state[1] < -2:
            action = 2
            if not self.is_collided(self.bots[bot_index].state, action):
                self.bots[bot_index].state[1] += self.drone_bot_speed
                self.bots[bot_index].state[1] = np.clip(
                    self.bots[bot_index].state[1], 0,  self.y_lim-1)

        elif self.bots[bot_index].state[2] - self.bots[bot_index].target_state[2] > 2:
            action = 5
            if not self.is_collided(self.bots[bot_index].state, action):
                self.bots[bot_index].state[2] -= self.drone_bot_speed
                self.bots[bot_index].state[2] = np.clip(
                    self.bots[bot_index].state[2], 0,  self.z_lim-1)

        elif self.bots[bot_index].state[2] - self.bots[bot_index].target_state[2] < -2:
            action = 4
            if not self.is_collided(self.bots[bot_index].state, action):
                self.bots[bot_index].state[2] += 1
                self.bots[bot_index].state[2] = np.clip(
                    self.bots[bot_index].state[2], 0,  self.z_lim-1)

        if np.all(self.prev_state == self.bots[bot_index].state):
            change_route = True

        return change_route

    def obstacle_move(self, obstacle_indice):

        self.obstacle_inside_area(obstacle_indice)

        if self.dynamic_obstacles == False:
            ...
        else:
            # action=0, x += 1.0
            if self.obstacle_movedirection[obstacle_indice] == 0:

                self.obstacle_points[obstacle_indice][0] += self.obstacle_vel
                self.obstacle_points[obstacle_indice][3] += self.obstacle_vel

            # action=1, x -= 1.0
            elif self.obstacle_movedirection[obstacle_indice] == 1:
                self.obstacle_points[obstacle_indice][0] -= self.obstacle_vel
                self.obstacle_points[obstacle_indice][3] -= self.obstacle_vel

            # action=2, y += 1.0
            elif self.obstacle_movedirection[obstacle_indice] == 2:

                self.obstacle_points[obstacle_indice][1] += self.obstacle_vel
                self.obstacle_points[obstacle_indice][4] += self.obstacle_vel

            # action=3, y -= 1.0
            elif self.obstacle_movedirection[obstacle_indice] == 3:
                self.obstacle_points[obstacle_indice][1] -= self.obstacle_vel
                self.obstacle_points[obstacle_indice][4] -= self.obstacle_vel

            # action=4, z += 1.0
            elif self.obstacle_movedirection[obstacle_indice] == 4:
                self.obstacle_points[obstacle_indice][2] += self.obstacle_vel
                self.obstacle_points[obstacle_indice][5] += self.obstacle_vel

            # action=5, z -= 1.0
            elif self.obstacle_movedirection[obstacle_indice] == 5:
                self.obstacle_points[obstacle_indice][2] -= self.obstacle_vel
                self.obstacle_points[obstacle_indice][5] -= self.obstacle_vel
            else:
                print("Invalid discrete action!")

        return

    def obstacle_inside_area(self, obstacle_indice):
        for _ in range(len(self.obstacle_points[obstacle_indice])):
            if self.obstacle_points[obstacle_indice][0] == 0:
                self.obstacle_movedirection[obstacle_indice] = 0
            elif self.obstacle_points[obstacle_indice][3] == self.x_lim:
                self.obstacle_movedirection[obstacle_indice] = 1
            elif self.obstacle_points[obstacle_indice][1] == 0:
                self.obstacle_movedirection[obstacle_indice] = 2
            elif self.obstacle_points[obstacle_indice][4] == self.y_lim:
                self.obstacle_movedirection[obstacle_indice] = 3
            elif self.obstacle_points[obstacle_indice][2] == 0:
                self.obstacle_movedirection[obstacle_indice] = 4
            elif self.obstacle_points[obstacle_indice][5] == self.z_lim:
                self.obstacle_movedirection[obstacle_indice] = 5
        return

    def get_tank_bot_des_grid(self, bot_index):
        self.prev_state = [self.tank_bots[bot_index].state[0],
                           self.tank_bots[bot_index].state[1], self.tank_bots[bot_index].state[2]]

        change_route = False

        if self.tank_bots[bot_index].state[0] - self.tank_bots[bot_index].target_state[0] > 2:
            action = 1
            if not self.is_collided(self.tank_bots[bot_index].state, action):
                self.tank_bots[bot_index].state[0] -= self.tank_bot_speed
                self.tank_bots[bot_index].state[0] = np.clip(
                    self.tank_bots[bot_index].state[0], 0,  self.x_lim-1)

        elif self.tank_bots[bot_index].state[0] - self.tank_bots[bot_index].target_state[0] < -2:
            action = 0
            if not self.is_collided(self.tank_bots[bot_index].state, action):
                self.tank_bots[bot_index].state[0] += self.tank_bot_speed
                self.tank_bots[bot_index].state[0] = np.clip(
                    self.tank_bots[bot_index].state[0], 0,  self.x_lim-1)

        elif self.tank_bots[bot_index].state[1] - self.tank_bots[bot_index].target_state[1] > 2:
            action = 3
            if not self.is_collided(self.tank_bots[bot_index].state, action):
                self.tank_bots[bot_index].state[1] -= self.tank_bot_speed
                self.tank_bots[bot_index].state[1] = np.clip(
                    self.tank_bots[bot_index].state[1], 0,  self.y_lim-1)

        elif self.tank_bots[bot_index].state[1] - self.tank_bots[bot_index].target_state[1] < -2:
            action = 2
            if not self.is_collided(self.tank_bots[bot_index].state, action):
                self.tank_bots[bot_index].state[1] += self.tank_bot_speed

                self.tank_bots[bot_index].state[1] = np.clip(
                    self.tank_bots[bot_index].state[1], 0,  self.y_lim-1)

        if np.all(self.prev_state == self.tank_bots[bot_index].state):
            change_route = True

        return change_route

    def get_drone_des_grid(self, drone_index, discrete_action):

        if discrete_action == 0:  # action=0, x += 1.0

            if not self.is_collided(self.quadrotors[drone_index].state, discrete_action):
                self.quadrotors[drone_index].state[0] += self.drone_speed
                self.quadrotors[drone_index].state[0] = np.clip(
                    self.quadrotors[drone_index].state[0], 0,  self.x_lim-1)

        elif discrete_action == 1:  # action=1, x -= 1.0
            if not self.is_collided(self.quadrotors[drone_index].state, discrete_action):
                self.quadrotors[drone_index].state[0] -= self.drone_speed
                self.quadrotors[drone_index].state[0] = np.clip(
                    self.quadrotors[drone_index].state[0], 0,  self.x_lim-1)

        elif discrete_action == 2:  # action=2, y += 1.0
            if not self.is_collided(self.quadrotors[drone_index].state, discrete_action):
                self.quadrotors[drone_index].state[1] += self.drone_speed
                self.quadrotors[drone_index].state[1] = np.clip(
                    self.quadrotors[drone_index].state[1], 0,  self.y_lim-1)

        elif discrete_action == 3:  # action=3, y -= 1.0

            if not self.is_collided(self.quadrotors[drone_index].state, discrete_action):
                self.quadrotors[drone_index].state[1] -= self.drone_speed
                self.quadrotors[drone_index].state[1] = np.clip(
                    self.quadrotors[drone_index].state[1], 0,  self.y_lim-1)

        elif discrete_action == 4:  # action=4, z += 1.0
            if not self.is_collided(self.quadrotors[drone_index].state, discrete_action):

                self.quadrotors[drone_index].state[2] += self.drone_speed
                self.quadrotors[drone_index].state[2] = np.clip(
                    self.quadrotors[drone_index].state[2], 0,  self.z_lim-1)

        elif discrete_action == 5:  # action=5, z -= 1.0
            if not self.is_collided(self.quadrotors[drone_index].state, discrete_action):

                self.quadrotors[drone_index].state[2] -= self.drone_speed
                self.quadrotors[drone_index].state[2] = np.clip(
                    self.quadrotors[drone_index].state[2], 0,  self.z_lim-1)

        else:
            print("Invalid discrete action!")
        agent_down = False
        if self.is_collided(self.quadrotors[drone_index].state, discrete_action):
            self.quadrotors[drone_index].is_alive = False
            self.quadrotors[drone_index].state = np.array([-1, -1, -1])
            agent_down = True
        drone_current_state = np.copy(self.quadrotors[drone_index].state)
        return drone_current_state, agent_down
        #############################################editedlines#################################

    def get_tank_des_grid(self, drone_index, discrete_action):

        if discrete_action == 0:  # action=0, x += 1.0
            if not self.is_collided(self.tanks[drone_index].state, discrete_action):
                self.tanks[drone_index].state[0] += self.tank_speed
                self.tanks[drone_index].state[0] = np.clip(
                    self.tanks[drone_index].state[0], 0,  self.x_lim-1)

        elif discrete_action == 1:  # action=1, x -= 1.
            if not self.is_collided(self.tanks[drone_index].state, discrete_action):
                self.tanks[drone_index].state[0] -= self.tank_speed
                self.tanks[drone_index].state[0] = np.clip(
                    self.tanks[drone_index].state[0], 0,  self.x_lim-1)

        elif discrete_action == 2:  # action=2, y += 1.0
            if not self.is_collided(self.tanks[drone_index].state, discrete_action):
                self.tanks[drone_index].state[1] += self.tank_speed
                self.tanks[drone_index].state[1] = np.clip(
                    self.tanks[drone_index].state[1], 0,  self.y_lim-1)

        elif discrete_action == 3:  # action=3, y -= 1.0
            if not self.is_collided(self.tanks[drone_index].state, discrete_action):
                self.tanks[drone_index].state[1] -= self.tank_speed
                self.tanks[drone_index].state[1] = np.clip(
                    self.tanks[drone_index].state[1], 0,  self.y_lim-1)

        else:
            print("Invalid discrete action!")
        agent_down = False

        if self.is_collided(self.tanks[drone_index].state, discrete_action):
            self.tanks[drone_index].is_alive = False
            self.tanks[drone_index].state = np.array([-1, -1, -1])
            agent_down = True

        tank_current_state = np.copy(self.tanks[drone_index].state)

        return tank_current_state, agent_down

    def get_closest_n_grids(self, current_pos, n):
        differences = current_pos-self.uncertainty_grids
        distances = np.sum(differences*differences, axis=1)
        sorted_indices = sorted(range(len(distances)),
                                key=lambda k: distances[k])

        return sorted_indices[0:n]

    def get_closest_grid(self, current_pos):
        differences = current_pos - self.uncertainty_grids
        distances = np.sum(differences*differences, axis=1)
        min_ind = np.argmin(distances)

        return min_ind

    def visualize(self, mode='human'):
        RESOLUTION = 6

        for obs_point in self.obstacle_points:

            x1, y1, z1, x2, y2, z2 = obs_point

            zex = np.linspace(z1, z2, RESOLUTION)
            iks = np.linspace(x1, x2, RESOLUTION)
            yeğ = np.linspace(y1, y2, RESOLUTION)

            for x, y in product(iks, yeğ):
                self.ax.scatter(x, y, zex, color='black')

        for obs_point in self.static_obstacle_points:
            x1, y1, z1, x2, y2, z2 = obs_point
            zex = np.linspace(z1, z2, RESOLUTION)
            iks = np.linspace(x1, x2, RESOLUTION)
            yeğ = np.linspace(y1, y2, RESOLUTION)

            for x, y in product(iks, yeğ):
                self.ax.scatter(x, y, zex, color='purple')

        # dıron
        print(self.n_tank_agents)
        for agent_ind in range(self.n_agents):
            # , self.tanks[agent_ind].state[2]
            x, y, z = self.quadrotors[agent_ind].state[0], self.quadrotors[agent_ind].state[1], 0
            self.ax.scatter(x, y, z, color='pink')

        # tenk
        for agent_ind in range(self.n_tank_agents):
            # , self.tanks[agent_ind].state[2]
            x, y, z = self.tanks[agent_ind].state[0], self.tanks[agent_ind].state[1], 0
            self.ax.scatter(x, y, z, color='orange')

        for agent_ind in range(self.n_tank_bots):
            # self.tank_bots[agent_ind].state[2]
            x, y, z = self.tank_bots[agent_ind].state[0], self.tank_bots[agent_ind].state[1], 0
            self.ax.scatter(x, y, z, color='red')

        for agent_ind in range(self.n_bots):
            # self.bots[agent_ind].state[2]
            x, y, z = self.bots[agent_ind].state[0], self.bots[agent_ind].state[1], 0
            self.ax.scatter(x, y, z, color='blue')

        self.ax.scatter(self.x_lim, self.y_lim, self.z_lim, color='white')
        self.ax.scatter(0, 0, 0, color='white')

        self.ax.scatter(self.x_lim/2, self.y_lim/2, self.z_lim, color='white')

        plt.pause(10)

        return

    def get_obstacle_indices(self):
        lst = []
        for location in self.obstacle_points:
            xyz = np.mgrid[location[0]:location[3]+0.1:1,
                           location[1]:location[4]+0.1:1,
                           location[2]:location[5]+0.1:2*1].reshape(3, -1).T
            lst.append(xyz)

        self.obstacle_positions = np.vstack((lst[i] for i in range(len(lst))))
        array_of_tuples = map(tuple, self.obstacle_positions)
        self.obstacle_positions = tuple(array_of_tuples)

        current_pos = list()
        for pos in self.obstacle_positions:
            current_pos.append(pos)

        return current_pos

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def is_collided(self, cstate, action) -> bool:
        state = cstate.copy()

        x = state[0]
        y = state[1]
        z = state[2]

        if action == 0:
            x += 1

        if action == 1:
            x -= 1

        if action == 2:
            y += 1

        if action == 3:
            y -= 1

        if action == 4:
            z += 1

        if action == 5:
            z -= 1

        for obstacle in self.obstacle_points:
            mox, moy, moz, ox, oy, oz = obstacle

            if (mox <= x <= ox) and (moy <= y <= oy) and (moz <= z <= oz):
                return True

        for obstacle in self.static_obstacle_points:
            mox, moy, moz, ox, oy, oz = obstacle
            if (mox <= x <= ox) and (moy <= y <= oy) and (moz <= z <= oz):
                return True

        return False

    def is_inside(self, state):
        x = state[0]
        y = state[1]
        z = state[2]

        for obstacle in self.obstacle_points:
            mox, moy, moz, ox, oy, oz = obstacle
            if (mox <= x <= ox) and (moy <= y <= oy) and (moz <= z <= oz):
                return True

        for obstacle in self.static_obstacle_points:
            mox, moy, moz, ox, oy, oz = obstacle
            if (mox <= x <= ox) and (moy <= y <= oy) and (moz <= z <= oz):
                return True

        return False

    def render(self, mode):
        self.visualize()