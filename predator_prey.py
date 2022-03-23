from cgitb import reset
import gym
from gym import spaces, error, utils
from gym.utils import seeding
from gym.envs.classic_control import rendering
from matplotlib.pyplot import ylim
import numpy as np
from os import path
import itertools

from sympy import true
from quadrotor_dynamics import Quadrotor, Drone, Bot, Tank, Tank1, Tank_Bot
from numpy.random import uniform, randint
from time import sleep
from collections import deque
import json


font = {'family': 'sans-serif',
        'weight': 'bold',
        'size': 14}


class QuadrotorFormation(gym.Env):

    def __init__(self, n_agents=1, n_bots=0,
                 n_tank_agents=0, n_tank_bots=1,
                 N_frame=5, visualization=True,
                 is_centralized=False, moving_target=True, nenvs=1):

        super(QuadrotorFormation, self).__init__()

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
        self.state = False
        self.num_envs = nenvs
        #################editedlines###########################

        self.n_tank_agents = n_tank_agents
        self.n_tank_bots = n_tank_bots

        ################editedlinesarefinished#################

        self.visualization = visualization
        self.is_centralized = is_centralized
        self.moving_target = moving_target
        self.action_dict = {0: "Xp", 1: "Xn", 2: "Yp", 3: "Yn"}

        self.quadrotors = []
        #################################Obstacles#########################

       # mainlist = []

        # infile = open('obstacle.txt','r')
        # for line in infile:
        #    mainlist.append(line.strip().split(','))

        # infile.close()

        # obstacle_coordinates=[ list( map(int,i) ) for i in mainlist ]

        self.obstacle_points = np.array([[26, 5, 0, 37, 10, 4]])

        self.obstacle_indices = None
        self.obstacle_pos_xy = None

        #self.current_pos = self.get_obstacle_indices()

        # editedlines###########################""

        self.tanks = list()

        ################editedlinesarefinished#################
        self.viewer = None
        self.dtau = 1e-3

        if self.is_centralized:
            self.action_space = spaces.Discrete(
                self.n_action**(self.n_agents+n_tank_agents))
        else:
            self.action_space = spaces.Discrete(self.n_action)

        # intitialize grid information
        self.x_lim = 40  # grid x limit
        self.y_lim = 40  # grid y limit
        self.z_lim = 12

        self.obs_shape = self.x_lim * self.y_lim * self.z_lim + \
            (self.max_drone_agents + self.max_tank_agents +
             self.max_drone_bots + self.max_tank_bots)*3

        self.observation_space = spaces.Box(low=-255, high=255,
                                            shape=(40, 3), dtype=np.float32)

        self.lim_values = [self.x_lim, self.y_lim, self.z_lim]
        self.grid_res = 1.0  # resolution for grids
        self.out_shape = 82  # width and height for uncertainty matrix
        self.N_closest_grid = 4
        self.neighbour_grids = 8

        X, Y, Z = np.mgrid[0: self.x_lim: self.grid_res,
                           0: self.y_lim: self.grid_res,
                           0:self.z_lim: self.grid_res]
        self.uncertainty_grids = np.vstack(
            (X.flatten(), Y.flatten(), Z.flatten())).T

        self.N_frame = N_frame  # Number of frames to be stacked
        self.frame_update_iter = 2
        self.iteration = None
        self.agents_stacks = [deque([], maxlen=self.N_frame)
                              for _ in range(self.n_agents)]
        self.bots_stacks = [deque([], maxlen=self.N_frame)
                            for _ in range(self.n_bots)]

        # editedline######################3
        self.tank_agents_stacks = [
            deque([], maxlen=self.N_frame) for _ in range(self.n_agents)]
        self.tank_bots_stacks = [deque([], maxlen=self.N_frame)
                                 for _ in range(self.n_bots)]
        ##################editedlinesarefinished##########

        self.action_list = []
        for p in itertools.product([0, 1, 2, 3, 4, 5], repeat=1):
            self.action_list.append(p)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):  # iteration, is_centralized):
        if True:
            iteration = None
            is_centralized = False

        # Environment Step
        self.iteration = iteration
        done = False
        #####################CollisionEtce#########################
        cubic_env = np.zeros((self.x_lim, self.y_lim, self.z_lim))

        drone_total_explored_indices = []
        tank_total_explored_indices = []

        for i in range(self.n_agents):
            drone_total_explored_indices.append([])

        for i in range(self.n_tank_agents):
            tank_total_explored_indices.append([])

        drone_obstacle_collision = np.zeros(self.n_agents)
        tank_obstacle_collision = np.zeros(self.n_tank_agents)

        drone_current_pos = np.array([[self.quadrotors[i].state[0], self.quadrotors[i].state[1],
                                     self.quadrotors[i].state[2]] for i in range(self.n_agents)])

        tank_current_pos = np.array([[self.tanks[i].state[0], self.tanks[i].state[1],
                                     self.tanks[i].state[2]] for i in range(self.n_tank_agents)])

        total_indices = np.arange(self.uncertainty_grids.shape[0])

        self.no_obstacle_indices = np.setdiff1d(
            total_indices, self.obstacle_indices)

        for i in range(self.n_agents):

            current_pos = [drone_current_pos[i, 0],
                           drone_current_pos[i, 1], drone_current_pos[i, 2]]
            differences = current_pos - self.uncertainty_grids
            distances = np.sum(differences * differences, axis=1)
            sorted_indices = sorted(
                range(len(distances)), key=lambda k: distances[k])
            # print ("\n Agent: ", (i+1))
            # print ("drone_position: ", self.quadrotors[i].state[0:4])
            # print ("drone_grid_indices: ", sorted_indices[0:4])
            # print ("drone_grid_positions: ", self.uncertainty_grids[sorted_indices[0:4]])

    ###############################################################
        # Is reward ###########################3
        reward_list = np.ones(self.n_agents) * (-2)
        tank_reward_list = np.ones(self.n_tank_agents) * (-2)

        action_1 = action
        tank_action = 2

        if is_centralized:
            agents_actions = self.action_list[action_1]
            tank_agents_actions = self.action_list[tank_action]
        else:
            if self.n_agents:
                agents_actions = np.reshape(action_1, (self.n_agents,))
            if self.n_tank_agents:
                tank_agents_actions = np.reshape(
                    tank_action, (self.n_tank_agents,))

        if self.n_agents:
            for agent_ind in range(self.n_agents):
                if self.quadrotors[agent_ind].is_alive:
                    current_action = agents_actions[agent_ind]
                    drone_current_state = self.get_drone_des_grid(
                        agent_ind, current_action)
        ###########################editedline################################
        if self.n_tank_agents:
            for agent_ind in range(self.n_tank_agents):
                if self.tanks[agent_ind].is_alive:
                    tank_current_action = tank_agents_actions[agent_ind]
                    tank_current_state = self.get_tank_des_grid(
                        agent_ind, tank_current_action)
        ###########################editedlinesfinished#########################
        #???????????????????????????????????????????????????????????????????????????????????????????????#
        if self.moving_target:
            for bot_ind in range(self.n_bots):
                if self.bots[bot_ind].is_alive:
                    route_change = self.get_bot_des_grid(bot_ind)

                    if np.linalg.norm(self.bots[bot_ind].state - self.bots[bot_ind].target_state) < 3.5 or route_change:
                        target_pos = [self.np_random.randint(low=0, high=self.x_lim - 1), self.np_random.randint(
                            low=0, high=self.y_lim - 1), self.np_random.randint(low=0, high=self.z_lim - 1)]
                        self.bots[bot_ind].target_state = target_pos

            ###########################editedline################################

            for bot_ind in range(self.n_tank_bots):
                if self.tank_bots[bot_ind].is_alive:
                    route_change_tank = self.get_tank_bot_des_grid(bot_ind)

                    if np.linalg.norm(self.tank_bots[bot_ind].state-self.tank_bots[bot_ind].target_state) < 3.5 or route_change_tank:
                        target_pos = [self.np_random.randint(
                            low=0, high=self.x_lim - 1), self.np_random.randint(low=0, high=self.y_lim - 1), 0]
                        self.tank_bots[bot_ind].target_state = target_pos

            ###########################editedlinesfinished#########################
        for agent_ind in range(self.n_agents):
            for other_agents_ind in range(self.n_agents):
                if agent_ind != other_agents_ind:
                    collision_distance = np.linalg.norm(
                        self.quadrotors[agent_ind].state-self.quadrotors[other_agents_ind].state)

                    if (collision_distance <= 2) and self.quadrotors[agent_ind].is_alive and self.quadrotors[other_agents_ind].is_alive:
                        # done = True
                        self.quadrotors[agent_ind].is_alive = False
                        self.quadrotors[other_agents_ind].is_alive = False
                        self.quadrotors[agent_ind].state = np.array(
                            [-1, -1, -1])
                        self.quadrotors[other_agents_ind].state = np.array(
                            [-1, -1, -1])
                        reward_list[agent_ind] -= 300
                        reward_list[other_agents_ind] -= 300

            if not self.quadrotors[agent_ind].is_alive:
                reward_list[agent_ind] += 2

            if not done:
                for bot_ind in range(self.n_tank_bots):
                    drone_distance = np.linalg.norm(
                        self.quadrotors[agent_ind].state-self.tank_bots[bot_ind].state)
                    if drone_distance <= 3 and self.tank_bots[bot_ind].is_alive:
                        reward_list[agent_ind] += 100
                        self.tank_bots[bot_ind].is_alive = False
                        self.tank_bots[bot_ind].state = np.array([-1, -1, -1])
                        self.quadrotors[agent_ind].is_alive = False

            if not done:
                for bot_ind in range(self.n_bots):
                    drone_distance = np.linalg.norm(
                        self.quadrotors[agent_ind].state-self.bots[bot_ind].state)
                    if drone_distance <= 3 and self.bots[bot_ind].is_alive:
                        reward_list[agent_ind] += 100
                        self.bots[bot_ind].is_alive = False
                        self.bots[bot_ind].state = np.array([-1, -1, -1])
        # 3editedlines###################################################3
        for agent_ind in range(self.n_tank_agents):
            for other_agents_ind in range(self.n_tank_agents):
                if agent_ind != other_agents_ind:
                    collision_distance = np.linalg.norm(
                        self.tanks[agent_ind].state-self.tanks[other_agents_ind].state)

                    if (collision_distance <= 3) and self.tanks[agent_ind].is_alive and self.tanks[other_agents_ind].is_alive:
                        # done = True
                        self.tanks[agent_ind].is_alive = False
                        self.tanks[other_agents_ind].is_alive = False
                        self.tanks[agent_ind].state = np.array([-1, -1, -1])
                        self.tanks[other_agents_ind].state = np.array(
                            [-1, -1, -1])
                        tank_reward_list[agent_ind] -= 300
                        tank_reward_list[other_agents_ind] -= 300

            if not self.tanks[agent_ind].is_alive:
                tank_reward_list[agent_ind] += 2

            if not done:
                for bot_ind in range(self.n_tank_bots):
                    drone_distance = np.linalg.norm(
                        self.tanks[agent_ind].state-self.tank_bots[bot_ind].state)

                    if drone_distance <= 3 and self.tank_bots[bot_ind].is_alive:
                        tank_reward_list[agent_ind] += 100
                        self.tank_bots[bot_ind].is_alive = False
                        self.tank_bots[bot_ind].state = np.array([-1, -1, -1])
        #########################################finished#######################################################
#############################################BURDAKİOLAYISORRRRRRRRRRRRRRRRRRRRRRRRRRR####################################################################

       # if (len(self.bots) and not self.bots[0].is_alive and not self.bots[1].is_alive):
       #     done = True
            """
			for agent_ind in range(self.n_agents):
				reward_list[agent_ind] += 25
			"""
    ####################################################################################
        # if ( len(self.tank_bots) and not self.tank_bots[0].is_alive and not self.tank_bots[1].is_alive ):
        #    done = True
            """
			for agent_ind in range(self.n_agents):
				reward_list[agent_ind] += 25
			"""
    #################################################################################

        agent = any(map(lambda x: x.is_alive, self.quadrotors))
        tank_agent = any(map(lambda x: x.is_alive, self.tanks))

        bot_agent = any(map(lambda x: x.is_alive, self.bots))
        tank_bot_agent = any(map(lambda x: x.is_alive, self.tank_bots))

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

        for obstacle in self.obstacle_points:
            mox, moy, moz, ox, oy, oz = obstacle
            cubic_env[mox:ox][moy:oy][moz:oz] = 1

        if (not agent and bot_agent) or (not bot_agent and not tank_bot_agent) or (not agent and not tank_agent):
            done = True

        if done and bot_agent:
            reward_list -= 50 * sum(map(lambda x: x.is_alive, self.bots))

        if self.visualization:
            self.visualize()

        if self.current_step > 123:
            done = True
            reward_list[0] -= 100

        self.current_step += 1

        if done:
            self.state = True

        return (self.get_observation(),
                reward_list[0],
                done,
                {})

        # return cubic_env, self.get_observation(), reward_list/100, tank_reward_list/100, done, {},\
        #     [self.quadrotors[i].state for i in range(self.n_agents)], [self.bots[j].state for j in range(self.n_bots)] ,\
        #         [self.tanks[i].state for i in range(self.n_tank_agents)], [self.tank_bots[j].state for j in range(self.n_tank_bots)]

    def get_observation(self):
        # ??????????????????????????????????????????????????????????Statei bi kontrol et
        state = -1*np.array(np.ones((self.max_drone_agents, 3)))
        tank_state = -1*np.array(np.ones((self.max_tank_agents, 3)))
        bot_state = -1*np.array(np.ones((self.max_drone_bots, 3)))
        Tank_bot_state = -1*np.array(np.ones((self.max_tank_bots, 3)))

        for agent_ind in range(self.n_agents):
            if self.quadrotors[agent_ind].is_alive:
                state[agent_ind] = np.array(self.quadrotors[agent_ind].state)

        # 3

        for agent_ind in range(self.n_tank_agents):
            if self.tanks[agent_ind].is_alive:
                tank_state[agent_ind] = np.array(self.tanks[agent_ind].state)

        # 3

        for agent_ind in range(self.n_bots):
            if self.bots[agent_ind].is_alive:

                bot_state[agent_ind] = np.array(self.bots[agent_ind].state)
        # 3

        for agent_ind in range(self.n_tank_bots):
            if self.tank_bots[agent_ind].is_alive:

                Tank_bot_state[agent_ind] = np.array(
                    self.tank_bots[agent_ind].state)

        observationlist = np.concatenate(
            [state, tank_state, bot_state, Tank_bot_state])
        ##########################################################################################
        self.observation_space = observationlist
        return observationlist

    def generate_agent_position(self):
        self.quadrotors = []

        for i in range(0, self.n_agents):
            while True:
                current_pos = [self.np_random.randint(low=0, high=self.x_lim-1), self.np_random.randint(
                    low=0, high=self.y_lim-1), self.np_random.randint(low=0, high=self.z_lim-1)]
                state0 = [current_pos[0], current_pos[1], current_pos[2]]
                if not self.is_inside(state0):
                    break

            self.quadrotors.append(Drone(state0))
    ##############################    beginning  ###################################

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
    ############################### ending  #################################

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
            # editedlines##########################################################################3

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
            ######################################3finishedline############################################

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
        # linebegin###################################3

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

        ###############################linefinish##########################################
        return collision

    def reset(self):
        self.generate_agent_position()
        self.generate_bot_position()
        self.generate_tank_agent_position()
        self.generate_tank_bot_position()
        self.iteration = 1

        self.current_step = 0
        collision = self.check_collision()

        if collision:
            return self.reset()
        else:
            pass

        return np.zeros((40, 3))

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
                self.bots[bot_index].state[0] -= 2
                self.bots[bot_index].state[0] = np.clip(
                    self.bots[bot_index].state[0], 0,  self.x_lim-1)

        elif self.bots[bot_index].state[0] - self.bots[bot_index].target_state[0] < -2:
            action = 0
            if not self.is_collided(self.bots[bot_index].state, action):
                self.bots[bot_index].state[0] += 2

                self.bots[bot_index].state[0] = np.clip(
                    self.bots[bot_index].state[0], 0,  self.x_lim-1)

        elif self.bots[bot_index].state[1] - self.bots[bot_index].target_state[1] > 2:
            action = 3
            if not self.is_collided(self.bots[bot_index].state, action):
                self.bots[bot_index].state[1] -= 2

                self.bots[bot_index].state[1] = np.clip(
                    self.bots[bot_index].state[1], 0,  self.x_lim-1)

        elif self.bots[bot_index].state[1] - self.bots[bot_index].target_state[1] < -2:
            action = 2
            if not self.is_collided(self.bots[bot_index].state, action):
                self.bots[bot_index].state[1] += 2
                self.bots[bot_index].state[1] = np.clip(
                    self.bots[bot_index].state[1], 0,  self.x_lim-1)

        elif self.bots[bot_index].state[2] - self.bots[bot_index].target_state[2] > 2:
            action = 5
            if not self.is_collided(self.bots[bot_index].state, action):
                self.bots[bot_index].state[2] -= 2
                self.bots[bot_index].state[2] = np.clip(
                    self.bots[bot_index].state[2], 0,  self.x_lim-1)

        elif self.bots[bot_index].state[2] - self.bots[bot_index].target_state[2] < -2:
            action = 4
            if not self.is_collided(self.bots[bot_index].state, action):
                self.bots[bot_index].state[2] += 2
                self.bots[bot_index].state[2] = np.clip(
                    self.bots[bot_index].state[2], 0,  self.x_lim-1)

        if np.all(self.prev_state == self.bots[bot_index].state):
            change_route = True

        return change_route

#####################################editedlinesbelow#####################################################
    def get_tank_bot_des_grid(self, bot_index):
        self.prev_state = [self.tank_bots[bot_index].state[0],
                           self.tank_bots[bot_index].state[1], self.tank_bots[bot_index].state[2]]
        change_route = False

        if self.tank_bots[bot_index].state[0] - self.tank_bots[bot_index].target_state[0] > 2:
            action = 1
            if not self.is_collided(self.tank_bots[bot_index].state, action):

                self.tank_bots[bot_index].state[0] -= 1
                self.tank_bots[bot_index].state[0] = np.clip(
                    self.tank_bots[bot_index].state[0], 0,  self.x_lim-1)

        elif self.tank_bots[bot_index].state[0] - self.tank_bots[bot_index].target_state[0] < -2:
            action = 0
            if not self.is_collided(self.tank_bots[bot_index].state, action):
                self.tank_bots[bot_index].state[0] += 1
                self.tank_bots[bot_index].state[0] = np.clip(
                    self.tank_bots[bot_index].state[0], 0,  self.x_lim-1)

        elif self.tank_bots[bot_index].state[1] - self.tank_bots[bot_index].target_state[1] > 2:
            action = 3
            if not self.is_collided(self.tank_bots[bot_index].state, action):
                self.tank_bots[bot_index].state[1] -= 1
                self.tank_bots[bot_index].state[1] = np.clip(
                    self.tank_bots[bot_index].state[1], 0,  self.y_lim-1)

        elif self.tank_bots[bot_index].state[1] - self.tank_bots[bot_index].target_state[1] < -2:
            action = 2
            if not self.is_collided(self.tank_bots[bot_index].state, action):
                self.tank_bots[bot_index].state[1] += 1

                self.tank_bots[bot_index].state[1] = np.clip(
                    self.tank_bots[bot_index].state[1], 0,  self.y_lim-1)

        if np.all(self.prev_state == self.tank_bots[bot_index].state):
            change_route = True

        return change_route

        ##################editedlinesarefinshed##################################

    def get_drone_des_grid(self, drone_index, discrete_action):

        if discrete_action == 0:  # action=0, x += 1.0

            if not self.is_collided(self.quadrotors[drone_index].state, discrete_action):
                self.quadrotors[drone_index].state[0] += 2 * self.grid_res
                self.quadrotors[drone_index].state[0] = np.clip(
                    self.quadrotors[drone_index].state[0], 0,  self.x_lim-1)

        elif discrete_action == 1:  # action=1, x -= 1.0
            if not self.is_collided(self.quadrotors[drone_index].state, discrete_action):
                self.quadrotors[drone_index].state[0] -= 2 * self.grid_res
                self.quadrotors[drone_index].state[0] = np.clip(
                    self.quadrotors[drone_index].state[0], 0,  self.x_lim-1)

        elif discrete_action == 2:  # action=2, y += 1.0
            if not self.is_collided(self.quadrotors[drone_index].state, discrete_action):
                self.quadrotors[drone_index].state[1] += 2 * self.grid_res
                self.quadrotors[drone_index].state[1] = np.clip(
                    self.quadrotors[drone_index].state[1], 0,  self.y_lim-1)

        elif discrete_action == 3:  # action=3, y -= 1.0

            if not self.is_collided(self.quadrotors[drone_index].state, discrete_action):
                self.quadrotors[drone_index].state[1] -= 2 * self.grid_res
                self.quadrotors[drone_index].state[1] = np.clip(
                    self.quadrotors[drone_index].state[1], 0,  self.y_lim-1)

        elif discrete_action == 4:  # action=4, z += 1.0
            if not self.is_collided(self.quadrotors[drone_index].state, discrete_action):

                self.quadrotors[drone_index].state[2] += 2 * self.grid_res
                self.quadrotors[drone_index].state[2] = np.clip(
                    self.quadrotors[drone_index].state[2], 0,  self.z_lim-1)

        elif discrete_action == 5:  # action=5, z -= 1.0
            if not self.is_collided(self.quadrotors[drone_index].state, discrete_action):

                self.quadrotors[drone_index].state[2] -= 2 * self.grid_res
                self.quadrotors[drone_index].state[2] = np.clip(
                    self.quadrotors[drone_index].state[2], 0,  self.z_lim-1)

        else:
            print("Invalid discrete action!")

        drone_current_state = np.copy(self.quadrotors[drone_index].state)
        return drone_current_state
        #############################################editedlines#################################

    def get_tank_des_grid(self, drone_index, discrete_action):

        if discrete_action == 0:  # action=0, x += 1.0

            if not self.is_collided(self.tanks[drone_index].state, discrete_action):

                self.tanks[drone_index].state[0] += self.grid_res
                self.tanks[drone_index].state[0] = np.clip(
                    self.tanks[drone_index].state[0], 0,  self.x_lim-1)

        elif discrete_action == 1:  # action=1, x -= 1.
            if not self.is_collided(self.tanks[drone_index].state, discrete_action):

                self.tanks[drone_index].state[0] -= self.grid_res
                self.tanks[drone_index].state[0] = np.clip(
                    self.tanks[drone_index].state[0], 0,  self.x_lim-1)

        elif discrete_action == 2:  # action=2, y += 1.0
            if not self.is_collided(self.tanks[drone_index].state, discrete_action):
                self.tanks[drone_index].state[1] += self.grid_res
                self.tanks[drone_index].state[1] = np.clip(
                    self.tanks[drone_index].state[1], 0,  self.y_lim-1)

        elif discrete_action == 3:  # action=3, y -= 1.0
            if not self.is_collided(self.tanks[drone_index].state, discrete_action):

                self.tanks[drone_index].state[1] -= self.grid_res
                self.tanks[drone_index].state[1] = np.clip(
                    self.tanks[drone_index].state[1], 0,  self.y_lim-1)

        else:
            print("Invalid discrete action!")

        tank_current_state = np.copy(self.tanks[drone_index].state)

        return tank_current_state
    #############################finishedlines###############################################################################

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

    def visualize(self, agent_pos_dict=None, mode='human'):
        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(0,
                                   self.x_lim, 0, self.y_lim)
            fname = path.join(path.dirname(__file__), "assets/dr.png")
            fname2 = path.join(path.dirname(__file__), "assets/plane2.png")
            fname3 = path.join(path.dirname(__file__), "assets/tank1.png")
            fname4 = path.join(path.dirname(__file__), "assets/tank2.png")

            self.drone_transforms = []
            self.drones = []

            self.prey_transforms = []
            self.preys = []

            self.tank_transforms = []
            self.tank = []

            self.tank_prey_transforms = []
            self.tank_preys = []

            for i in range(len(self.obstacle_points)):
                obstacle = rendering.make_polygon([(self.obstacle_points[i][0], self.obstacle_points[i][1]),
                                                   (self.obstacle_points[i][0],
                                                  self.obstacle_points[i][4]),
                                                   (self.obstacle_points[i][3],
                                                    self.obstacle_points[i][4]),
                                                   (self.obstacle_points[i][3], self.obstacle_points[i][1])])

                obstacle_transform = rendering.Transform()
                obstacle.add_attr(obstacle_transform)
                obstacle.set_color(.8, .3, .3)
                self.viewer.add_geom(obstacle)

            for i in range(self.n_agents):
                self.drone_transforms.append(rendering.Transform())
                self.drones.append(rendering.Image(fname, 2., 2.))
                self.drones[i].add_attr(self.drone_transforms[i])

            for i in range(self.n_bots):
                self.prey_transforms.append(rendering.Transform())
                self.preys.append(rendering.Image(fname2, 2., 2.))
                self.preys[i].add_attr(self.prey_transforms[i])

            for i in range(self.n_tank_agents):
                self.tank_transforms.append(rendering.Transform())
                self.tank.append(rendering.Image(fname3, 2., 2.))
                self.tank[i].add_attr(self.tank_transforms[i])

            for i in range(self.n_tank_bots):
                self.tank_prey_transforms.append(rendering.Transform())
                self.tank_preys.append(rendering.Image(fname4, 2., 2.))
                self.tank_preys[i].add_attr(self.tank_prey_transforms[i])

        for i in range(self.n_bots):
            if self.bots[i].is_alive:
                self.viewer.add_onetime(self.preys[i])
                self.prey_transforms[i].set_translation(
                    self.bots[i].state[0], self.bots[i].state[1])
                self.prey_transforms[i].set_rotation(self.bots[i].psid)

        for i in range(self.n_agents):
            if self.quadrotors[i].is_alive:
                self.viewer.add_onetime(self.drones[i])
                self.drone_transforms[i].set_translation(
                    self.quadrotors[i].state[0], self.quadrotors[i].state[1])
                self.drone_transforms[i].set_rotation(self.quadrotors[i].psi)

        for i in range(self.n_tank_bots):
            if self.tank_bots[i].is_alive:
                self.viewer.add_onetime(self.tank_preys[i])
                self.tank_prey_transforms[i].set_translation(
                    self.tank_bots[i].state[0], self.tank_bots[i].state[1])
                self.tank_prey_transforms[i].set_rotation(
                    self.tank_bots[i].psid)

        for i in range(self.n_tank_agents):
            if self.tanks[i].is_alive:
                self.viewer.add_onetime(self.tank[i])
                self.tank_transforms[i].set_translation(
                    self.tanks[i].state[0], self.tanks[i].state[1])
                self.tank_transforms[i].set_rotation(self.tanks[i].psi)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


###############################################################################################


    def get_obstacle_indices(self):
        obstacle_indices = []
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
            # current_ind = self.get_closest_grid(current_pos)
            # obstacle_indices.append(current_ind)

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

        return False

    def is_inside(self, state):
        x = state[0]
        y = state[1]
        z = state[2]

        for obstacle in self.obstacle_points:
            mox, moy, moz, ox, oy, oz = obstacle
            if (mox <= x <= ox) and (moy <= y <= oy) and (moz <= z <= oz):

                return True
        return False

    def render(self, mode):
        self.visualize()
