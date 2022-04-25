from re import I
import time
import random

from predator_prey import QuadrotorFormation

env = QuadrotorFormation(n_agents=2, n_tank_agents=2, n_bots=2, n_tank_bots=2)
env.reset()

i = 0
while True:
    env.visualize()
    step_res = env.step([random.randint(0, 3) for _ in range(4)])

    #print("<DRONE STATES>\n", step_res)

    i += 1
