import numpy as np
import random
import matplotlib.pyplot as plt

GRID_DIM = 50
OBSTACLE_CHANCE = 0.3
GOAL_REWARD = 150
MOVE_REWARD = -2
OBSTACLE_PENALTY = -5
DISCOUNT = 0.95
CONVERGENCE_CRITERION = 0.005
MOVEMENTS = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1)]

class GridWorld:
    def __init__(self, grid_size=GRID_DIM, obstacle_prob=OBSTACLE_CHANCE):
        self.grid_size = grid_size
        self.obstacle_prob = obstacle_prob
        self.grid = np.zeros((grid_size, grid_size))
        self.start = (0, 0)
        self.goal = (grid_size - 1, grid_size - 1)
        self.init_grid()

    def init_grid(self):
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if random.random() < self.obstacle_prob and (i, j) not in [self.start, self.goal]:
                    self.grid[i, j] = -1
        self.grid[self.start] = 0
        self.grid[self.goal] = 0

    def is_valid(self, state):
        x, y = state
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size and self.grid[x, y] != -1:
            return True
        return False

    def get_next_state(self, state, action):
        x, y = state
        dx, dy = action
        next_state = (x + dx, y + dy)
        if self.is_valid(next_state):
            return next_state
        return state

    def get_reward(self, state):
        if state == self.goal:
            return GOAL_REWARD
        elif self.grid[state] == -1:
            return OBSTACLE_PENALTY
        return MOVE_REWARD

class ValueIterationAgent:
    def __init__(self, grid_world, discount_factor=DISCOUNT, threshold=CONVERGENCE_CRITERION):
        self.grid_world = grid_world
        self.discount_factor = discount_factor
        self.threshold = threshold
        self.value_table = np.zeros((grid_world.grid_size, grid_world.grid_size))

    def value_iteration(self):
        while True:
            delta = 0
            new_value_table = np.copy(self.value_table)

            for x in range(self.grid_world.grid_size):
                for y in range(self.grid_world.grid_size):
                    state = (x, y)
                    if state == self.grid_world.goal or self.grid_world.grid[state] == -1:
                        continue
                    
                    value_max = float('-inf')
                    for action in MOVEMENTS:
                        next_state = self.grid_world.get_next_state(state, action)
                        reward = self.grid_world.get_reward(next_state)
                        value = reward + self.discount_factor * self.value_table[next_state]
                        value_max = max(value_max, value)

                    new_value_table[x, y] = value_max
                    delta = max(delta, abs(self.value_table[x, y] - new_value_table[x, y]))

            self.value_table = new_value_table
            if delta < self.threshold:
                break

    def get_policy(self):
        policy = np.zeros((self.grid_world.grid_size, self.grid_world.grid_size), dtype=(int, 2))
        for x in range(self.grid_world.grid_size):
            for y in range(self.grid_world.grid_size):
                state = (x, y)
                if state == self.grid_world.goal or self.grid_world.grid[state] == -1:
                    continue

                best_action = None
                best_value = float('-inf')

                for action in MOVEMENTS:
                    next_state = self.grid_world.get_next_state(state, action)
                    reward = self.grid_world.get_reward(next_state)
                    value = reward + self.discount_factor * self.value_table[next_state]

                    if value > best_value:
                        best_value = value
                        best_action = action

                policy[x, y] = best_action

        return policy

def visualize_grid(grid_world, policy=None):
    grid = grid_world.grid
    fig, ax = plt.subplots()
    ax.imshow(grid, cmap='gray')
    start_x, start_y = grid_world.start
    goal_x, goal_y = grid_world.goal
    ax.text(start_y, start_x, 'S', color='green', ha='center', va='center', fontsize=14, fontweight='bold')
    ax.text(goal_y, goal_x, 'G', color='blue', ha='center', va='center', fontsize=14, fontweight='bold')
    if policy is not None:
        for x in range(grid_world.grid_size):
            for y in range(grid_world.grid_size):
                if grid_world.grid[x, y] == -1 or (x, y) == grid_world.goal:
                    continue
                dx, dy = policy[x, y]
                ax.arrow(y, x, dy * 0.3, dx * -0.3, head_width=0.3, head_length=0.3, fc='red', ec='red')

    plt.show()

grid_world = GridWorld()
agent = ValueIterationAgent(grid_world)

agent.value_iteration()

optimal_policy = agent.get_policy()
visualize_grid(grid_world, optimal_policy)
