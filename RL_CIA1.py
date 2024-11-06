import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


class Bandit:
    def __init__(self, payoff_probs):
        self.actions = range(len(payoff_probs))
        self.pay_offs = payoff_probs

    def sample(self, action):
        return 1 if random.random() <= self.pay_offs[action] else 0


def random_agent(bandit, iterations):
    for i in range(iterations):
        action = random.choice(bandit.actions)
        reward = bandit.sample(action)
        yield action, reward


def optimal_agent(bandit, iterations):
    for i in range(iterations):
        action = bandit.pay_offs.index(max(bandit.pay_offs))
        reward = bandit.sample(action)
        yield action, reward


def initial_explore_agent(bandit, iterations, initial_rounds=10):
    pay_offs = {}
    best_action = -1

    for i in range(iterations):
        if i < initial_rounds:
            action = random.choice(bandit.actions)
        else:
            if best_action == -1:
                means = {action: np.mean(rewards) for action, rewards in pay_offs.items()}
                best_action = max(means, key=means.get)
            action = best_action
        
        reward = bandit.sample(action)
        if action not in pay_offs:
            pay_offs[action] = []
        pay_offs[action].append(reward)

        yield action, reward


def epsilon_greedy_agent(bandit, iterations, epsilon=0.2, initial_rounds=1):
    pay_offs = {}

    for i in range(iterations):
        if random.random() < epsilon or i < initial_rounds:
            action = random.choice(bandit.actions)
        else:
            means = {action: np.mean(rewards) for action, rewards in pay_offs.items()}
            action = max(means, key=means.get)
        
        reward = bandit.sample(action)
        if action not in pay_offs:
            pay_offs[action] = []
        pay_offs[action].append(reward)

        yield action, reward


def decaying_epsilon_greedy_agent(bandit, iterations, epsilon=0.2, initial_rounds=1, decay=0.999):
    pay_offs = {}

    for i in range(iterations):
        if random.random() < epsilon or i < initial_rounds:
            action = random.choice(bandit.actions)
        else:
            means = {action: np.mean(rewards) for action, rewards in pay_offs.items()}
            action = max(means, key=means.get)
        
        reward = bandit.sample(action)
        if action not in pay_offs:
            pay_offs[action] = []
        pay_offs[action].append(reward)

        epsilon *= decay

        yield action, reward


def plot_results(methods, bandit, number_of_iterations=200, number_of_trials=1000):
    plt.figure()

    all_rewards = []
    for method in methods:
        total_rewards = []
        cumulative_rewards = []

        for trial in range(number_of_trials):
            total_reward = 0
            cumulative_reward = []

            for action, reward in method(bandit, number_of_iterations):
                total_reward += reward
                cumulative_reward.append(total_reward)

            total_rewards.append(total_reward)
            cumulative_rewards.append(cumulative_reward)

        df_rewards = pd.DataFrame({
            'x': np.arange(1, number_of_iterations + 1),
            'y': np.concatenate(cumulative_rewards)
        })

        sns.lineplot(x='x', y='y', data=df_rewards, label=method.__name__)

        print(f"{method.__name__}: Mean reward over trials = {np.mean(total_rewards)}")

    plt.title(f"Cumulative reward for each algorithm over {number_of_iterations} iterations and {number_of_trials} trials.")
    plt.xlabel("Iterations")
    plt.ylabel("Cumulative reward")
    plt.legend()

    plt.savefig("Iterations.pdf", bbox_inches='tight')
    plt.savefig("Iterations.svg", bbox_inches='tight')
    plt.show()


pay_offs = [0.25, 0.3, 0.5, 0.1, 0.3, 0.25, 0]
bandit = Bandit(pay_offs)

methods = [random_agent, initial_explore_agent, epsilon_greedy_agent, decaying_epsilon_greedy_agent, optimal_agent]

plot_results(methods, bandit)
