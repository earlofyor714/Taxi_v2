import numpy as np
from collections import defaultdict


class Agent:
    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))

        # Variables I've added
        self.alpha = 0.02
        self.gamma = 1.0  # Chosen by "popular choice"
        self.i_episode = 0

    def update_policy(self, Q_s, state, epsilon):
        policy = np.ones(self.nA) * epsilon / self.nA
        policy[np.argmax(Q_s)] = 1 - epsilon + epsilon / self.nA
        return policy

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """

        # Using an epsilon-greedy policy
        epsilon = 1 / ((self.i_episode / 0.7) + 1)
        # epsilon = (20000 - self.i_episode) / 20000
        policy = self.update_policy(self.Q[state], state, epsilon)
        return np.random.choice(np.arange(self.nA), p=policy)

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        old_Q = self.Q[state][action]
        expected_policy = self.update_policy(self.Q[next_state], next_state, 0.005)
        Sarsa_value = np.dot(self.Q[next_state], expected_policy)
        self.Q[state][action] += self.alpha * (reward + self.gamma * Sarsa_value - old_Q)
        if done:
            self.i_episode += 1