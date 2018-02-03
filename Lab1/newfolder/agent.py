import numpy as np

class RandomAgent:
    def __init__(self):
        self.attribut = 0
    def act(self, observation):
        return np.random.randint(0,9)
    def reward(self, observation, action, reward):
        pass
Agent = RandomAgent