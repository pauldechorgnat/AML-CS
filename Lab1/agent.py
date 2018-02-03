import numpy as np

"""
Contains the definition of the agent that will run in an
environment.
"""

class RandomAgent:
    def __init__(self):
        """Init a new agent.
        """

    def act(self, observation):
        """Acts given an observation of the environment.

        Takes as argument an observation of the current state, and
        returns the chosen action.
        """
        return np.random.randint(0,9)

    def reward(self, observation, action, reward):
        """Receive a reward for performing given action on
        given observation.

        This is where your agent can learn.
        """
        pass
      
class EpsilonGreedyAgent:
    def __init__(self, epsilon = .1):
        self.number_of_activation = np.zeros(10)
        self.avg_reward = np.zeros(10)# np.random.uniform(size = 10)
        self.epsilon = epsilon
    
    def act(self, observation):
        # here the observation is only a random variable depending on epsilon
        if np.random.uniform()<self.epsilon:
            # with probability epsilon we pick a random arm
            arm_to_pick = np.random.randint(0,9)
        else:
            # with probability 1-epsilon we pick (one of the best arm)
            arm_to_pick = np.argmax(self.avg_reward)
        
        # updating the number of activation of this arm
        self.number_of_activation[arm_to_pick] +=1
        return arm_to_pick
    
    def reward(self, observation, action, reward):
        # updating the score
        self.avg_reward[action] = (reward+ (self.number_of_activation[action]-1)*self.avg_reward[action])/self.number_of_activation[action]
        pass
    
class OptimisticEpsilonGreedyAgent:
    # exacttly the same thing but this one expect the good one to be decided very quick
    def __init__(self, epsilon = .9):
        self.number_of_activation = np.zeros(10)
        self.avg_reward = np.zeros(10)#np.random.uniform(size = 10)
        self.epsilon = epsilon
    
    def act(self, observation):
        # here the observation is only a random variable depending on epsilon
        if np.random.uniform()<self.epsilon:
            # with probability epsilon we pick a random arm
            arm_to_pick = np.random.randint(0,9)
        else:
            # with probability 1-epsilon we pick (one of the best arm)
            arm_to_pick = np.argmax(self.avg_reward)
        
        # updating the number of activation of this arm
        self.number_of_activation[arm_to_pick] +=1
        return arm_to_pick
    
    def reward(self, observation, action, reward):
        # updating the score
        self.avg_reward[action] = (reward+ (self.number_of_activation[action]-1)*self.avg_reward[action])/self.number_of_activation[action]
        pass
    
class SoftMaxAgent:
    def __init__(self, decay_factor = .1):
        self.number_of_activation = np.zeros(10)
        self.avg_reward = np.zeros(10)#np.random.uniform(size = 10)
        self.decay_factor = decay_factor
    
    def act(self, observation):
        # computing softmax probabilities
        softmax_probabilities = np.exp(self.avg_reward/self.decay_factor)
        softmax_probabilities = softmax_probabilities/sum(softmax_probabilities)

        # defining the arm to pick
        arm_to_pick = int(np.random.choice(a = np.arange(10), size = 1, p = softmax_probabilities))
        
        # updating the number of activation of this arm
        self.number_of_activation[arm_to_pick] +=1
        return arm_to_pick
    
    def reward(self, observation, action, reward):
        # updating the avg reward
        self.avg_reward[action] = (reward+ (self.number_of_activation[action]-1)*self.avg_reward[action])/self.number_of_activation[action]
        pass
    
class UCBAgent:
    def __init__(self):
        self.number_of_activation = np.ones(10)#initializing at 1 to avoid dividing by 0
        self.avg_reward = np.zeros(10)#np.random.uniform(size = 10)
        self.time = 0
    
    def act(self, observation):
        # computing Bts (k)
        self.time += 1
        bts = self.avg_reward + np.sqrt(2 * np.log(self.time)/self.number_of_activation)

        # defining the arm to pick
        arm_to_pick = np.argmax(bts)
        
        # updating the number of activation of this arm
        self.number_of_activation[arm_to_pick] += 1
        return arm_to_pick
    
    def reward(self, observation, action, reward):
        # updating the avg reward
        self.avg_reward[action] = (reward+ (self.number_of_activation[action]-1)*self.avg_reward[action])/self.number_of_activation[action]
        pass
    
class ScaledUCBAgent:
    def __init__(self):
        self.number_of_activation = np.zeros(10)
        self.avg_reward = np.zeros(10)#np.random.uniform(size = 10)
        self.time = 0
        self.max_reward = 0
        self.min_reward = 0
    
    def act(self, observation):
        # pulling one time each arm
        if self.time <len(self.avg_reward):
            action = self.time %len(self.avg_reward)
            self.time +=1
            self.number_of_activation[action] += 1
            return action
        else :
            self.min_reward = np.min(self.avg_reward)
            self.max_reward = np.max(self.avg_reward)
            
            delta_reward = self.max_reward-self.min_reward
            # computing Bts (k)
            self.time += 1
            bts = self.avg_reward + np.sqrt(delta_reward/2 * np.log(self.time)/self.number_of_activation)#/np.log(self.time)

            # defining the arm to pick
            action = np.argmax(bts)
        
            # updating the number of activation of this arm
            self.number_of_activation[action] += 1
            return action
    
    def reward(self, observation, action, reward):
        # updating the avg reward
        if self.time < len(self.avg_reward):
            self.avg_reward[action] = reward
        else :
            if reward > self.max_reward:
                self.max_reward = reward
            if reward < self.min_reward:
                self.min_reward = reward
                
            self.avg_reward[action] = (reward+ (self.number_of_activation[action]-1)*self.avg_reward[action])/self.number_of_activation[action]
        pass
# Choose which Agent is run for scoring
#Agent = RandomAgent
#Agent = EpsilonGreedyAgent
#Agent = OptimisticEpsilonGreedyAgent
#Agent = SoftMaxAgent
#Agent = UCBAgent
Agent = ScaledUCBAgent