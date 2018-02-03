import numpy as np

class NewAgent:
    def __init__(self):
        """Init a new agent.
        """
        self.step = 0
        self.game = 0
        
        self.p = 5
        self.k = 5
        
        self.states_coordinates = None
        
        self.previous_action = None
        self.previous_position = None
        self.previous_reward = None
        
        self.learning_games = 180
        
        self.GAMMA = .2
        self.ALPHA = .1
        self.LAMBDA = .1
        
        self.greediness = .1
        
        self.e = None
        self.W = None
        
        self.x_min = None
        self.x_max = None
        
        
        self.victories = 0

    def reset(self, x_range):
        """Reset the state of the agent for the start of new game.

        Parameters of the environment do not change when starting a new 
        episode of the same game, but your initial location is randomized.

        x_range = [xmin, xmax] contains the range of possible values for x

        range for vx is always [-20, 20]
        """
        [xmin, xmax] = x_range
        
        self.x_max = xmax
        self.x_min = xmin
        
        self.game += 1
        self.step = 0
                
        # if game == 1 we need to create a grid
        if self.game ==1:
            self.states_coordinates = np.array([[[-xmin + i*(xmax-xmin)/self.p , -20 + 40*j/self.k] for i in range(self.p+1)] for j in range(self.k+1)])
            self.W = np.zeros((self.p+1, self.k+1,3))
        self.e = np.zeros((self.p+1, self.k+1,3))
        pass

    def act(self, observation):
        """Acts given an observation of the environment.

        Takes as argument an observation of the current state, and
        returns the chosen action.

        observation = (x, vx)
        """
        (x, vx) = observation
        
        phi_current = self.compute_phi(x, vx)
        Qs = [np.sum(phi_current*self.W[:,:,i]) for i in range(3)] #for i in range(3)]                
        
        best_action = np.argmax(Qs)  
        #print(best_action)
        #print(self.W[:,:,0])
        if (self.game < self.learning_games) and (np.random.rand() < self.greediness):
            action = np.random.randint(low = -1, high = 2, dtype = 'int')
        else:
            action = best_action -1
        return action
        
    def compute_phi(self, x, vx):
        phi_ij = np.exp(-np.power((self.states_coordinates[:,:,0]-x)/(self.x_max-self.x_min), 2)) *\
        np.exp(-np.power((self.states_coordinates[:,:,1]-vx)*self.k/40.,2))
        #max_phi = np.sum(phi_ij)
        
        #phi_ij = phi_ij / max_phi
        #print(phi_ij)
        return phi_ij /np.max(phi_ij)
        

    def reward(self, observation, action, reward):
        """Receive a reward for performing given action on
        given observation.

        This is where your agent can learn.
        """
        (x, vx) = observation
        phi_current = self.compute_phi(x, vx)
        
        self.e[:,:,action + 1] +=  phi_current
        
        if reward > 0:
            self.victories += 1
        
        if self.step >0:
            
            (x_prev, vx_prev) = self.previous_position
            phi_previous = self.compute_phi(x_prev, vx_prev)
            
            Q_current = np.sum(self.W[:,:,action+1]*phi_current)
            Q_previous = np.sum(self.W[:,:,self.previous_action+1]*phi_previous)
            
            delta = reward + self.GAMMA * Q_current - Q_previous
            
            self.W += delta * self.e * self.ALPHA
            self.e *= self.LAMBDA * self.GAMMA * self.e
            
        self.step += 1
        self.previous_action = action
        self.previous_position = observation
        self.previous_reward = reward
        
        pass
Agent = NewAgent