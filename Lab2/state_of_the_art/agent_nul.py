import numpy as np

class QLearnerAgent:
    def __init__(self):
        """Init a new agent.
        """
        self.time = 0 # 0 is not doing anything
        self.discount_factor = 10
        self.max_x = 0
        self.max_y = 0
        self.previous_position = (0,0)
        self.previous_action = 0
        self.game = 0
        self.number_of_visits = {}
        
        self.states = {}
        
        
    def reset(self):
        """Reset the internal state of the agent, for a new run.

        You need to reset the internal state of your agent here, as it
        will be started in a new instance of the environment.

        You must **not** reset the learned parameters.
        
        """
        self.game +=1
        self.time = 0
        pass

    def act(self, observation):
        """Acts given an observation of the environment.

        Takes as argument an observation of the current state, and
        returns the chosen action.

        Here, for the Wumpus world, observation = ((x,y), smell, breeze, charges)
        """
        
        ((x,y), smell, breeze, charges) = observation
        charges = int(charges>0)
        smell = int(smell)
        breeze = int(breeze)
        
        # adding the new position to the known states
        if (x,y) not in self.states.keys():
            self.states[(x,y)] = -np.ones((2,2,2, 8))
#            self.number_of_visits[(x,y)] =  np.zeros((2,2,2, 8))
        
        # updating next states
        if x > self.max_x:
            self.max_x = x
        if y > self.max_y:
            self.max_y = y
            
        self.time +=1
        
        possible_actions = [i for i in range(4)]
#        if x == 0:
#            possible_actions.remove(2)
#        if y==0:
#            possible_actions.remove(0)
    
            
        if self.game <1:
            # during the first game, we are going to explore randomly
            action = np.random.choice(a = possible_actions)
        else :
            
            expected_rewards = self.states[(x,y)][smell, breeze, charges, possible_actions]
            next_expected_rewards = []
            for action in possible_actions:
                ((next_x, next_y), next_charge) = self.compute_next_state(observation, action+1)
                if (next_x, next_y) in self.states.keys():
                    next_expected_rewards.append(np.mean(self.states[(next_x,next_y)][:,:,next_charge]))
                else:
                    next_expected_rewards.append(-1)
            expected_rewards = expected_rewards + self.discount_factor* np.array(next_expected_rewards)
            action = np.argmax(expected_rewards)+1
            
#        self.number_of_visits[(x,y)][smell, breeze, charges, action-1] +=1
            
        return action
    
    def compute_next_state(self, observation, action):
        ((x,y), smell, breeze, charges) = observation
        
#        last_charge = int(charges <= 1)
        charges = int(charges>0)
        smell = int(smell)
        breeze = int(breeze)
        
        next_x = x
        next_y = y
        next_charge = charges
        
#        if action > 4:
#            next_charge = int(last_charge != 1)
#        else:
        if (action == 1) :
            next_y = min([self.max_y, y+1])
        elif action == 2:
            next_y = max([0, y-1])
        elif action == 3:
            next_x = min([0, x-1])
        else :
            next_x = min([self.max_x, x+1])
        return ((next_x, next_y), next_charge)
                
                
        

    def reward(self, observation, action, reward):
        """Receive a reward for performing given action on
        given observation.

        This is where your agent can learn.
        """
        ((x,y), smell, breeze, charges) = observation
        ((next_x, next_y), next_charges) = self.compute_next_state(observation, action)
        
        charges = int(charges>0)
        smell = int(smell)
        breeze = int(breeze)
                
                
        self.states[(x,y)][smell, breeze, charges, action - 1] +=reward
        
#        
#        state_reward = self.states[(x,y)][smell, breeze, charges, action - 1] 
#        state_visits = self.number_of_visits[(x,y)][smell, breeze, charges, action - 1] 
#        
#        new_reward = (reward + (state_visits-1)*state_reward)/state_visits
#        
#        self.states[(x,y)][smell, breeze, charges, action - 1]  = new_reward
        
        
        pass
    
Agent = QLearnerAgent