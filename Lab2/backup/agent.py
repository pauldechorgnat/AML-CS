import numpy as np

"""
Contains the definition of the agent that will run in an
environment.
"""

ACT_UP    = 1
ACT_DOWN  = 2
ACT_LEFT  = 3
ACT_RIGHT = 4
ACT_TORCH_UP    = 5
ACT_TORCH_DOWN  = 6
ACT_TORCH_LEFT  = 7
ACT_TORCH_RIGHT = 8


class RandomAgent:
    def __init__(self):
        """Init a new agent.
        """

    def reset(self):
        """Reset the internal state of the agent, for a new run.

        You need to reset the internal state of your agent here, as it
        will be started in a new instance of the environment.

        You must **not** reset the learned parameters.
        """
        pass

    def act(self, observation):
        """Acts given an observation of the environment.

        Takes as argument an observation of the current state, and
        returns the chosen action.

        Here, for the Wumpus world, observation = ((x,y), smell, breeze, charges)
        """
        return np.random.randint(1,9)

    def reward(self, observation, action, reward):
        """Receive a reward for performing given action on
        given observation.

        This is where your agent can learn.
        """
        pass

class NonPositionalAgent:
    def __init__(self):
        """Init a new agent.
        """
        self.time = 0 # 0 is not doing anything
        self.transition_matrix = np.ones((2, 2, 2, 8))
        self.transition_matrix /= self.transition_matrix#.sum(axis = 3)

    def reset(self):
        """Reset the internal state of the agent, for a new run.

        You need to reset the internal state of your agent here, as it
        will be started in a new instance of the environment.

        You must **not** reset the learned parameters.
        
        """
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
        
        probabilities = self.transition_matrix[smell, breeze, charges, :]
        
        if self.time <100:
            self.time +=1
            return np.random.randint(low = 1, high = 9)
        else : 
            self.time +=1
            return np.argmax(probabilities)+1
        
        

    def reward(self, observation, action, reward):
        """Receive a reward for performing given action on
        given observation.

        This is where your agent can learn.
        """
        ((x,y), smell, breeze, charges) = observation
        charges = int(charges>0)
        smell = int(smell)
        breeze = int(breeze)
        
        action_index = action -1
        
        self.transition_matrix[smell, breeze, charges, action_index] += reward
        
        
        
        
        pass
    
class QLearnerAgent:
    def __init__(self):
        """Init a new agent.
        """
        self.time = 0 # 0 is not doing anything
        self.transition_matrix = np.ones((2, 2, 2, 8))
        self.transition_matrix /= self.transition_matrix#.sum(axis = 3)
        self.discount_factor = .8
        self.max_x = 0
        self.max_y = 0
        self.reached_x_boundary = False
        self.reached_y_boundary = False
        self.previous_position = (0,0)
        self.previous_action = 0
        self.game = 0
        self.serious_mode = False
        
        
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
        
        
        probabilities = self.transition_matrix[smell, breeze, charges, :]
        
        # updating our knowledge of the size of the map
        if not self.reached_x_boundary:
            if self.max_x <x:
                self.max_x = x
        if not self.reached_y_boundary:
            if self.max_y <y :
                self.max_y = y
        
        if self.time >1 :
            if self.reached_x_boundary == False:
                if (x == self.previous_position[0]) and (self.previous_action == 4):
                    self.reached_x_boundary = True
                    self.max_x == x
            if self.reached_y_boundary == False:
                if (y == self.previous_position[1]) and (self.previous_action == 4):
                    self.reached_y_boundary = True
                    self.max_y == y
        
        possible_actions = [i for i in range(1, 9)]    
        
        # handling boundaries
#        if (self.reached_x_boundary == True) and (self.max_x == x): # no use of going further 
#            possible_actions.remove(4)
#            possible_actions.remove(8)
#        elif x == 0:
#            possible_actions.remove(3)
#            possible_actions.remove(7)
#                
#        
#        if (self.reached_y_boundary == True) and (self.max_y == y):
#            possible_actions.remove(1)
#            possible_actions.remove(5)
#        elif y == 0:
#            possible_actions.remove(2)
#            possible_actions.remove(6)
#            
        # handling the absence of smell 
        if smell == 0:
            for i in range(5, 9):
                try :
                    possible_actions.remove(i)
                except ValueError:
                    continue
        
        self.time +=1
        self.previous_position = (x, y)
        
        
        
        if self.game <=5:
            action = np.random.choice(a = possible_actions)
            self.previous_action = action
            return action
        else : 
            possible_probabilities = np.exp(np.array([probabilities[i - 1] for i in possible_actions])/100.)
##            possible_probabilities -= possible_probabilities.max()
#            possible_probabilities = np.exp(possible_probabilities/10.)
            possible_probabilities/=possible_probabilities.sum()            
            action = np.random.choice(a = possible_actions, p = possible_probabilities)
            self.previous_action = action
            return action
    
    def compute_next_state(self, observation, action):
        ((x,y), smell, breeze, charges) = observation
        
        last_charge = int(charges <= 1)
        charges = int(charges>0)
        smell = int(smell)
        breeze = int(breeze)
        
        next_x = x
        next_y = y
        next_charge = charges
        
        if action > 4:
            if last_charge == 1:
                next_charge = 0
            else :
                next_charge = 1
        else:
            if (action == 1) :
                if (self.max_y == y) and (self.reached_y_boundary):
                    next_y == y
                else :
                    next_y = y-1
            elif action == 2:
                if y == 0:
                    next_y = 0
                else : 
                    next_y = y-1
            elif action == 3:
                if x == 0:
                    next_x = 0
                else :
                    next_x = x - 1
            elif (action == 4) :
                if (self.max_x == x) and self.reached_x_boundary:
                    next_x = x
                else : 
                    next_x = x + 1
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
        
        action_index = action - 1
        
        next_state_reward = 0
                
        
        self.transition_matrix[smell, breeze, charges, action_index] += self.discount_factor*reward
        
        
        
        
        pass
    
class SeriousQLearnerAgent:
    def __init__(self):
        """Init a new agent.
        """
        self.time = 0 # 0 is not doing anything
        self.transition_matrix = np.ones((2, 2, 8))
        self.transition_matrix /= self.transition_matrix#.sum(axis = 3)
        self.discount_factor = .8
        self.max_x = 0
        self.max_y = 0
        self.reached_x_boundary = False
        self.reached_y_boundary = False
        self.previous_position = (0,0)
        self.previous_action = 0
        self.game = 0
        self.serious_mode = False
        self.max_charges = 0
        self.charges = 0
        
        
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
        smell = int(smell)
        breeze = int(breeze)
        
        if (self.game == 0) and (self.time == 0):
            # changing the size of the transition_matrix so that we take into account the number of charges
            self.max_charges = charges
            self.transition_matrix = np.tile(self.transition_matrix, reps = self.max_charges+1).reshape(1, 1, 2, 2, self.max_charges+1, 8)
            self.charges = charges
            
        
        
        
        
        probabilities = self.transition_matrix[1, 1, smell, breeze, charges, :]
        
        # updating our knowledge of the size of the map
        if not self.reached_x_boundary:
            if self.max_x <x:
                self.max_x = x
        if not self.reached_y_boundary:
            if self.max_y <y :
                self.max_y = y
        
        if self.time >1 :
            if self.reached_x_boundary == False:
                if (x == self.previous_position[0]) and (self.previous_action == 4):
                    self.reached_x_boundary = True
                    self.max_x == x
            if self.reached_y_boundary == False:
                if (y == self.previous_position[1]) and (self.previous_action == 4):
                    self.reached_y_boundary = True
                    self.max_y == y
            # entering serious mode
            if self.reached_x_boundary and self.reached_y_boundary:
                self.serious_mode = True
                self.transition_matrix = np.tile(self.transition_matrix, reps = (self.max_x+1)*(self.max_y+1)).reshape(self.max_x+1, self.max_y+1, 2, 2, self.max_charges+1, 8)
                self.transition_matrix[0, :,:,:,:,2] = 0
                self.transition_matrix[-1,:,:,:,:,3] = 0
                self.transition_matrix[:,0,:,:,:,1] = 0
                self.transition_matrix[:,-1,:,:,:,0] = 0
                self.transition_matrix[:,:,:,:,0,3:] = 0
            
        if self.serious_mode == False:
            possible_actions = [i for i in range(1, 9)]    
            
            # handling boundaries
            if (self.reached_x_boundary == True) and (self.max_x == x): # no use of going further 
                possible_actions.remove(4)
                possible_actions.remove(8)
            elif x == 0:
                possible_actions.remove(3)
                possible_actions.remove(7)
                    
            
            if (self.reached_y_boundary == True) and (self.max_y == y):
                possible_actions.remove(1)
                possible_actions.remove(5)
            elif y == 0:
                possible_actions.remove(2)
                possible_actions.remove(6)
                
            # handling the absence of smell 
            if smell == 0:
                for i in range(5, 9):
                    try :
                        possible_actions.remove(i)
                    except ValueError:
                        continue
                    
            if self.game <=10:
                action = np.random.choice(a = possible_actions)
                
            else : 
                possible_probabilities = np.exp(np.array([probabilities[i - 1] for i in possible_actions])/100.)
    ##            possible_probabilities -= possible_probabilities.max()
    #            possible_probabilities = np.exp(possible_probabilities/10.)
                possible_probabilities/=possible_probabilities.sum()            
                action = np.random.choice(a = possible_actions, p = possible_probabilities)
        else:
            action = np.argmax(self.transition_matrix[x, y, smell, breeze, charges, :])+1
            
        self.time +=1
        self.previous_position = (x, y)
        self.previous_action = action
    
        return action
    
    def compute_next_state(self, observation, action):
        ((x,y), smell, breeze, charges) = observation
        
        last_charge = int(charges <= 1)
        charges = int(charges>0)
        smell = int(smell)
        breeze = int(breeze)
        
        next_x = x
        next_y = y
        next_charge = charges
        
        if action > 4:
            if last_charge == 1:
                next_charge = 0
            else :
                next_charge = 1
        else:
            if (action == 1) :
                if (self.max_y == y) and (self.reached_y_boundary):
                    next_y == y
                else :
                    next_y = y-1
            elif action == 2:
                if y == 0:
                    next_y = 0
                else : 
                    next_y = y-1
            elif action == 3:
                if x == 0:
                    next_x = 0
                else :
                    next_x = x - 1
            elif (action == 4) :
                if (self.max_x == x) and self.reached_x_boundary:
                    next_x = x
                else : 
                    next_x = x + 1
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
        
        action_index = action - 1
        
        next_state_reward = 0
                
        
        self.transition_matrix[smell, breeze, charges, action_index] += self.discount_factor*reward
        
        
        
        
        pass
    

class NewQLearnerAgent:
    def __init__(self):
        """Init a new agent.
        """
        self.time = 0 # 0 is not doing anything
        self.expected_reward_matrix = -np.ones((2, 2, 2, 8))
        self.number_of_visits = np.ones((2,2,2,8))
        self.discount_factor = .8
        
        self.max_x = 0
        self.max_y = 0
        
        self.reached_x_boundary = False
        self.reached_y_boundary = False
        self.previous_position = (0,0)
        self.previous_action = 0
        self.game = 0
        self.serious_mode = False
        
        
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
        
        
        probabilities = self.expected_reward_matrix[smell, breeze, charges, :]
        
        possible_actions = [i for i in range(1, 9)]               
        # handling the absence of smell 
        if smell == 0:
            for i in range(5, 9):
                try :
                    possible_actions.remove(i)
                except ValueError:
                    continue
    
        if self.game <=5:
            action = np.random.choice(a = possible_actions)
            
        else : 
            possible_probabilities = np.exp(np.array([probabilities[i - 1] for i in possible_actions])/100.)
            possible_probabilities/=possible_probabilities.sum()            
            action = np.random.choice(a = possible_actions, p = possible_probabilities)
            
        self.previous_action = action
        self.time +=1
        self.previous_position = (x, y)
        self.number_of_visits[breeze, smell, charges, action - 1] += 1
        
        return action
    
    def compute_next_state(self, observation, action):
        ((x,y), smell, breeze, charges) = observation
        
        last_charge = int(charges <= 1)
        charges = int(charges>0)
        smell = int(smell)
        breeze = int(breeze)
        
        next_x = x
        next_y = y
        next_charge = charges
        
        if action > 4:
            if last_charge == 1:
                next_charge = 0
            else :
                next_charge = 1
        else:
            if (action == 1) :
                if (self.max_y == y) and (self.reached_y_boundary):
                    next_y = y
                else :
                    next_y = y-1
            elif action == 2:
                if y == 0:
                    next_y = 0
                else : 
                    next_y = y-1
            elif action == 3:
                if x == 0:
                    next_x = 0
                else :
                    next_x = x - 1
            elif (action == 4) :
                if (self.max_x == x) and self.reached_x_boundary:
                    next_x = x
                else : 
                    next_x = x + 1
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
        
        action_index = action - 1
        
        previous_visits = self.number_of_visits[smell, breeze, charges, action_index] 
        previous_reward = self.expected_reward_matrix[smell, breeze, charges, action_index] 
        previous_reward = (previous_reward*(previous_visits-1) + reward)/previous_visits
        
        self.number_of_visits[smell, breeze, charges, action_index]  = previous_reward       
        
        
        pass
    
class DicoQLearnerAgent:
    def __init__(self):
        """Init a new agent.
        """
        self.time = 0 # 0 is not doing anything
        self.transition_matrix = np.ones((2, 2, 2, 8))
        self.transition_matrix /= self.transition_matrix#.sum(axis = 3)
        self.discount_factor = .8
        self.max_x = 0
        self.max_y = 0
        self.reached_x_boundary = False
        self.reached_y_boundary = False
        self.previous_position = (0,0)
        self.previous_action = 0
        self.game = 0
        self.serious_mode = False
        
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
        # updating our knowledge of the size of the map
        if self.max_x <x:
            self.max_x = x
        if self.max_y <y :
            self.max_y = y
        
        
        
        if (x,y) not in self.states.keys():
            self.states[(x,y)] = 0
        
        
        expected_rewards = []
        for action in range(1,9):
            ((next_x, next_y), next_charges) = self.compute_next_state(observation, action)
            if (next_x, next_y) in self.states.keys():
                expected_rewards.append(self.states[(next_x, next_y)])
            else: 
                expected_rewards.append(0)
        if charges == 0:
            expected_rewards = expected_rewards[1:5]
        
        action = np.argmax(expected_rewards) + 1
        
        self.time +=1
        
        
        return action
    
    def compute_next_state(self, observation, action):
        ((x,y), smell, breeze, charges) = observation
        
        last_charge = int(charges <= 1)
        charges = int(charges>0)
        smell = int(smell)
        breeze = int(breeze)
        
        next_x = x
        next_y = y
        next_charge = charges
        
        if action > 4:
            if last_charge == 1:
                next_charge = 0
            else :
                next_charge = 1
        else:
            if (action == 1) :
                if (self.max_y == y) and (self.reached_y_boundary):
                    next_y == y
                else :
                    next_y = y-1
            elif action == 2:
                if y == 0:
                    next_y = 0
                else : 
                    next_y = y-1
            elif action == 3:
                if x == 0:
                    next_x = 0
                else :
                    next_x = x - 1
            elif (action == 4) :
                if (self.max_x == x) and self.reached_x_boundary:
                    next_x = x
                else : 
                    next_x = x + 1
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
                
                
        
        self.states[(x,y)] += reward
        
        
        
        
        pass

Agent = RandomAgent
Agent = NonPositionalAgent
Agent = QLearnerAgent
#Agent = SeriousQLearnerAgent
#Agent = NewQLearnerAgent
Agent = DicoQLearnerAgent