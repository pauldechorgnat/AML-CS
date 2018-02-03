# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 14:00:55 2018

@author: Paul
"""

import numpy as np

class QLearnerAgentGreedy:
    def __init__(self):
        self.game = 0
        self.states = {}
        self.number_of_visits = {}
        self.x_max = 0
        self.y_max = 0
        self.alpha = .5
        self.gamma = .9
        self.previous_state = ()
        self.previous_reward = 0
        self.previous_action = 0
        
    def act(self, observation):
        
        ((x,y), smell, breeze, charges) = observation
        charges = int(charges>0)
        smell = int(smell)
        breeze = int(breeze)
        
        # adding the new position to the known states
        if (x,y) not in self.states.keys():
            self.states[(x,y)] = -np.ones((2,2,2, 4))
            self.number_of_visits[(x,y)] = np.zeros((2,2,2, 4))
            
        action = np.argmax(self.states[(x,y)][ smell, breeze, charges, :])
        self.number_of_visits[(x,y)][ smell, breeze, charges, action] += 1
        self.step += 1
        
        
        return action + 1
    
    def reset(self):
        self.step = 0
        self.game += 1
        pass
    
    def reward(self,observation, action, reward):
        
        ((x,y), smell, breeze, charges) = observation
        charges = int(charges>0)
        smell = int(smell)
        breeze = int(breeze)
        
        if self.step >1:
            ((x_prev, y_prev), smell_prev, breeze_prev, charges_prev) = self.previous_state
            current_state_Q = self.states[(x,y)][smell, breeze, charges, action]
            previous_state_Q = self.states[(x_prev,y_prev)][smell_prev, breeze_prev, int(charges_prev>0), self.previous_action]
            
            updated_reward = previous_state_Q + self.alpha * (reward + self.gamma *current_state_Q - previous_state_Q)
            
            self.states[(x_prev,y_prev)][smell_prev, breeze_prev, int(charges_prev>0), self.previous_action] = updated_reward
        
        self.previous_action = action
        self.previous_state = observation
        pass

class QLearnerAgentUCB:
    def __init__(self):
        self.game = 0
        self.states = {}
        self.number_of_visits = {}
        self.alpha = .9
        self.gamma = .99
        self.previous_state = ()
        self.previous_action = 0
        
    def act(self, observation):
        
        ((x,y), smell, breeze, charges) = observation
        smell = int(smell)
        breeze = int(breeze)
        
        # adding the new position to the known states
        if (x,y) not in self.states.keys():
            self.states[(x,y)] = -np.ones((2,2,4))
            if x == 0:
                self.states[(x,y)][:,:,2] = -1000
            if y == 0:
                self.states[(x,y)][:,:,1] = -1000
            self.number_of_visits[(x,y)] = np.ones((2,2,4))
        
        possible_actions = [i for i in range(0, 4)]
        
        if ((self.step == 0) and (self.game == 0)) :#or (self.game <3):
            action = np.random.choice(a = possible_actions) +1
            self.step += 1
        else :
            self.step += 1
            ucb_score = self.states[(x,y)][smell, breeze, possible_actions] +\
            np.sqrt(2*np.log(self.step)/self.number_of_visits[(x,y)][ smell, breeze, possible_actions])
            action = np.argmax(ucb_score) +1
            
        self.number_of_visits[(x,y)][ smell, breeze, action-1] += 1
        
        
        return action 
    
    def reset(self):
        self.step = 0
        self.game += 1
        pass
    
    def reward(self,observation, action, reward):
        
        ((x,y), smell, breeze, charges) = observation
        
        smell = int(smell)
        breeze = int(breeze)
        
        if self.step >1:
            ((x_prev, y_prev), smell_prev, breeze_prev, charges_prev) = self.previous_state
            breeze_prev = int(breeze_prev)
            smell_prev = int(smell_prev)
            
            current_state_Q = self.states[(x,y)][smell, breeze, action-1]
            previous_state_Q = self.states[(x_prev,y_prev)][smell_prev, breeze_prev, self.previous_action]
            
            updated_reward = previous_state_Q + self.alpha * (reward + self.gamma *current_state_Q - previous_state_Q)
            
            self.states[(x_prev,y_prev)][smell_prev, breeze_prev, self.previous_action] = updated_reward
        
        self.previous_action = action - 1
        self.previous_state = observation
        pass
    
Agent = QLearnerAgentGreedy
Agent = QLearnerAgentUCB
            
            
        