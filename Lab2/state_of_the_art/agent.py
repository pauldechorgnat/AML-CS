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
        self.alpha = .9
        self.gamma = .5
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
            self.states[(x,y)] = -np.ones((2,2,2, 8))
            self.number_of_visits[(x,y)] = np.zeros((2,2,2, 8))
            
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
            
            updated_reward = previous_state_Q + self.alpha *(reward + self.gamma *current_state_Q - previous_state_Q)
            
            self.states[(x_prev,y_prev)][smell_prev, breeze_prev, int(charges_prev>0), self.previous_action] = updated_reward
        
        self.previous_action = action
        self.previous_state = observation
        pass
    
    
Agent=QLearnerAgentGreedy
            
            
            
        