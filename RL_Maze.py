# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 13:40:32 2023

@author: ATA
"""
import numpy as np
import random
from RLDataLoader import Reward, Transition
class Maze:
    def __init__(self,states,rewards,transitions,actions,discount_facotr):
        #numbered list of states
        self.states = states
        #List of reward for being at each state, corresonding to self.states 
        self.rewards = rewards
        #List of all possible actions can be taken by maze
        self.actions = actions
        #List of transition probability matrices, each transition probability 
        #matrix is a numpy array corresponded to an action, according to 
        #actions in self.actions
        self.transitions = transitions
        # discount factor a number 0 < discount_factor < 1 (float)
        self.discount_facotr = discount_facotr
    
    def _evaluate_value_function(self,policy, transition_matrix):
        """ function to solve linear Bellman equation and calculate
        Value function for all states (matrix inversion O(n^3) complexity)
        
        Arguments:
        policy - list of length n including action to be taken at each step
        transition probability - matrix of nxn for transition 
        probabilties after taking action "a" according to policy. P(s' |s, a=policy ) 
        """
        
        return np.linalg.inv(np.eye(len(policy))-self.discount_facotr*transition_matrix).dot(np.array(self.rewards))
        
    def policy_iteration(self):
        """ function that uses policy iteration to find optimal policy to solve
        the maze, as described as below:
            
        --> initial random policy for states that agent can take action in it (i.e.
        every state other than dragon and Exit states) 
        --> evaluate V and for all states and Q for all state,all actions 
        at each state 
        --> improve policy: by taking action s.t. Q is maximum at each state
        --> repeat
        """
        
        # initialize random policy (by chosing random action for each state)
        policy_old = [0 for item in self.rewards]
        for i in range(len(self.rewards)):
            if self.rewards[i] == 0:
                policy_old[i] = random.choice(self.actions)
        
        #iterate to improve policy        
        while True:
            transition_matrix = np.zeros((len(self.states),len(self.states)))
            for row in range(len(self.states)):
                if policy_old[row] !=0:
                    transition_matrix[row,:] = self.transitions[policy_old[row]-1][row,:]
            V = self._evaluate_value_function(policy_old, transition_matrix)
            #calculate action value functions for each state , and all actions at each state
            Q =  np.zeros((len(self.states),len(self.actions)))
            policy_new = []
            for state_index in range(len(self.states)):
                for action_index in range(len(self.actions)):
                    Q[state_index,action_index] = self.transitions[action_index][state_index,:].dot(V)
                policy_new.append(np.argmax(Q[state_index,:])+1)
                
            #condition to end the iteration    
            if policy_old == policy_new:
                break
            else:
                policy_old = policy_new
                
        return policy_new
    
    def Value_Iteration(self):
        """ function that use Value iteration approach to find optimal value 
        function and then extract optimal policy"""
        
        V_old = np.zeros((len(self.states),1))
        while True:
            best_Q = []
            best_actions =[]
            Q =  np.zeros((len(self.states),len(self.actions)))
            for state_index in range(len(self.states)):
                for action_index in range(len(self.actions)):
                    Q[state_index,action_index] = self.transitions[action_index][state_index,:].dot(V_old)
                best_Q.append(np.max(Q[state_index,:]))
                best_actions.append(np.argmax(Q[state_index,:])+1)
            V_new = np.array(self.rewards) + self.discount_facotr * np.array(best_Q)
            
            if np.linalg.norm(V_new-V_old) < 0.1:
                break
            else:
                V_old = V_new
                
        return V_new, best_actions
    
def Main():
    
    R = Reward()
    P = Transition()
    S = list(range(len(R)))
    A = list(range(1,1+len(P)))
    gamma = 0.99
    M = Maze(S, R, P, A, gamma)
    
    optimal_policy = M.policy_iteration()
    print("##################### Policy Iteration #####################")
    print('actions:    1: go West    2:go North   3: go East  4: go South')
    print(optimal_policy)
    
    
    optimal_values, optimal_actions = M.Value_Iteration()
    print("##################### Value Iteration #####################")
    print(optimal_values)
    print('Optimal Policy')
    print('actions:    1: go West    2:go North   3: go East  4: go South')
    print(optimal_actions)
    
if __name__ == "__main__":       
    Main()