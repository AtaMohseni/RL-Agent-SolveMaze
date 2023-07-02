# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 12:57:22 2023

@author: ATA
"""
import numpy as np
def Reward():
    try:
        fhand = open("./data/rewards.txt")
    except:
        print ('the file does not exist')
        return None
    rewards = []
    for line in fhand:
        rewards.append(int(line.strip()))
    return rewards
def Transition():
    try:
        fhand1 = open("./data/prob_a1.txt")
        fhand2 = open("./data/prob_a2.txt")
        fhand3 = open("./data/prob_a3.txt")
        fhand4 = open("./data/prob_a4.txt")
    except:
        print ('the file does not exist')
        return None
    number_of_states = len(Reward())
    T = []
    transition1 = np.zeros((number_of_states,number_of_states))
    transition2 = np.zeros((number_of_states,number_of_states))
    transition3 = np.zeros((number_of_states,number_of_states))
    transition4 = np.zeros((number_of_states,number_of_states))
    for line in fhand1:
        row_index = int(line.split()[0])-1
        col_index = int(line.split()[1])-1
        transition1[row_index,col_index] = float(line.split()[2])
    fhand1.close() 
    T.append(transition1)
    for line in fhand2:
        row_index = int(line.split()[0])-1
        col_index = int(line.split()[1])-1
        transition2[row_index,col_index] = float(line.split()[2])
    fhand2.close() 
    T.append(transition2)
    for line in fhand3:
        row_index = int(line.split()[0])-1
        col_index = int(line.split()[1])-1
        transition3[row_index,col_index] = float(line.split()[2])
    fhand3.close() 
    T.append(transition3)
    for line in fhand4:
        row_index = int(line.split()[0])-1
        col_index = int(line.split()[1])-1
        transition4[row_index,col_index] = float(line.split()[2])
    fhand4.close() 
    T.append(transition4)
    return T