#!/bin/python

import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')

# Q table: 
# Rows: state space
# Cols: action space
Q = np.zeros([env.observation_space.n, env.action_space.n])

# Learning parameters
lr = 0.85
gamma = 0.99
num_episodes = 2000

R_list = []

for i in range(num_episodes):
    s = env.reset()
    R_episode = 0
    d = False    
    
    for _ in range(99):
        a = np.argmax(Q[s,:] + np.random.randn(1, env.action_space.n)*(1./(i+1)))

        s1, R, done, info = env.step(a)
        
        Q[s,a] = Q[s,a] + lr*(R + gamma*np.max(Q[s1,:]) - Q[s,a])
        R_episode += R
        s = s1
        if done:
            break
    R_list.append(R_episode)

print('Score over time: ' + str(sum(R_list)/num_episodes))
print('Final Q-Table Values')
print(Q)

#plt.plot(range(num_episodes), R_list)
#plt.show()
