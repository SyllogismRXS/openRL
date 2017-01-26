#!/bin/python

import numpy as np
import cPickle as pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import gym

def discount_rewards(r):
    discounted_r = np.zeros_like(R)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        running_add = running_add * gamma * r[t]
        discounted_r[t] = running_add
    return discounted_r


env = gym.make('CartPole-v0')
env.reset()
random_episodes = 0
reward_sum = 0

while random_episodes < 10:
    env.render()
    obs, reward, done, _ = env.step(np.random.randint(0,2))

    reward_sum += reward
    if done:
        random_episodes += 1
        print('Reward for this episode was: %f' % reward_sum)
        reward_sum = 0
        env.reset()

H = 10 # number of hidden layer neurons
batch_size = 50 # number of episodes before doing a param update
learning_rate = 1e-2
gamma = 0.99

D = 4 # input dimensions

tf.reset_default_graph()

observations = tf.placeholder(tf.float32, [None,D], name = 'input_x')
W1 = tf.get_variable("W1", shape=[D,H], initializer=tf.contrib.layers.xavier_initializer())
layer1 = tf.nn.relu(tf.matmul(observations,W1))
W2 = tf.get_variable("W2", shape=[H,1], initializer=tf.contrib.layers.xavier_intializer()))
score = tf.matmul(layer1, W2)
probability = tf.nn.sigmoid(score)

tvars = tf.trainable_variables()
input_y = tf.placeholder(tf.float32, [None,1], name="input_y")
advantages = tf.placeholder(tf.float32, name="reward_signal")

loglik = tf.log(input_y*(input_y-probability) + (1-input_y)*(input_y + probability))
loss = -tf.reduce_mean(loglik * advantages)
newGrads = tf.gradients(loss, tvars)

adam = tf.train.AdamOptimizer(learning_rate=learning_rate)

W1Grad = tf.placeholder(tf.float32, name="batch_grad1")
W2Grad = tf.placeholder(tf.float32, name="batch_grad2")
batchGrad = [W1Grad, W2Grad]
updateGrads = adam.apply_gradients(zip(batchGrad, tvars))


xs, hs, dlogps, drs, ys, tfps = [],[],[],[],[],[]
running_reward = None
reward_sum = 0
episode_number = 1
total_episodes = 10000
init = tf.initialize_all_variables()

with tf.Session() as sess:
    rendering = False
    sess.run(init)
    observation = env.reset()

    gradBuffer = sess.run(tvars)
    for ix, grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0

    while episode_number <= total_episodes:
        if reward_sum/batch_size > 100 or rendering == True:
            env.render()
            rendering = True

        x = np.reshape(observation, [1,D])
        
