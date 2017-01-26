#!/bin/python

import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random

env = gym.make('FrozenLake-v0')

tf.reset_default_graph()

