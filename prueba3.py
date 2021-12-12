import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from rl.agents import CEMAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

from PongGame_Env import *

env = PongGame(game_speed = 2)

states = env.observation_space.shape
actions = env.action_space.n

model = Sequential()    
model.add(Dense(24, activation='relu', input_shape=states))
model.add(Dense(24, activation='relu'))
model.add(Dense(actions, activation='linear'))

model.summary()

policy = BoltzmannQPolicy()
memory = SequentialMemory(limit=50000, window_length=1)
dqn = CEMAgent(model=model, memory=memory, 
                nb_actions=actions, nb_steps_warmup=10)

dqn.compile()
dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)

_ = dqn.test(env, nb_episodes=15, visualize=True)