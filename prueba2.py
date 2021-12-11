from PongGame_Env import *
from numpy import mean

env = PongGame(game_speed = 2)

episodes = 30
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0 
    
    while not done:
        env.render()
        action = random.choice([0, 1, 2])
        n_state, reward, done, info = env.step(action)
    print('Episode:{} Score:{}'.format(episode, reward))

# states = env.observation_space.shape[0]
# actions = env.action_space.n


# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Flatten
# from tensorflow.keras.optimizers import Adam

# model = Sequential()
# model.add(Flatten(input_shape=(1,states)))
# model.add(Dense(24, activation='relu'))
# model.add(Dense(24, activation='relu'))
# model.add(Dense(actions, activation='linear'))

# model.summary()


# from rl.agents.dqn import DQNAgent
# from rl.policy import BoltzmannQPolicy
# from rl.memory import SequentialMemory  

# policy = BoltzmannQPolicy()
# memory = SequentialMemory(limit=50000, window_length=1)
# dqn = DQNAgent(model=model, memory=memory, policy=policy, 
#             nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)

# dqn.compile(Adam(lr=1e-3), metrics=['mae'])
# dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)


# scores = dqn.test(env, nb_episodes=10, visualize=True)
# print(mean(scores.history['episode_reward']))

# dqn.save_weights('dqn_weights.h5f', overwrite=True)