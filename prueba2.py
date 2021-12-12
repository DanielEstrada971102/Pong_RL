from PongGame_Env import *
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3 import PPO

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


# Train the agent
model = PPO('MlpPolicy', env, verbose=1)
model.learn(10000)

# Test the trained agent
obs = env.reset()
n_steps = 20
for step in range(n_steps):
  action, _ = model.predict(obs, deterministic=True)
  print("Step {}".format(step + 1))
  print("Action: ", action)
  obs, reward, done, info = env.step(action)
  print('obs=', obs, 'reward=', reward, 'done=', done)
  env.render(mode='console')
  if done:
    print("Goal reached!", "reward=", reward)
    break
