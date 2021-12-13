# Exploración inicial del entorno 
# se desarrollan 10 episodeos ejecutando acciones aleatorias
import os
from gym.envs.classic_control.cartpole import CartPoleEnv
from stable_baselines3 import A2C

# Ruta donde esta el modelo entrenado
A2C_path = os.path.join("Training", "Saved_Models", "A2C")

env = CartPoleEnv()

#-------------- Comportamiento sin entrenar, decisiones aleatorias -----------

# episode = 1

# while 1:
#     env.reset()
#     done = 0
#     score = 0

#     while not done:
#         action = env.action_space.sample() # accione aleatoria
#         obs, reward, done, _ = env.step(action)
#         env.render()
#         score += reward
    
#     print("Episode %d, Score %d"%(episode, score))
#     episode += 1

# env.close() 

#-------------- Comportamiento de un agente A2 entrenado con -------------------

model = A2C.load(A2C_path)

episode = 1

while 1:
    obs = env.reset()
    done = 0
    score = 0

    while not done:
        action, _ = model.predict(obs) # accione aleatoria
        obs, reward, done, _ = env.step(action)
        env.render()
        score += reward
    
    print("Episode %d, Score %d"%(episode, score))
    episode += 1

env.close() 