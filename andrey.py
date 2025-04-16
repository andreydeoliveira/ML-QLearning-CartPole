import gymnasium as gym
import numpy as np

# Hiperparâmetros
alpha = 0.1    # Taxa de aprendizado
gamma = 0.99   # Desconto futuro
epsilon = 0.99  # Probabilidade de explorar ações aleatórias
epsilon_min = 0.01
epsilon_decay = 0.995
bins = 30      # Quantidade de divisões para discretizar os estados

env = gym.make('CartPole-v1', render_mode='human')

q_table = np.zeros([bins] * 4 + [env.action_space.n])

def choose_action(state):
    if np.random.rand() < epsilon:  # Exploração
        return np.random.choice(env.action_space.n)
    else:  # Exploração
        return np.argmax(q_table[state])

def train_agend(episodes):
    for episode in range(episodes):
        print('treinando ' + str(episode))
        print(env.action_space.n)
        state, _ = env.reset()
        
        done = False
        
        while not done:
            action = choose_action(state)
            next_state, reward, done, _, _ = env.step(action)
            state = next_state
            
            print(reward)
        
    


train_agend(200)
