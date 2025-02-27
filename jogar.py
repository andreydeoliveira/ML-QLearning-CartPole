import gymnasium as gym
import numpy as np

bins = 30    

# Função para discretizar os estados (mesma função do treinamento)
def discretize_state(state, bins):
    upper_bounds = [4.8, 5, 0.418, 5]
    lower_bounds = [-4.8, -5, -0.418, -5]
    ratios = [(state[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(state))]
    new_state = [int(round((bins - 1) * ratios[i])) for i in range(len(state))]
    new_state = [min(bins - 1, max(0, s)) for s in new_state]
    return tuple(new_state)

# Carregar a tabela Q treinada
q_table = np.load('q_table.npy')

# Inicializa o ambiente
env = gym.make('CartPole-v1', render_mode='human')

# Função para jogar com o agente treinado
def play_game():
    state, _ = env.reset()
    state = discretize_state(state, bins)
    done = False
    total_reward = 0

    while not done:
        action = np.argmax(q_table[state])  # Ação com maior valor Q
        state, reward, done, truncated, _ = env.step(action)
        state = discretize_state(state, bins)
        env.render()
        total_reward += reward

    print(f'Fim do Jogo! Recompensa Total: {total_reward}')

# Jogar com o agente treinado
play_game()

env.close()
