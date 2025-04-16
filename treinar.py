import gymnasium as gym
import numpy as np

# Hiperparâmetros
alpha = 0.1    # Taxa de aprendizado
gamma = 0.99   # Desconto futuro
epsilon = 0.99  # Probabilidade de explorar ações aleatórias
epsilon_min = 0.01
epsilon_decay = 0.995
bins = 30      # Quantidade de divisões para discretizar os estados

# Função para discretizar os estados (como no treinamento)
def discretize_state(state, bins):
    upper_bounds = [4.8, 5, 0.418, 5]
    lower_bounds = [-4.8, -5, -0.418, -5]
    ratios = [(state[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(state))]
    new_state = [int(round((bins - 1) * ratios[i])) for i in range(len(state))]
    new_state = [min(bins - 1, max(0, s)) for s in new_state]
    return tuple(new_state)

# Inicializa o ambiente
# env = gym.make('CartPole-v1', render_mode='human')
env = gym.make('CartPole-v1', render_mode='rbg_array')

# Inicializa a tabela Q (com tamanho do estado discretizado e 2 ações possíveis)
q_table = np.zeros([bins] * 4 + [env.action_space.n])

# Função para escolher a ação (ε-greedy)
def choose_action(state):
    if np.random.rand() < epsilon:  # Exploração
        return np.random.choice(env.action_space.n)
    else:  # Exploração
        return np.argmax(q_table[state])

# Função de treinamento
def train_agent(episodes):
    global epsilon
    for episode in range(episodes):
        state, _ = env.reset()
        state = discretize_state(state, bins)
        done = False
        total_reward = 0

        while not done:
            action = choose_action(state)  # Escolhe a ação com base na política
            next_state, reward, done, truncated, _ = env.step(action)
            next_state = discretize_state(next_state, bins)

            # Atualiza a tabela Q com a fórmula do Q-Learning
            q_table[state][action] = q_table[state][action] + alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state][action])

            state = next_state
            total_reward += reward

        # Diminui o epsilon após cada episódio para reduzir a exploração
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        print(f'Episódio {episode + 1}, Recompensa Total: {total_reward}, Epsilon: {epsilon}')

# Treinamento do agente
train_agent(20000)  # Treinar por 1000 episódios

# Salvar a tabela Q treinada
np.save('q_table.npy', q_table)
