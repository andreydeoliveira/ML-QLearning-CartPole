import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset()

for _ in range(1000):
    env.render()
    action = env.action_space.sample()  # Ação aleatória
    observation, reward, done, truncated, info = env.step(action)
    
    if done or truncated:
        observation, info = env.reset()

env.close()
