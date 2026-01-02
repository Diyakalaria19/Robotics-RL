import numpy as np
import gymnasium as gym
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle  # built-in, no need for pickle5
#import time

# ----------------------------
# 1. Create FrozenLake environment for training
# ----------------------------
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)  # no render_mode, runs silently

state_space = env.observation_space.n
action_space = env.action_space.n

# ----------------------------
# 2. Initialize Q-table
# ----------------------------
def initialize_q_table(state_space, action_space):
    return np.zeros((state_space, action_space))

Qtable = initialize_q_table(state_space, action_space)

# ----------------------------
# 3. Define policies
# ----------------------------
def greedy_policy(Qtable, state):
    return np.argmax(Qtable[state])

def epsilon_greedy_policy(Qtable, state, epsilon):
    if random.uniform(0, 1) > epsilon:
        return greedy_policy(Qtable, state)
    else:
        return env.action_space.sample()

# ----------------------------
# 4. Hyperparameters
# ----------------------------
n_training_episodes = 50000
learning_rate = 0.7
gamma = 0.95
max_steps = 99

max_epsilon = 1.0
min_epsilon = 0.05
decay_rate = 0.0005

# ----------------------------
# 5. Training function
# ----------------------------
def train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable):
    for episode in tqdm(range(n_training_episodes)):
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
        state, info = env.reset()
        terminated = False
        truncated = False

        for step in range(max_steps):
            action = epsilon_greedy_policy(Qtable, state, epsilon)
            new_state, reward, terminated, truncated, info = env.step(action)
            # Q-Learning update
            Qtable[state][action] = Qtable[state][action] + learning_rate * (
                reward + gamma * np.max(Qtable[new_state]) - Qtable[state][action]
            )
            state = new_state
            if terminated or truncated:
                break
    return Qtable

# ----------------------------
# 6. Train the agent silently
# ----------------------------
Qtable = train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable)

# ----------------------------
# Print and visualize Q-table
# ----------------------------
np.set_printoptions(precision=2, suppress=True)
print("Q-table after training:")
print(Qtable)

# Optional heatmap
plt.imshow(Qtable, cmap="cool", interpolation="nearest")
plt.colorbar()
plt.title("Q-Table Heatmap")
plt.xlabel("Actions (0=L,1=D,2=R,3=U)")
plt.ylabel("States")
plt.show()

# ----------------------------
# 7. Evaluate agent silently
# ----------------------------
def evaluate_agent(env, max_steps, n_eval_episodes, Qtable):
    rewards = []
    for _ in range(n_eval_episodes):
        state, info = env.reset()
        total_reward = 0
        terminated = False
        truncated = False
        for step in range(max_steps):
            action = greedy_policy(Qtable, state)
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        rewards.append(total_reward)
    return np.mean(rewards), np.std(rewards)

mean_reward, std_reward = evaluate_agent(env, max_steps, 100, Qtable)
print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")

env.close()  # close training environment

# ----------------------------
# 8. Watch trained agent live
# ----------------------------
watch_env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode="human")

state, info = watch_env.reset()
terminated = False
truncated = False
step = 0
print("Watching trained agent...")

while not (terminated or truncated) and step < max_steps:
    watch_env.render()  # display environment live
    action = greedy_policy(Qtable, state)
    state, reward, terminated, truncated, info = watch_env.step(action)
    step += 1
    #time.sleep(0.3)  # slow down so you can see movements

#time.sleep(2)  # pause on final frame
print(f"Episode finished in {step} steps, reward: {reward}")
watch_env.close()

# ----------------------------
# 9. Optional: Save Q-table
# ----------------------------
# with open("Qtable.pkl", "wb") as f:
#     pickle.dump(Qtable, f)
