import numpy as np
import pygame
import sys
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from src.flappy_bird import FlappyBirdEnv
import matplotlib.pyplot as plt

 # Training parameters
EPISODES = 2500
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.95
EPSILON = 1.0
EPSILON_DECAY = 0.99
EPSILON_MIN = 0.01

def discretize_state(state):
    bird_y, pipe_x, pipe_height = state
    bird_y_bin = min(int(bird_y // 20), 29)
    pipe_x_bin = max(0, min(int(pipe_x // 50), 23))
    pipe_height_bin = min(int((pipe_height - 100) // 20), 14)
    return (bird_y_bin, pipe_x_bin, pipe_height_bin)

env = FlappyBirdEnv()
state_space_size = (30, 24, 15)
action_space_size = 2
Q_table = np.zeros(state_space_size + (action_space_size,))
best_Q_table = np.copy(Q_table)
best_score = -float('inf')

 # Training loop
for episode in range(EPISODES):
    state = env.reset()
    discrete_state = discretize_state(state)
    done = False
    total_reward = 0

    while not done:
        if np.random.random() < EPSILON:
            action = np.random.randint(0, 2)
        else:
            action = np.argmax(Q_table[discrete_state])

        next_state, reward, done = env.step(action)
        next_discrete_state = discretize_state(next_state)

        old_q = Q_table[discrete_state + (action,)]
        next_max_q = np.max(Q_table[next_discrete_state])
        new_q = old_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * next_max_q - old_q)
        Q_table[discrete_state + (action,)] = new_q

        discrete_state = next_discrete_state
        total_reward += reward

    if env.score > best_score:
        best_score = env.score
        best_Q_table = np.copy(Q_table)
        print(f"New best score at Episode {episode}: {best_score}")

    EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)
    if episode % 1000 == 0:
        print(f"Episode {episode}, Reward: {total_reward}, Score: {env.score}")

 # Save the best Q-Table
np.save("models/best_q_table_beginner.npy", best_Q_table)
print(f"Training complete. Best Q-Table saved with best score: {best_score}")

 # Test Agent 10 times and plot
print("Testing Agent performance over 10 episodes...")
agent_scores = []
for i in range(10):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                exit()
        discrete_state = discretize_state(state)
        action = np.argmax(best_Q_table[discrete_state])
        state, reward, done = env.step(action)
        env.render()
        total_reward += reward
    agent_scores.append(env.score)
    print(f"Test Episode {i}, Reward: {total_reward}, Score: {env.score}")

 # Plot and save the score graph
plt.plot(range(1, 11), agent_scores, marker='o', linestyle='-', color='b')
plt.xlabel('Test Episode')
plt.ylabel('Score')
plt.title('Agent Performance (Play across 2500 times)')
plt.grid(True)
plt.savefig("results/beginner_mode_scores.png")
print("Score graph saved to 'results/beginner_mode_scores.png'")
plt.close()

env.close()