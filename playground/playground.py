import numpy as np
import pygame
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from src.flappy_bird import FlappyBirdEnv
import time
import matplotlib.pyplot as plt

def discretize_state(state):
    bird_y, pipe_x, pipe_height = state
    bird_y_bin = min(int(bird_y // 20), 29)
    pipe_x_bin = max(0, min(int(pipe_x // 50), 23))
    pipe_height_bin = min(int((pipe_height - 100) // 20), 14)
    return (bird_y_bin, pipe_x_bin, pipe_height_bin)

env = FlappyBirdEnv()

 # Load Q-Table
best_Q_table = np.load("../models/best_q_table_beginner.npy")
print("Loaded best_q_table_beginner.npy for Agent")

 # Countdown function for user
def countdownUser():
    env.reset()
    for i in range(3, 0, -1):
        env.WIN.blit(env.BACKGROUND, (0, 0))
        countdown_text = env.font.render(f"Your turn first. Press 'SpaceBar' to fly. Get Ready: {i}", True, env.GOLD, (50, 50, 50))
        env.WIN.blit(countdown_text, (env.WIDTH // 2 - 250, env.HEIGHT // 2 - 10))
        pygame.display.update()
        time.sleep(1)

def countdownAgent():
    env.reset()
    for i in range(3, 0, -1):
        env.WIN.blit(env.BACKGROUND, (0, 0))
        countdown_text = env.font.render(f"Agent's turn. Get Ready: {i}", True, env.GOLD, (50, 50, 50))
        env.WIN.blit(countdown_text, (env.WIDTH // 2 - 150, env.HEIGHT // 2 - 10))
        pygame.display.update()
        time.sleep(1)

 # Function for user vs agent round
def play_round(round_num):
    # User plays
    print(f"\nRound {round_num} - User's turn...")
    countdownUser()
    state = env.reset()
    done = False
    user_score = 0
    while not done:
        action = 0  # Default: do not jump
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                return False  # Indicates exit
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                action = 1  # User presses space to jump
        state, reward, done = env.step(action)
        env.render()
        if reward > 0:
            user_score = env.score

    # Agent plays
    print(f"Round {round_num} - Agent's turn...")
    countdownAgent()
    state = env.reset()
    done = False
    agent_score = 0
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                return False  # Indicates exit
        discrete_state = discretize_state(state)
        action = np.argmax(best_Q_table[discrete_state])
        state, reward, done = env.step(action)
        env.render()
        if reward > 0:
            agent_score = env.score

    # Show single round result
    print(f"Round {round_num} - User Score: {user_score}, Agent Score: {agent_score}")
    winner = "User" if user_score > agent_score else "Agent" if agent_score > user_score else "Tie"
    print(f"Round {round_num} - Winner: {winner}")

    env.WIN.blit(env.BACKGROUND, (0, 0))
    result_text = env.font.render(f"User: {user_score} vs Agent: {agent_score}", True, env.GOLD, (50, 50, 50))
    winner_text = env.font.render(f"Winner: {winner}", True, env.GOLD)
    env.WIN.blit(result_text, (env.WIDTH // 2 - 150, env.HEIGHT // 2 - 40))
    env.WIN.blit(winner_text, (env.WIDTH // 2 - 100, env.HEIGHT // 2 + 10))
    pygame.display.update()
    time.sleep(3)  # Show result for 3 seconds

    return user_score, agent_score

 # Main loop: multiple rounds
print("Starting Flappy Bird Challenge...")
user_scores = []
agent_scores = []
round_num = 1

while True:
    result = play_round(round_num)
    if result is False:  # User closed window
        break
    user_score, agent_score = result
    user_scores.append(user_score)
    agent_scores.append(agent_score)
    round_num += 1

    # Show continue or quit prompt
    env.WIN.blit(env.BACKGROUND, (0, 0))
    continue_text = env.font.render("Press 'c' to continue, 'q' to quit", True, env.GOLD, (50, 50, 50))
    env.WIN.blit(continue_text, (env.WIDTH // 2 - 150, env.HEIGHT // 2))
    pygame.display.update()

    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                waiting = False
            elif event.type == pygame.KEYDOWN:
                print(f"Key pressed: {event.key}")  # Debug: print pressed key
                if event.key == pygame.K_c:  # Only check lowercase 'c'
                    waiting = False  # Continue next round

                elif event.key == pygame.K_q:  # Only check lowercase 'q'
                    waiting = False
                    pygame.quit()  # Ensure exit

    if not pygame.get_init():  # If window is closed
        break

# Statistics and analysis
if user_scores:  # Ensure at least one round played
    avg_user_score = np.mean(user_scores)
    avg_agent_score = np.mean(agent_scores)
    print(f"\nAnalysis:")
    print(f"Total Rounds: {len(user_scores)}")
    print(f"Average User Score: {avg_user_score:.2f}")
    print(f"Average Agent Score: {avg_agent_score:.2f}")

    # Plot user vs agent result graph
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(user_scores) + 1), user_scores, marker='o', linestyle='-', color='b', label='User Score')
    plt.plot(range(1, len(agent_scores) + 1), agent_scores, marker='o', linestyle='-', color='g', label='Agent Score')
    plt.xlabel('Round')
    plt.ylabel('Score')
    plt.title('User vs Agent Performance (2500)')
    plt.legend()
    plt.grid(True)
    plt.savefig("../results/user_vs_agent_scores.png")
    print("User vs Agent score graph saved to '../results/user_vs_agent_scores.png'")
    plt.close()

if not pygame.get_init():  # Ensure program exits if window is closed
    pygame.quit()
