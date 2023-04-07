#   /$$$$$$$$                                                      /$$                 /$$                
#  | $$_____/                                                     | $$                | $$                
#  | $$     /$$$$$$   /$$$$$$  /$$$$$$$$  /$$$$$$  /$$$$$$$       | $$        /$$$$$$ | $$   /$$  /$$$$$$ 
#  | $$$$$ /$$__  $$ /$$__  $$|____ /$$/ /$$__  $$| $$__  $$      | $$       |____  $$| $$  /$$/ /$$__  $$
#  | $$__/| $$  \__/| $$  \ $$   /$$$$/ | $$$$$$$$| $$  \ $$      | $$        /$$$$$$$| $$$$$$/ | $$$$$$$$
#  | $$   | $$      | $$  | $$  /$$__/  | $$_____/| $$  | $$      | $$       /$$__  $$| $$_  $$ | $$_____/
#  | $$   | $$      |  $$$$$$/ /$$$$$$$$|  $$$$$$$| $$  | $$      | $$$$$$$$|  $$$$$$$| $$ \  $$|  $$$$$$$
#  |__/   |__/       \______/ |________/ \_______/|__/  |__/      |________/ \_______/|__/  \__/ \_______/
#                                                                                                         
#                                                                                                         

# Libraries 

import gym 
import numpy as np
import random
from IPython.display import clear_output
import argparse


def train_agent(env, alpha = 0.8, gamma = 0.6, epsilon = 0.2):

    # Generating Q-table
    q_table = np.zeros([env.observation_space.n, env.action_space.n])

    for i in range(1, 100001):
        state = env.reset()[0]

        done = False
        
        while not done:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample() # Explore action space
            else:
                action = np.argmax(q_table[state]) # Exploit learned values

            next_state, reward, done, info, _ = env.step(action) 
            
            new_value = q_table[state, action] + alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state, action])
            q_table[state, action] = new_value
        
            state = next_state
            
        if i % 100 == 0:
            clear_output(wait=True)
            print(f"Episode: {i}")

    print("Training finished.\n")
    return q_table


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Generate Q-learning demonstration of the Frozen Lake.')
    parser.add_argument('--ep',default = 1, help='number of episodes you want to display')
    parser.add_argument('--alpha', default=0.8, help='Learning rate')
    parser.add_argument('--gamma', default=0.6, help='Discount factor')
    parser.add_argument('--epsilon', default=0.2, help='Exploratory factor')

    args = parser.parse_args()
    
    # Create environment
    env = gym.make("FrozenLake-v1", is_slippery = False)

    q_table = train_agent(env, alpha= float(args.alpha), gamma= float(args.gamma), epsilon= float(args.epsilon))

    episodes = int(args.ep)
    env = gym.make("FrozenLake-v1", render_mode = "human" , is_slippery = False)

    for _ in range(episodes): 
        state = env.reset()[0]
        done = False
        while not done:
            action = np.argmax(q_table[state])
            state, reward, done, info, _ = env.step(action)
            env.render()
    env.close()







