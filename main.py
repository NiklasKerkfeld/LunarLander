import time

import numpy as np
import gym
from DeepQAgent import DeepQAgent as Agent
from tqdm import tqdm


def main():
    train()

def train(epochs=1_000, render_every=100, save_every=100):
    env = gym.make("CartPole-v1")
    agent = Agent(4, 2, alpha=.2, random=.3)

    rewards = []
    for e in tqdm(range(1, epochs+1)):
        observation = env.reset()
        done = False
        sum_reward = 0
        while not done:
            if e % render_every == 0:
                env.render()
                time.sleep(0.1)
            action = agent.step(observation)
            observation, reward, done, _ = env.step(action)
            reward *= (not done)
            sum_reward += reward
            agent.update(observation, reward)
        rewards.append(sum_reward)
        if e % save_every == 0:
            agent.save("deepQ")
            print(np.max(rewards[e-save_every:e]), np.average(rewards[e-save_every:e]))

    env.close()


if __name__ == '__main__':
    main()

