import time
import numpy as np
import gym


from DoubleDeepQAgent import DoubleDeepQAgent as Agent
from Helper import LoadingBar, plot


def main():
    train()


def train(epochs=1_000, render_every=100, save_every=100):
    env = gym.make("CartPole-v1")
    agent = Agent(4, 2)

    rewards = []
    loadingBar = LoadingBar()

    for e in range(1, epochs+1):
        loadingBar(step=e-1, all=epochs, scores=rewards, loss=agent.losses, version=agent.version, random=agent.epsilon, gamma=agent.gamma)
        observation = env.reset()
        done = False
        sum_reward = 0
        while not done:
            if e % render_every == 0:
                env.render()
                time.sleep(0.02)

            action = agent.step(observation)

            observation, reward, done, _ = env.step(action)
            sum_reward += reward
            reward = reward if not done else -reward

            agent.update(observation, reward, done)

        rewards.append(sum_reward)
        if e % save_every == 0:
            agent.save("DoubledeepQ")
    env.close()

    plot([rewards, agent.losses], ['rewards', 'loss'])


if __name__ == '__main__':
    main()

