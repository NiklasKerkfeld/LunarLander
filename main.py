import time
import numpy as np
import gym


from DeepQAgent import DeepQAgent as Agent
from Helper import LoadingBar, plot
from tqdm import tqdm


def main():
    train()


def train(epochs=1_000, render_every=100, save_every=500):
    env = gym.make("CartPole-v1")
    agent = Agent(4, 2, alpha=.2, epsilon=1, epsilon_min=.02, theta=3e-4)

    rewards = []
    loadingBar = LoadingBar()

    step_time, update_time = [], []


    for e in range(1, epochs+1):
        loadingBar(step=e, all=epochs+1, scores=rewards, loss=agent.losses, version=agent.version, random=agent.epsilon, gamma=agent.gamma)
        observation = env.reset()
        done = False
        sum_reward = 0
        while not done:
            if e % render_every == 0:
                env.render()
                time.sleep(0.1)

            action = agent.step(observation)

            observation, reward, done, _ = env.step(action)
            sum_reward += reward
            reward = reward if not done else -reward

            agent.update(observation, reward, done)

        rewards.append(sum_reward)
        if e % save_every == 0:
            agent.save("deepQ")
    env.close()

    plot([rewards, agent.losses], ['rewards', 'loss'])


if __name__ == '__main__':
    main()

