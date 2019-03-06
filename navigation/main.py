import argparse
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from unityagents import UnityEnvironment

from navigation.dqn_agent import Agent

CHECKPOINT_PATH = str(Path() / 'navigation' / 'checkpoint.pth')


def not_trained_mode(agent, env, brain_name, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    episode = 0

    eps = eps_start  # initialize epsilon
    while True:  # loop over episodes
        env_info = env.reset(train_mode=True)[brain_name]
        episode += 1
        state = env_info.vector_observations[0]  # get the current state
        score = 0
        while True:  # loop over t
            # action = np.random.randint(action_size)        # select an action
            action = agent.act(state, eps)  # select an action
            env_info = env.step(action)[brain_name]  # send the action to the environment
            next_state = env_info.vector_observations[0]  # get the next state
            reward = env_info.rewards[0]  # get the reward
            done = env_info.local_done[0]  # see if episode has finished
            agent.step(state, action, reward, next_state, done)
            score += reward  # update the score
            state = next_state  # roll over the state to next time step
            if done:  # exit loop if episode finished
                break

        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_window)), end="")
        if episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_window)))
        if np.mean(scores_window) >= 13.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode - 100,
                                                                                         np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), CHECKPOINT_PATH)
            break

    # plot the scores
    fig = plt.figure()
    fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()


def trained_mode(agent, env, brain_name):
    agent.qnetwork_local.load_state_dict(torch.load(CHECKPOINT_PATH))
    env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
    state = env_info.vector_observations[0]  # get the current state
    score = 0  # initialize the score
    while True:
        action = agent.act(state, 0)  # select an action
        env_info = env.step(action)[brain_name]  # send the action to the environment
        next_state = env_info.vector_observations[0]  # get the next state
        reward = env_info.rewards[0]  # get the reward
        done = env_info.local_done[0]  # see if episode has finished
        score += reward  # update the score
        state = next_state  # roll over the state to next time step
        if done:  # exit loop if episode finished
            break

    print("Score: {}".format(score))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trained', help='Load a trained agent', action='store_true')
    parser.add_argument('--environment', help='Path to the environment', default="Banana_Linux/Banana.x86_64")
    args = parser.parse_args()

    env = UnityEnvironment(file_name=args.environment)

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents in the environment
    print('Number of agents:', len(env_info.agents))

    # number of actions
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)

    # examine the state space
    state = env_info.vector_observations[0]
    print('States look like:', state)
    state_size = len(state)
    print('States have length:', state_size)

    agent = Agent(state_size=state_size, action_size=action_size, seed=0)

    if args.trained:
        trained_mode(agent, env, brain_name)
    else:
        not_trained_mode(agent, env, brain_name)
    env.close()


if __name__ == '__main__':
    main()
