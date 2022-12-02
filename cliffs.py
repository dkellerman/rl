#!/usr/bin/env python

'''
Cliffs is a simple game to demo RL algorithms. The agent starts at the bottom left
and must find a way to the goal on the bottom right, by walking around the cliffs.

----------
----------
----------
A////////G

'''

import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

# game config
size = (10, 4)
width, height = size
cliffs = tuple([(x, height - 1) for x in range(1, width - 1)])
goal = (width - 1, height - 1)
action_map = dict(up=(0, -1), down=(0, 1), left=(-1, 0), right=(1, 0))
rewards = dict(cliff=-100, step=-1, goal=0)
max_steps = 200
verbose = False

# game state
agent = None
history = None
step_ct = None
random_ct = None
Q = defaultdict(lambda: 0)

# hyperparameters
learning_rate = .5
epsilon = .1
discount = 1.0
use_sarsa = True


def debug(*args):
    if verbose:
        print(*args)


def reset():
    debug('resetting')
    global agent, history, step_ct, random_ct
    agent = (0, height - 1)
    history = [agent]
    step_ct = 0
    random_ct = 0


def get_state():
    return agent


def log_state():
    print('State:')
    for y in range(height):
        row = []
        for x in range(width):
            sq = (x, y)
            if sq == agent:
                row.append('A')
            elif sq in history:
                row.append('*')
            elif sq == goal:
                row.append('G')
            elif sq in cliffs:
                row.append('/')
            else:
                row.append('-')
        print(''.join(row))

    if agent == goal:
        print('Won!')
    elif agent in cliffs:
        print('Lost!')
    elif step_ct > max_steps:
        print('Timed out!')
    else:
        print('Still playing...')


def get_reward():
    if agent == goal:
        return rewards['goal']
    elif agent in cliffs:
        return rewards['cliff']
    else:
        return rewards['step']


def is_done():
    return (step_ct > max_steps) or (agent in cliffs) or (agent == goal)


def get_action_space():
    actions = []
    for action, (dx, dy) in action_map.items():
        x, y = agent
        if 0 <= x + dx < width and 0 <= y + dy < height:
            actions.append(action)
    return actions


def get_policy_action(state, use_epsilon=True, lookahead=False):
    global random_ct
    actions = get_action_space()
    scores = [Q[(state, a)] for a in actions]
    if use_epsilon and (random.random() < epsilon):
        idx = random.choice([i for i, _ in enumerate(scores)])
        action = actions[idx]
        if not lookahead:
            random_ct += 1
    else:
        score = random.choice([s for s in scores if s == max(scores)])
        action = actions[scores.index(score)]
    return action


def step():
    global step_ct, agent, Q
    state = get_state()
    action = get_policy_action(state)
    qval = Q[(state, action)]

    # log all action values
    debug('step', step_ct + 1, state, 'action space:')
    for a in get_action_space():
        debug('\t', a, Q[(state, a)], '[*]' if a == action else '')

    # take action, make observation
    dx, dy = action_map.get(action)
    agent = (agent[0] + dx, agent[1] + dy)
    observation, reward, done = get_state(), get_reward(), is_done()

    # log action result
    debug('\t', 'taken:', action, 'reward:', reward, 'done:', done)

    # look ahead to update Q
    if not done:
        next_action = get_policy_action(
            observation, use_epsilon=use_sarsa, lookahead=True)
        next_qval = Q[(observation, next_action)]
    else:
        next_qval = 0

    Q[(state, action)] = qval + (learning_rate *
                                 (reward + ((discount * next_qval) - qval)))
    debug('\tupdated qval:', Q[(state, action)])

    step_ct += 1
    history.append(agent)

    return reward, done


def run_episode():
    reset()
    tot_rewards = 0
    while True:
        reward, done = step()
        tot_rewards += reward
        if done:
            break
    return tot_rewards


def train(episode_ct):
    episode_rewards = []
    tot_steps = 0
    tot_random = 0
    for _ in tqdm(range(episode_ct)):
        ep_tot_rewards = run_episode()
        episode_rewards.append(ep_tot_rewards)
        tot_steps += step_ct
        tot_random += random_ct
    return dict(
        rewards=episode_rewards,
        tot_steps=tot_steps,
        tot_random=tot_random,
    )


def log_training_stats(stats):
    print('tot steps', stats['tot_steps'])
    print('tot random', stats['tot_random'],
          stats['tot_random'] / stats['tot_steps'])
    print('Q s/a pair count', len(Q.values()))

    y = np.array(stats['rewards'])
    y = np.average(y.reshape(-1, 100), axis=1)  # average over n episodes
    x = np.arange(len(y))

    plt.plot(x, y)
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.ylim(-140, 0)
    plt.show()


def play():
    # temporarily remove randomization for single play and increase logging
    global epsilon, verbose
    _old_e = epsilon
    epsilon = 0
    verbose = True

    run_episode()

    epsilon = _old_e
    verbose = False


if __name__ == '__main__':
    episode_ct = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    stats = train(episode_ct)
    play()
    log_state()
    log_training_stats(stats)
