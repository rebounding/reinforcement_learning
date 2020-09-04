import gym
import argparse

import numpy as np

n_states = 40 # magic number for q-table size

def obs_to_state(env, obs):
    '''
    Maps an observation to state
    '''
    # quantify the continous state space into discrete space
    env_low = env.observation_space.low
    env_high = env.observation_space.high
    env_dx = (env_high - env_low) / n_states
    a = int((obs[0] - env_low[0])/env_dx[0]) # position
    b = int((obs[1] - env_low[1])/env_dx[1]) # velocity
    return a, b

def egreedy_policy(q_values, env, obs, epsilon=0.1):
    '''
    Choose an action based on a epsilon greedy policy.
    A random action is selected with epsilon probability, else select the best action.
    '''
    if np.random.random() < epsilon:
        return np.random.choice(3) # magic number for action space
    else:
        a, b = obs_to_state(env, obs)
        return np.argmax(q_values[a][b])

def q_learning(env, num_episodes=5000, render=True, exploration_rate=0.1,
        learning_rate=0.5, gamma=1.0):
    q_values = np.zeros((n_states, n_states, 3))
    ep_rewards = []

    for i in range(num_episodes):
        state = env.reset()
        done = False
        reward_sum = 0

        ## eta: learning rate is decreased at each episode
        eta = max(0.003, learning_rate * (0.85 ** (i//100)))

        while not done:
            # Choose action
            action = egreedy_policy(q_values, env, state, exploration_rate)
            # Do the action
            next_state, reward, done, _ = env.step(action)
            reward_sum += reward
            # Update q_values
            a_, b_ = obs_to_state(env, next_state)
            td_target = reward + gamma * np.max(q_values[a_][b_])
            a, b = obs_to_state(env, state)
            td_error = td_target - q_values[a][b][action]
            q_values[a][b][action] += eta * td_error
            # Update state
            state = next_state

            if render:
                env.render()
        
        if i % 200 == 0:
            print('Iteration #%d -- Total reward = %d.' %(i + 1, reward_sum))

        ep_rewards.append(reward_sum)

    return ep_rewards, q_values

def run_episode(env, policy=None, render=False):
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    for _ in range(10000):
        if render:
            env.render()
        if policy is None:
            action = env.action_space.sample()
        else:
            a,b = obs_to_state(env, obs)
            action = policy[a][b]
        obs, reward, done, _ = env.step(action)
        total_reward += 1.0 ** step_idx * reward
        step_idx += 1
        if done:
            break
    return total_reward

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('game', nargs="?", default="MountainCar-v0")
    args = parser.parse_args()

    env = gym.make(args.game)
    env.seed(0)
    np.random.seed(0)

    q_learning_rewards, q_values = q_learning(env, num_episodes=1200, render=False)

    solution_policy = np.argmax(q_values, axis=2)
    solution_policy_scores = [run_episode(env, solution_policy, False) for _ in range(100)]
    print("Average score of solution = ", np.mean(solution_policy_scores))
    for _ in range(2):
        run_episode(env, solution_policy, True)
    env.close()
