import gym
import argparse

from collections import deque
import random
import abc

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class DoubleDeepQNetwork:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # hyperparameters
        self.gamma = 0.95  # discount rate on future rewards
        self.epsilon = 1.0  # exploration rate
        self.epsilon_decay = 0.995  # the decay of epsilon after each training batch
        self.epsilon_min = 0.1  # the minimum exploration rate
        self.batch_size = 32  # size of mini batch

        # deep q model
        self.model = self.build_model()
        self.model_target = self.build_model()
        self.memory = deque(maxlen=2000)

    @abc.abstractmethod
    def build_model(self):
        return None

    def select_action(self, state, is_training=True):
        if is_training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        return np.argmax(self.model.predict(state)[0])

    def record(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0

        mini_batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in mini_batch:
            #print("state: {} | action: {} | reward: {} | next_state: {} | done: {}"
                #.format(state, action, reward, next_state, done))
            y = self.model.predict(state)
            q = self.model_target.predict(next_state)
            next_action = np.argmax(self.model.predict(next_state), axis=1)

            target = reward
            if not done:
                target = (reward + self.gamma * q[0][next_action])
            y[0][action] = target
            self.model.fit(state, y, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class CartPoleAgent(DoubleDeepQNetwork):

    def build_model(self):
        model = Sequential()
        model.add(Dense(12, activation='relu', input_dim=self.state_size))
        model.add(Dense(12, activation='relu'))
        model.add(Dense(self.action_size))
        model.compile(Adam(lr=0.001), 'huber_loss')

        return model

def run_episodes(env=None, agent=None, num_episodes=1000, num_maxstep=200,
        update_target_network=16, is_training=False):
    assert(env != None and agent != None)
    episode_reward_history = []

    for i_episode in range(num_episodes):
        episode_reward = 0
        step_count = 0
        state = env.reset()
        state = state.reshape(1, env.observation_space.shape[0])
        for t in range(num_maxstep):
            if not is_training:
                env.render()
            action = agent.select_action(state, is_training)
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.reshape(1, env.observation_space.shape[0])

            if is_training:
                agent.record(state, action, reward, next_state, done)

            episode_reward += reward
            step_count += 1
            state = next_state
            if done:
                break
        
        if is_training:
            # train the agent based on a sample of past experiences
            agent.replay()

            # update running reward to check condition for solving
            episode_reward_history.append(episode_reward)
            if len(episode_reward_history) > 10:
                del episode_reward_history[:1]
            running_reward = np.mean(episode_reward_history)

            if running_reward > 195:  # condition to consider the task solved
                print("Solved at episode {}!".format(i_episode + 1))
                break

        if i_episode % update_target_network == 0:
            # update the the target network with new weights
            agent.model_target.set_weights(agent.model.get_weights())
            # Log details
            template = "running reward: {:.2f} at episode {}, step count {}"
            print(template.format(episode_reward, i_episode + 1, step_count))

        print('episode: {}/{} | reward: {} | step: {} | epsilon: {:.3f}'
                .format(i_episode + 1, num_episodes, episode_reward, step_count, agent.epsilon))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('game', nargs="?", default="CartPole-v0")
    args = parser.parse_args()

    env = gym.make(args.game)
    env.seed(42)
    np.random.seed(42)

    agent = CartPoleAgent(4, 2) # CartPole's states and actions
    run_episodes(env, agent, num_episodes=2000, is_training=True)
    run_episodes(env, agent, num_episodes=5, is_training=False)

    env.close()
