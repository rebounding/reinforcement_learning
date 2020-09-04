import gym
import argparse

## random agent
class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('game', nargs="?", default="CartPole-v0")
    args = parser.parse_args()

    env = gym.make(args.game)
    num_episodes = 5
    num_maxstep = 100

    agent = RandomAgent(env.action_space)

    reward = 0
    done = False

    for i_episode in range(num_episodes):
        observation = env.reset()
        for t in range(num_maxstep):
            env.render()
            action = agent.act(observation, reward, done)
            observation, reward, done, info = env.step(action)
            print('episode {}-step {}, taking action {}, observation {}'.format(i_episode, t, action, observation))
        env.close()
