import numpy as np
import gym
from gym import spaces
import jpype

class brick_game_env(gym.Env):
    def __init__(self):
        super(brick_game_env, self).__init__()
        self.board_size = (14, 10)
        self.action_space = spaces.MultiDiscrete([self.board_size[0] * self.board_size[1]] * 2)
        self.observation_space = spaces.Box(low=0, high=60, shape=self.board_size, dtype=np.int32)
        self.state = None
        self.reset()
        jpype.startJVM()


    def reset(self):
        self.state = np.random.randint(1, 61, size=self.board_size) # todo
        return self.state


    def step(self, action):
        idx1, idx2 = action
        reward = 0
        done = False
        info = {}

        if self.can_remove(idx1, idx2):
            reward = 1
            self.state[idx1 // self.board_size[1]][idx1 % self.board_size[1]] = 0
            self.state[idx2 // self.board_size[1]][idx2 % self.board_size[1]] = 0
        else:
            reward = -1

        if np.all(self.state == 0):
            done = True

        return self.state, reward, done, info

    def can_remove(self, idx1, idx2):
        # 这里应该实现检查两个图标是否可以消除的逻辑
        return True

    def render(self, mode='human'):
        print(self.state)

    def close(self):
        jpype.shutdownJVM()
        print("close")

        
if __name__ == '__main__':
    env = brick_game_env()