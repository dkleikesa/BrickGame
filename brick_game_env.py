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
        jpype.startJVM(classpath=["/Users/jianzhangli/src/BrickGame/lib2.jar"])
        self.brick_game = jpype.JClass('com.sogou.iot.lib2.BrickGameTipHelper')


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

    def run(self):
        array =  [[1, 0, 0, 1, 2, 1, 0, 1],
                 [4, 1, 0, 1, 1, 3, 1, 0],
                [1, 0, 1, 12, 1, 1, 1, 1],
                [13, 14, 15, 16, 1, 1, 1, 1]]

        result = self.brick_game.findMoves(array)
        python_list = [[int(x) for x in ja] for ja in result]

        # 现在 python_list 包含了 Python 中的整数列表，您可以打印它
        print(python_list)



if __name__ == '__main__':
    env = brick_game_env()
    env.run()