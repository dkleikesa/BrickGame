import numpy as np
import jpype
import os
import gym
from gym import spaces
from jpype import JArray, JInt, startJVM, getDefaultJVMPath

game_grid_path = "/Users/jianzhangli/src/BrickGame/full_grid"

class brick_game_env(gym.Env):
    def __init__(self):
        super(brick_game_env, self).__init__()
        self.board_size = (14, 10)
        self.action_space =  spaces.Tuple((
    spaces.Discrete(self.board_size[0] * self.board_size[1]),
    spaces.Discrete(self.board_size[0] * self.board_size[1]),
    spaces.Discrete(self.board_size[0] * self.board_size[1])
))
        self.observation_space = spaces.Box(low=0, high=59, shape=(self.board_size[0] * self.board_size[1],), dtype=np.int32)
        self.game_list = []
        self.state = None
        self.reset()
        jpype.startJVM(classpath=["/Users/jianzhangli/src/BrickGame/lib2.jar"])
        self.brick_game = jpype.JClass('com.sogou.iot.lib2.BrickGameTipHelper')


        def read_game(path):
            array = []
            with open(path, "r") as file:
                for line in file:
                    row = [int(x) for x in line.strip().split(",")]
                    array.append(row)
            return array

        def write_game(path, array):
            with open(path, "w") as file:
                for row in array:
                    file.write(",".join(str(x) for x in row) + "\n")


    def reset(self):
        # 遍历目录下的所有文件和子目录
        for root, dirs, files in os.walk(game_grid_path):
            for filename in files:
                filepath = os.path.join(root, filename)
                if filepath.endswith(".csv"):
                    self.game_list.append(np.genfromtxt(filepath,dtype=np.int32, delimiter=','))
                    print(filepath)
        self.state = np.copy(self.game_list[0].flatten())
        return self.state


    def step(self, action):
        idx1, idx2,idx3 = action
        reward = 0
        done = False
        info = {}
        python_action = [idx1 // self.board_size[1], idx1 % self.board_size[1],
                       idx2 // self.board_size[1],idx2 % self.board_size[1],
                       idx3 // self.board_size[1], idx3 % self.board_size[1]]
        java_action = JArray(JInt)(python_action)

        state_2d = self.state.reshape(self.board_size)
        java_array = JArray(JArray(JInt))(state_2d.tolist())

        value = self.brick_game.actionStep(java_array,java_action)
        self.state =np.array(java_array).flatten()
        # int actionStep(int[][] grid, int[] action)
        #2：游戏胜利 , 1：消除成功 ,-1：无法移动 , -2：游戏失败
        if value == 2:
            done = True
            reward = 2
        elif value == 1:
            done = False
            reward = 1
        elif value == -1:
            done = False
            reward = -1
        else:
            done = True
            reward = -2

        return self.state, reward, done, info

    def render(self, mode='human'):
        print('render')
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