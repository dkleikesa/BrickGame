import numpy as np
import jpype
import os
from jpype import JArray, JInt
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step
from tf_agents.specs import tensor_spec
import random

game_grid_path = "/Users/jianzhangli/src/BrickGame/full_grid"


class brick_game_env_custom(py_environment.PyEnvironment):
    def __init__(self, train):
        super(brick_game_env_custom, self).__init__()
        self.train = train
        self.board_size = (14, 10)
        self.step_count_all = 0
        self.game_count = 0
        self.current_game_file_name = None
        self.step_count_current = 0
        self.success_step_current = 0
        self._action_spec = (tensor_spec.BoundedTensorSpec((),np.int32,0,self.board_size[0] * self.board_size[1] - 1),
                             tensor_spec.BoundedTensorSpec((),np.int32,0,self.board_size[0] * self.board_size[1] - 1),
                             tensor_spec.BoundedTensorSpec((),np.int32,0,self.board_size[0] * self.board_size[1] - 1))

        self._observation_spec = array_spec.BoundedArraySpec(shape=(self.board_size[0] * self.board_size[1],),
                                                             dtype=np.int32, minimum=0, maximum=59, name='observation')
        self.game_list = {}
        self.success_game_steps = []
        self._state = None
        jpype.startJVM(classpath=["/Users/jianzhangli/src/BrickGame/lib2.jar"])
        self.brick_game = jpype.JClass('com.sogou.iot.lib2.BrickGameTipHelper')
        for root, dirs, files in os.walk(game_grid_path):
            for filename in files:
                filepath = os.path.join(root, filename)
                if filepath.endswith(".csv"):
                    self.game_list[filename] = np.genfromtxt(filepath, dtype=np.int32, delimiter=',')
                    print("read game:" + filepath)

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self.game_count += 1
        self.step_count_current = 0
        self.success_step_current = 0
        self.success_game_steps = []
        #key = list(self.game_list.keys())[0]
        key, value = random.choice(list(self.game_list.items()))
        self.current_game_file_name = key
        self._state = np.copy(value.flatten())
        print("reset -> total game count：" + str(self.game_count) +" "+key)
        print("reset -> grid：\n" + str(self.game_list[key]))
        return time_step.restart(self._state)

    def _step(self, action):
        self.step_count_current += 1
        self.step_count_all += 1
        idx1, idx2, idx3 = action
        reward = 0
        python_action = [idx1 // self.board_size[1], idx1 % self.board_size[1],
                         idx2 // self.board_size[1], idx2 % self.board_size[1],
                         idx3 // self.board_size[1], idx3 % self.board_size[1]]
        java_action = JArray(JInt)(python_action)

        state_2d = self._state.reshape(self.board_size)
        java_array = JArray(JArray(JInt))(state_2d.tolist())

        value = self.brick_game.actionStep(java_array, java_action)
        self._state = np.array(java_array).flatten()
        # int actionStep(int[][] grid, int[] action)
        # 2：游戏胜利 , 1：消除成功 ,-1：无法移动 , -2：游戏失败
        if value == 2:
            done = True
            reward = self.step_count_current / 1000+20
            np.savetxt(f'game_steps_{self.current_game_file_name}.csv', self.success_game_steps, delimiter=',',fmt='%d')
        elif value == 1:
            done = False
            reward = self.step_count_current / 1000
            self.success_step_current += 1
            self.success_game_steps.append(python_action)
        elif value == -1:
            done = False
            reward = -(self.step_count_current / 1000)
        else:
            done = True
            reward = -(self.step_count_current / 1000)-20

        if done:
            if reward > 0:
                print("game success! game num:" + str(self.game_count) + " step count:" + str(
                    self.step_count_all) + " total step count:" + str(self.step_count_all))
            else:
                print("game fail! game num:" + str(self.game_count) + " step count:" + str(
                    self.step_count_all) + " total step count:" + str(self.step_count_all))
            return time_step.termination(self._state, reward)
        else:
            if self.step_count_current % 1000 == 0:
                print(
                    "game is running! game num:" + str(self.game_count) + " step count:" + str(
                        self.step_count_current) + " brick num:" + str(
                       np.count_nonzero(self._state)) + " success step count:" + str(self.success_step_current))
            return time_step.transition(self._state, reward)

    def render(self, mode='human'):
        print('render')
        print(self._state)

    def close(self):
        jpype.shutdownJVM()
        print("close")

    def run(self):
        array = [[1, 0, 0, 1, 2, 1, 0, 1],
                 [4, 1, 0, 1, 1, 3, 1, 0],
                 [1, 0, 1, 12, 1, 1, 1, 1],
                 [13, 14, 15, 16, 1, 1, 1, 1]]

        result = self.brick_game.findMoves(array)
        python_list = [[int(x) for x in ja] for ja in result]

        # 现在 python_list 包含了 Python 中的整数列表，您可以打印它
        print(python_list)


if __name__ == '__main__':
    env = brick_game_env_custom()
    env.run()
