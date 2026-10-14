import numpy as np
import jpype
import os
from jpype import JArray, JInt
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step
from tf_agents.specs import tensor_spec
import random

game_grid_path = "full_grid"


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
        self.same_brick_step_count = 0
        self.from_to_same_line_step_count = 0
        self.from_to_same_line_can_move_step_count = 0
        self.to_pair_same_line_step_count = 0
        self.value_cache = []
        self._action_spec = (
            tensor_spec.BoundedTensorSpec((), np.int32, 0, self.board_size[0] * self.board_size[1] - 1),
            tensor_spec.BoundedTensorSpec((), np.int32, 0, self.board_size[0] * self.board_size[1] - 1),
            tensor_spec.BoundedTensorSpec((), np.int32, 0, self.board_size[0] * self.board_size[1] - 1))

        self._observation_spec = array_spec.BoundedArraySpec(shape=(self.board_size[0]*self.board_size[1],),
                                                             dtype=np.float32,  name='observation')
        size =self.board_size[0] * self.board_size[1]
        items_spec = array_spec.BoundedArraySpec(shape=(size, 3), dtype=np.int32, name='itmes')
        mask_spec = array_spec.BoundedArraySpec(shape=(size,), dtype=np.int32, name='mask')
        status_spec = array_spec.BoundedArraySpec(shape=(2,), dtype=np.int32, name='status')
        self.state_spec = {"status":status_spec, "mask":mask_spec, "items":items_spec}
        self.game_list = {}
        self.success_game_steps = []
        self.grid = None
        self._state = None
        jpype.startJVM(classpath=["lib2.jar"])
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
        # key = list(self.game_list.keys())[0]
        key, value = random.choice(list(self.game_list.items()))
        self.current_game_file_name = key
        self.grid = np.copy(value)
        print("reset -> total game count：" + str(self.game_count) + " " + key)
        print("reset -> grid：\n" + str(self.game_list[key]))
        self._state = self.to_state(self.grid)
        return time_step.restart(self._state) #todo

    def to_state(self ,grid):
        size = self.board_size[0]*self.board_size[1]
        brick_num = np.count_nonzero(grid)
        items = np.empty((size, 3),dtype= np.int32)
        i = 0
        for index, value in np.ndenumerate(grid):
            items[i,0] =index[0]
            items[i, 1] = index[1]
            items[i, 2] = value
            i = i+1

        status = np.array([size,brick_num],dtype= np.int32)
        mask = np.where(grid != 0, 1, grid).flatten()
        return {'status':status,
                'mask':mask,
                'items':items

        }


    def _step(self, action):
        self.step_count_current += 1
        self.step_count_all += 1
        idx1, idx2, idx3 = action
        reward = 0
        python_action = [idx1 // self.board_size[1], idx1 % self.board_size[1],
                         idx2 // self.board_size[1], idx2 % self.board_size[1],
                         idx3 // self.board_size[1], idx3 % self.board_size[1]]
        java_action = JArray(JInt)(python_action)

        java_array = JArray(JArray(JInt))(self.grid.tolist())

        value = self.brick_game.actionStep(java_array, java_action)
        self.grid = np.array(java_array)
        GAME_SUCCESS = 2  # 游戏胜利
        ACTION_SUCCESS = 1  # 消除成功
        ACTION_FAIL = -1  # 无法移动
        GAME_FAIL = -2  # 游戏失败
        FROM_PAIR_SAME_BRICK = -3  # from 和 pair是相同的砖块
        FROM_TO_SAME_LINE = -4  # from 和 to在一条直线上，但是不可移动
        FROM_TO_SAME_LINE_CAN_MOVE = -5  # from 和to是可移动的
        TO_PAIR_SAME_LINE = -6  # // to和pair在一条直线上，但是中间有砖块
        done = False
        if value == GAME_SUCCESS:
            done = True
            reward = 100
            np.savetxt(f'game_steps_{self.current_game_file_name}.csv', self.success_game_steps, delimiter=',',
                       fmt='%d')
        elif value == ACTION_SUCCESS:
            done = False
            reward = 20
            self.success_step_current += 1
            self.success_game_steps.append(python_action)
        elif value == ACTION_FAIL:
            done = False
            reward = -1
        elif value == GAME_FAIL:
            done = True
            reward = -2
        elif value == FROM_PAIR_SAME_BRICK:
            done = False
            reward = 5
            self.same_brick_step_count += 1
        elif value == FROM_TO_SAME_LINE:
            done = False
            reward = 6
            self.from_to_same_line_step_count += 1;
        elif value == FROM_TO_SAME_LINE_CAN_MOVE:
            done = False
            reward = 7
            self.from_to_same_line_can_move_step_count += 1
        elif value == TO_PAIR_SAME_LINE:
            done = False
            reward = 8
            self.to_pair_same_line_step_count += 1

        if done:
            if reward > 0:
                print("game success! game num:" + str(self.game_count) + " step count:" + str(
                    self.step_count_all) + " total step count:" + str(self.step_count_all))
            else:
                print("game fail! game num:" + str(self.game_count) + " step count:" + str(
                    self.step_count_all) + " total step count:" + str(self.step_count_all))
            self._state = self.to_state(self.grid)
            return time_step.termination(self._state, reward)
        else:
            if self.step_count_current % (64*20) == 0:
                print(
                    "game is running! game num:" + str(self.game_count) + " step count:" + str(
                        self.step_count_current) + " brick num:" + str(
                        np.count_nonzero(self.grid)) + " success step count:" + str(self.success_step_current) +
                    " same_brick_step_count:" + str(
                        self.same_brick_step_count) + " from_to_same_line_step_count:" + str(
                        self.from_to_same_line_step_count) + " from_to_same_line_can_move_step_count:" + str(
                        self.from_to_same_line_can_move_step_count) + " to_pair_same_line_step_count:" + str(
                        self.to_pair_same_line_step_count))
                self.value_cache.clear()
                self.value_cache.append(self.same_brick_step_count)
                self.value_cache.append(self.from_to_same_line_step_count)
                self.value_cache.append(self.from_to_same_line_can_move_step_count)
                self.value_cache.append(self.to_pair_same_line_step_count)
                self.same_brick_step_count = 0
                self.from_to_same_line_step_count = 0
                self.from_to_same_line_can_move_step_count = 0
                self.to_pair_same_line_step_count = 0
            self._state = self.to_state(self.grid)
            return time_step.transition(self._state, reward)

    def render(self, mode='human'):
        print('render')
        print(self.grid)

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
    env = brick_game_env_custom(True)
    env.reset()
    print(str(env.to_state(env.grid)))
