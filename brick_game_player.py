import tensorflow as tf
import tf_agents
import numpy as np
from tf_agents.networks import actor_distribution_network, value_network
from tf_agents.agents.ppo import ppo_agent
from tf_agents.environments import tf_py_environment
from brick_game_env_custom import brick_game_env_custom
from tf_agents.trajectories import trajectory
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import time_step as ts
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import suite_gym
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import suite_gym
from brick_game_env import brick_game_env

# 创建环境
train_env = brick_game_env_custom(True)
train_env = tf_py_environment.TFPyEnvironment(train_env)
#env = brick_game_env()
#train_env = tf_py_environment.TFPyEnvironment(suite_gym.wrap_env(env))
# 创建Actor和Value网络
actor_net = actor_distribution_network.ActorDistributionNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    # 指定其他参数，比如 fc_layer_params，如果你的网络需要。
)

value_net = value_network.ValueNetwork(
    input_tensor_spec = train_env.observation_spec(),
  )



# 创建PPO Agent
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-3)
train_step_counter = tf.Variable(0)

agent = ppo_agent.PPOAgent(
    time_step_spec=train_env.time_step_spec(),
    action_spec=train_env.action_spec(),
    optimizer = optimizer,
    actor_net = actor_net,
    value_net = value_net,
    num_epochs = 5,
    normalize_observations = False,
    train_step_counter  = train_step_counter)
agent.initialize()


replay_buffer_capacity = 1000  # 回放缓冲区的大小

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,  # 通常是环境的批次大小
    max_length=replay_buffer_capacity)

# 训练Agent
def train_agent(num_iterations):
    train_env.reset()
    collect_step(train_env, agent.collect_policy)
    iterator = iter(replay_buffer.as_dataset(
        num_parallel_calls=4,
        sample_batch_size=64,  # 每个批次样本的大小
        num_steps=2).prefetch(3))  # num_steps 是为 TD 或其他多步算法配置多步数

    for _ in range(num_iterations):
        # 从回放缓冲区获得一个批次
        trajectories, _ = next(iterator)
        # 在一个序列批次上训练代理
        train_loss = agent.train(experience=trajectories)

    # for _ in range(num_iterations):
    #     traj = collect_step(train_env, agent.collect_policy)
    #     trajs.append(traj)
    #     train_loss = agent.train(traj)
        # step = train_step_counter.numpy()
        # if step % 100 == 0:
        #         print('step = {0}: loss = {1}'.format(step, train_loss.loss))


def collect_step(environment, policy):
    time_step = environment.current_time_step()
    for _ in range(replay_buffer_capacity):
        action_step = policy.action(time_step)
        next_time_step = environment.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)
        replay_buffer.add_batch(traj)  # 收集数据到回放缓冲区
        time_step = next_time_step
    # time_step = environment.current_time_step()
    # action_step = policy.action(time_step)
    # next_time_step = environment.step(action_step.action)
    # traj = trajectory.from_transition(time_step, action_step, next_time_step)
    # return traj
# 训练模型
train_agent(1000)
#
# # 测试模型
# time_step = eval_env.reset()
# while not time_step.is_last():
#     action_step = agent.policy.action(time_step)
#     time_step = eval_env.step(action_step.action)
#     print(time_step.observation)


# if __name__ == '__main__':
#     for root, dirs, files in os.walk(game_grid_path):
#         for file in files:
#             file_path = os.path.join(root, file)
#             print("File:", str(read_game(file_path)))
#
#     print('main')

