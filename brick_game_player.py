import tensorflow as tf
import tf_agents
import numpy as np
import math
from tf_agents.networks import actor_distribution_network, value_network
from tf_agents.agents.ppo import ppo_agent
from tf_agents.environments import tf_py_environment
from brick_game_env_custom import brick_game_env_custom
from tf_agents.trajectories import trajectory
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import time_step
from tf_agents.policies import policy_saver
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import suite_gym
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import suite_gym
from brick_game_env import brick_game_env

tf.config.threading.set_inter_op_parallelism_threads(0)
tf.config.threading.set_intra_op_parallelism_threads(0)
# 创建环境
env = brick_game_env_custom(True)
train_env = tf_py_environment.TFPyEnvironment(env)
eval_env = tf_py_environment.TFPyEnvironment(env)
# env = brick_game_env()
# train_env = tf_py_environment.TFPyEnvironment(suite_gym.wrap_env(env))
# 创建Actor和Value网络
actor_net = actor_distribution_network.ActorDistributionNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    # 指定其他参数，比如 fc_layer_params，如果你的网络需要。
)

value_net = value_network.ValueNetwork(
    input_tensor_spec=train_env.observation_spec(),
)

# 创建PPO Agent
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-3)
train_step_counter = tf.Variable(0)

agent = ppo_agent.PPOAgent(
    time_step_spec=train_env.time_step_spec(),
    action_spec=train_env.action_spec(),
    optimizer=optimizer,
    actor_net=actor_net,
    value_net=value_net,
    num_epochs=5,
    normalize_observations=False,
    train_step_counter=train_step_counter)
agent.initialize()

replay_buffer_capacity = 64 * 100  # 回放缓冲区的大小
num_epochs = 5

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,  # 通常是环境的批次大小
    max_length=replay_buffer_capacity)

# 创建检查点对象
checkpoint = tf.train.Checkpoint(agent=agent)
checkpoint_manager = tf.train.CheckpointManager(
    checkpoint, directory='./checkpoints', max_to_keep=5)

tf_policy_saver = policy_saver.PolicySaver(agent.policy)


# 训练Agent
def train_agent(num_iterations):
    checkpoint_manager.restore_or_initialize()
    for _ in range(num_epochs):
        train_env.reset()
        while not train_env.current_time_step().is_last():
            collect_step(train_env, agent.collect_policy, num_iterations)
            dataset = replay_buffer.as_dataset(
                num_parallel_calls=tf.data.AUTOTUNE,
                sample_batch_size=64,  # 每个批次样本的大小
                num_steps=4,
                single_deterministic_pass=False).prefetch(tf.data.AUTOTUNE)  # num_steps 是为 TD 或其他多步算法配置多步数
            # 计算数据集中的批次数量
            train_loss = None
            for trajectories, _ in dataset:
                # 在一个序列批次上训练代理
                train_loss = agent.train(experience=trajectories)
            checkpoint_manager.save()
            tf_policy_saver.save('saved_policy')
            print("train_loss:" + str(train_loss.loss.numpy()))


def collect_step(environment, policy, steps):
    time_step = environment.current_time_step()
    for _ in range(steps):
        action_step = policy.action(time_step)
        next_time_step = environment.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)
        replay_buffer.add_batch(traj)  # 收集数据到回放缓冲区
        time_step = next_time_step


def test_agent():
    checkpoint_manager.restore_or_initialize()
    time_step = eval_env.reset()
    total_step = 0
    success_step = 0
    while not time_step.is_last():
        action_step = agent.collect_policy.action(time_step)
        time_step: TimeStep = eval_env.step(action_step.action)
        total_step += 1
        time_step += 1

    print("total_step:" + str(total_step) + " success_step:" + str(success_step))


def convert_model_to_mobile():
    saved_model_dir = 'saved_policy'
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    # 选择优化策略，这里使用默认的优化
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # 进行转换
    tflite_model = converter.convert()
    # 保存TFLite模型
    with open(saved_model_dir + 'tflite/banzhuan_ppo_model.tflite', 'wb') as f:
        f.write(tflite_model)


if __name__ == '__main__':
    train_agent(replay_buffer_capacity)
    convert_model_to_mobile()
    test_agent()
    print('main')
