import os
import tensorflow as tf
from tf_agents.environments import suite_gym
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import value_network
from tf_agents.agents.ppo import ppo_agent
from tf_agents.utils import common
from tf_agents.environments import tf_py_environment

game_grid_path = "/Users/jianzhangli/src/BrickGame/full_grid"


def read_game(path):
    array = []
    with open(path, "r") as file:
        for line in file:
            row = [int(x) for x in line.strip().split(",")]
            array.append(row)
    return array


def write_game(path,array):
    with open(path, "w") as file:
        for row in array:
            file.write(",".join(str(x) for x in row) + "\n")

# 创建环境
env = brick_game_env()
train_env = tf_py_environment.TFPyEnvironment(suite_gym.wrap_env(env))
eval_env = tf_py_environment.TFPyEnvironment(suite_gym.wrap_env(env))

# 创建Actor和Value网络
actor_net = actor_distribution_network.ActorDistributionNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    fc_layer_params=(100,))

value_net = value_network.ValueNetwork(
    train_env.observation_spec(),
    fc_layer_params=(100,))

# 创建PPO Agent
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)
train_step_counter = tf.Variable(0)

agent = ppo_agent.PPOAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    optimizer,
    actor_net=actor_net,
    value_net=value_net,
    num_epochs=10,
    train_step_counter=train_step_counter)
agent.initialize()

# 训练Agent
def train_agent(num_iterations):
    time_step = train_env.reset()
    for _ in range(num_iterations):
        collect_step(train_env, agent.policy)
        experience = agent.collect_data_spec.trajectory
        train_loss = agent.train(experience)

def collect_step(environment, policy):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)
    return traj

# 训练模型
train_agent(1000)

# 测试模型
time_step = eval_env.reset()
while not time_step.is_last():
    action_step = agent.policy.action(time_step)
    time_step = eval_env.step(action_step.action)
    print(time_step.observation)


if __name__ == '__main__':
    for root, dirs, files in os.walk(game_grid_path):
        for file in files:
            file_path = os.path.join(root, file)
            print("File:", str(read_game(file_path)))

    print('main')

