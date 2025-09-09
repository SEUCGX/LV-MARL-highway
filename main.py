import copy
import os
import time
import torch
from matplotlib import pyplot as plt
from scipy.io import savemat
from torch.utils.tensorboard import SummaryWriter
from Config import Config
from SACagent import SACAgent
from base_function import my_env, set_seed
from util import Evaluation

# 训练的时候控制20个车，评估的时候控制1000个
seed = 1902
if __name__ == '__main__':
    # 配置
    conf = Config(seed)
    conf.train_step = 1
    conf.target_entropy = 0.1
    conf.gamma = 0.9
    conf.n_epochs = 10000
    conf.IfConv = True
    conf.use_rule = False
    conf.norm_type = "layer_norm"  # 控制是否使用归一化层
    env_info = {"controlled_vehicles": 20, "n_actions": 9, "obs_shape": 4 * 3 * 7, "multiprocess": False,
                "other_vehicles": True, "other_vehicles_number": 1}
    conf.set_env_info(env_info)
    set_seed(conf.seed)
    env = my_env(conf)
    conf_test = Config(seed)
    env_info = {"controlled_vehicles": 1000, "n_actions": 9, "obs_shape": 4 * 3 * 7, "multiprocess": False,
                "other_vehicles": True, "other_vehicles_number": 1, "video_address": "sac1000_Result.gif",
                "data_address": "sac1000_Result.mat", "test_time": True}
    conf_test.set_env_info(env_info)
    env_eval = my_env(conf_test)
    env_eval.chang_test_time(True)  # 这样才有动画
    env_eval.reset(seed=1)
    ####
    start_time = time.time()
    obs, info = env.reset(seed=seed)
    # 智能体, 可以选择不一样
    agent = SACAgent(conf)
    # writer = SummaryWriter("runs/dqn_experiment_" + time.strftime("%Y%m%d-%H%M%S"))
    episode_data = {"Episode": [], "Win_rate": [], "Total_Reward": []}
    evaluation_data = {"Speed": [], "Safe_rate": [], "Total_Reward": []}
    result_dir = "test_model/proposed2"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    print("Training begin!")
    max_total_rewards = -100
    max_rate = 0
    for i_episode in range(conf.n_epochs):
        observations, info = env.reset()
        total_reward = 0
        truncated = False
        win = []
        while not truncated:
            actions = []
            for observation in observations:
                action = agent.select_action(observation, False)
                actions.append(action)  # 不用列表生成式，方便调参
            next_observations, rewards, truncated, info = env.step(tuple(actions))
            dones = info["agents_dones"]
            for observation, next_observation, reward, action, done in zip(observations, next_observations, rewards, actions, dones):
                agent.push_memory(observation, action, reward, next_observation, done)

            observations = copy.deepcopy(env.get_obs())  # 不能直接copy，因为会清理车
            agent.mul_optimize_model()
            total_reward += info["global_reward"]
            win.append(info["win_rate"])
        win_rate = sum(win) / len(win)
        agent.update_epsilon()

        if i_episode % agent.target_update == 0:
            agent.update_target_network()

        if (i_episode + 0) % conf.evaluate_per_epoch == 0:

            total_rewards, steps, speeds, safe_rate = Evaluation(agent, env_eval)
            evaluation_data["Speed"].append(sum(speeds) / len(speeds))
            evaluation_data["Safe_rate"].append(safe_rate)
            evaluation_data["Total_Reward"].append(sum(total_rewards) / len(total_rewards))
            print(
                f"Episode {i_episode}, Test Speed: {speeds}, Test Reward: {total_rewards}, test safe rate: {safe_rate}")
            print(f"Episode {i_episode}, train Reward: {total_reward / 20}, train safe rate: {win_rate}")
            tr = sum(total_rewards) / len(total_rewards)
            if tr > max_total_rewards:
                env_eval.save()  # 相当于执行一些保存的操作
                max_total_rewards = tr
                max_rate = safe_rate
            # 恢复原样
            env_eval.clear_frames()
            end_time = time.time()
            run_time = end_time - start_time
            start_time = end_time
            print(f"运行时间: {run_time}s当前算法平均奖励值：{max_total_rewards:.4f},安全率{max_rate:.3f}")
        episode_data["Episode"].append(i_episode)
        episode_data["Win_rate"].append(win_rate)
        episode_data["Total_Reward"].append(total_reward)

    print(f"最终所提出算法平均奖励值：{max_total_rewards:.4f},安全率{max_rate:.3f}")
    full_path = __file__
    filename_with_extension = os.path.basename(full_path)
    filename_without_extension = os.path.splitext(filename_with_extension)[0]

    # 生成并保存 episode_data
    episode_dir = f"0318/episode_results"
    os.makedirs(episode_dir, exist_ok=True)  # 自动创建目录（如果不存在）
    data_name = f"{episode_dir}/{filename_without_extension}_episode_data.mat"
    savemat(data_name, episode_data)
    print(f"OK! {data_name}")

    # 生成并保存 evaluation_data
    evaluation_dir = f"0318/evaluation_data"
    os.makedirs(evaluation_dir, exist_ok=True)  # 自动创建目录（如果不存在）
    data_name = f"{evaluation_dir}/{filename_without_extension}_evaluation_data.mat"
    savemat(data_name, evaluation_data)
    print(f"OK! {data_name}")
