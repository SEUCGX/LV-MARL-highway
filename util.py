import copy
import imageio
import numpy as np

def Evaluation(agent, env):
    total_rewards = []
    steps = []
    speeds = []
    agent_number = 0
    crash_number = 0  # 算安全率的
    for i_episode in range(agent.conf.evaluate_epoch):
        observations, info = env.reset(seed=i_episode)  # reset不会清空之前保存的动画！！！
        total_reward = 0
        truncated = False
        step = 0
        speed = 0
        agent_number += len(observations)
        frames = []
        while not truncated:
            actions = []
            for observation in observations:
                action_mask = np.ones(agent.conf.n_actions)
                action = agent.select_action(observation, True, action_mask)
                actions.append(action)
            next_observations, rewards, truncated, info = env.step(tuple(actions))
            # frames.append(env.render())
            dones = info["agents_dones"]
            for observation, next_observation, reward, action, done in zip(observations, next_observations, rewards,
                                                                           actions, dones):
                agent.push_memory(observation, action, reward, next_observation, done)

            observations = copy.deepcopy(env.get_obs())  # 不能直接copy原先的观测，因为会清理车
            total_reward += info["global_reward"]
            speed += info["average_speed"]
            step += 1
            crash_number += sum(info["crashed"])

        total_rewards.append(total_reward)
        steps.append(step)
        speeds.append(speed / step)

    safe_rate = 1 - (crash_number / agent_number)
    return total_rewards, steps, speeds, safe_rate
