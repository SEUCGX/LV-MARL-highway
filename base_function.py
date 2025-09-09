import copy
import random
import numpy as np
import torch
import math
import gymnasium as gym
from ma_highway_env import MaHighwayEnv # 这是一个从highway-env中下载后修改的环境


def is_ellipse_neighbor(p, a1, a2, b):
    # 用一个两个椭圆区域来判断点是否在附近
    # error是以ego为中心的坐标点，相当于以(0,0)为中心
    # a1是长半轴的后面(x<0)
    # a2是长半轴的前面(x>0)
    x = p[0]
    y = p[1]
    if x < 0:
        t1 = x ** 2 / (a1 ** 2)
    else:  # 等于0无所谓
        t1 = x ** 2 / (a2 ** 2)

    t2 = y ** 2 / (b ** 2)
    if (t1 + t2) < 1:
        return True  # 是邻居
    else:
        return False

class my_env(gym.Env):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.env = make_env(self.conf)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.set_mode("rgb_array")

    def step(self, actions):
        observation, reward, terminated, truncated, info = self.env.step(actions)
        info["state"] = self.get_state()
        info["action_mask"] = np.ones((len(observation), 5))
        info["average_speed"] = sum(info["speed"]) / len(info["speed"])
        info["global_reward"] = np.mean(reward)
        info["win_rate"] = 1 - sum(info["agents_dones"]) / len(info["agents_dones"])

        info["regional_rewards"] = np.empty(len(self.env.controlled_vehicles))
        for i, ego_vehicle in enumerate(self.env.controlled_vehicles):
            test = []
            for other_vehicle in self.env.controlled_vehicles:
                if ego_vehicle is not other_vehicle:
                    error = ego_vehicle.position - other_vehicle.position  # (x,y)
                    if is_ellipse_neighbor(error, self.env.config["reward_type"]["a1"], self.env.config["reward_type"]["a2"],
                                           self.env.config["reward_type"]["b"]):
                        test.append(self.env.config["reward_type"]["w"] * self.env._agent_reward(actions, other_vehicle))
            info["regional_rewards"][i] = self.env._agent_reward(actions, ego_vehicle) + sum(test) / (len(test) + 1e-9)

        return tuple(observation), info["regional_rewards"], truncated, info

    def reset(self, seed=None):
        if seed is None:
            seed = random.randint(1, 1000000000)
        observation, info = self.env.reset(seed=seed)
        info["state"] = self.get_state()
        info["action_mask"] = np.ones((len(observation), 5))
        return observation, info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    def save(self):
        return self.env.save()

    def clear_frames(self):
        return self.env.clear_frames()

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def set_mode(self, mode="human"):
        self.env.render_mode = mode

    def get_state(self):
        max_x = int(self.env.config["Terminal_position"] / 20)
        states = np.zeros((4, self.env.config["lanes_count"], max_x + 1))
        for vehicle in self.env.road.vehicles:
            p = vehicle.position
            index_x = np.clip(int(np.trunc(p[0] / 20)), 0, max_x)
            index_y = np.clip(int(np.trunc((p[1] + 2) / self.env.ROAD_WIDTH)), 0, self.env.config["lanes_count"] - 1)
            states[0:2, index_y, index_x] = vehicle.position / [self.env.config["Terminal_position"],
                                                                self.env.ROAD_WIDTH * self.env.config["lanes_count"]]
            states[2, index_y, index_x] = vehicle.speed * np.cos(vehicle.heading)
            states[3, index_y, index_x] = vehicle.speed * np.sin(vehicle.heading)
        return copy.deepcopy(states)

    def get_obs(self):
        return self.env.get_obs()

    def get_state_shape(self):
        return 4 * self.env.config["lanes_count"] * (int(self.env.config["Terminal_position"] / 20) + 1)

    def get_controlled_number(self):
        return len(self.env.controlled_vehicles)

    def chang_test_time(self, test_time):
        self.env.config["test_time"] = test_time


def make_env(conf):
    env = MaHighwayEnv()
    config = {
        "reward_type": {"type": conf.reward_type,
                        "a1": conf.reward_a1,
                        "a2": conf.reward_a2,
                        "b": conf.reward_b,
                        "w": conf.reward_w, },
        "duration": 20,
        "multiprocess": conf.multiprocess,
        "cpu_num": 8,
        "video_fps": 10,
        "video_address": conf.video_address,
        "data_address": conf.data_address,
        "SAVE_IMAGES": False,
        "save_video": True,
        "directory": "result",
        "screen_width": 1000,
        "screen_height": 1000,
        "scaling": 5,
        "spawn_probability": 5,  # 生成概率大于1意味着可以后续有超过一辆车被生成
        "test_time": conf.test_time,  # 这个变量决定了是否记录中间的动画过程
        "Terminal_position": 8000,
        "controlled_vehicles": conf.controlled_vehicles,
        "obs_position": np.array([200, 0]),
        # 奖励值
        "collision_reward": -10,
        "right_lane_reward": 0.0,
        "high_speed_reward": 1,
        "headway_reward": 0.1,  # 新增的车头距离
        # 有没有其他车
        "other_vehicles_number": conf.other_vehicles_number,
        "other_vehicles": conf.other_vehicles
    }
    env.configure(config)
    env.reset(seed=conf.seed)
    return env


def lmap(v: float, x, y) -> float:
    """Linear map of value v with range x to desired range y."""
    return y[0] + (v - x[0]) * (y[1] - y[0]) / (x[1] - x[0])


def set_seed(random_seed=11):
    np.set_printoptions(suppress=True)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    if torch.cuda.device_count() > 1:
        torch.cuda.manual_seed_all(random_seed)
    else:
        torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_tensor(data):
    return torch.tensor(data, dtype=torch.float)


def to_array(data):
    return data.cpu().detach().numpy()
