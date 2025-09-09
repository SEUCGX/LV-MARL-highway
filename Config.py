import torch


class Config:
    def __init__(self, seed):
        self.multiprocess = None
        self.controlled_vehicles = None
        self.n_agents = None
        self.obs_shape = None
        self.episode_limit = None
        self.seed = None
        self.state_shape = None
        self.n_actions = None
        self.cuda = True
        self.seed = seed
        self.train_step = 3
        self.n_epochs = 2000  # 20000
        self.evaluate_epoch = 1  # 评估多少个
        self.evaluate_per_epoch = 200  # 每多少个epoch评估一次
        self.batch_size = 1024  # 256
        self.buffer_size = 15000
        self.save_frequency = 500000  # 5000
        self.gamma = 0.93
        self.max_grad_norm = 10  # prevent gradient explosion
        self.update_target_params = 50  # 200
        self.reward_type = "regionalR"
        self.reward_a2 = 10  # 长半轴，前面
        self.reward_b = 4  # 短边
        self.reward_a1 = 40  # 长半轴，后面
        self.reward_w = 0.7
        self.video_address = "Result.gif"  # 保存视频的地址
        self.data_address = "Result.mat"  # 保存数据的地址
        if self.cuda:
            self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

        self.hidden_size_1 = 512
        self.hidden_size_2 = 128
        # 是否使用卷积层
        self.IfConv = False
        self.out_channels = 16
        self.conv_dim = self.out_channels * 3 * 5
        # 优化相关
        self.optimizer = "rmsprop"  # "RMS"
        self.lr = 1e-3
        # epsilon greedy
        self.start_epsilon = 1
        self.end_epsilon = 0.05
        self.epsilon_decay = 0.995  # 50000
        self.test_time = False  # 是否记录中间的动画等等
        # 其他车辆
        self.other_vehicles_number = 0
        self.other_vehicles = False
        # sac特有参数
        self.actor_lr = 1e-3
        self.critic_lr = 1e-3
        self.alpha_lr = 1e-3
        self.target_entropy = 0.1  # -1
        self.tau = 0.005
        self.use_rule = False  # 评估的时候是否使用规则
        self.norm_type = "None"

    def set_env_info(self, env_info):
        self.n_actions = env_info.get("n_actions", 0)
        self.state_shape = env_info.get("state_shape", 0)
        self.obs_shape = env_info.get("obs_shape", 4 * 3 * 7)
        self.episode_limit = env_info.get("episode_limit", 20)
        self.controlled_vehicles = env_info.get("controlled_vehicles", 50)
        self.multiprocess = env_info.get("multiprocess", False)
        self.other_vehicles = env_info.get("other_vehicles", False)
        self.other_vehicles_number = env_info.get("other_vehicles_number", 0)
        self.video_address = env_info.get("video_address", "Result.gif")
        self.data_address = env_info.get("data_address", "Result.mat")
