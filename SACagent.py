import collections
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init


class FeatureNet(torch.nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.norm_type = conf.norm_type

        if not conf.IfConv:
            self.f1 = nn.Identity()
            self.normalize = nn.Identity()
        else:
            self.f1 = nn.Conv2d(in_channels=4, out_channels=self.conf.out_channels, kernel_size=3, stride=1,
                                padding=(1, 0))
            init.kaiming_normal_(self.f1.weight, mode='fan_out')
            if self.norm_type == "layer_norm":
                self.normalize = nn.LayerNorm([conf.out_channels, 3, 5])
            else:
                self.normalize = nn.Identity()

    def forward(self, x):
        return self.normalize(self.f1(x))


class PolicyNet(torch.nn.Module):
    def __init__(self, conf):
        super(PolicyNet, self).__init__()
        self.conf = conf
        if conf.IfConv:
            state_dim = conf.conv_dim
        else:
            state_dim = conf.obs_shape
        action_dim = conf.n_actions
        hidden_dim1 = conf.hidden_size_1
        hidden_dim2 = conf.hidden_size_2
        self.feature_net = FeatureNet(conf)
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim1)
        self.fc2 = torch.nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = torch.nn.Linear(hidden_dim2, action_dim)
        self.init_weights()

        # normal
        self.norm_type = conf.norm_type
        if self.norm_type == "layer_norm":
            self.normalize1 = nn.LayerNorm(conf.hidden_size_1)
            self.normalize2 = nn.LayerNorm(conf.hidden_size_2)
        else:
            self.normalize1 = nn.Identity()
            self.normalize2 = nn.Identity()

    def forward(self, x):
        x = self.feature_net(x)
        x = F.relu(self.normalize1(self.fc1(x.view(x.shape[0], -1))))
        x = F.relu(self.normalize2(self.fc2(x)))
        return F.softmax(self.fc3(x), dim=1)

    def init_weights(self):
        # 使用 He 初始化方法 (适合ReLU激活函数)
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc3.weight, mode='fan_in', nonlinearity='linear')  # 最后一层使用线性初始化
        # 也可以为偏置项设置一个常数，例如0
        self.fc1.bias.data.fill_(0)
        self.fc2.bias.data.fill_(0)
        self.fc3.bias.data.fill_(0)


class QValueNet(torch.nn.Module):
    def __init__(self, conf):
        super(QValueNet, self).__init__()
        self.conf = conf
        if conf.IfConv:
            state_dim = conf.conv_dim
        else:
            state_dim = conf.obs_shape
        action_dim = conf.n_actions
        hidden_dim1 = conf.hidden_size_1
        hidden_dim2 = conf.hidden_size_2
        self.feature_net = FeatureNet(conf)
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim1)
        self.fc2 = torch.nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = torch.nn.Linear(hidden_dim2, action_dim)
        self.init_weights()

        # normal
        self.norm_type = conf.norm_type
        if self.norm_type == "layer_norm":
            self.normalize1 = nn.LayerNorm(conf.hidden_size_1)
            self.normalize2 = nn.LayerNorm(conf.hidden_size_2)
        else:
            self.normalize1 = nn.Identity()
            self.normalize2 = nn.Identity()

    def forward(self, x):
        x = self.feature_net(x)
        x = F.relu(self.normalize1(self.fc1(x.view(x.shape[0], -1))))
        x = F.relu(self.normalize2(self.fc2(x)))
        return self.fc3(x)

    def init_weights(self):
        # 使用 He 初始化方法 (适合ReLU激活函数)
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc3.weight, mode='fan_in', nonlinearity='linear')  # 最后一层使用线性初始化
        # 也可以为偏置项设置一个常数，例如0
        self.fc1.bias.data.fill_(0)
        self.fc2.bias.data.fill_(0)
        self.fc3.bias.data.fill_(0)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)


class SACAgent:
    def __init__(self, conf):
        # 策略网络
        self.target_update = 50  # 没啥用，对齐格式
        self.conf = conf
        self.device = conf.device
        self.replay_buffer = ReplayBuffer(conf.buffer_size)
        actor_lr = conf.actor_lr
        self.actor = PolicyNet(conf).to(self.device)
        # 第一个Q网络
        self.critic_1 = QValueNet(conf).to(self.device)
        # 第二个Q网络
        self.critic_2 = QValueNet(conf).to(self.device)
        self.target_critic_1 = QValueNet(conf).to(self.device)  # 第一个目标Q网络
        self.target_critic_2 = QValueNet(conf).to(self.device)  # 第二个目标Q网络

        # 令目标Q网络的初始参数和Q网络一样
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=conf.critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=conf.critic_lr)

        # 使用alpha的log值,可以使训练结果比较稳定
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True  # 可以对alpha求梯度
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=conf.alpha_lr)
        self.target_entropy = conf.target_entropy  # 目标熵的大小
        self.gamma = conf.gamma
        self.tau = conf.tau

    def select_action(self, state, evaluate=False, action_mask=None):
        if action_mask is None:
            action_mask = np.ones(self.conf.n_actions)
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        probs = self.actor(state)
        probs = probs * torch.tensor(np.array(action_mask)).to(self.device)
        probs = probs / probs.sum()
        action_dist = torch.distributions.Categorical(probs)
        if evaluate:
            # 评估模式：选择概率最大的动作
            action = torch.argmax(probs).item()
        else:
            # 训练模式：从分布中采样一个动作
            action = action_dist.sample().item()
        return action

    # 计算目标Q值,直接用策略网络的输出概率进行期望计算
    def calc_target(self, rewards, next_states, dones):
        next_probs = self.actor(next_states)
        next_log_probs = torch.log(next_probs + 1e-8)
        entropy = -torch.sum(next_probs * next_log_probs, dim=1, keepdim=True)
        q1_value = self.target_critic_1(next_states)
        q2_value = self.target_critic_2(next_states)
        min_qvalue = torch.sum(next_probs * torch.min(q1_value, q2_value),
                               dim=1,
                               keepdim=True)
        next_value = min_qvalue + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)  # 动作不再是float类型
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        # 更新两个Q网络
        td_target = self.calc_target(rewards, next_states, dones)
        critic_1_q_values = self.critic_1(states).gather(1, actions)
        critic_1_loss = torch.mean(
            F.mse_loss(critic_1_q_values, td_target.detach()))
        critic_2_q_values = self.critic_2(states).gather(1, actions)
        critic_2_loss = torch.mean(
            F.mse_loss(critic_2_q_values, td_target.detach()))
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # 更新策略网络
        probs = self.actor(states)
        log_probs = torch.log(probs + 1e-8)
        # 直接根据概率计算熵
        entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True)  #
        q1_value = self.critic_1(states)
        q2_value = self.critic_2(states)
        min_qvalue = torch.sum(probs * torch.min(q1_value, q2_value),
                               dim=1,
                               keepdim=True)  # 直接根据概率计算期望
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - min_qvalue)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新alpha值
        alpha_loss = torch.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)

    def update_epsilon(self):
        # 暂时没什么用
        pass

    def push_memory(self, observation, action, reward, next_observation, done):
        return self.replay_buffer.add(observation, action, reward, next_observation, done)

    def mul_optimize_model(self):
        for i in range(self.conf.train_step):
            self.optimize_model()

    def optimize_model(self):
        if self.replay_buffer.size() > self.conf.batch_size:
            b_s, b_a, b_r, b_ns, b_d = self.replay_buffer.sample(self.conf.batch_size)
            transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
            self.update(transition_dict)

    def update_target_network(self):
        pass
