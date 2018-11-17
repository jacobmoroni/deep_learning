import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
from tqdm import tqdm

class PolicyNetwork(nn.Module):
  def __init__(self, state_dimension, action_dimension):
    super(PolicyNetwork, self).__init__()
    self.policy_net = nn.Sequential( nn.Linear(state_dimension, 10),
                                   nn.ReLU(),
                                   nn.Linear(10,10),
                                   nn.ReLU(),
                                   nn.Linear(10,10),
                                   nn.ReLU(),
                                   nn.Linear(10,action_dimension))
    self.policy_softmax = nn.Softmax(dim=1)

  def forward(self,x):
    scores = self.policy_net(x)
    return self.policy_softmax(scores)


class ValueNetwork(nn.Module):
  def __init__(self, state_dimension):
    super(ValueNetwork, self).__init__()
    self.value_net = nn.Sequential( nn.Linear(state_dimension, 10),
                                   nn.ReLU(),
                                   nn.Linear(10,10),
                                   nn.ReLU(),
                                   nn.Linear(10,10),
                                   nn.ReLU(),
                                   nn.Linear(10,1))

  def forward(self,x):
    return self.value_net(x)

class AdvantageDataset(Dataset):
  def __init__(self, experience):
    super(AdvantageDataset, self).__init__()
    self._exp = experience
    self._num_runs = len(experience)
    self._length = reduce(lambda acc, x: acc + len(x), experience, 0)

  def __getitem__(self, index):
    idx = 0
    seen_data = 0
    current_exp = self._exp[0]
    while seen_data + len(current_exp) - 1 < index:
      seen_data += len(current_exp)
      idx += 1
      current_exp = self._exp[idx]
    chosen_exp = current_exp[index - seen_data]
    return chosen_exp[0], chosen_exp[4]

  def __len__(self):
    return self._length

class PolicyDataset(Dataset):
  def __init__(self, experience):
    super(PolicyDataset, self).__init__()
    self._exp = experience
    self._num_runs = len(experience)
    self._length = reduce(lambda acc, x: acc + len(x), experience, 0)

  def __getitem__(self, index):
    idx = 0
    seen_data = 0
    current_exp = self._exp[0]
    while seen_data + len(current_exp) - 1 < index:
      seen_data += len(current_exp)
      idx += 1
      current_exp = self._exp[idx]
    chosen_exp = current_exp[index - seen_data]
    return chosen_exp

  def __len__(self):
    return self._length

def calculate_returns(trajectories, gamma):
  for i, trajectory in enumerate(trajectories):
    current_reward = 0
    for j in reversed(range(len(trajectory))):
      state, probs, action_index, reward = trajectory[j]
      ret = reward + gamma * current_reward
      trajectories[i][j] = (state, probs, action_index, reward, ret)
      current_reward = ret

def calculate_advantages(trajectories, value_net):
  for i, trajectory in enumerate(trajectories):
    for j, exp in enumerate(trajectory):
      advantage = exp[4] - value_net(torch.from_numpy(exp[0]).float().unsqueeze(0))[0,0]#missing something here
      trajectories[i][j] =(exp[0], exp[1], exp[2], exp[3], exp[4], advantage )


def render_now():
  for i in range(300):
    env.render()
    action = policy(torch.from_numpy(s).float().view(1,-1))
    action_index = np.random.multinomial(1, action.detach().numpy().reshape((num_actions)))
    action_index = np.argmax(action_index)
    env.step(action_index)
# env = gym.make('CartPole-v0')
# states= 2
# actions = 3
# policy = PolicyNetwork(4, 2)
# value = ValueNetwork(4)

env = gym.make('MountainCar-v0')
num_states= 2
num_actions = 3
policy = PolicyNetwork(num_states, num_actions)
value = ValueNetwork(num_states)


policy_optim = optim.Adam(policy.parameters(), lr=1e-2, weight_decay=0.01)
value_optim = optim.Adam(value.parameters(), lr=1e-3, weight_decay=1)
value_criteria = nn.MSELoss()

# Hyperparameters
epochs = 300 #1000
env_samples = 100
episode_length = 2000# 200
gamma = 0.9
value_epochs = 2
policy_epochs = 5
batch_size = 32
policy_batch_size = 256
epsilon = 0.2
standing_time_list = []
loss_list = []

loop = tqdm(total=epochs,position=0, leave=False)

for epoch in range(epochs):

  # generate rollouts
  rollouts = []
  standing_length = 0
  max_x_total = 0
  for _ in range(env_samples):
    current_rollout = []
    s = env.reset()
    max_x = -10
    min_x = 0
    for i in range(episode_length):
      action = policy(torch.from_numpy(s).float().view(1,-1))
      action_index = np.random.multinomial(1, action.detach().numpy().reshape((num_actions)))
      action_index = np.argmax(action_index)
      s_prime, r, t, _ = env.step(action_index)
      if s_prime[0] > max_x:
          max_x = s_prime[0]
          r = 3
      if s_prime[0] < min_x:
          min_x = s_prime[0]
          r = 1
      if t:
          r = 7000
      current_rollout.append((s, action.detach().reshape(-1), action_index, r))
      standing_length += 1
      # print (s_prime)


      if t:
        break

      s = s_prime
    max_x_total += max_x
    rollouts.append(current_rollout)

#     print('avg standing time:', standing_length / env_samples)
  avg_max_x = max_x_total / env_samples
  avg_standing_time = standing_length / env_samples
  standing_time_list.append(avg_standing_time)
  calculate_returns(rollouts, gamma)

  # Approximate the value function
  value_dataset = AdvantageDataset(rollouts)
  value_loader = DataLoader(value_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
  for _ in range(value_epochs):
    # train value network
    total_loss = 0
    for state, returns in value_loader:
      value_optim.zero_grad()
      returns = returns.unsqueeze(1).float()
      expected_returns = value(state.float())
      loss = value_criteria(expected_returns, returns)
      total_loss += loss.item()
      loss.backward()
      value_optim.step()
#       print ("total value loss:",total_loss)
    loss_list.append(total_loss)

  calculate_advantages(rollouts, value)

  # Learn a policy
  policy_dataset = PolicyDataset(rollouts)
  policy_loader = DataLoader(policy_dataset, batch_size=policy_batch_size, shuffle=True, pin_memory=True)
  for _ in range(policy_epochs):
    # train policy network
    for state, probs, action_index, reward, ret, advantage in policy_loader:
      policy_optim.zero_grad()
      current_batch_size = reward.size()[0]
      advantage = ret.float()
      p = policy(state.float())
      ratio = p[range(current_batch_size), action_index] / probs[range(current_batch_size), action_index]#something else goes here]

      lhs = ratio * advantage
      rhs = torch.clamp(ratio, 1-epsilon, 1+epsilon) * advantage
      loss = -torch.mean(torch.min(lhs,rhs))

      loss.backward()
      policy_optim.step()

  if (epoch % 50 == 0 & epoch != 0):
    plt.figure()
    plt.plot(standing_time_list)
    plt.title("Average Standing Time over Time")

    plt.figure()
    plt.plot(loss_list)
    plt.title("Loss over time")

    plt.show()


 
#   print('avg standing time:', standing_length / env_samples)
  # loop.set_description('epoch:{},loss:{:.4f},standing time:{:.5f}'.format(epoch,total_loss,avg_standing_time))
  loop.set_description('epoch:{},loss:{:.4f},max_x:{:.5f}'.format(epoch,total_loss, avg_max_x))
  loop.update(1)

loop.close()
plt.figure()
plt.plot(standing_time_list)
plt.title("Average Standing Time over Time")

plt.figure()
plt.plot(loss_list)
plt.title("Loss over time")

plt.show()
render_now()


