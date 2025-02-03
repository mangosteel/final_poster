from model.actor import Actor
from model.critic import Critic
from sars_buffer import SizeLimmitedSarsPushBuffer
from train_callback import TrainCallback
import numpy as np
import torch.nn as nn
import torch
from update_utils import soft_update

class Trainer:
  def __init__ (self, actor: Actor, critic: Critic, target_actor: Actor, target_critic: Critic, sars_buffer: SizeLimmitedSarsPushBuffer, gamma: float, device):
    self.device = device
    self.actor = actor
    self.critic = critic
    self.target_actor = target_actor
    self.target_critic = target_critic
    self.sars_buffer = sars_buffer
    self.gamma = gamma
    self.critic_criterion = nn.MSELoss(reduction='mean').to(self.device, dtype=torch.float32)
    self.critic_optimizer = torch.optim.Adam(params=self.critic.parameters(), lr=0.001)
    self.actor_optimizer = torch.optim.Adam(params=self.actor.parameters(), lr=0.001)
  
  def train (self): # 이거 자체가 발동이 안되는 거 같은디?
    try:
      sars_data = self.sars_buffer.sample(100) # 샘플이 안모이나? 버퍼를 점검해보자...
    except:
      
      return ([-1234, -1234]) # 이거만 나옴.
      

    with torch.no_grad():  # 근데 이제 state : 11 >> 17 / 기존 4개 >> 10개  / 
      target_next_state = torch.from_numpy(sars_data[:, 12: 22]).to(self.device, dtype=torch.float32) # 기존(100,4) , (100,10)
      target_next_action = self.target_actor(target_next_state) #10차원 되야함.
      q_target = torch.from_numpy(sars_data[:, [11]]).to(self.device, dtype=torch.float32) \
        + self.gamma * (1.0 - torch.from_numpy(sars_data[:, [22]]).to(self.device, dtype=torch.float32)) \
        * self.target_critic(torch.cat([target_next_state, target_next_action], dim=1))
    q = self.critic(torch.from_numpy(sars_data[:, 0: 11]).to(self.device, dtype=torch.float32)) # 크리틱 11개 이거땜에 오류난거..
    critic_loss = self.critic_criterion(q, q_target)
    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    self.critic_optimizer.step()

    actor_loss = -self.critic(
      torch.cat([
        torch.from_numpy(sars_data[:, 0: 10]).to(self.device, dtype=torch.float32),
        self.actor(torch.from_numpy(sars_data[:, 0: 10]).to(self.device, dtype=torch.float32))
      ], dim=1)
    ).mean()   # 이거 값을 한번 나중에 보자...
    #print("actor_loss:{}".format(actor_loss))
    self.actor_optimizer.zero_grad()
    actor_loss.backward()
    self.actor_optimizer.step()

    soft_update(self.target_critic, self.critic, 0.001)
    soft_update(self.target_actor, self.actor, 0.001)
    #print("loss_sample: {}".format(actor_loss.detach().cpu().numpy(), critic_loss.detach().cpu().numpy()) ) 안나옴..
    return (actor_loss.detach().cpu().numpy(), critic_loss.detach().cpu().numpy())

class TrainWithCallback (Trainer):
  def __init__ (self, trainer: Trainer, callbacks: list[TrainCallback]):
    self.trainer = trainer
    self.callbacks = callbacks
  
  def train (self):
    actor_critic_loss = self.trainer.train()
    for callback in self.callbacks:
      callback.handle(actor_critic_loss)
