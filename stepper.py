from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.articulations import ArticulationView
import numpy as np
from trainer import Trainer
from sars_buffer import SizeLimmitedSarsPushBuffer
from agent_over_manager import AgentOverManager

from env.env import get_imu_data  # 서로서로 가져오는건 충돌>> 메서드를 다른 모듈에 저장해서 새로들고오기
xn , yn = 5 ,1
reward_cum = np.zeros((xn*yn,1)) # reward 초기화
#step_num=0

class Stepper:
  def __init__ (self, world: SimulationContext, cartpoles: ArticulationView, render: bool):
    self.world = world
    self.cartpoles = cartpoles
    self.render = render
    
   

  def step (self, a: np.ndarray):
    self.cartpoles.set_joint_efforts(efforts=np.reshape(a, (-1, 1)) * 100, joint_indices=np.repeat(0, a.shape[0]))
    self.world.step(render=self.render)
    

class StepForTrain (Stepper):
  def __init__ (self, cartpoles: ArticulationView, stepper: Stepper, trainer: Trainer, agent_over_manager: AgentOverManager, sars_buffer: SizeLimmitedSarsPushBuffer, agent_num: int,env):
    self.cartpoles = cartpoles
    self.stepper = stepper
    self.trainer = trainer
    self.sars_buffer = sars_buffer
    self.agent_num = agent_num
    self.agent_over_manager = agent_over_manager
    self.env = env
    self.reward_mean=0
    self.storage = []
    self.imu_storage = []

  def step (self, a: np.ndarray[np.ndarray[float]]): # 센서데이터에 대한 종료조건도 추가해야될것...
    global reward_cum  #step_num
    #step_num +=1
    prev_joint_positions = self.cartpoles.get_joint_positions(joint_indices=[0, 1])
    prev_joint_velocities = self.cartpoles.get_joint_velocities(joint_indices=[0, 1])
    prev_linear_acceleration = np.array([get_imu_data(f"/World/Cartpole_{i}/pole/IMU")[0] for i in range(self.agent_num)])
    prev_angular_velocity = np.array([get_imu_data(f"/World/Cartpole_{i}/pole/IMU")[1] for i in range(self.agent_num)])
    self.stepper.step(a)
    cur_joint_positions = self.cartpoles.get_joint_positions(joint_indices=[0, 1])
    cur_joint_velocities = self.cartpoles.get_joint_velocities(joint_indices=[0, 1])
    cur_linear_acceleration = np.array([get_imu_data(f"/World/Cartpole_{i}/pole/IMU")[0] for i in range(self.agent_num)]) # (5,3)
    cur_angular_velocity = np.array([get_imu_data(f"/World/Cartpole_{i}/pole/IMU")[1] for i in range(self.agent_num)])
	
    prev_over_table = np.array(self.agent_over_manager.over_table)
    prev_live_table = np.logical_not(prev_over_table)
    self.agent_over_manager.update(cur_joint_positions)
    cur_over_table = np.array(self.agent_over_manager.over_table)
    
    
    reward = np.zeros((self.agent_num, 1), dtype=np.float32)  # 보상값이 너무 낮게 나오므로 조금만 느슨하게 조건을 주자..
    reward += 0.8
    reward -= np.abs(cur_joint_positions[:, [0]]) * 0.02
    reward -= np.abs(cur_joint_velocities[:,[0]]) * 0.1  # joint velocity로 수정
    reward -= np.abs(cur_joint_positions[:, [1]]) * 0.02 # 1번 관절에 대한 정보도 넣어주자
    reward -= np.abs(cur_joint_velocities[:, [1]]) * 0.02
    reward -= np.abs(cur_linear_acceleration.mean(axis=1).reshape(-1,1)) * 0.01 # 내일은 이 센서 데이터값을 조금더  죽여보자..
    reward += np.abs(cur_angular_velocity.mean(axis=1).reshape(-1,1)) * 0.01 # sum >> mean으로 바꿔보자...
    reward[np.expand_dims(cur_over_table, axis=1)] -= 0.5 # 이게 원인 : 종료될때마다 보상을 엄청 낮게줌, 당연히 초반에는 많이 실패하니 보상값이
    # 엄청나게 낮아질수 밖에 없넹... 그러면 -0.8고정이 아닌 전체에서 빼는걸로 수정하자..  그리고 종료기준을 조금더 완화시켜 샘플수집속도를 높이자.. 
    #print(cur_over_table) 특정순간 all true나옴 이게 무엇을 의미할까?
    #print(reward)
    #print(cur_linear_acceleration) #   3차원성분의 평균으로 재구성하자..
    #print('joint_pos0:{}, joint_velo0:{}'.format(np.abs(cur_joint_positions[:, [0]]),np.abs(cur_joint_velocities[:,[0]]) ))
    #print()
    #print('joint_pos1:{}, joint_velo1:{}'.format(np.abs(cur_joint_positions[:, [1]]),np.abs(cur_joint_velocities[:,[1]]) ))
    #print()
    #print('linear:{}, angle:{}'.format(np.abs(cur_linear_acceleration.sum(axis=1).reshape(-1,1)),np.abs(cur_angular_velocity.sum(axis=1).reshape(-1,1) ) ))
    #print()   이건 관절데이터 값 확인 하기위한 출력코드...
    reward_cum += reward  # 아마도 보상이 축적되지않을까? (5,1) 그리고 그것을 이제 합쳐야지... 
    #print(reward_cum)  너무 -인데 이러면 수렴하는데 속도가 너무 느린데 일단 해보죠...
    self.reward_mean = np.mean(reward_cum,axis=0)
    #print("step: {}, reward_mean: {}".format(step_num,self.reward_mean)) 오케...
    self.storage.append(self.reward_mean) # 근데 이제는 스텝이 다 끝났을때 이걸 어떻게 처리하는지 이당.
    #self.imu_storage.append()
    
    new_sars = np.concatenate([   # sarsd : 11차원  / 이제 6차원데이터를 상태에 따로 추가...? 근데 어떻게 가져오지?
      prev_joint_positions[prev_live_table][:, [0]],
      prev_joint_velocities[prev_live_table][:, [0]],
      prev_joint_positions[prev_live_table][:, [1]],
      prev_joint_velocities[prev_live_table][:, [1]],  
      prev_linear_acceleration[prev_live_table],
      prev_angular_velocity[prev_live_table],
      a[prev_live_table][:, [0]],    # 11차원 까지 
      reward[prev_live_table][:, [0]],
      cur_joint_positions[prev_live_table][:, [0]],
      cur_joint_velocities[prev_live_table][:, [0]],
      cur_joint_positions[prev_live_table][:, [1]],
      cur_joint_velocities[prev_live_table][:, [1]],
      cur_linear_acceleration[prev_live_table],    #(5,3)
      cur_angular_velocity[prev_live_table],
      np.expand_dims(np.array(cur_over_table, dtype=np.float32), axis=1)[prev_live_table][:, [0]]
    ], axis=1)
    #print(new_sars.shape) 아니면 좀만 기다려 볼까?
    #print(prev_live_table)
    self.sars_buffer.push(new_sars)

    self.trainer.train()
