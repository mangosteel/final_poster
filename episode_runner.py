from omni.isaac.core.simulation_context import SimulationContext
from stepper import Stepper
from env.env import Env
import numpy as np
import torch
from model.actor import Actor
from model.critic import Critic
from agent_over_manager import AgentOverManager
from pxr import Usd
import omni
import time
from omni.isaac.sensor import _sensor
from env.env import get_imu_data
import os
#def get_imu_data(imu_prim_path):
   # """IMU 센서의 선형 가속도 및 각속도를 가져오는 함수"""
    
    # ✅ IMU 센서 인터페이스 가져오기
 #   _imu_sensor_interface = _sensor.acquire_imu_sensor_interface()
    
    # ✅ 센서 데이터 가져오기
  #  reading = _imu_sensor_interface.get_sensor_reading(
    #    imu_prim_path, use_latest_data=True, read_gravity=True  # ✅ 중력 값 포함 가능
   # )
    
    #if not reading.is_valid:
   #     print(f"❌ ERROR: IMU sensor at {imu_prim_path} returned invalid data!") # 종료되면 이게뜸...
   #     return None, None

  #  linear_acceleration = np.array([reading.lin_acc_x, reading.lin_acc_y, reading.lin_acc_z])
  #  angular_velocity = np.array([reading.ang_vel_x, reading.ang_vel_y, reading.ang_vel_z])

   # print(f"📡 IMU Data from {imu_prim_path}:")
  #  print(f"   🔹 Linear Acceleration: {linear_acceleration}")
  #  print(f"   🔹 Angular Velocity: {angular_velocity}")

   # return np.array(linear_acceleration, dtype=np.float32), np.array(angular_velocity, dtype=np.float32)

step_num = 0
url = "standalone_examples/cartpole_train/shark/reward.csv"
class EpisodeRunner:
  def __init__ (self, stepper: Stepper, env: Env, actor: Actor, critic: Critic, agent_over_manager: AgentOverManager, episode_num: int, simulation_seconds: float, device,train_monitor):
    self.device = device
    self.stepper = stepper  # stepfortrain 을 stepper로 받는다.
    self.env = env
    self.actor = actor
    self.critic = critic
    self.agent_over_manager = agent_over_manager
    self.episode_num = episode_num
    self.simulation_seconds = simulation_seconds
    self.train_monitor = train_monitor  # 수정

  def run_episode (self):
      global step_num ,url  # 4칸간격 유지...
      for _ in range(self.episode_num):   # 지금 10번 인데..
            self.env['world'].reset()
            self.train_monitor.start_new_episode() # 수정
            
            self.agent_over_manager.init()

            for _ in range(round(self.simulation_seconds / self.env['physics_dt'])): # 600번 반복
                step_num +=1
                if step_num > 600*self.episode_num-1:
                    array_data = np.array(self.stepper.storage)  # (60000,1) 어레이 예상.. stepper storage를 가져옴..
                    np.savetxt(url, array_data, delimiter=",", fmt="%d") # csv파일로 저장 넘파이를
                    print('save complete')  
                    
                    #print('weewewee') # 확인
                    
                else:
                    pass
                #print(step_num) # ok
                with torch.no_grad():
                
                
                     
                    self.env['world'].step(render=True)
                    time.sleep(0.01)
                    joint_positions = self.env['cartpoles'].get_joint_positions(joint_indices=[0, 1])
                    joint_velocities = self.env['cartpoles'].get_joint_velocities(joint_indices=[0, 1])
                    #print(self.stepper.storage) # 일단 출력성공
                    #print(os.getcwd()) # 현재위치..
                    #print(joint_velocities)
                    # ✅ IMU 데이터 가져오기
                    imu_data = []
                    for i in range(len(self.env['imus'])):
                        imu_prim_path = f"/World/Cartpole_{i}/pole/IMU"   # 경로에 cart를 추가해보자
                        linear_acceleration, angular_velocity = get_imu_data(imu_prim_path)
                        #print(linear_acceleration, angular_velocity)

                        if linear_acceleration is None or angular_velocity is None:  # 이게 적용됨... 
                            imu_data.append((np.zeros(3), np.zeros(3)))  # 기본값 사용
                        else:
                            imu_data.append((linear_acceleration, angular_velocity))

                    imu_data = np.array(imu_data)
                    linear_acceleration, angular_velocity = imu_data[:, 0], imu_data[:, 1] # (5,3) : 아마도 1개당 xyz성분가속도인듯
                    #print(linear_acceleration)  # 아마 추가입력은+6으로 수정해야할것. 이것도 안받아짐.0으로됨.
                    #print(angular_velocity) #근데 데이터가 받아지긴 한듯?? 나오는거 보니.. 근데 데이터가 안받아짐 0으로 만 이루어짐..

                    # ✅ 기존 state에 IMU 데이터 추가 ,  상태 : 10차원...
                    state = torch.from_numpy(
                        np.concatenate([
                            joint_positions[:, [0]],  # Cart 위치 (5,1)  >> state : 10차원
                            joint_velocities[:, [0]],  # Cart 속도 (5,1)
                            joint_positions[:, [1]],  # Pole 각도 (5,1)
                            joint_velocities[:, [1]],  # Pole 각속도 (5,1)
                            linear_acceleration, # 선형 가속도 , (5,3) , xyz성분..
                            angular_velocity       # 각속도 , (5,3)
                        ], axis=1)
                    ).to(self.device)
                    #print(state.shape)

                    action = self.actor(state) + torch.from_numpy(
                        np.clip(np.random.normal(size=(self.env['xn'] * self.env['yn'], 1)) * 0.2, -0.4, 0.4)
                    ).to(self.device)
                    #print("action:{}".format(action)) # 이건 계산됨, 신경망 돌아가는디?
                self.stepper.step(action.detach().cpu().numpy())

    
    
		
        
        
