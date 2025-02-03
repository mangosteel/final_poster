import numpy as np

class AgentOverManager:
  over_table: np.ndarray[np.bool8]

  def __init__ (self, agent_num):
    self.agent_num = agent_num

  def init (self):
    self.over_table = np.zeros((self.agent_num, ), np.bool8)
  
  def update (self, joint_positions: np.ndarray[np.ndarray[np.float32]]): # 종료기준을 완화 시킴
    over_test = self.over_table | (np.abs(joint_positions[:, 0]) > 5.0) | (np.logical_not((np.abs(joint_positions[:, 1]) <= (np.pi / 1.5)) | (np.abs(joint_positions[:, 1] - 2 * np.pi) <= (np.pi / 1.5)) | (np.abs(joint_positions[:, 1] + 2 * np.pi) <= (np.pi / 1.5))))
    self.over_table[over_test] = True
    
    
