from typing import TypedDict
import numpy as np

# Sars type -> s_t, a, r, s_tp1, done -> 4 * [float] + 1 * [float] + 1 * [float] + 4 * [float] + 1 *[float] = 11 * [float]
# 기존은 11차원세트 >> 근데 6개의 추가 센서데이터 고려하면 각  s에 6개씩 추가되어야 하므로... 그리고 r의기준도 새로 세워야...
class SarsBuffer:
  data: np.ndarray[np.ndarray[float]]

  def __init__ (self):
    self.data = np.zeros((0, 23), dtype=np.float32) #11 + 12이면 ..23?
  
  def sample (self, sample_num: int):
    if self.data.shape[0] < sample_num:
      raise Exception('SarsBuffer.sample failed: self.data.shape[0] < sample_num ({} < {})'.format(self.data.shape[0], sample_num))
    else:
      #print(6666)
      return self.data[np.random.choice(self.data.shape[0], (sample_num, ), replace=False)]

  def push (self, data: np.ndarray[np.ndarray[float]]):
    self.data = np.concatenate([self.data, data], axis=0)
    #print(self.data.shape) 38이상 안쌓임.. push과정에서 문제가?
  
  def pop (self, indices: np.ndarray[int]):
    if indices.shape[0] > self.data.shape[0]:
      raise Exception('SarsBuffer.pop: self.indices.shape[0] > self.data.shape[0] ({} > {})'.format(indices.shape[0], self.data.shape[0]))

    self.data = np.delete(self.data, indices, axis=0)

class SizeLimmitedSarsPushBuffer:
  sars_buffer: SarsBuffer
  size_limit: int

  def __init__ (self, sars_buffer: SarsBuffer, size_limit: int):
    self.sars_buffer = sars_buffer
    self.size_limit = size_limit
  
  def push (self, data: np.ndarray[np.ndarray[float]]):
    self.sars_buffer.push(data)
    if self.sars_buffer.data.shape[0] > self.size_limit:
      indices = np.random.choice(self.sars_buffer.data.shape[0], (self.sars_buffer.data.shape[0] - self.size_limit, ), replace=False)
      self.sars_buffer.pop(indices)
  
  def sample (self, sample_num):
    return self.sars_buffer.sample(sample_num)
