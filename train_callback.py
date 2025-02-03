from model.actor import Actor
from model.critic import Critic
from safetensors.torch import load_model, save_model
import os
mango=0
class TrainCallback:
  def handle (self, actor_critic_loss):
    pass

class TrainMonitor (TrainCallback):
  train_num: int

  def __init__ (self):
    self.train_num = 0
    #self.mango = False  # 내 생각엔 이게 나중에 콜백할떄 한번도 init되니깐 그떄 false로 다시 바뀌는듯...
    self.epi_num = 0
    #self.mango=0  #2번 init되니깐 이런 일이 발생한다. : 원인... 

  def start_new_episode(self):   #에피소드가 새로 시작될 때 호출 > 혹시 100번마다 출력이라서 안되는거 아닐까?
    global mango
    mango = False  # 0 >> false
    mango = True
    #print(888)   # 출력을 확인함  1
    #print(mango) # true 확인.  2    # 원인은 trainmonitor가 2번 init되면서, mango값이 바뀌는 것이였다(false).. 그래서 글로벌변수를 통해서 아예
                                   # init의 영향을 받지 않도록한 결과 곧바로 에피소드가 업데이트 된다.


  def handle (self, actor_critic_loss):
    global mango
    self.train_num += 1
    if mango:   #수정  이게 실행이 안되는 중...
        self.epi_num +=1
        #print(24)
        #print(self.epi_num) #이게 지금 출력이 안되는듯... 왜일까? 
        mango=False  # 중첩후 항상 초기화!
    else:
        pass
        #print(mango) # 3 false >> 아마도 true > false로 조건문에 도달하기전에 바뀌는듯...
        #print(999)   #4
    if self.train_num %100==0:
      #print(mango) #  여기선 false  5
      #print(12) #  6
      print('actor loss, critic_loss: {:.4f}, {:.4f}, step: {}, epi: {}'.format(*actor_critic_loss,self.train_num,self.epi_num))
      print()
      #print(777)  # 손실이 또 안나오넹..? 
      #print(actor_critic_loss) # 이제 마지막 단계...


class TrainSaver (TrainCallback):
  def __init__ (self, actor: Actor, critic: Critic, actor_target: Actor, critic_target: Critic):
    self.train_num = 0
    self.actor = actor
    self.critic = critic
    self.actor_target = actor_target
    self.critic_target = critic_target

  def handle (self, actor_critic_loss):
    self.train_num += 1
    if self.train_num % 100000 == 0: # 조금만 크기를 줄여보자....
      project_path = os.path.join(os.getcwd(), 'standalone_examples/cartpole_train')
      save_model(self.actor, os.path.join(project_path, 'ckpt/actor_{}.safetensors'.format(self.train_num)))
      save_model(self.critic, os.path.join(project_path, 'ckpt/critic_{}.safetensors'.format(self.train_num)))
      save_model(self.actor_target, os.path.join(project_path, 'ckpt/actor_target_{}.safetensors'.format(self.train_num)))
      save_model(self.critic_target, os.path.join(project_path, 'ckpt/critic_target_{}.safetensors'.format(self.train_num)))
      print('saved')

class TrainEvaluator (TrainCallback):
  def handle (self, actor_critic_loss):
    pass
    # print('train evaluator')
