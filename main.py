import numpy as np
from omni.isaac.kit import SimulationApp
import os

CONFIG = {
  "headless": True,
}
simulation_app = SimulationApp(launch_config=CONFIG)


from omni.isaac.core import World
from env.env import Env, generate_cartpoles, initialize_env, CartpoleOriginPositionsGenerator

physics_dt = 1.0 / 120.0
rendering_dt = 1.0 / 30.0
xn = 1
yn = 5
assets_root_path = 'omniverse://localhost/NVIDIA/Assets/Isaac/4.2/'
cartpole_usd_path = os.path.join(assets_root_path, 'Isaac/Robots/Cartpole/cartpole.usd')

world = World(stage_units_in_meters=1.0)
cartpole_origin_positions = CartpoleOriginPositionsGenerator(xn, yn).generate()
cartpoles,imus = generate_cartpoles(cartpole_origin_positions)
initialize_env(world, cartpoles, physics_dt, rendering_dt)
env: Env = {
  'world': world,
  'cartpoles': cartpoles,
  'cartpole_origin_positions': cartpole_origin_positions,
  'physics_dt': physics_dt,
  'rendering_dt': rendering_dt,
  'xn': xn,
  'yn': yn,
  'imus':imus
}

epi_num = 5   # 에피소드 횟수.
import torch
from safetensors.torch import load_file
from model.actor import Actor
from model.critic import Critic
from update_utils import hard_update
# os.getcwd() : .../isaac-sim-4.2.0 까지
project_path = os.path.join(os.getcwd(), 'standalone_examples/cartpole_train/ckpt')
actor_weights = os.path.join(project_path, 'actor_{}.safetensors'.format(epi_num*600))
critic_weights = os.path.join(project_path, 'critic_{}.safetensors'.format(epi_num*600))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

actor = Actor().to(device)
critic = Critic().to(device)
target_actor = Actor().to(device)
target_critic = Critic().to(device)



hard_update(target_actor, actor)
hard_update(target_critic, critic)


from sars_buffer import SarsBuffer, SizeLimmitedSarsPushBuffer

sars_buffer = SizeLimmitedSarsPushBuffer(SarsBuffer(), 10000)


from episode_runner import EpisodeRunner
from stepper import Stepper, StepForTrain
from trainer import TrainWithCallback, Trainer
from train_callback import TrainSaver, TrainEvaluator, TrainMonitor
from agent_over_manager import AgentOverManager

agent_over_manager = AgentOverManager(env['xn'] * env['yn'])
episode_runner = EpisodeRunner(
  StepForTrain(
    env['cartpoles'], Stepper(env['world'], env['cartpoles'], True),
    TrainWithCallback(
      Trainer(actor, critic, target_actor, target_critic, sars_buffer, 0.99, device),
      callbacks=[TrainMonitor(), TrainSaver(actor, critic, target_actor, target_critic), TrainEvaluator()]
    ),
    agent_over_manager,
    sars_buffer,
    env['xn'] * env['yn'],env
  ), env, actor, critic, agent_over_manager, epi_num, 5, device,TrainMonitor() # 60000 iteration! >> 6000
)
episode_runner.run_episode()

simulation_app.close()
