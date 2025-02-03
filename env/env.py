import os
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.cloner import Cloner
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.articulations import ArticulationView
import numpy as np
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core import World
from typing import TypedDict
from omni.isaac.sensor import IMUSensor
import omni
import time
from pxr import UsdPhysics
from omni.isaac.core import World
from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf
from omni.isaac.sensor import _sensor
world = World(stage_units_in_meters=1.0)
num=0
class Env (TypedDict):
  world: SimulationContext
  cartpoles: ArticulationView
  physics_dt: float
  rendering_dt: float
  xn: int
  yn: int
  imus:list
  
def get_imu_data(imu_prim_path):
    global num
    """IMU 센서의 선형 가속도 및 각속도를 가져오는 함수"""
    num +=1
    #  IMU 센서 인터페이스 가져오기
    _imu_sensor_interface = _sensor.acquire_imu_sensor_interface()
    
    #  센서 데이터 가져오기
    reading = _imu_sensor_interface.get_sensor_reading(
        imu_prim_path, use_latest_data=True, read_gravity=True  #  중력 값 포함 가능
    )
    
    if not reading.is_valid:
        print(f"x ERROR: IMU sensor at {imu_prim_path} returned invalid data!") # 종료되면 이게뜸...
        return None, None

    linear_acceleration = np.round(np.array([reading.lin_acc_x, reading.lin_acc_y, reading.lin_acc_z]),4)
    angular_velocity = np.round(np.array([reading.ang_vel_x, reading.ang_vel_y, reading.ang_vel_z]),4)
    
    if num %5000==0:
        print(f" * IMU Data from {imu_prim_path}:")
        print(f"   * Linear Acceleration: {linear_acceleration}, step:{num} ")
        print(f"   * Angular Acceleration: {angular_velocity}, step:{num}")
        print()

    return np.array(linear_acceleration, dtype=np.float32), np.array(angular_velocity, dtype=np.float32) # 센서 데이터  튜플로  생성...
      
def add_imu_sensor(cartpole_prim_path: str):
    global world
    """각 CartPole에 IMU 센서를 추가하는 함수"""
    imu_path = f"{cartpole_prim_path}/IMU"
    imu_parent_path = f"{cartpole_prim_path}" # cart에 한번 추가해보자!
    stage = omni.usd.get_context().get_stage()
    #print(imu_parent_path,888) # imu >> cart가 또 들어감..
    parent_prim = stage.GetPrimAtPath(imu_parent_path)
    
    

    if not parent_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        print("x Cart에 RigidBodyAPI가 적용되지 않았습니다. IMU 데이터가 갱신되지 않을 수 있습니다.")
    else:
        print("o Cart에 RigidBodyAPI가 적용되어 있습니다.")
        print()

    UsdPhysics.RigidBodyAPI.Apply(parent_prim)  # 여기서 오류발생.. 아마 .../cart까지를 경로로 가져야 할듯? ㅇㅋ
    
    
    imu_sensor = IMUSensor(
        prim_path=imu_path,
        name=f"imu_sensor_{cartpole_prim_path.split('_')[-1]}",
        frequency=100,  # Hz
        translation=np.array([0.0, 0.2, 0.0]),  # 센서 위치
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        linear_acceleration_filter_size = 10,
    angular_velocity_filter_size = 10,
    orientation_filter_size = 10)  # 방향 (쿼터니언)
    world.step(render=True)
    time.sleep(0.05) 
    value = imu_sensor.get_current_frame()
    #print(value)
    #print(777)
    
    
    imu_prim = stage.GetPrimAtPath(imu_path)

    if imu_prim and imu_prim.IsValid():
        imu_prim.CreateAttribute("linearAcceleration", Sdf.ValueTypeNames.Float3, False).Set(Gf.Vec3f(0, 0, 0))
        imu_prim.CreateAttribute("angularVelocity", Sdf.ValueTypeNames.Float3, False).Set(Gf.Vec3f(0, 0, 0))
        print(f"o IMU sensor added at {imu_path} with attributes") 
    else:
        print(f"x ERROR: IMU sensor was not created correctly at {imu_path}")
        
        
    imu_prim = stage.GetPrimAtPath(imu_path)

    if imu_prim and imu_prim.IsValid():
        print("o IMU 센서가 존재합니다.")
        print(f"o IMU Type: {imu_prim.GetTypeName()}")
        print(f"o 선형 가속도 존재 여부: {imu_prim.HasAttribute('linearAcceleration')}")
        print(f"o 각속도 존재 여부: {imu_prim.HasAttribute('angularVelocity')}")
        print(f"o 센서 활성화 여부: {imu_prim.HasAttribute('enabled') and imu_prim.GetAttribute('enabled').Get()}")
        print()
    else:
        print(f"x ERROR: IMU 센서가 존재하지 않습니다! {imu_path}")
    
    
    return imu_sensor

class CartpoleOriginPositionsGenerator:
  def __init__ (self, xn: int, yn: int):
    self.xn = xn
    self.yn = yn
  
  def generate (self):
    ret: np.ndarray = np.zeros((self.xn * self.yn, 3), dtype=np.float32)
    for y in range(self.yn):
      for x in range(self.xn):
        ret[y * self.xn + x][0] = x
        ret[y * self.xn + x][1] = 7 * y
    
    return ret

def generate_cartpoles (cartpole_origin_positions: np.ndarray[np.ndarray[np.float32]]):
  assets_root_path = 'omniverse://localhost/NVIDIA/Assets/Isaac/2023.1.1/'
  cartpole_usd_path = os.path.join(assets_root_path, 'Isaac/Robots/Cartpole/cartpole.usd')
  add_reference_to_stage(cartpole_usd_path, '/World/Cartpole_0')

  Articulation(prim_path='/World/Cartpole_0', name='cartpole')

  cloner = Cloner()
  cloner.filter_collisions('/physicsScene', '/World', list(map(lambda x: '/World/Cartpole_{}'.format(x), list(range(0, cartpole_origin_positions.shape[0])))))
  target_paths = cloner.generate_paths("/World/Cartpole", cartpole_origin_positions.shape[0])
  cloner.clone(source_prim_path="/World/Cartpole_0", prim_paths=target_paths, positions=cartpole_origin_positions)
  world = omni.isaac.core.SimulationContext.instance() # 수정
  world.step(render=True) #
  world.play()
  cartpoles = ArticulationView('/World/Cartpole_*', 'cartpole_view')
  
  imus = []
  for i in range(cartpole_origin_positions.shape[0]): # 이건 되는듯.
      imu_prim_path = f"/World/Cartpole_{i}/pole" # 경로 cart 로 수정 / 이게 원인이였다... ㅋ
      imu = add_imu_sensor(imu_prim_path)
      imus.append(imu)
      
      
      
      if omni.usd.get_context().get_stage().GetPrimAtPath(imu_prim_path):
          print(f"o IMU sensor successfully added at {imu_prim_path}")  #일단 여기 까진 된듯?
      else:
          print(f"x Failed to add IMU sensor at {imu_prim_path}")
          print()


    

  return cartpoles , imus

def initialize_env (world: SimulationContext, cartpoles: ArticulationView, physics_dt: float, rendering_dt: float):
  world.set_simulation_dt(physics_dt, rendering_dt)
  world.scene.add(cartpoles)
