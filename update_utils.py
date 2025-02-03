def soft_update (target, origin, tau: float):
  for target_param, origin_param in zip(target.parameters(), origin.parameters()):
    target_param.data.copy_(target_param.data * (1.0 - tau) + origin_param.data * tau)

def hard_update (target, origin):
  for target_param, origin_param in zip(target.parameters(), origin.parameters()):
    target_param.data.copy_(target_param.data)
