## checking imports working fine
from queryscaler.server.queryscaler_environment import QueryscalerEnvironment
env = QueryscalerEnvironment()
obs = env.reset()
print(obs.task_description)