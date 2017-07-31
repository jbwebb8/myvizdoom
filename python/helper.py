from agent import *
from memory import *
import json

agent_types = {"agent": Agent.Agent,
               "dqn": DQNAgent.DQNAgent,
               "ddqn": DDQNAgent.DDQNAgent,
               "a3c": A3CAgent.A3CAgent}

memory_types = {"standard": ReplayMemory.ReplayMemory,
                "prioritized": PrioritizedReplayMemory.PrioritizedReplayMemory}

def create_agent(agent_filename, **kwargs):
    agent_file = json.loads(open(agent_filename).read())
    agent_type = agent_file["agent_args"]["type"]
    return agent_types[agent_type](agent_file=agent_filename, **kwargs)

def create_memory(memory_type, **kwargs):
    return memory_types[memory_type](**kwargs)