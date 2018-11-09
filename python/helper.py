import json

###############################################################################
# Methods to return class instances
# Note: Import statements (and dictionaries) are listed inside methods to avoid
# circular imports
###############################################################################

def create_agent(agent_filename, **kwargs):
    from agent import Agent, DQNAgent, DDQNAgent, DRQNAgent, DDRQNAgent, \
        ACERAgent, PositionEncoder, DecoderAgent
    agent_types = {"agent": Agent.Agent,
                   "dqn": DQNAgent.DQNAgent,
                   "ddqn": DDQNAgent.DDQNAgent,
                   "drqn": DRQNAgent.DRQNAgent,
                   "ddrqn": DDRQNAgent.DDRQNAgent,
                   "acer": ACERAgent.ACERAgent,
                   "position": PositionEncoder.PositionEncoder,
                   "decoder": DecoderAgent.DecoderAgent}
    agent_file = json.loads(open(agent_filename).read())
    agent_type = agent_file["agent_args"]["type"]
    return agent_types[agent_type](agent_file=agent_filename, **kwargs)

def create_network(network_filename, **kwargs):
    from network import DQNetwork, DRQNetwork, ACNetwork, PositionEncoder, DecoderNetwork
    network_types = {"dqn": DQNetwork.DQNetwork,
                     "dueling_dqn": DQNetwork.DQNetwork,
                     "drqn": DRQNetwork.DRQNetwork,
                     "dueling_drqn": DRQNetwork.DRQNetwork,
                     "ac": ACNetwork.ACNetwork,
                     "position": PositionEncoder.PositionEncoder,
                     "decoder": DecoderNetwork.DecoderNetwork}
    net_file = json.loads(open(network_filename).read())
    net_type = net_file["global_features"]["type"].lower()
    return network_types[net_type](network_file=network_filename, **kwargs)

def create_memory(memory_type, **kwargs):
    from memory import ReplayMemory, PrioritizedReplayMemory, PositionReplayMemory
    memory_types = {"standard": ReplayMemory.ReplayMemory,
                    "prioritized": PrioritizedReplayMemory.PrioritizedReplayMemory,
                    "position": PositionReplayMemory.PositionReplayMemory}
    return memory_types[memory_type](**kwargs)

def create_wrapper(env_type, **kwargs):
    from env import Gridworld, Atari
    env_types = {"gridworld": Gridworld.Gridworld,
                 "atari": Atari.Atari}
    return env_types[env_type](**kwargs)

###############################################################################
# Dictionaries encoding DoomGame features
###############################################################################

game_buttons = {"ATTACK":         0,
                "USE":            1,
                "JUMP":           2,
                "CROUCH":         3,
                "TURN180":        4,
                "ALTATTACK":      5,
                "RELOAD":         6,
                "ZOOM":           7,
                "SPEED":          8,
                "STRAFE":         9,
                "MOVE_RIGHT":     10,
                "MOVE_LEFT":      11,
                "MOVE_BACKWARD":  12,
                "MOVE_FORWARD":   13,
                "TURN_RIGHT":     14,
                "TURN_LEFT":      15,
                "LOOK_UP":        16,
                "LOOK_DOWN":      17,
                "MOVE_UP":        18,
                "MOVE_DOWN":      19,
                "LAND":           20,

                "SELECT_WEAPON1": 21,
                "SELECT_WEAPON2": 22,
                "SELECT_WEAPON3": 23,
                "SELECT_WEAPON4": 24,
                "SELECT_WEAPON5": 25,
                "SELECT_WEAPON6": 26,
                "SELECT_WEAPON7": 27,
                "SELECT_WEAPON8": 28,
                "SELECT_WEAPON9": 29,
                "SELECT_WEAPON0": 30,

                "SELECT_NEXT_WEAPON":          31,
                "SELECT_PREV_WEAPON":          32,
                "DROP_SELECTED_WEAPON":        33,

                "ACTIVATE_SELECTED_ITEM":      34,
                "SELECT_NEXT_ITEM":            35,
                "SELECT_PREV_ITEM":            36,
                "DROP_SELECTED_ITEM":          37,

                "LOOK_UP_DOWN_DELTA":          38,
                "TURN_LEFT_RIGHT_DELTA":       39,
                "MOVE_FORWARD_BACKWARD_DELTA": 40,
                "MOVE_LEFT_RIGHT_DELTA":       41,
                "MOVE_UP_DOWN_DELTA":          42}

def get_game_button_indices(button_names):
    if type(button_names) is not list:
        button_names = list(button_names)
    return [game_buttons[name] for name in button_names]

def get_game_button_names(button_indices):
    if type(button_indices) is not list:
        button_indices = list(button_indices)
    reverse_dict = {}
    for key, value in game_buttons.items():
        reverse_dict[value] = key
    return [reverse_dict[index] for index in button_indices]