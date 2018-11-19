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

GAME_BUTTONS = {"ATTACK":         0,
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
    return [GAME_BUTTONS[name] for name in button_names]

def get_game_button_names(button_indices):
    if type(button_indices) is not list:
        button_indices = list(button_indices)
    reverse_dict = {}
    for key, value in GAME_BUTTONS.items():
        reverse_dict[value] = key
    return [reverse_dict[index] for index in button_indices]

from vizdoom import GameVariable

GAME_VARIABLES = {'KILLCOUNT': GameVariable.KILLCOUNT,
                  'ITEMCOUNT': GameVariable.ITEMCOUNT,
                  'SECRETCOUNT': GameVariable.SECRETCOUNT,
                  'FRAGCOUNT': GameVariable.FRAGCOUNT,
                  'DEATHCOUNT': GameVariable.DEATHCOUNT,
                  'HITCOUNT': GameVariable.HITCOUNT,
                  'HITS_TAKEN': GameVariable.HITS_TAKEN,
                  'DAMAGECOUNT': GameVariable.DAMAGECOUNT,
                  'DAMAGE_TAKEN': GameVariable.DAMAGE_TAKEN,
                  'HEALTH': GameVariable.HEALTH,
                  'ARMOR': GameVariable.ARMOR,
                  'DEAD': GameVariable.DEAD,
                  'ON_GROUND': GameVariable.ON_GROUND,
                  'ATTACK_READY': GameVariable.ATTACK_READY,
                  'ALTATTACK_READY': GameVariable.ALTATTACK_READY,
                  'SELECTED_WEAPON': GameVariable.SELECTED_WEAPON,
                  'SELECTED_WEAPON_AMMO:' GameVariable.SELECTED_WEAPON_AMMO,

                  'AMMO0': GameVariable.AMMO0,
                  'AMMO1': GameVariable.AMMO1,
                  'AMMO2': GameVariable.AMMO2,
                  'AMMO3': GameVariable.AMMO3,
                  'AMMO4': GameVariable.AMMO4,
                  'AMMO5': GameVariable.AMMO5,
                  'AMMO6': GameVariable.AMMO6,
                  'AMMO7': GameVariable.AMMO7,
                  'AMMO8': GameVariable.AMMO8,
                  'AMMO9': GameVariable.AMMO9,
                  'WEAPON0': GameVariable.WEAPON0,
                  'WEAPON1': GameVariable.WEAPON1,
                  'WEAPON2': GameVariable.WEAPON2,
                  'WEAPON3': GameVariable.WEAPON3,
                  'WEAPON4': GameVariable.WEAPON4,
                  'WEAPON5': GameVariable.WEAPON5,
                  'WEAPON6': GameVariable.WEAPON6,
                  'WEAPON7': GameVariable.WEAPON7,
                  'WEAPON8': GameVariable.WEAPON8,
                  'WEAPON9': GameVariable.WEAPON9,
                  'POSITION_X': GameVariable.POSITION_X,
                  'POSITION_Y': GameVariable.POSITION_Y,
                  'POSITION_Z': GameVariable.POSITION_Z,
                  'ANGLE': GameVariable.ANGLE,
                  'PITCH': GameVariable.PITCH,
                  'ROLL': GameVariable.ROLL,
                  'VIEW_HEIGHT': GameVariable.VIEW_HEIGHT,
                  'VELOCITY_X': GameVariable.VELOCITY_X,
                  'VELOCITY_Y': GameVariable.VELOCITY_Y,
                  'VELOCITY_Z': GameVariable.VELOCITY_Z,

                  'CAMERA_POSITION_X': GameVariable.CAMERA_POSITION_X,
                  'CAMERA_POSITION_Y': GameVariable.CAMERA_POSITION_Y,
                  'CAMERA_POSITION_Z': GameVariable.CAMERA_POSITION_Z,
                  'CAMERA_ANGLE': GameVariable.CAMERA_ANGLE,
                  'CAMERA_PITCH': GameVaraible.CAMERA_PITCH,
                  'CAMERA_ROLL': GameVariable.CAMERA_ROLL,
                  'CAMERA_FOV': GameVariable.CAMERA_FOV,
                  'PLAYER_NUMBER': GameVariable.PLAYER_NUMBER,
                  'PLAYER_COUNT': GameVariable.PLAYER_COUNT,
                  'PLAYER1_FRAGCOUNT': GameVariable.PLAYER1_FRAGCOUNT,
                  'PLAYER2_FRAGCOUNT': GameVariable.PLAYER2_FRAGCOUNT,
                  'PLAYER3_FRAGCOUNT': GameVariable.PLAYER3_FRAGCOUNT,
                  'PLAYER4_FRAGCOUNT': GameVariable.PLAYER4_FRAGCOUNT,
                  'PLAYER5_FRAGCOUNT': GameVariable.PLAYER5_FRAGCOUNT,
                  'PLAYER6_FRAGCOUNT': GameVariable.PLAYER6_FRAGCOUNT,
                  'PLAYER7_FRAGCOUNT': GameVariable.PLAYER7_FRAGCOUNT,
                  'PLAYER8_FRAGCOUNT': GameVariable.PLAYER8_FRAGCOUNT,
                  'PLAYER9_FRAGCOUNT': GameVariable.PLAYER9_FRAGCOUNT,
                  'PLAYER10_FRAGCOUNT': GameVariable.PLAYER10_FRAGCOUNT,
                  'PLAYER11_FRAGCOUNT': GameVariable.PLAYER11_FRAGCOUNT,
                  'PLAYER12_FRAGCOUNT': GameVariable.PLAYER12_FRAGCOUNT,
                  'PLAYER13_FRAGCOUNT': GameVariable.PLAYER13_FRAGCOUNT,
                  'PLAYER14_FRAGCOUNT': GameVariable.PLAYER14_FRAGCOUNT,
                  'PLAYER15_FRAGCOUNT': GameVariable.PLAYER15_FRAGCOUNT,
                  'PLAYER16_FRAGCOUNT': GameVariable.PLAYER16_FRAGCOUNT,
                  
                  # User (ACS) variables
                  # USER0 is reserved for reward
                  'USER1': GameVariable.USER1,
                  'USER2': GameVariable.USER2,
                  'USER3': GameVariable.USER3,
                  'USER4': GameVariable.USER4,
                  'USER5': GameVariable.USER5,
                  'USER6': GameVariable.USER6,
                  'USER7': GameVariable.USER7,
                  'USER8': GameVariable.USER8,
                  'USER9': GameVariable.USER9,
                  'USER10': GameVariable.USER10,
                  'USER11': GameVariable.USER11,
                  'USER12': GameVariable.USER12,
                  'USER13': GameVariable.USER13,
                  'USER14': GameVariable.USER14,
                  'USER15': GameVariable.USER15,
                  'USER16': GameVariable.USER16,
                  'USER17': GameVariable.USER17,
                  'USER18': GameVariable.USER18,
                  'USER19': GameVariable.USER19,
                  'USER20': GameVariable.USER20,
                  'USER21': GameVariable.USER21,
                  'USER22': GameVariable.USER22,
                  'USER23': GameVariable.USER23,
                  'USER24': GameVariable.USER24,
                  'USER25': GameVariable.USER25,
                  'USER26': GameVariable.USER26,
                  'USER27': GameVariable.USER27,
                  'USER28': GameVariable.USER28,
                  'USER29': GameVariable.USER29,
                  'USER30': GameVariable.USER30,
                  'USER31': GameVariable.USER31,
                  'USER32': GameVariable.USER32,
                  'USER33': GameVariable.USER33,
                  'USER34': GameVariable.USER34,
                  'USER35': GameVariable.USER35,
                  'USER36': GameVariable.USER36,
                  'USER37': GameVariable.USER37,
                  'USER38': GameVariable.USER38,
                  'USER39': GameVariable.USER39,
                  'USER40': GameVariable.USER40,
                  'USER41': GameVariable.USER41,
                  'USER42': GameVariable.USER42,
                  'USER43': GameVariable.USER43,
                  'USER44': GameVariable.USER44,
                  'USER45': GameVariable.USER45,
                  'USER46': GameVariable.USER46,
                  'USER47': GameVariable.USER47,
                  'USER48': GameVariable.USER48,
                  'USER49': GameVariable.USER49,
                  'USER50': GameVariable.USER50,
                  'USER51': GameVariable.USER51,
                  'USER52': GameVariable.USER52,
                  'USER53': GameVariable.USER53,
                  'USER54': GameVariable.USER54,
                  'USER55': GameVariable.USER55,
                  'USER56': GameVariable.USER56,
                  'USER57': GameVariable.USER57,
                  'USER58': GameVariable.USER58,
                  'USER59': GameVariable.USER59,
                  'USER60': GameVariable.USER60}

def get_game_variables(gv_keys):
    if type(gv_keys) is not list:
        gv_keys = list(gv_keys)
    return [GAME_VARIABLES[key] for key in gv_keys]