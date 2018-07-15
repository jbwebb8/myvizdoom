from helper import create_network
import numpy as np
import json
from copy import deepcopy

class DecoderAgent():

    NET_JSON_DIR = "../networks/"
    MAIN_SCOPE = "decoder_network"
    DEFAULT_AGENT_ARGS = {"agent_name":         "default",
                          "net_file":           "default",
                          "alpha":              0.00025}

    def __init__(self, 
                 main_agent,
                 game=None,
                 output_directory=None,
                 agent_file=None,
                 params_file=None,
                 train_mode=True):
    
        # Set up input/output
        self.main_agent = main_agent
        if game is None:
            self.game = main_agent.game
        else:
            self.game = game
        if output_directory is None:
            self.net_dir = main_agent.net_dir + "decoder_net/"
        else:
            if not output_directory.endswith("/"): 
                output_directory += "/"
            self.net_dir = output_directory + "decoder_net/"

        # Load learning and network parameters
        if agent_file is not None:
            self._load_agent_file(agent_file)
        else:
            self.agent_name         = kwargs.pop("agent_name", 
                                                 self.DEFAULT_AGENT_ARGS["agent_name"])
            self.net_file           = kwargs.pop("net_name", 
                                                 self.DEFAULT_AGENT_ARGS["net_file"])
            self.alpha              = kwargs.pop("alpha", 
                                                 self.DEFAULT_AGENT_ARGS["alpha"])                                 
            # TODO: finish rest of default agent args
        
        # Save readable network pointers
        if not self.net_file.startswith(self.NET_JSON_DIR):
            self.net_file = self.NET_JSON_DIR + self.net_file
        if not self.net_file.endswith(".json"):
            self.net_file += ".json"
        self.params_file = params_file

        # Create decoder network
        self.network = create_network(self.net_file,
                                      encoding_network=self.main_agent.network,
                                      results_dir=self.net_dir,
                                      params_file=self.params_file,
                                      learning_rate=self.alpha,
                                      decode_list=self.decode_list,
                                      prediction_len=self.pred_len,
                                      scope=self.MAIN_SCOPE)
    
        # Create separate replay memory buffer to store positions 
        # in addition to transition variables
        self.train_mode = train_mode
        if self.train_mode:
            self.memory = self.main_agent.create_memory(self.rm_type)
            self.memory.add_auxiliary_variables([2]) # position

    def _load_agent_file(self, agent_file):
        """Grabs arguments from agent file"""
        # Open JSON file
        if not agent_file.lower().endswith(".json"): 
            raise Exception("No agent JSON file.")
        agent = json.loads(open(agent_file).read())

        # Convert "None" string to None type (not supported in JSON)
        def recursive_search(d, keys):
            if isinstance(d, dict):
                for k, v in zip(d.keys(), d.values()):
                    keys.append(k)
                    recursive_search(v, keys)
                if len(keys) > 0: # avoids error at end
                    keys.pop()
            else:
                if d == "None":
                    t = agent 
                    for key in keys[:-1]:
                        t = t[key]
                    t[keys[-1]] = None
                keys.pop()

        recursive_search(agent, [])

        # TODO: implement get method to catch KeyError
        self.agent_name = agent["agent_args"]["name"]
        self.agent_type = agent["agent_args"]["type"]
        self.decode_list = agent["agent_args"]["decode_list"]
        self.net_file = agent["network_args"]["name"]
        self.net_type = agent["network_args"]["type"]
        self.alpha = agent["network_args"]["alpha"]
        self.epsilon = agent["learning_args"]["epsilon"]
        self.batch_size = agent["learning_args"]["batch_size"]
        self.rm_type = agent["memory_args"]["replay_memory_type"]
        self.rm_capacity = agent["memory_args"]["replay_memory_size"]
        self.rm_start_size = agent["memory_args"]["replay_memory_start_size"]
        self.pred_len = agent["memory_args"]["prediction_length"]
    
    def _get_targets(self, positions=None, actions=None):
        targets = []
        for d in self.decode_list:
            if d == "positions":
                targets.append(positions)
            elif d == "actions":
                targets.append(actions)
        
        return targets

    def perform_learning_step(self):
        # Get current state and position
        s1 = deepcopy(self.main_agent.state)
        x = self.main_agent.track_position()[1:3]
        
        # Make and get action
        r = self.main_agent.make_action()
        a = self.main_agent.track_action(index=True)

        # Get new state if not terminal.
        isterminal = self.game.is_episode_finished()
        if not isterminal:
            # Get new state
            self.main_agent.update_state()
            s2 = deepcopy(self.main_agent.state)
        else:
            # Terminal state set to zero
            s2 = self.main_agent.get_zero_state()

        # Add transition to replay memory if action was not random
        # TODO: how to incorporate n-step?
        self.memory.add_transition(s1, a, s2, isterminal, r, x)

        # Learn from replay memory
        if self.rm_start_size <= self.memory.size:            
            # Learn from minibatch of replay memory samples
            offset = ( ((self.pred_len - (self.memory.capacity % self.pred_len))
                        * self.memory.lap) % self.pred_len )
            idx_start = np.arange(offset, self.memory.capacity, self.pred_len)
            s1, a, s2, isterminal, r, w, idx, x = self.memory.get_sample(self.batch_size,
                                                                            idx_start=idx_start)
            s1 = [s1[0][::self.pred_len], s1[1][::self.pred_len]]
            w = w[::self.pred_len]
            targets = self._get_targets(positions=x, actions=a)
            _ = self.network.learn(s1, targets=targets, weights=w)
    
    def get_loss(self, state, position=None, action=None):
        targets = self._get_targets(positions=position, actions=action)
        return self.network.get_loss(state, targets=targets)

    def get_prediction(self, state):
        return self.network.predict(state)
    
    def get_test_batch(self, states, positions=None, actions=None):
        targets = self._get_targets(positions=positions, actions=actions)
        return [states] + targets

    def save_model(self, model_name, global_step=None, save_meta=True,
                   save_summaries=True, test_batch=None):
        self.network.save_model(model_name,
                                global_step=global_step,
                                save_meta=save_meta,
                                save_summaries=save_summaries,
                                test_batch=test_batch)