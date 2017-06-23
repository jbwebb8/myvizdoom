class Network:
    """
    Pre-constructed neural networks

    Input:
    - name: name of pre-constructed network

    Output:
    - Returns a Tensorflow network
    """
    def _init_(self, name):
        self.name = name
        self.network = _create_network()
    
    def _create_network(self):
        if (self.name == "dqn_basic"):
            return _create_dqn_basic()
        else:
            raise NameError("No network exists for ", self.name, ".")
        
    def _create_dqn_basic(self):
        # TODO: implement base DQN (2 layer, 8 features) using Tensorflow
    
    # TODO: Think how to modularize network creation. Different name for every network
    # or broad names that allow user to specify details (like features, layers, etc.)
