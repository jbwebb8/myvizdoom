from network.DQNetwork import DQNetwork
import numpy as np

class DRQNetwork(DQNetwork):

    def __init__(self, 
                 phi, 
                 num_channels, 
                 num_outputs, 
                 output_directory, 
                 session,
                 batch_size=32,
                 train_mode=True, 
                 learning_rate=None, 
                 network_file=None, 
                 params_file=None, 
                 scope=""):
    
        DQNetwork.__init__(self,
                           phi=phi,
                           num_channels=num_channels,
                           num_outputs=num_outputs,
                           output_directory=output_directory,
                           session=session,
                           train_mode=train_mode,
                           learning_rate=learning_rate,
                           network_file=network_file,
                           params_file=params_file,
                           scope=scope)
    
        # RNN components
        self.rnn_states = self.graph_dict["rnn_states"]
        self.rnn_init_states = self.graph_dict["rnn_init_states"]
        self.batch_size = self.graph_dict["batch_size"][0]
        self.train_batch_size = batch_size
        self.reset_rnn_state()
    
    def reset_rnn_state(self, batch_size=1):
        #feed_dict = {s: np.zeros([batch_size] + s.get_shape().as_list()[1:]) for s in self.state}
        feed_dict = {self.batch_size: batch_size}
        self.rnn_current_states = self.sess.run(self.rnn_init_states,
                                                feed_dict=feed_dict)

    def update_rnn_state(self, s1, batch_size=1):
        s1 = self._check_state(s1)
        feed_dict = {s_: s for s_, s in zip(self.state, s1)}
        feed_dict.update({self.batch_size: batch_size})
        self.rnn_current_states = self.sess.run(self.rnn_states, 
                                                feed_dict=feed_dict)

    
    # NOTE: do not need to override learn function because rnn state
    # always initialized with zeros during training

    # Create DRQNetwork class that inherits DQNetwork. Override learn and other
    # functions that would need to feed the RNN state. Need to figure out how to 
    # let DRQNAgent inherit from DQNAgent class but create DRQNetworks for its
    # network (and target network).
    # 
    # Solved: replaced target_network = DQNetwork...
    # with target_network = create_network...
    #
    # Try without overriding functions; instead, feed rnn states with
    # _check_train_mode function
    # 
    # Problem is getting current rnn state(s) after running each function
    # 
    # What if we overrode _check_state to additionally update the initial rnn states
    # by feeding in the given agent state (which provides batch_size)
    #
    # Need to feed batch_size into network (since it will then infer trace_length)
    # but how to do that given generic Network creation from generic Agent class,
    # which does not have batch_size (only subclass DQAgent does)?