from network.Network import Network

class ACNetwork(Network):

    def __init__(self, phi, num_channels, num_actions, output_directory,
                 session, train_mode=True, learning_rate=None, 
                 network_file=None, params_file=None, scope=""):
        # Create base network
        Network.__init__(self,
                         phi=phi,
                         num_channels=num_channels,
                         num_actions=num_actions,
                         output_directory=output_directory,
                         session=session,
                         train_mode=train_mode,
                         learning_rate=learning_rate,
                         network_file=network_file,
                         params_file=params_file,
                         scope=scope)
        
        # Set reserved variables
        self.pi = self.graph_dict["pi"][0]
        self.v = self.graph_dict["V"][0]
        self.actions = self.graph_dict["actions"][0]
        self.q_sa = self.graph_dict["q_sa"][0]
        self.loss_pi = self.graph_dict["loss_pi"][0]
        self.loss_v = self.graph_dict["loss_v"][0]
        self.IS_weights = self.graph_dict["IS_weights"][0]
        self.optimizer = self.graph_dict["optimizer"][0]
        self.train_step = self.graph_dict["train_step"][0]
    
    def learn(self, s1, a, q_sa, weights=None):
        s1 = self._check_state(s1)
        a = self._check_actions(a)
        if weights is None:
            weights = np.ones(a.shape[0])
        feed_dict={self.state: s1, self.actions: a, 
                   self.q_sa: q_sa, self.IS_weights: weights}
        learn_fns = [self.pi_loss, self.v_loss, 
                     self.pi_train_step, self.v_train_step]
        pi_l, v_l, pi_ts, v_ts = self.sess.run(learn_fns,
                                               feed_dict=feed_dict)
        return pi_l, v_l

    def get_value_output(self, s):
        feed_dict = {self.state: s}
        return self.sess.run(self.v, feed_dict=feed_dict)

    def get_policy_output(self, s):
        feed_dict = {self.state: s}
        return self.sess.run(self.pi, feed_dict=feed_dict)