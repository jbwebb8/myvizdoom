from network.Network import Network
import numpy as np

class DQNetwork(Network):
    """
    Reserved names and name scopes:
    - state
    - Q
    - actions
    - target_q
    - loss
    - IS_weights
    - optimizer
    - train_step
    - best_action
    """
    def __init__(self, phi, num_channels, num_actions, output_directory, 
                 session, train_mode=True, learning_rate=None, 
                 network_file=None, params_file=None, scope=""):
        # Initialize basic network settings
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
        
        # Set Q-learning variables
        self.q = self.graph_dict["Q"][0]
        self.actions = self.graph_dict["actions"][0]
        self.target_q = self.graph_dict["target_q"][0]
        self.loss = self.graph_dict["loss"][0]
        self.IS_weights = self.graph_dict["IS_weights"][0]
        self.optimizer = self.graph_dict["optimizer"][0]
        self.train_step = self.graph_dict["train_step"][0]
        self.best_a = self.graph_dict["best_action"][0]
    
    def learn(self, s1, a, target_q, weights=None):
        s1 = self._check_state(s1)
        a = self._check_actions(a)
        if weights is None:
            weights = np.ones(a.shape[0])
        feed_dict = {s_: s for s_, s in zip(self.state, s1)} 
        feed_dict.update({self.actions: a, 
                          self.target_q: target_q, 
                          self.IS_weights: weights})
        feed_dict = self._check_train_mode(feed_dict)
        loss_, train_step_ = self.sess.run([self.loss, self.train_step],
                                           feed_dict=feed_dict)
        return loss_
    
    def get_q_values(self, s1_):
        s1_ = self._check_state(s1_)
        feed_dict={s_: s for s_, s in zip(self.state, s1_)}
        feed_dict = self._check_train_mode(feed_dict)
        return self.sess.run(self.q, feed_dict=feed_dict)
    
    def get_best_action(self, s1_):
        s1_ = self._check_state(s1_)
        feed_dict={s_: s for s_, s in zip(self.state, s1_)}
        feed_dict = self._check_train_mode(feed_dict)
        return self.sess.run(self.best_a, feed_dict=feed_dict)

    def save_model(self, model_name, global_step=None, save_meta=True,
                   save_summaries=True, test_batch=None):
        self.saver.save(self.sess, self.params_dir + model_name, 
                        global_step=global_step,
                        write_meta_graph=save_meta)
        if save_summaries:
            var_sum_ = self.sess.run(self.var_sum)
            self.writer.add_summary(var_sum_, global_step)
            if test_batch is not None:
                s1, a, target_q, w, _ = test_batch
                s1 = self._check_state(s1)
                a = self._check_actions(a)
                feed_dict=({s_: s for s_, s in zip(self.state, s1)} 
                           + {self.actions: a, 
                              self.target_q: target_q, 
                              self.IS_weights: weights})
                feed_dict = self._check_train_mode(feed_dict)
                neur_sum_ = self.sess.run(self.neur_sum,
                                          feed_dict=feed_dict)
                self.writer.add_summary(neur_sum_, global_step)
                grad_sum_ = self.sess.run(self.grad_sum,
                                          feed_dict=feed_dict)
                self.writer.add_summary(grad_sum_, global_step)
            # TODO: implement event accumulator to save files (esp. histograms)
            # to CSV files.
            #self.ea.Reload()
            #print(self.ea.Tags())