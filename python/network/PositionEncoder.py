from network.Network import Network
import numpy as np

class PositionEncoder(Network):
    """
    Reserved names and name scopes:
    - state
    - position
    - loss
    - optimizer
    - train_step
    """
    def __init__(self, phi, num_channels, num_outputs, output_directory, 
                 session, train_mode=True, learning_rate=None, 
                 network_file=None, params_file=None, scope=""):
        # Initialize basic network settings
        Network.__init__(self,
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
        
        # Set variables from graph_dict
        self.pred_position = self.graph_dict["POS"][0]
        self.position = self.graph_dict["position"][0]
        self.loss = self.graph_dict["loss"][0]
        self.IS_weights = self.graph_dict["IS_weights"][0]
        self.optimizer = self.graph_dict["optimizer"][0]
        self.train_step = self.graph_dict["train_step"][0]

    def learn(self, s1, position, weights=None):
        s1 = self._check_state(s1)
        if weights is None:
            weights = np.ones([position.shape[0], 2])
        feed_dict = {self.state: s1, self.position: position, 
                        self.IS_weights: weights}
        feed_dict = self._check_train_mode(feed_dict)
        loss, train_step = self.sess.run([self.loss, self.train_step],
                                            feed_dict=feed_dict)
        return loss

    def save_model(self, model_name, global_step=None, save_meta=True,
                save_summaries=True, test_batch=None):
        self.saver.save(self.sess, self.params_dir + model_name, 
                        global_step=global_step,
                        write_meta_graph=save_meta)
        if save_summaries:
            if test_batch is None:
                raise ValueError("Test batch must be provided "
                                 + "to save summaries.")
            s1, position, w = test_batch
            s1 = self._check_state(s1)
            feed_dict = {self.state: s1, self.position: position, 
                            self.IS_weights: w}
            feed_dict = self._check_train_mode(feed_dict)
            sum_list_ = self.sess.run(self.sum_list, feed_dict=feed_dict)
            self.writer.add_summary(sum_list_, global_step)
            self.writer.flush()
            # TODO: implement event accumulator to save files (esp. histograms)
            # to CSV files.
            #self.ea.Reload()
            #print(self.ea.Tags())