from network.Network import Network
from network.NetworkBuilder import NetworkBuilder
import tensorflow as tf
import numpy as np

class DecoderNetwork():
    # Redefine either to be full subclass of Network (i.e. utilize NetworkBuilder)
    # or take an instance of a Network (sub)class as a parameter
    def __init__(self,
                 network_file,
                 encoding_network,
                 results_dir,
                 learning_rate=0.001,
                 params_file=None,
                 decode_list=["actions", "positions"],
                 prediction_len=1,
                 scope="decoder_network"):
        self.encoding_net = encoding_network
        self.sess = encoding_network.sess
        self.results_dir = results_dir
        self.learning_rate = learning_rate
        self.decode_list = decode_list
        self.pred_len = prediction_len
        self.scope = scope
        with tf.name_scope(self.scope):
            # Load JSON file with NetworkBuilder
            if not network_file.lower().endswith(".json"): 
                raise SyntaxError("File format not supported for network settings. " \
                                    "Please use JSON file.")
            self.name = network_file[0:-5]
            builder = NetworkBuilder(self, network_file)
            builder.graph_dict.update(self.encoding_net.graph_dict)
            self.graph_dict, self.data_format = builder.load_json(network_file)
            
            # Get target, prediction, and loss variables
            self.target_list = []
            self.pred_list = []
            self.loss_list = []
            for d in self.decode_list:
                try:
                    t, p, l = self._get_decoder_variables(d)
                    if t is not None: self.target_list.append(t)
                    if p is not None: self.pred_list.append(p)
                    if l is not None: self.loss_list.append(l)   
                except KeyError as e:
                    print("Warning: Key " + str(e) + " not found in builder."
                          + " Ignoring \"%s\" in decode list." % d)
            self.loss = self.graph_dict["loss_tot"][0]
            self.loss_list.append(self.loss)

            # Get other training variables
            self.IS_weights = self.graph_dict["IS_weights"][0]
            self.optimizer = self.graph_dict["optimizer"][0]
            self.train_step = self.graph_dict["train_step"][0]

            # Add summaries
            with tf.name_scope("summaries"):
                sum_list = builder.add_summaries(loss_list=self.loss_list)

            # Create objects for saving
            self.saver = tf.train.Saver(max_to_keep=None)
            self.graph = tf.get_default_graph()
            self.sum_list = tf.summary.merge(sum_list)
            self.writer = tf.summary.FileWriter(self.results_dir, self.graph)
        
        # Initialize variables or load parameters if provided
        if params_file is not None:
            self.saver.restore(self.sess, params_file)
        else:
            var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                         scope=self.scope)
            self.sess.run(tf.variables_initializer(var_list))
        
        # Get functions to check decoder inputs
        self._check_fn_list = [self._get_check_fn(d) for d in self.decode_list
                               if self._get_check_fn(d) is not None]
    
    def _get_check_fn(self, decoder_name):
        if decoder_name == "positions":
            return self._check_positions
        elif decoder_name == "actions":
            return self._check_actions
        elif decoder_name == "r":
            return None
    
    def _get_decoder_variables(self, decoder_name):
        if decoder_name.lower() == "positions":
            target = self.graph_dict["target_pos"][0]
            pred = self.graph_dict["POS"][0]
            loss = self.graph_dict["loss_pos"][0]
            return target, pred, loss
        elif decoder_name.lower() == "r":
            #target = self.graph_dict["target_r"][0]
            target = None
            pred = self.graph_dict["R"][0]
            loss = self.graph_dict["loss_r"][0]
            return target, pred, loss
        elif decoder_name.lower() == "actions":
            target = self.graph_dict["target_act"][0]
            pred = self.graph_dict["ACT_softmax"][0]
            loss = self.graph_dict["loss_act"][0]
            return target, pred, loss
    
    def _check_positions(self, x):
        try:
            ndim = x.ndim
        except AttributeError: # not numpy array
            x = np.asarray(x)
            ndim = x.ndim
        if ndim < 2:
            if len(x) > 3:
                print("Warning: position arrays must be 2D. Failure to do so may result"
                      + "in an indexing error.")
            return np.reshape(x, [1] + list(x.shape))
        else:
            return x
    
    def _check_actions(self, a):
        try:
            ndim = a.ndim
        except AttributeError: # not numpy array
            a = np.asarray(a)
            ndim = a.ndim
        if ndim > 1:    
            a = np.squeeze(a)

        return a

    def learn(self, s1, targets, weights=None):
        s1 = self.encoding_net._check_state(s1)
        feed_dict = {s_: s for s_, s in zip(self.encoding_net.state, s1)}
        for i, t in enumerate(targets):
            t = self._check_fn_list[i](t)
            feed_dict.update({self.target_list[i]: t})
        if weights is None:
            weights = np.ones(s1[0].shape[0])
        feed_dict.update({self.IS_weights: weights})
        feed_dict = self.encoding_net._check_train_mode(feed_dict)
        loss_, train_step_ = self.sess.run([self.loss, self.train_step],
                                           feed_dict=feed_dict)
        return loss_
    
    def get_loss(self, s1, targets, weights=None, all_losses=False):
        s1 = self.encoding_net._check_state(s1)
        feed_dict = {s_: s for s_, s in zip(self.encoding_net.state, s1)}
        for i, t in enumerate(targets):
            t = self._check_fn_list[i](t)
            feed_dict.update({self.target_list[i]: t})
        if weights is None:
            weights = np.ones(s1[0].shape[0])
        feed_dict.update({self.IS_weights: weights})
        feed_dict = self.encoding_net._check_train_mode(feed_dict)
        if all_losses:
            run_list = self.loss_list
        else:
            run_list = [self.loss]
        loss_ = self.sess.run(run_list, feed_dict=feed_dict)
        return loss_
    
    def predict(self, s1):
        s1 = self.encoding_net._check_state(s1)
        feed_dict = {s_: s for s_, s in zip(self.encoding_net.state, s1)}
        feed_dict = self.encoding_net._check_train_mode(feed_dict)
        pred_list_ = self.sess.run(self.pred_list,
                                  feed_dict=feed_dict)
        return pred_list_
    
    def save_model(self, model_name, global_step=None, save_meta=True,
                   save_summaries=True, test_batch=None):
        self.saver.save(self.sess, self.results_dir + model_name, 
                        global_step=global_step,
                        write_meta_graph=save_meta)
        if save_summaries and test_batch is not None:
            s1 = test_batch[0]
            targets = test_batch[1:]
            s1 = self.encoding_net._check_state(s1)
            feed_dict={s_: s for s_, s in zip(self.encoding_net.state, s1)}
            for i, t in enumerate(targets):
                t = self._check_fn_list[i](t)
                feed_dict.update({self.target_list[i]: t})
            weights = np.ones(s1[0].shape[0])
            feed_dict.update({self.IS_weights: weights})
            feed_dict = self.encoding_net._check_train_mode(feed_dict)
            #for s in self.sum_list:
            #    s_ = self.sess.run(s, feed_dict=feed_dict)
            #    self.writer.add_summary(s_, global_step)
            s_ = self.sess.run(self.sum_list, feed_dict=feed_dict)
            self.writer.add_summary(s_, global_step)