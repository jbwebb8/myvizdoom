from vizdoom import *
import numpy as np
from helper import create_agent
import argparse
import os, errno, sys
from shutil import copy
from time import time, sleep
from random import random, randint
from copy import deepcopy


# Command line arguments
parser = argparse.ArgumentParser(description="Decode an agent network.")
parser.add_argument("--main-agent", default=None, metavar="",
                    help="json file containing main agent setup")
parser.add_argument("--main-params", default=None, metavar="", 
                    help="TF filename (no extension) containing network \
                          parameters for main agent")
parser.add_argument("--decoder-agent", default=None, metavar="",
                    help="json file containing main agent setup")
parser.add_argument("--decoder-params", default=None, metavar="", 
                    help="TF filename (no extension) containing network \
                          parameters for decoder agent")
parser.add_argument("--config-file", default=None, metavar="",
                    help="config file for scenario")
parser.add_argument("--results-dir", default=None, metavar="",
                    help="directory where results will be saved")
parser.add_argument("-a", "--action-set", default=None, metavar="", 
                    help="name of action set available to agent")
parser.add_argument("-e", "--epochs", type=int, default=100, metavar="", 
                    help="number of epochs to train")
parser.add_argument("-s", "--learning-steps", type=int, default=5000, metavar="", 
                    help="learning steps per epoch")
parser.add_argument("-t", "--test-steps", type=int, default=1000, metavar="", 
                    help="test steps per epoch")
parser.add_argument("-f", "--save-freq", type=int, default=0, metavar="", 
                    help="save params every x epochs")
parser.add_argument("-n", "--name", default="train", metavar="", 
                    help="experiment name (for saving files)")
parser.add_argument("-d", "--description", default="training", metavar="", 
                    help="description of experiment")
args = parser.parse_args()

# Custom error message for required arguments (default is vague)
required_args = {"--main-agent": args.main_agent, 
                 "--decoder-agent": args.decoder_agent, 
                 "--config-file": args.config_file,
                 "--results-dir": args.results_dir}
for flag, arg in required_args.items():
    assert arg is not None, "Argument %s is required." % flag

# Experiment setup
main_agent_file_path = args.main_agent
main_params_file_path = args.main_params
decoder_agent_file_path = args.decoder_agent
decoder_params_file_path = args.decoder_params
config_file_path = args.config_file
results_dir = args.results_dir
if not results_dir.endswith("/"): 
    results_dir += "/"
action_set = args.action_set
epochs = args.epochs
learning_steps_per_epoch = args.learning_steps
test_steps_per_epoch = args.test_steps
save_freq = args.save_freq
if save_freq == 0: save_freq = epochs
exp_name = args.name
exp_descr = args.description

# Makes directory if does not already exist
def make_directory(folders):
    for f in folders:
        try:
            os.makedirs(f)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

# Saves txt file of important experimental settings
# and copies (small) configuration files
def save_exp_details(folder, main_agent, decoder_agent):
    f = open(folder + "settings.txt", "w+")
    f.write("Name: " + exp_name + "\n")
    f.write("Description: " + exp_descr + "\n")
    f.write("Main agent file: " + main_agent_file_path + "\n")
    main_net_file_path = main_agent.net_file
    f.write("Main network file: " + main_net_file_path + "\n")
    f.write("Main params file: " + str(main_params_file_path) + "\n")
    f.write("Decoder agent file: " + decoder_agent_file_path + "\n")
    decoder_net_file_path = decoder_agent.net_file
    f.write("Decoder network file: " + decoder_net_file_path + "\n")
    f.write("Decoder params file: " + str(decoder_params_file_path) + "\n")
    f.write("Config file: " + config_file_path + "\n")
    f.write("Action set: " + str(action_set) + "\n")
    f.write("Epochs: " + str(epochs) + "\n")
    f.write("Learning steps per epoch: " + str(learning_steps_per_epoch) + "\n")
    f.write("Test steps per epoch: " + str(test_steps_per_epoch))
    files_to_copy = [main_agent_file_path, main_net_file_path, 
                     decoder_agent_file_path, decoder_net_file_path, 
                     config_file_path]
    for fp in files_to_copy:
        new_fp = folder + fp.split("/")[-1]
        while os.path.exists(new_fp):
            t = new_fp.split(".")
            new_fp = '.'.join(['.'.join(t[0:-1]) + '_1', t[-1]])
        copy(fp, new_fp)

# Initializes DoomGame from config file
def initialize_vizdoom(config_file):
    print("Initializing doom... "), sys.stdout.flush()
    game = DoomGame()
    game.load_config(config_file)
    game.init()
    return game  

# Make output directories
details_dir = results_dir + "details/"
game_dir = results_dir + "game_data/"
max_dir = results_dir + "max_data/"
make_directory([results_dir, details_dir, game_dir, max_dir])

# Create game and agent
game = initialize_vizdoom(config_file_path) 
agent = create_agent(main_agent_file_path,
                     game=game, 
                     params_file=main_params_file_path,
                     action_set=action_set,
                     output_directory=results_dir,
                     train_mode=False)

# Build decoder network
decoder = create_agent(decoder_agent_file_path,
                       main_agent=agent,
                       game=agent.game,
                       output_directory=None,
                       params_file=decoder_params_file_path,
                       train_mode=True)

# Save experimental details
save_exp_details(details_dir, agent, decoder)

# Train and test decoder for specified number of epochs
loss_history = []
time_start = time()
for epoch in range(epochs):
    print("\nEpoch %d\n-------" % (epoch + 1))

    # Training
    print("Training...")
    train_episodes_finished = 0
    agent.initialize_new_episode()
    learning_step = 0
    while learning_step < learning_steps_per_epoch:
        # With probability epsilon make a random action.
        # Otherwise, choose best action.
        if random() <= decoder.epsilon:
            random_action = True
        else:
            random_action = False
        
        # Repeatedly select random or best action for length of predictions.
        # Only learn from memory during non-random action selection to maintain
        # 1:1 learning step to memory storage
        for _ in range(decoder.pred_len):
            if random_action:                
                # Make random action
                a = randint(0, agent.num_actions - 1)
                r = agent.make_action(action=agent.actions[a])
            
            else:
                # Perform learning step
                decoder.perform_learning_step()

                # Increment counter
                learning_step += 1
                if learning_step % 100 == 0:
                    print("Learning step %d of %d.        " 
                            % (learning_step, learning_steps_per_epoch), 
                            end='\r')
                if learning_step == learning_steps_per_epoch:
                    break
            
            # Initialize new episode if game ended
            if game.is_episode_finished():
                agent.initialize_new_episode()
                train_episodes_finished += 1
                break
                
    print("%d training episodes played." % train_episodes_finished)
    
    # Testing
    print("\nTesting...")
    save_epoch = (epoch + 1 == epochs or (epoch + 1) % save_freq == 0)
    agent.initialize_new_episode()
    zero_state = agent.get_zero_state() # cheap way to get state shape(s)
    states = [np.zeros([test_steps_per_epoch] + list(s.shape)) for s in zero_state]
    targets_pos = np.zeros([test_steps_per_epoch, 2], dtype=np.float32)
    targets_act = np.zeros([test_steps_per_epoch, 1], dtype=np.int16)
    
    test_step = 0
    while test_step < test_steps_per_epoch:
        # With probability epsilon make a random action; otherwise, choose
        # best action.
        epsilon = decoder.epsilon
        if random() <= epsilon:
            random_action = True
        else:
            random_action = False
        
        # Repeatedly select random or best action for length of predictions
        for _ in range(decoder.pred_len):
            if random_action:
                # Make random action
                a = randint(0, agent.num_actions - 1)
                r = agent.make_action(action=agent.actions[a])
            
            else:
                # Update and get current state, position
                agent.update_state()
                s = deepcopy(agent.state)
                x = agent.track_position()[1:3]

                # Make and get action
                r = agent.make_action()
                a = agent.track_action(index=True)

                # Save states and targets
                states[0][test_step] = s[0]
                states[1][test_step] = s[1]
                targets_pos[test_step] = x
                targets_act[test_step] = a
                
                # Increment counter
                test_step += 1
                if test_step % 100 == 0:
                    print("Game tick %d of max %d.        " 
                          % (test_step, test_steps_per_epoch), 
                          end='\r')
                if test_step == test_steps_per_epoch:
                    break
            
            # Initialize new episode if game ended
            if game.is_episode_finished():
                agent.initialize_new_episode()
                break
    
    # Run test data after all test steps complete
    # (speed at expense of storage)
    # Also note that test trajectories are non-overlapping
    s_ = [states[0][::decoder.pred_len], states[1][::decoder.pred_len]]
    loss = decoder.get_loss(state=s_, 
                            position=targets_pos, 
                            action=targets_act)[0]
    preds = decoder.get_prediction(s_)

    # Update loss history and save episode history
    loss_history.append(loss)
    print("\nEpoch %d loss: %.2f" % (epoch + 1, loss))

    # Save results after specified number of epochs; 
    # otherwise store temporarily after each epoch
    model_filename = exp_name + "_model"
    test_batch = decoder.get_test_batch(states=s_, 
                                        positions=targets_pos,
                                        actions=targets_act)
    if save_epoch: 
        print("Saving network... ", end="")
        global_step = epoch + 1
    else:
        print("Stashing network... ", end="")
        global_step = None        
    decoder.save_model(model_filename, 
                       global_step=global_step, 
                       save_meta=(epoch == 0), 
                       save_summaries=True,
                       test_batch=test_batch)

    print("Done.")
    print("Total elapsed time: %.2f minutes" % ((time() - time_start) / 60.0))

# Close game
game.close()
print("======================================")