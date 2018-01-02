"""
This module contains class definitions for open ai gym environments.
"""

import os
import collections
import argparse
import random
from datetime import datetime
import time
from functools import reduce
import time

import numpy as np
import tensorflow as tf

from environment import MinesweeperEnvironment


from qnetwork import DeepQNetwork, update_target_network
from replay_memory import ReplayMemory, ScreenHistory


class QAgent:
    """An environment class for open ai gym atari games using the screen.

    Attributes:
        _display : bool
            Display the game visually
        _screen (:obj: 'array', :obj: 'float') : The screen output (rgb)
        _reward (float) : amount of reward achieved by the previous action. 
                          The scale varies between environments, 
                          but the goal is always to increase your total reward.
        _done (bool) : Whether it's time to reset the environment again. 
                       Most (but not all) tasks are divided up into well-defined
                       episodes, and done being True indicates the episode has 
                       terminated.
        _random_start (int) : How long we let the agent take random actions in a
                              new game.
        screen_width (int) : The width of the screen after resizing.
        screen_height (int) : The height of the screen after resizing.
        _action_repeat (int) : The number of time-steps an action is repeated.
        env (:obj:) : The open ai gym environment object
    """

    def __init__(self, params):

        self.params = params # These are the parameters collected for the agent.

        # Load environmnet

        self.game = MinesweeperEnvironment(
                        self.params.input_height,
                        self.params.input_width,
                        self.params.mines_min,
                        self.params.mines_max,
                        self.params.show_game,
                        self.params.reward_recent_update)

        # Initialize two Q-Value Networks 
        # Q-network for training.

        self.dqn_train = DeepQNetwork(
                            params=self.params,
                            num_actions=self.game.num_actions,
                            network_name="qnetwork-train",
                            trainable=True)

        if self.params.is_train:

            # Q-Network for predicting target Q-values
            self.dqn_target = DeepQNetwork(
                                params=self.params,
                                num_actions=self.game.num_actions,
                                network_name="qnetwork-target",
                                trainable=False)
            
            # Initialize replay memory for storing experience to sample batches from
            self.replay_mem = ReplayMemory(
                                self.params.replay_capacity, 
                                self.params.history_length,
                                self.params.nchannels,
                                self.params.batch_size,
                                self.params.input_height,
                                self.params.input_width,
                                self.params.game,
                                self.params.memory_checkpoint,
                                self.params.restore_memory,
                                self.params.output_dir)

        # Small structure for storing the last four screens
        self.history = ScreenHistory(self.params)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        self.checkpoint_dir    = os.path.abspath(os.path.join(self.params.output_dir, "checkpoints_" + self.params.game))
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "model")
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.train_iteration    = 0
        self.count_actions      = np.zeros(self.game.num_actions)   # Count per action (only greedy)
        self.count_act_random   = 0  # Count of random actions
        self.count_act_greedy   = 0  # Count of greedy actions
        self.win_rate = 0.0 # For atari

        # Histories of qvalues and loss for running average
        self.qvalues_hist = collections.deque([0]*self.params.interval_summary,  maxlen=self.params.interval_summary)
        self.loss_hist    = collections.deque([10]*self.params.interval_summary, maxlen=self.params.interval_summary)

        self.epsilon = 0

    def fit(self):

        screen, reward, is_done = self.game.new_game()
        for _ in range(self.params.history_length):
            self.history.add(screen)

        # Initialize the TensorFlow session
        gpu_options = tf.GPUOptions(
           per_process_gpu_memory_fraction=self.params.gpu_memory
        )

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

            # Initialize the TensorFlow session
            init = tf.global_variables_initializer()
            sess.run(init)

            # Only save trainable variables and the global iteration to disk
            tf_vars_to_save = tf.trainable_variables() + [self.dqn_train.global_iteration]
            saver = tf.train.Saver(tf_vars_to_save, max_to_keep=200)

            if self.params.model_file is not None:
                # Load pre-trained model from disk
                model_path = os.path.join(self.checkpoint_dir, self.params.model_file)
                saver.restore(sess, model_path)
                self.train_iteration, learning_rate = sess.run([self.dqn_train.global_iteration, self.dqn_train.learning_rate])
                print("Restarted training from model file. iteration = %06i, Learning Rate = %.5f" % (self.train_iteration, learning_rate))

            # Initialize summary writer
            self.dqn_train.build_summary_writer(sess)

            # Initialize the target Q-Network fixed with the same weights
            update_target_network(sess, "qnetwork-train", "qnetwork-target")

            for iteration in range(self.params.num_iterations): # Iteration is also how many times we added to replay
                                                                # self.train_iteration is the true train iteration
                self._sel_move(sess, iteration)
                self._train(sess, iteration, saver)

            print("Finished training Q-network.")

    def _sel_move(self, sess, iteration):

        if self.params.is_train:
            replay_mem_size = self.replay_mem.num_examples()
            if replay_mem_size < self.params.train_start and iteration % 1000 == 0:
                print("Initializing replay memory %i/%i" % (iteration, self.params.train_start))

        # self.epsilon Greedy Exploration: with the probability of self.epsilon
        # choose a random action, otherwise go greedy with the action
        # having the maximal Q-value. Note the minimum episolon of 0.1
        if self.params.is_train:
            self.epsilon = max(self.params.min_epsilon, 1.0-float(self.train_iteration*self.params.train_freq) / float(self.params.epsilon_step))
        else:
            self.epsilon = self.params.eval_epsilon

        ################################################################
        ####################### SELECT A MOVE ##########################
        ################################################################

        # Either choose a random action or predict the action using the Q-network
        do_random_action = (random.random() < self.epsilon)
        if do_random_action or (self.params.is_train and replay_mem_size < self.params.train_start):
            action_id = random.randrange(self.game.num_actions)
            self.count_act_random += 1
        else:

            # Get the last screens from the self.history and perform
            # feed-forward through the network to compute Q-values
            feed_dict  = { self.dqn_train.pl_screens: self.history.get() }

            qvalues    = sess.run(self.dqn_train.qvalues, feed_dict=feed_dict)

            # Choose the best action based on the approximated Q-values
            qvalue_max = np.max(qvalues[0])
            action_id  = np.argmax(qvalues[0])

            self.count_act_greedy += 1
            self.count_actions[action_id] += 1
            self.qvalues_hist.append(qvalue_max)

        self._move(action_id)

    def _move(self, action_id):

        ################################################################
        ####################### PLAY THE MOVE ##########################
        ################################################################

        # Play the selected action (either random or predicted) on the self.game game
        # Note that the action is performed for k = 4 frames (frame skipping)
        screen, cumulative_reward, is_done = self.game.act(action_id)

        # Perform reward clipping and add the example to the replay memory
        # This is done with Huber loss now
        #cumulative_reward = min(+1.0, max(-1.0, cumulative_reward))

        # Add the screen to short term self.history and replay memory
        self.history.add(screen)

        # Add experience to replay memory
        if self.params.is_train:
            self.replay_mem.add(action_id, cumulative_reward, screen, is_done)

        # Check if we are game over, and if yes, initialize a new game
        if is_done:
            screen, reward, is_done = self.game.new_game()
            if self.params.is_train:
                self.replay_mem.add(0, reward, screen, is_done)
                self.history.add(screen)

    def _train(self, sess, iteration, saver):

        ################################################################
        ###################### TRAINING MODEL ##########################
        ################################################################


        if self.params.is_train and iteration > self.params.train_start and iteration % self.params.train_freq == 0:

            screens, actions, rewards, screens_1, dones = self.replay_mem.sample_batch()

            # Below, we perform the Double-DQN update.


            # First, we need to determine the best actions
            # in the train network
            qvalues_train = sess.run(
                self.dqn_train.qvalues,
                feed_dict={ self.dqn_train.pl_screens: screens_1 }
            )

             # Find the best actions for each using the train network
             # which will be used with the q-values form the target network
            actions_target = np.argmax(qvalues_train, 1)

            # We use this to evalute the q-value for some state
            # Now,we get the q-values for all actions given the states
            # We then later sort out the q-values from the target network
            # using the best actions from the train network

            qvalues_target = sess.run(
                self.dqn_target.qvalues,
                feed_dict={ self.dqn_target.pl_screens: screens_1 }
            )


            # Inputs for trainable Q-network
            feed_dict = {
                self.dqn_train.pl_screens   : screens,
                self.dqn_train.pl_actions   : actions,
                self.dqn_train.pl_rewards   : rewards,
                self.dqn_train.pl_dones : dones,
                #self.dqn_train.pl_qtargets  : np.max(qvalues_target, axis=1),
                self.dqn_train.pl_qtargets  : qvalues_target,
                self.dqn_train.pl_actions_target : actions_target,
            }

            # Actual training operation
            _, loss, self.train_iteration = sess.run([self.dqn_train.train_op,
                                            self.dqn_train.loss,
                                            self.dqn_train.global_iteration],
                                            feed_dict=feed_dict)

            # Running average of the loss
            self.loss_hist.append(loss)

             # Check if the returned loss is not NaN
            if np.isnan(loss):
                print("[%s] Training failed with loss = NaN." %
                      datetime.now().strftime("%Y-%m-%d %H:%M"))

            # Once every n = 10000 frames update the Q-network for predicting targets
            if self.train_iteration % self.params.network_update_rate == 0:
                print("[%s] Updating target network." % datetime.now().strftime("%Y-%m-%d %H:%M"))
                update_target_network(sess, "qnetwork-train", "qnetwork-target")

            self._evaluate(sess, feed_dict)
            self._print_save(sess, feed_dict, saver)


    def _evaluate(self, sess, feed_dict):

        ################################################################
        ####################### MODEL EVALUATION #######################
        ################################################################

        if self.params.is_train and self.train_iteration % self.params.eval_frequency == 0 or self.train_iteration == 0:

            eval_total_reward = 0
            eval_num_episodes = 0
            eval_num_wins = 0
            eval_num_rewards = 0
            eval_episode_max_reward = 0
            eval_episode_reward = 0
            eval_actions = np.zeros(self.game.num_actions)


            # We store all of these parameters temporarily so this evaluation does not
            # affect model evaluation

            tmp_episode_step          = self.game._episode_step
            tmp_episode_number        = self.game._episode_number
            tmp_episode_reward        = self.game._episode_reward
            tmp_max_reward_episode    = self.game._max_reward_episode
            tmp_global_step           = self.game._global_step
            tmp_global_reward         = self.game._global_reward
            tmp_recent_reward         = self.game._recent_reward
            tmp_recent_episode_number = self.game._recent_episode_number
            tmp_recent_games_won      = self.game._recent_games_won
            tmp_games_won             = self.game._games_won
            tmp_reward_recent_update  = self.game.reward_recent_update

            prev_action_id = -1
            prev_episode_num = -1 # Just has to be different intially than prev
            action_id = -1 
            eval_num_episodes = 0

            # Initialize new game without random start moves
            screen, reward, done = self.game.new_game()

            for _ in range(self.params.history_length):
                self.history.add(screen)

            #for eval_iterations in range(self.params.eval_iterations):
            while eval_num_episodes < self.params.eval_iterations: # Play eval_iterations games
                prev_action_id = action_id

                # if random.random() < self.params.eval_epsilon:
                #     # Random action
                #     action_id = random.randrange(self.game.num_actions)
                #else:
                # Greedy action
                # Get the last screens from the self.history and perform
                # feed-forward through the network to compute Q-values
                feed_dict_eval  = { self.dqn_train.pl_screens: self.history.get() }
                qvalues = sess.run(self.dqn_train.qvalues, feed_dict=feed_dict_eval)

                # Choose the best action based on the approximated Q-values
                qvalue_max = np.max(qvalues[0])
                action_id  = np.argmax(qvalues[0])

                # Skip this action if we are in the same game
                if prev_action_id == action_id and prev_episode_num == eval_num_episodes:
                    action_id = random.randrange(self.game.num_actions)

                prev_episode_num = eval_num_episodes

                # Keep track of how many of each action is performed
                eval_actions[action_id] += 1

                # Perform the action
                screen, reward, done = self.game.act(action_id)
                self.history.add(screen)

                eval_episode_reward += reward
                if reward > 0:
                    eval_num_rewards += 1

                if reward == self.game.env.rewards["win"]:
                    eval_num_wins += 1

                if done:
                    # Note max reward is from playin gthe games
                    eval_total_reward += eval_episode_reward
                    eval_episode_max_reward = max(eval_episode_reward, eval_episode_max_reward)
                    eval_episode_reward = 0
                    eval_num_episodes += 1

                    screen, reward, done = self.game.new_game()
                    for _ in range(self.params.history_length):
                        self.history.add(screen)

            # Send statistics about the environment to TensorBoard
            eval_update_ops = [
                self.dqn_train.eval_rewards.assign(eval_total_reward),
                self.dqn_train.eval_win_rate.assign((eval_num_wins / eval_num_episodes)*100),
                self.dqn_train.eval_num_rewards.assign(eval_num_rewards),
                self.dqn_train.eval_max_reward.assign(eval_episode_max_reward),
                self.dqn_train.eval_num_episodes.assign(eval_num_episodes),
                self.dqn_train.eval_actions.assign(eval_actions / np.sum(eval_actions))

            ]
            sess.run(eval_update_ops)
            summaries = sess.run(self.dqn_train.eval_summary_op, feed_dict=feed_dict)
            self.dqn_train.train_summary_writer.add_summary(summaries, self.train_iteration)

            print("[%s] Evaluation Summary" % datetime.now().strftime("%Y-%m-%d %H:%M"))
            print("  Total Reward: %i" % eval_total_reward)
            print("  Max Reward per Episode: %i" % eval_episode_max_reward)
            print("  Num Episodes: %i" % eval_num_episodes)
            print("  Num Rewards: %i" % eval_num_rewards)
            print("  Win Rate: %.1f" % ((eval_num_wins / eval_num_episodes)*100))

            self.win_rate = (eval_num_wins / eval_num_episodes)*100


            self.game._episode_step          = tmp_episode_step
            self.game._episode_number        = tmp_episode_number
            self.game._episode_reward        = tmp_episode_reward
            self.game._max_reward_episode    = tmp_max_reward_episode
            self.game._global_step           = tmp_global_step
            self.game._global_reward         = tmp_global_reward
            self.game._recent_reward         = tmp_recent_reward
            self.game._recent_episode_number = tmp_recent_episode_number
            self.game._recent_games_won      = tmp_recent_games_won
            self.game._games_won             = tmp_games_won
            self.game.reward_recent_update   = tmp_reward_recent_update



    def _print_save(self, sess, feed_dict, saver):

        ################################################################
        ###################### PRINTING / SAVING #######################
        ################################################################

        # Write a training summary to disk
        # This is what controls how often we write to disk
        if self.params.is_train and self.train_iteration % self.params.interval_summary == 0:

            # Send statistics about the environment to TensorBoard
            update_game_stats_ops = [
                self.dqn_train.avg_reward_per_game.assign(self.game.avg_reward_per_episode()),
                self.dqn_train.max_reward_per_game.assign(self.game.max_reward_per_episode),
                self.dqn_train.avg_moves_per_game.assign(self.game.avg_steps_per_episode()),
                self.dqn_train.total_reward_replay.assign(self.replay_mem.total_reward()),
                self.dqn_train.num_games_played.assign(self.game.episode_number),
                self.dqn_train.moves.assign(self.game.global_step),
                self.dqn_train.actions_random.assign(self.count_act_random),
                self.dqn_train.actions_greedy.assign(self.count_act_greedy),
            ]
            sess.run(update_game_stats_ops)

            # Build and save summaries
            summaries = sess.run(self.dqn_train.train_summary_op, feed_dict=feed_dict)


            # Here we set train_iteration on x-axis
            self.dqn_train.train_summary_writer.add_summary(summaries, self.train_iteration)

            # Here we set number of moves on x-axis
            #self.dqn_train.train_summary_writer.add_summary(summaries, self.game.global_step)

            avg_qvalue = avg_loss = 0
            for i in range(len(self.qvalues_hist)):
                avg_qvalue += self.qvalues_hist[i]
                avg_loss   += self.loss_hist[i]

            avg_qvalue /= float(len(self.qvalues_hist))
            avg_loss   /= float(len(self.loss_hist))

            learning_rate = sess.run(self.dqn_train.learning_rate)

            format_str = "[%s] It. %06i, Replay = %i, epsilon = %.4f, "\
                         "Episodes = %i, Steps = %i, Avg.R = %.3f, "\
                         "Max.R = %.3f, Win = %.1f, Avg.Q = %.4f, Avg.Loss = %.6f, lr = %.6f"
            print(format_str % (datetime.now().strftime("%Y-%m-%d %H:%M"), self.train_iteration,
                                self.replay_mem.num_examples(), self.epsilon, self.game.episode_number, self.game.global_step,
                                self.game.avg_reward_per_episode(),
                                self.game.max_reward_per_episode, self.win_rate,
                                avg_qvalue, avg_loss, learning_rate))

        # Write model checkpoint to disk
        if self.params.is_train and self.train_iteration % self.params.interval_checkpoint == 0:
            path = saver.save(sess, self.checkpoint_prefix, global_step=self.train_iteration)
            print("[%s] Saving TensorFlow model checkpoint to disk." %
                  datetime.now().strftime("%Y-%m-%d %H:%M"))

            sum_actions = float(reduce(lambda x, y: x+y, self.count_actions))
            action_str = ""
            for action_id, action_count in enumerate(self.count_actions):
                action_perc = action_count/sum_actions if not sum_actions == 0 else 0
                action_str += "<%i, %s, %i, %.2f> " % \
                              (action_id, self.game.action_to_string(action_id),
                               action_count, action_perc)

            format_str = "[%s] Q-Network Actions Summary: NumRandom: %i, NumGreedy: %i, %s"
            print(format_str % (datetime.now().strftime("%Y-%m-%d %H:%M"),
                                self.count_act_random, self.count_act_greedy, action_str))


    def play_mine(self):
        # Show the game being played visually

        # Initialize a new game and store the screens in the self.history
        screen, reward, is_done = self.game.new_game()
        for _ in range(self.params.history_length):
            self.history.add(screen)

        # Initialize the TensorFlow session
        gpu_options = tf.GPUOptions(
           per_process_gpu_memory_fraction=self.params.gpu_memory
        )

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

            # Initialize the TensorFlow session
            init = tf.global_variables_initializer()
            sess.run(init)

            # Only save trainable variables and the global iteration to disk
            tf_vars_to_save = tf.trainable_variables() + [self.dqn_train.global_iteration]
            saver = tf.train.Saver(tf_vars_to_save, max_to_keep=200)

            if self.params.model_file is not None:
                # Load pre-trained model from disk
                model_path = os.path.join(self.checkpoint_dir, self.params.model_file)
                saver.restore(sess, model_path)


            prev_action_id = -1
            prev_episode_num = -1 # Just has to be different intially than prev
            action_id = -1 
            eval_num_episodes = 0

            while self.game.episode_number < self.params.num_games:
                if self.params.show_game:
                    inp = input("Enter input (ROW,COL)")

                prev_action_id = action_id

                feed_dict_eval  = { self.dqn_train.pl_screens: self.history.get() }
                qvalues = sess.run(self.dqn_train.qvalues, feed_dict=feed_dict_eval)

                # Choose the best action based on the approximated Q-values
                qvalue_max = np.max(qvalues[0])
                action_id  = np.argmax(qvalues[0])

                # Skip this action if we are in the same game
                if prev_action_id == action_id and prev_episode_num == eval_num_episodes:
                    if self.params.show_game:
                        print("Network repeated an action")
                    action_id = random.randrange(self.game.num_actions)

                prev_episode_num = eval_num_episodes

                # Perform the action
                screen, reward, done = self.game.act(action_id)
                self.history.add(screen)

                # Stop printing in the minesweeper environment
                # if reward == self.game.env.rewards["win"]:
                #     eval_num_wins += 1

                if done:
                    eval_num_episodes += 1

                    screen, reward, done = self.game.new_game()
                    for _ in range(self.params.history_length):
                        self.history.add(screen)


    def test_mine(self):

        # Tests a single model

        # Initialize a new game and store the screens in the self.history
        screen, reward, is_done = self.game.new_game()
        for _ in range(self.params.history_length):
            self.history.add(screen)

        # Initialize the TensorFlow session
        gpu_options = tf.GPUOptions(
           per_process_gpu_memory_fraction=self.params.gpu_memory
        )

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

            # Initialize the TensorFlow session
            init = tf.global_variables_initializer()
            sess.run(init)
            # Only save trainable variables and the global iteration to disk
            tf_vars_to_save = tf.trainable_variables() + [self.dqn_train.global_iteration]
            saver = tf.train.Saver(tf_vars_to_save, max_to_keep=200)

            # if self.params.model_file is not None:
            #     # Load pre-trained model from disk
            #     model_path = os.path.join(self.checkpoint_dir, self.params.model_file)
            #     saver.restore(sess, model_path)
            model_path = os.path.join(self.checkpoint_dir, self.params.model_file)
            saver.restore(sess, model_path)

            prev_action_id = -1
            prev_episode_num = -1 # Just has to be different intially than prev
            action_id = -1 
            eval_num_episodes = 0

            eval_total_reward = 0
            eval_num_episodes = 0
            eval_num_wins = 0
            eval_num_rewards = 0
            eval_episode_max_reward = 0
            eval_episode_reward = 0
            eval_actions = np.zeros(self.game.num_actions)

            # Initialize new game without random start moves
            screen, reward, done = self.game.new_game()

            for _ in range(self.params.history_length):
                self.history.add(screen)

            #for eval_iterations in range(self.params.eval_iterations):
            while eval_num_episodes < self.params.eval_iterations: # Play eval_iterations games
                prev_action_id = action_id

                feed_dict_eval  = { self.dqn_train.pl_screens: self.history.get() }
                qvalues = sess.run(self.dqn_train.qvalues, feed_dict=feed_dict_eval)

                # Choose the best action based on the approximated Q-values
                qvalue_max = np.max(qvalues[0])
                action_id  = np.argmax(qvalues[0])

                # Skip this action if we are in the same game
                if prev_action_id == action_id and prev_episode_num == eval_num_episodes:
                    action_id = random.randrange(self.game.num_actions)

                prev_episode_num = eval_num_episodes


                # Perform the action
                screen, reward, done = self.game.act(action_id)
                self.history.add(screen)

                eval_episode_reward += reward
                if reward > 0:
                    eval_num_rewards += 1

                if reward == self.game.env.rewards["win"]:
                    eval_num_wins += 1

                if done:
                    # Note max reward is from playin gthe games
                    eval_total_reward += eval_episode_reward
                    eval_episode_max_reward = max(eval_episode_reward, eval_episode_max_reward)
                    eval_episode_reward = 0
                    eval_num_episodes += 1

                    screen, reward, done = self.game.new_game()
                    for _ in range(self.params.history_length):
                        self.history.add(screen)

            print("  Win Rate: %.2f" % ((eval_num_wins / eval_num_episodes)*100))

    def evaluate_mine(self):
        # Test a number of models using the naming scheme
        # to find the best model in a range

        # Initialize a new game and store the screens in the self.history
        screen, reward, is_done = self.game.new_game()
        for _ in range(self.params.history_length):
            self.history.add(screen)

        # Initialize the TensorFlow session
        gpu_options = tf.GPUOptions(
           per_process_gpu_memory_fraction=self.params.gpu_memory
        )

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            max_name = 800000
            min_name = 680000
            current_name = min_name
            best_model = min_name
            best_win_rate = 0
            current_win_rate = 0

            # Initialize the TensorFlow session
            init = tf.global_variables_initializer()
            sess.run(init)
            # Only save trainable variables and the global iteration to disk
            tf_vars_to_save = tf.trainable_variables() + [self.dqn_train.global_iteration]
            saver = tf.train.Saver(tf_vars_to_save, max_to_keep=200)


            while current_name <= max_name:

                print("Restoring: ", current_name)


                # if self.params.model_file is not None:
                #     # Load pre-trained model from disk
                #     model_path = os.path.join(self.checkpoint_dir, self.params.model_file)
                #     saver.restore(sess, model_path)
                model_path = os.path.join(self.checkpoint_dir, 'model-' + str(current_name))
                saver.restore(sess, model_path)

                prev_action_id = -1
                prev_episode_num = -1 # Just has to be different intially than prev
                action_id = -1 
                eval_num_episodes = 0

                eval_total_reward = 0
                eval_num_episodes = 0
                eval_num_wins = 0
                eval_num_rewards = 0
                eval_episode_max_reward = 0
                eval_episode_reward = 0
                eval_actions = np.zeros(self.game.num_actions)

                # Initialize new game without random start moves
                screen, reward, done = self.game.new_game()

                for _ in range(self.params.history_length):
                    self.history.add(screen)

                #for eval_iterations in range(self.params.eval_iterations):
                while eval_num_episodes < self.params.eval_iterations: # Play eval_iterations games
                    prev_action_id = action_id

                    feed_dict_eval  = { self.dqn_train.pl_screens: self.history.get() }
                    qvalues = sess.run(self.dqn_train.qvalues, feed_dict=feed_dict_eval)

                    # Choose the best action based on the approximated Q-values
                    qvalue_max = np.max(qvalues[0])
                    action_id  = np.argmax(qvalues[0])

                    # Skip this action if we are in the same game
                    if prev_action_id == action_id and prev_episode_num == eval_num_episodes:
                        action_id = random.randrange(self.game.num_actions)

                    prev_episode_num = eval_num_episodes


                    # Perform the action
                    screen, reward, done = self.game.act(action_id)
                    self.history.add(screen)

                    eval_episode_reward += reward
                    if reward > 0:
                        eval_num_rewards += 1

                    if reward == self.game.env.rewards["win"]:
                        eval_num_wins += 1

                    if done:
                        # Note max reward is from playin gthe games
                        eval_total_reward += eval_episode_reward
                        eval_episode_max_reward = max(eval_episode_reward, eval_episode_max_reward)
                        eval_episode_reward = 0
                        eval_num_episodes += 1

                        screen, reward, done = self.game.new_game()
                        for _ in range(self.params.history_length):
                            self.history.add(screen)

                current_win_rate = (eval_num_wins / eval_num_episodes)*100

                print("  Win Rate: %.2f" % (current_win_rate))

                if current_win_rate > best_win_rate:
                    best_win_rate = current_win_rate
                    best_model = current_name

                current_name = current_name + 20000

            print("Best model is: ", best_model)