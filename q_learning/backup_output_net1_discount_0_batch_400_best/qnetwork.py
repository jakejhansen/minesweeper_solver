"""
This module contains a function to update another neural network by copying
all trainable variables and a class for a deep Q network.
"""

import os
import numpy as np
import tensorflow as tf

# Based on
# https://github.com/tomrunia/DeepReinforcementLearning-Atari

def update_target_network(sess, network_name_train, network_name_target):
    """This helper method copies all the trainable weights and biases from
    one DeepQNetwork to another. This method is used for synchronisation
    of the train and target Q-networks

    Parameters
    ----------
    sess: tensorflow session.
        The current session in tensorflow.
    network_name_train: str
        Name of Q-Network for training.s
    network_name_target: str
        Name of Q-Network for predicting target Q-values

    Returns
    -------
    None

    """

    # Retrieve current variable scope and set reuse flag to True
    tf.get_variable_scope().reuse_variables()

    # Get list of values in "network_name_train" in the collection
    # GraphKeys.TRAINABLE_VARIABLES
    vars_source = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=network_name_train
    )

    copy_ops  = [] # Where we copy the variables to
    check_ops = [] # Contains a list of booleans determining if variables are equal

    for v in vars_source:
        # Note the [0:-2] to cut of the device placement
        v_source = tf.get_variable(v.name[0:-2])

        # Remove variable prefix (network name)
        var_name = v.name[v.name.find("/"):]
        v_target = tf.get_variable(network_name_target + var_name[0:-2])

        copy_ops.append(v_target.assign(v_source))
        check_ops.append(tf.equal(v_target, v_source))

    # Actual copying all the variables, check if the values are equal
    sess.run(copy_ops)
    check_res = sess.run(check_ops)
    for res in check_res:
        if not np.all(res): # Failed to copy variables
            raise ValueError("Verification of tf.equal(var_train, var_target) failed.")


class DeepQNetwork:
    """A class for replay memories used during training of deep Q-networks.

    Parameters
    ----------
    replay_capacity : int
        The maximum number of samples in replay memory
    batch_size : Bool
        The number of frames to return.

    Attributes:
        network_name : str
            The name of the  network.
        trainable : bool
            If this network is supposed to be trained.
            True for the Q-network we train and False for the targer Q-network.
        model_file : str
            The name of the file containing the trained model parameters.
        output_dir : str
            The name of the output directory.
        input_size : int
            The size of the screen input.
        input_depth : int
            The input depth, this is the number of frames used, expects 4.
        filter_sizes: :obj: 'list' of :obj: 'int'
            A list containing the filter sizes in conv layers.
        filter_strides: :obj: 'list' of :obj: 'int'
            A list containing the filter strides in conv layers.
        num_filters: :obj: 'list' of :obj: 'int'
            A list containing the number of filters in conv layers.
        num_hidden: :obj: 'list' of :obj: 'int'
            A list containing the number of hidden units in feedforward layers.
        num_actions: int
            The number of actions available.
        dueling_type: str
            The type of dueling.
        batch_size: int
            Size of the training batch.
        learning_rate_base: float
            The start learning rate.
        learning_rate_decay: float
            The decay rate of the learning rate. 
            We decay every larning_rate_iteration.
        learning_rate_step: int
            The number of iterations to anneal the learning rate.
        learning_rate_min: float
            The minimum learning rate.
        clip_delta: bool
            If True use error clipping.
        batch_accumulator: str
            Either "mean" or "sum", determines if we reduce sum or mean of
            delta_square

    """


    def __init__(self, params, num_actions, network_name, trainable):

        self.params = params

        # Network information
        self.network_name        = network_name
        self.trainable           = trainable

        # Paths and models
        self.model_file          = params.model_file
        self.output_dir          = params.output_dir

        # Input to the Q-network
        self.input_height         = params.input_height      # 84
        self.input_width          = params.input_width      # 84

        self.env_type = params.env

        if self.env_type == 'minesweeper':
            self.input_depth          = params.nchannels  # 4
        else:
            self.input_depth          = params.history_length  # 4


        self.network_type = params.network_type

        # CNN layer architecture
        self.filter_sizes        = params.filter_sizes    # 8, 4, 3
        self.filter_strides      = params.filter_strides  # 4, 2, 1
        self.num_filters         = params.num_filters     # 32, 64, 64
        self.num_hidden          = params.num_hidden      # 512
        self.num_actions         = num_actions            # depends on game
        self.dueling_type        = params.dueling_type

        # Training Parameters
        self.batch_size          = params.batch_size             # 32
        self.learning_rate_base  = params.learning_rate          # 0.0025
        self.learning_rate_decay = params.learning_rate_decay    # 0.96
        self.learning_rate_step  = params.learning_rate_step     # 450000
        self.learning_rate_min   = params.learning_rate_minimum  # 0.00025
        self.discount            = params.discount               # 0.99
        self.network_update_rate = params.network_update_rate    # 10000
        self.clip_delta          = params.clip_delta             # 1.0
        self.batch_accumulator   = params.batch_accumulator      # mean or sum

        # Init TensorFlow graph
        self._init_graph()

    def _init_graph(self):
        """Initializes the computational graph in Tensorflow and builds the
        neural network.
        If trainable is True, we also build the loss part of the graph.

        Returns
        -------
        None

        """

        # Initialize operation for creating/writing summary to disk
        self.train_summary_op     = None
        self.eval_summary_op      = None
        self.train_summary_writer = None

        with tf.variable_scope(self.network_name):

            self.global_iteration    = tf.Variable(0, name='global_iteration', 
                                    trainable=False)

            # Placeholders
            self.pl_screens = tf.placeholder(tf.float32, 
                                    shape=[None, 
                                            self.input_height, 
                                            self.input_width, 
                                            self.input_depth], 
                                            name="screens")

            # For feedforward networks
            self.pl_flat = tf.placeholder(tf.float32, 
                                    shape=[None, 
                                            self.input_height*self.input_width*self.input_depth], 
                                            name="flat")


            # Contain q-targest for all batches per action, None for batches
            self.pl_qtargets = tf.placeholder(tf.float32, shape=[None, self.num_actions], 
                                    name="qtargets")

            self.pl_actions = tf.placeholder(tf.int64, shape=[None], 
                                    name="actions")

            self.pl_actions_target = tf.placeholder(tf.int64, shape=[None],  # Actions from train-network used to select q-values in target network
                                    name="actions_target")

            self.pl_rewards = tf.placeholder(tf.float32, shape=[None], 
                                    name="rewards")

            self.pl_dones   = tf.placeholder(tf.float32, shape=[None], 
                                    name="dones")

            # Convert strings to lists
            self.filter_sizes   = list(map(int, self.filter_sizes.split(",")))
            self.filter_strides = list(map(int, self.filter_strides.split(",")))
            self.num_filters    = list(map(int, self.num_filters.split(",")))

            # Initialize the CNN
            if self.network_type == 'conv':
                self.qvalues = self._build_cnn()
            elif self.network_type == 'fc':
                self.qvalues = self._build_fc()

            # Initialize the loss function if we have the trainable Q-network.
            # Otherwise all parameters are copied.
            if self.trainable:

                self._build_loss()

                # Learning rate decay
                # When training a model, it is often recommended to lower the 
                # learning rate as the training progresses. 
                # This function applies an exponential decay function to a 
                # provided initial learning rate.
                # Uses global_iteration to compute the decayed learning rate.
                self.learning_rate   = tf.train.exponential_decay(
                    self.learning_rate_base, self.global_iteration,
                    self.learning_rate_step, self.learning_rate_decay,
                    staircase=True
                )

                self.learning_rate = tf.maximum(self.learning_rate, 
                                        self.learning_rate_min)

                # Adam Optimizer
                self.optimizer = tf.train.AdamOptimizer(
                                    learning_rate=self.learning_rate,
                                    epsilon=1.5e-4) # From Rainbow paper

                self.train_op_optimizer = self.optimizer.minimize(
                                            self.loss, 
                                            global_step=self.global_iteration # Steps refer to something else in reinforcement, so we call iteration
                                        )

                # Keep track of running average of the loss when minimizing 
                # the loss
                # tf.control_dependencies means that 
                # self.train_op = tf.group(self.loss_moving_avg_op)
                # is only run after self.loss_moving_avg_op has been executed
                with tf.control_dependencies([self.train_op_optimizer]):
                    self.train_op = tf.group(self.loss_moving_avg_op)

    def _build_cnn(self):
        """Builds a convolutional network. When we add new layers, it is
        important to set trainable=self.trainable since these parameters needs
        to be copied to the target nework and it uses the set of trianable.

        Returns
        -------
        :obj: 'ndarray' of :obj: 'float'
            The q-values in the output as a list, corresponding to each possible
            action.


        """

        with tf.variable_scope("Network"):

            if self.env_type == 'atari':

                conv1 = tf.layers.conv2d(
                    inputs=self.pl_screens, 
                    filters=32, 
                    kernel_size=[8, 8],
                    strides=(4, 4),
                    activation=tf.nn.relu,
                    padding="valid",  # Same as in keras
                    kernel_initializer=tf.contrib.layers.xavier_initializer(seed=self.params.seed),
                    bias_initializer= tf.constant_initializer(self.params.bias_init),
                    trainable=self.trainable,
                    name="conv1"
                    )

                conv2 = tf.layers.conv2d(
                    inputs=conv1, 
                    filters=64, 
                    kernel_size=[4, 4],
                    strides=(2, 2),
                    activation=tf.nn.relu,
                    padding="valid", 
                    kernel_initializer=tf.contrib.layers.xavier_initializer(seed=self.params.seed),
                    bias_initializer= tf.constant_initializer(self.params.bias_init),
                    trainable=self.trainable,
                    name="conv2"
                    )

                conv3 = tf.layers.conv2d(
                    inputs=conv2, 
                    filters=64, 
                    kernel_size=[3, 3],
                    strides=(1, 1),
                    activation=tf.nn.relu,
                    padding="valid", 
                    kernel_initializer=tf.contrib.layers.xavier_initializer(seed=self.params.seed),
                    bias_initializer= tf.constant_initializer(self.params.bias_init),
                    trainable=self.trainable,
                    name="conv3"
                    )

            elif self.env_type == 'minesweeper':
                print("Building small network")

                # Make the  network smaller

                conv1 = tf.layers.conv2d(
                    inputs=self.pl_screens, 
                    filters=18, 
                    kernel_size=[5, 5],
                    strides=(1, 1),
                    activation=tf.nn.relu,
                    padding="same",  # Same as in keras
                    kernel_initializer=tf.contrib.layers.xavier_initializer(seed=self.params.seed),
                    bias_initializer= tf.constant_initializer(self.params.bias_init),
                    trainable=self.trainable,
                    name="conv1"
                    )

                conv2 = tf.layers.conv2d(
                    inputs=conv1, 
                    filters=36, 
                    kernel_size=[3, 3],
                    strides=(1,1),
                    activation=tf.nn.relu,
                    padding="same",  # Same as in keras
                    kernel_initializer=tf.contrib.layers.xavier_initializer(seed=self.params.seed),
                    bias_initializer= tf.constant_initializer(self.params.bias_init),
                    trainable=self.trainable,
                    name="conv2"
                    )

                # conv3 = tf.layers.conv2d(
                #     inputs=conv2, 
                #     filters=32, 
                #     kernel_size=[2, 2],
                #     strides=(1,1),
                #     activation=tf.nn.relu,
                #     padding="valid",  # Same as in keras
                #     kernel_initializer=tf.contrib.layers.xavier_initializer(seed=self.params.seed),
                #     bias_initializer= tf.constant_initializer(self.params.bias_init),
                #     trainable=self.trainable,
                #     name="conv3"
                #     )


            # To implement dueling, see https://github.com/dbobrenko/reinforceflow/blob/master/reinforceflow/nets/dueling.py

            if self.dueling_type is None:

                conv_flatten = tf.contrib.layers.flatten(
                    inputs=conv2
                    )

                fc1 = tf.layers.dense(
                        inputs=conv_flatten, 
                        units=288, 
                        activation=tf.nn.relu, 
                        kernel_initializer=tf.contrib.layers.xavier_initializer(seed=self.params.seed),
                        bias_initializer= tf.constant_initializer(self.params.bias_init),
                        trainable=self.trainable,
                        name="fc1"
                    )

                fc2 = tf.layers.dense(
                        inputs=fc1, 
                        units=220, 
                        activation=tf.nn.relu, 
                        kernel_initializer=tf.contrib.layers.xavier_initializer(seed=self.params.seed),
                        bias_initializer= tf.constant_initializer(self.params.bias_init),
                        trainable=self.trainable,
                        name="fc2"
                    )


                fc3 = tf.layers.dense(
                        inputs=fc2, 
                        units=220, 
                        activation=tf.nn.relu, 
                        kernel_initializer=tf.contrib.layers.xavier_initializer(seed=self.params.seed),
                        bias_initializer= tf.constant_initializer(self.params.bias_init),
                        trainable=self.trainable,
                        name="fc3"
                    )


                qvalues = tf.layers.dense(
                            inputs=fc3, 
                            units=self.num_actions, 
                            kernel_initializer=tf.contrib.layers.xavier_initializer(seed=self.params.seed),
                            bias_initializer= tf.constant_initializer(self.params.bias_init),
                            trainable=self.trainable
                        )

            # Dueling is enabled
            else:
                #We take the output from the final convolutional layer and 
                # split it into separate advantage and value streams.

                streamAC, streamVC = tf.split(conv3,2,3)

                streamA = tf.contrib.layers.flatten(streamAC)
                streamV = tf.contrib.layers.flatten(streamVC)

                qvalues = self._add_dueling(streamA, streamV)

        return qvalues


    def _build_fc(self):
        """Builds a fully-connected network. When we add new layers, it is
        important to set trainable=self.trainable since these parameters needs
        to be copied to the target nework and it uses the set of trianable.

        Returns
        -------
        :obj: 'ndarray' of :obj: 'float'
            The q-values in the output as a list, corresponding to each possible
            action.


        """

        with tf.variable_scope("Network"):

            if self.env_type == 'atari':
                pass

            elif self.env_type == 'minesweeper':
                print("Building small network")

                fc1 = tf.layers.dense(
                        inputs=self.pl_flat,
                        units=6*6*4,
                        activation=tf.nn.relu,
                        kernel_initializer=tf.contrib.layers.xavier_initializer(seed=self.params.seed),
                        bias_initializer= tf.constant_initializer(self.params.bias_init),
                        trainable=self.trainable,
                        name='fc1'
                    )

                fc2 = tf.layers.dense(
                        inputs=fc1,
                        units=150,
                        activation=tf.nn.relu,
                        kernel_initializer=tf.contrib.layers.xavier_initializer(seed=self.params.seed),
                        bias_initializer= tf.constant_initializer(self.params.bias_init),
                        trainable=self.trainable,
                        name='fc2'
                    )

                fc3 = tf.layers.dense(
                        inputs=fc2,
                        units=150,
                        activation=tf.nn.relu,
                        kernel_initializer=tf.contrib.layers.xavier_initializer(seed=self.params.seed),
                        bias_initializer= tf.constant_initializer(self.params.bias_init),
                        trainable=self.trainable,
                        name='fc3'
                    )



            if self.dueling_type is None:

                fc4 = tf.layers.dense(
                        inputs=fc3, 
                        units=150, 
                        activation=tf.nn.relu, 
                        kernel_initializer=tf.contrib.layers.xavier_initializer(seed=self.params.seed),
                        bias_initializer= tf.constant_initializer(self.params.bias_init),
                        trainable=self.trainable,
                        name="fc4"
                    )

                qvalues = tf.layers.dense(
                            inputs=fc4, 
                            units=self.num_actions, 
                            kernel_initializer=tf.contrib.layers.xavier_initializer(seed=self.params.seed),
                            bias_initializer= tf.constant_initializer(self.params.bias_init),
                            trainable=self.trainable
                        )

            # Dueling is enabled
            else:

                qvalues = self._add_dueling(fc3, fc3)

        return qvalues

    def _add_dueling(self, streamA, streamV):
        """Adds dueling to a network.
        See:
        https://github.com/devsisters/DQN-tensorflow/blob/master/dqn/agent.py
        https://github.com/dbobrenko/reinforceflow/blob/master/reinforceflow/nets/dueling.py
        https://github.com/devsisters/DQN-tensorflow/blob/master/dqn/agent.py
        https://github.com/gokhanettin/dddqn-tf/blob/master/dddqn.py
        https://github.com/awjuliani/DeepRL-Agents/blob/master/Double-Dueling-DQN.ipynb

        And the original paper
        https://arxiv.org/pdf/1511.06581.pdf

        Parameters
        ----------
        streamA :obj: 'ndarray' of :obj: 'float'
                    The input to the advantage part.
        streamV :obj: 'ndarray' of :obj: 'float'
                    The input to value part.

        (Deprecated)
        net_input :obj: 'ndarray' of :obj: 'float'
            The input to the part of the network implementing dueling

        Returns
        -------
        :obj: 'ndarray' of :obj: 'float'
            The q-values in the output as a list, corresponding to each possible
            action.

        """

        adv_in = tf.layers.dense(
            inputs=streamA, 
            units=512, 
            activation=tf.nn.relu, 
            kernel_initializer=tf.contrib.layers.xavier_initializer(seed=self.params.seed),
            bias_initializer= tf.constant_initializer(self.params.bias_init),
            trainable=self.trainable,
            name="adv_in")

        adv_out = tf.layers.dense(
            inputs=adv_in, 
            units=self.num_actions, 
            activation=None, 
            kernel_initializer=tf.contrib.layers.xavier_initializer(seed=self.params.seed),
            bias_initializer= tf.constant_initializer(self.params.bias_init),
            trainable=self.trainable,
            name="adv_out")

        value_in = tf.layers.dense(
            inputs = streamV,
            units=512, 
            activation=tf.nn.relu, 
            kernel_initializer=tf.contrib.layers.xavier_initializer(seed=self.params.seed),
            bias_initializer= tf.constant_initializer(self.params.bias_init),
            trainable=self.trainable,
            name="value_in")

        value_out = tf.layers.dense(
            inputs=value_in, 
            units=1, 
            activation=tf.nn.relu, 
            kernel_initializer=tf.contrib.layers.xavier_initializer(seed=self.params.seed),
            bias_initializer= tf.constant_initializer(self.params.bias_init),
            trainable=self.trainable,
            name="value_out")


        # caculate the Q(s,a;theta)
        # dueling_type == 'avg'
        # Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-Avg_a(A(s,a;theta)))
        # dueling_type == 'max'
        # Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-max_a(A(s,a;theta)))
        # dueling_type == 'naive'
        # Q(s,a;theta) = V(s;theta) + A(s,a;theta)

        if self.dueling_type == 'naive':

            qvalues = value_out + adv_out

        # Mean is what is suggested in the paper
        elif self.dueling_type == 'mean':

            qvalues = tf.add(value_out, 
                        tf.subtract(adv_out, 
                            tf.reduce_mean(adv_out, 1, keep_dims=True)))

        elif self.dueling_type == 'max':

            qvalues = tf.add(value_out, 
                        tf.subtract(adv_out, 
                            tf.reduce_max(adv_out, 1, keep_dims=True)))

        else:

            raise ValueError("Unknown dueling type '%s'. Available: 'naive', \
                        'mean', 'max'." % self.dueling_type)

        return qvalues

    def _build_loss(self):
        """Build the loss function for the neural network.

        """

        with tf.variable_scope("loss"):

            # Compute y_j = r_j + discount*max_qvalue
            self.tf_discount = tf.constant(self.discount)
            #self.pl_qtargets = tf.reduce_max(self.pl_qtargets, reduction_indices=[1])


            # self.pl_actions_train
            #self.pl_qtargets = self.pl_qtargets[range(self.batch_size), self.pl_actions_target]

            # Get Q-values for the argmax actions from actions train
            # For this, we use the one-hot encoding
            self.actions_one_hot_target = tf.one_hot(
                                    self.pl_actions_target, self.num_actions, 1.0, 0.0
                                    )

            # Finally, we compute the q-vaue of the target network
            # for the actions which maximixed the 
            self.qtarget = tf.reduce_sum(
                                    tf.multiply(
                                        self.pl_qtargets, 
                                        self.actions_one_hot_target), 
                                        reduction_indices=1
                                )

            if self.env_type == 'atari':

                # In atari, done is bad since the games should last as long as possible
                # for mine-sweeper, done is ambigious, and we represent loss by a strong negative
                # reward, and win is simply progressing by a certain number of steps
                self.qtarget = tf.add(
                                    self.pl_rewards, 
                                    tf.multiply(1.0-self.pl_dones, 
                                        tf.multiply(self.tf_discount, self.qtarget))
                                )

            elif self.env_type == 'minesweeper':

                self.qtarget = tf.add(
                                        self.pl_rewards, 
                                        tf.multiply(self.tf_discount, self.qtarget)
                                    )

            # Select Q-values for given actions
            # The actions are a 1D array, we one-hot encode with the number of actions
            # last two are on-value, off-value
            # Then we just have a matrix with 1s where an action was performed
            self.actions_one_hot = tf.one_hot(
                                    self.pl_actions, self.num_actions, 1.0, 0.0
                                    )


            # Below we obtain the loss by taking the sum of squares difference 
            # between the target and prediction Q values.
            # Compute finally sum along all rows by summing column elements
            self.qvalue_pred = tf.reduce_sum(
                                    tf.multiply(
                                        self.qvalues, 
                                        self.actions_one_hot), 
                                        reduction_indices=1
                                )
            
            # Difference between target (true) and predicted Q-network output
            self.delta = tf.subtract(self.qtarget, self.qvalue_pred)

            if self.clip_delta:
                # Perform clipping of the error term, 
                # default clipping is to (-1, +1) range
                self.delta_square = tf.losses.huber_loss(self.qtarget, self.qvalue_pred)
            else:
                # No error clipping
                # td-error
                self.delta_square  = tf.square(self.delta)

        # Actual loss
        if self.batch_accumulator == "sum":
           self.loss = tf.reduce_sum(self.delta_square)
        else:
           self.loss = tf.reduce_mean(self.delta_square)


        # Running average of the loss for TensorBoard
        self.loss_moving_avg    = tf.train.ExponentialMovingAverage(decay=0.999)
        self.loss_moving_avg_op = self.loss_moving_avg.apply([self.loss])


    def build_summary_writer(self, sess):
        """Build summary writer of the progress.

        """

        # Compute the average Q-value
        avg_qvalues_train  = tf.reduce_mean(self.qvalues)
        avg_qvalues_target = tf.reduce_mean(tf.reduce_mean(self.pl_qtargets))
        avg_reward_batch   = tf.reduce_mean(tf.reduce_mean(self.pl_rewards))

        # Summaries for training
        training_summaries = [
            tf.summary.scalar("train/qvalues_train_avg", avg_qvalues_train),
            tf.summary.scalar("train/qvalues_target_avg", avg_qvalues_target),
            tf.summary.scalar("train/avg_reward_batch", avg_reward_batch),
            tf.summary.scalar("train/loss", self.loss),
            tf.summary.scalar("train/loss_average", self.loss_moving_avg.average(self.loss)),
            tf.summary.scalar("train/learning_rate", self.learning_rate),
            tf.summary.histogram("train/delta", self.delta)
        ]
        training_summaries_merged = tf.summary.merge(training_summaries)

        # Environment related summaries
        with tf.variable_scope("environment"):
            self.avg_reward_per_game = tf.Variable(0.0, trainable=False, name="avg_reward_per_game")
            self.max_reward_per_game = tf.Variable(0.0, trainable=False, name="max_reward_per_game")
            self.avg_moves_per_game  = tf.Variable(0.0, trainable=False, name="avg_moves_per_game")
            self.num_games_played    = tf.Variable(0.0, trainable=False, name="num_games_played")
            self.moves               = tf.Variable(0.0, trainable=False, name="num_moves_played")
            self.total_reward_replay = tf.Variable(0.0, trainable=False, name="reward_in_replay_memory")
            self.actions_random      = tf.Variable(0.0, trainable=False, name="num_actions_random")
            self.actions_greedy      = tf.Variable(0.0, trainable=False, name="num_actions_greedy")

        environment_summaries = [
            tf.summary.scalar("environment/avg_reward_per_game", self.avg_reward_per_game),
            tf.summary.scalar("environment/max_reward_per_game", self.max_reward_per_game),
            tf.summary.scalar("environment/num_games_played", self.num_games_played),
            tf.summary.scalar("environment/moves", self.moves),
            tf.summary.scalar("environment/avg_moves_per_game", self.avg_moves_per_game),
            tf.summary.scalar("environment/reward_in_replay_memory", self.total_reward_replay),
            tf.summary.scalar("actions/num_actions_random", self.actions_random),
            tf.summary.scalar("actions/num_actions_greedy", self.actions_greedy),
            #tf.summary.image("screens", self.pl_screens, max_outputs=10) # This only works with atari for some reason
        ]
        environment_summaries_merged = tf.summary.merge(environment_summaries)

        # Environment related summaries
        with tf.variable_scope("evaluation"):
            self.eval_rewards      = tf.Variable(0.0, trainable=False, name="total_reward")
            self.eval_win_rate     = tf.Variable(0.0, trainable=False, name="win_rate")
            self.eval_num_rewards  = tf.Variable(0.0, trainable=False, name="num_rewards")
            self.eval_max_reward   = tf.Variable(0.0, trainable=False, name="max_reward")
            self.eval_num_episodes = tf.Variable(0.0, trainable=False, name="num_episodes")
            self.eval_actions      = tf.Variable(np.zeros(self.num_actions), trainable=False, name="actions")


        evaluation_summaries = [
            tf.summary.scalar("evaluation/total_reward", self.eval_rewards),
            tf.summary.scalar("evaluation/win_rate", self.eval_win_rate),
            tf.summary.scalar("evaluation/num_rewards", self.eval_num_rewards),
            tf.summary.scalar("evaluation/max_reward", self.eval_max_reward),
            tf.summary.scalar("evaluation/num_episodes", self.eval_num_episodes),
            tf.summary.histogram("evaluation/actions", self.eval_actions)
        ]

        # Evaluation Summaries for TensorBoard
        self.eval_summary_op = tf.summary.merge(evaluation_summaries)

        # Training summaries for TensorBoard
        self.train_summary_op = tf.summary.merge([training_summaries_merged,
                                                  environment_summaries_merged])

        train_summary_dir = os.path.join(self.output_dir, "summaries_" +  self.params.game)
        self.train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
