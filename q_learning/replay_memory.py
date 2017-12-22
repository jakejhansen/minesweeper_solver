"""
This module contains class definitions for memories used during training of the
Q-network.
"""

import os
import numpy as np
import cv2
import h5py
import shutil


class ScreenHistory(object):
    """A simple class to store the last N screen images in the batch. When we actually do
    train, we use the replay memory, this class is used to interact with the
    environment and obtain new rewards based on actions etc.

    Based on: https://github.com/devsisters/DQN-tensorflow/blob/master/dqn/history.py

    Parameters
    ----------
    params:
        arguments

    Attributes
    ----------
    screens : :obj: 'list' of :obj: 'ndarray' of :obj: 'float'
        The screen output (grayscale) collected in a small history with the
        newest in the highest index.

    """


    def __init__(self, params):

        self.history_length = params.history_length
        self.nchannels = params.nchannels

        if self.nchannels == 1:
            self.screens = np.zeros(
                [params.history_length, params.input_height, params.input_width], 
                dtype=np.float32
            )
        else:
            self.screens = np.zeros(
                [params.history_length, params.input_height, params.input_width, self.nchannels], 
                dtype=np.float32
            )



    def add(self, screen):
        """Adds a new screen to the screen buffer by first deleting the oldest
        screen in the buffer and then inserting.

        Returns
        -------
        None

        """

        self.screens = np.delete(self.screens, 0, axis=0)
        self.screens = np.insert(self.screens, self.history_length-1, screen, axis=0)


    def reset(self):
        """Reset the screen buffer.

        Returns
        -------
        None

        """
        self.screens *= 0.0


    def get(self):
        """Returns the screen buffer.

        If we have inputted arrays of size (2, 2), we have that self.screens
        is of (4, 2, 2). We then convert to (1, 2, 2, 4).
        So we have 4 channels with each corresponding to a time-step, and the
        other two dimensions just correspond to specific positions in the
        general image.

        Example
        -------
        >>> test = array([[[ 0.,  0.],
                [ 0.,  0.]],

               [[ 1.,  2.],
                [ 3.,  4.]],

               [[ 1.,  2.],
                [ 3.,  4.]],

               [[ 2.,  3.],
                [ 4.,  5.]]])

        >>> np.transpose(test, [1, 2, 0])
        array([[[ 0.,  1.,  1.,  2.],   --> (time-series for (0,0))
                [ 0.,  2.,  2.,  3.]],  --> (time-series for (0,1))

               [[ 0.,  3.,  3.,  4.],   --> (time-series for (1,0))
                [ 0.,  4.,  4.,  5.]]]) --> (time-sereis for (1,1))

        So in each row we will have a time-sequence at a position in the array
        and the highest dimension just corresponds to all the time-series for
        the row in the original.

        Returns
        -------
        :obj: 'list' of :obj: 'ndarray' of :obj: 'float'

        """

        # For mine-sweeper and others with no history
        if self.history_length == 4 or self.nchannels == 1: # This is for atari, need to clean it up
            screen = np.expand_dims(np.transpose(self.screens, [1, 2, 0]), axis=0)
        elif self.nchannels == 2:
            return self.screens

        return screen


class ReplayMemory(object):
    """A class for replay memories used during training of deep Q-networks.

    Based on https://github.com/devsisters/DQN-tensorflow/blob/master/dqn/replay_memory.py

    Parameters
    ----------
    replay_capacity : int
        The maximum number of samples in replay memory
    batch_size : Bool
        The number of frames to return.

    Attributes:
    replay_capacity : int
        The maximum number of samples in replay memory
    counter : int
        The next place in the replay buffer where we can place an element.
    current : int
        current position in the replay buffer, we wrap around when we reach
        the end.
    screens : :obj: 'ndarray' of :obj: 'float'
        The screen output (grayscale)
    rewards : :obj: 'list' of :obj: 'float'
        The screen output (grayscale)
    dones : :obj: 'list' of :obj: 'bool'
        The screen output (grayscale)
    batch_size : int
        The size of the batch
    batch_screens_pre : :obj: 'ndarray' of :obj: 'float'
        The set of screens prior to a corresponding set of actions.
    batch_screens_post : :obj: 'ndarray' of :obj: 'float'
        The set of screens after a corresponding set of actions.

    """


    def __init__(
        self, replay_capacity=int(1e6), history_length=4, nchannels=1, batch_size=32, 
        screen_height=84, screen_width=84, file_name = None,
        memory_checkpoint = int(1e6), restore_memory=False, output_dir = None
    ):

        self.replay_capacity = replay_capacity
        self.counter = 0
        self.current = 0

        self.history_length = history_length
        self.nchannels = nchannels

        self.output_dir = output_dir

        self.restore_memory = restore_memory

        self.memory_checkpoint = memory_checkpoint

        self.screen_height  = screen_height
        self.screen_width   = screen_width

        self.file_name = file_name

        # Pre-initialization for batches
        self.batch_size = batch_size


        # We do np.zeros since it seems faster than np.emtpy when filling, see
        # https://stackoverflow.com/questions/43145332/numpy-array-of-zeros-or-empty
        # https://stackoverflow.com/questions/26213199/np-fullsize-0-vs-np-zerossize-vs-np-empty

        if self.nchannels == 1:
            self.screens    = np.zeros(
                                [self.replay_capacity, screen_height, screen_width], 
                                dtype=np.float16
                            )

            # Change 4 to self.history_length
            self.batch_screens_pre  = np.zeros(
                                        [self.batch_size, self.history_length, screen_height, screen_width], 
                                        dtype=np.float16)

            self.batch_screens_post = np.zeros(
                                        [self.batch_size, self.history_length, screen_height, screen_width], 
                                        dtype=np.float16)

        else:
            # We have removed self.history_length for these, maybe add nchannels here and then do
            # the transpotition in the end

            self.screens    = np.zeros(
                                [self.replay_capacity, screen_height, screen_width, nchannels], 
                                dtype=np.float16
                            )

            # Change 4 to self.history_length
            self.batch_screens_pre  = np.zeros(
                                        [self.batch_size, screen_height, screen_width, nchannels], 
                                        dtype=np.float16)

            self.batch_screens_post = np.zeros(
                                        [self.batch_size, screen_height, screen_width, nchannels], 
                                        dtype=np.float16)


        self.actions    = np.zeros(self.replay_capacity, dtype=np.uint8)
        self.rewards    = np.zeros(self.replay_capacity, dtype=np.float16)
        self.dones      = np.zeros(self.replay_capacity, dtype=np.bool)

        if self.file_name:
            self.checkpoint_dir = os.path.abspath(os.path.join(self.output_dir, "replay_" + self.file_name))
            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)

        # Check if there is a replay buffer stored
        if os.path.exists(self.checkpoint_dir) and self.restore_memory:
            self.load_memory()



    def add(self, action, reward, post_screen, is_done):
        """Add a sample to the replay memory. The previous screen is already in
        the memory so only the screen after performing the action is required 
        (this is post_screen)

        Parameters
        ----------
        action: int
            The action performed.
        reward: float
            The reward gained form performing an action (clipped)
        post_screen: :obj: 'array' of :obj: 'float'
            The screen after performing an action.
        is_done: bool
            True for game is over, False otherwise.

        Returns
        -------
        None

        """
        self.rewards[self.current]      = reward
        self.actions[self.current]      = action

        # Same as [1,:,:] but faster
        self.screens[self.current, ...] = post_screen
        self.dones[self.current]        = is_done

        self.counter = max(self.counter, self.current+1)
        self.current = (self.current + 1) % self.replay_capacity

        #Check if we should save now
        # We use current since we also want to save newer episodes.
        if not (self.current + 1) % self.memory_checkpoint:
            self.save_memory()


    # Rename this to getState()
    def state(self, index):
        """Returns the current state of the game, that is the screens at this 
        index. This method is based on the following implementation: 
        https://goo.gl/1U2eFn

        Parameters
        ----------
        index: int
            index of the state in replay memory

        Returns
        -------
        None

        """

        assert self.counter > 0, "replay memory is empy, use at least --random_steps 1"

        # if is not in the beginning of matrix
        index = index % self.counter
        if index >= (self.history_length-1): # history length - 1
            # use faster slicing
            screens = self.screens[(index-(self.history_length-1)):(index + 1),...]
        else:
            # otherwise normalize indexes and use slower list based access
            indexes = [(index - i) % self.counter for i in reversed(range(self.history_length))]
            screens = self.screens[indexes,...]

        return screens

    def sample_batch(self):
        """Randomly samples a batch from the replay memory based on the
        given batch_size.
        This method also checks if the sample is already in the batch when
        doing random sampling.

        Returns
        -------
        :obj: 'list' of 
            :obj: 'list' of :obj: 'ndarray' of :obj: 'float'
                Sets of screens before an action.
            :obj: 'list' of :obj: 'int'
                Sets of actions.
            :obj: 'list' of :obj: 'float'
                Sets of rewards for these actions.
            :obj: 'list' of :obj: 'ndarray' of :obj: 'float'
                Sets of screens after an action.
            :obj: 'list' of :obj: 'bool'
                Sets containing if game finished.

        """

        assert self.counter > self.history_length
        assert self.counter > self.batch_size # We need to have more

        # Indices of samples in batch.
        samples_in_batch = []
        while len(samples_in_batch) < self.batch_size:

            # find random index
            while True:
                # sample one index (ignore states wraping over 
                index = np.random.randint(self.history_length, self.counter-1)
                if index >= self.current > (index-self.history_length):
                    # Sample not within range, continue
                    continue

                # if wraps over current pointer, then get new one
                if index >= self.current and index - self.history_length < self.current:
                    continue
                # Sample is already in the batch, get a new one
                if index in samples_in_batch:
                    continue

                # Check there is a done state within the index range
                # The last state can't be a done state
                if self.dones[(index-self.history_length):index].any():
                    # Found done state
                    continue

                # Found a valid sample
                break

            # Add the sample to the batch
            self.batch_screens_pre[len(samples_in_batch), ...]  = \
                 self.state(index-1)

            # ... is the same as :,:,:
            self.batch_screens_post[len(samples_in_batch), ...] = \
                 self.state(index)

            samples_in_batch.append(index)


        # Note that we permute axis of the tensors from [32,4,84,84] to 
        # [32,84,84,4]
        # In order to have it as channels for atari
        # This always has to be 3 for atari!
        # This is the same thing we do in screen history
        if self.nchannels == 1:
            return_screens_pre = np.transpose(self.batch_screens_pre, (0, 2, 3, 1))
            #print(return_screens_pre.shape)
            return_screens_post = np.transpose(self.batch_screens_post, (0, 2, 3, 1))

        else:
            return_screens_pre = self.batch_screens_pre
            return_screens_post = self.batch_screens_post

        actions = self.actions[samples_in_batch]
        rewards = self.rewards[samples_in_batch].astype(np.float16)
        dones = self.dones[samples_in_batch].astype(np.float16)

        return return_screens_pre, actions, rewards, return_screens_post, dones

    def num_examples(self):
        """int: Returns the number of examples."""
        return min(self.counter, self.replay_capacity)


    def action_counts(self):
        """ndarray of ints: Returns the number of bins in actions."""
        return np.bincount(self.actions)


    def total_reward(self):
        """float: Returns the total reward."""
        return np.sum(self.rewards)

    def save_memory(self):   
        """Saves a replay buffer to h5.

        Parameters
        ----------
        file_name: str
            The name of the game

        """
        print()
        print("Saving replay buffer to memory.")
        print()

        # Remove if it exists
        if os.path.exists(self.checkpoint_dir):
            shutil.rmtree(self.checkpoint_dir)
            os.makedirs(self.checkpoint_dir)
        else:
            os.makedirs(self.checkpoint_dir)

        with open(self.checkpoint_dir + os.sep + self.file_name + '.txt', 'w') as f:
            f.write('%d\n' % self.counter)
            f.write('%d\n' % self.current)

        with h5py.File(self.checkpoint_dir + os.sep + self.file_name + '_screens.h5', 'w') as hf:
            hf.create_dataset(self.file_name,  data=self.screens[0:(self.counter-1), ...], chunks=True, compression="gzip", compression_opts=4)

        with h5py.File(self.checkpoint_dir + os.sep + self.file_name + '_actions.h5', 'w') as hf:
            hf.create_dataset(self.file_name,  data=self.actions[0:(self.counter-1), ...], chunks=True, compression="gzip", compression_opts=4)

        with h5py.File(self.checkpoint_dir + os.sep + self.file_name + '_rewards.h5', 'w') as hf:
            hf.create_dataset(self.file_name,  data=self.rewards[0:(self.counter-1), ...], chunks=True, compression="gzip", compression_opts=4)

        with h5py.File(self.checkpoint_dir + os.sep + self.file_name + '_dones.h5', 'w') as hf:
            hf.create_dataset(self.file_name,  data=self.dones[0:(self.counter-1), ...], chunks=True, compression="gzip", compression_opts=4)

    def load_memory(self):
        """Loads a replay buffer from h5.

        Parameters
        ----------
        self.file_name: str
            The name of the game

        """

        print()
        print("Loading replay buffer from memory")
        print()

        with open(self.checkpoint_dir + os.sep + self.file_name + '.txt', 'r') as f:
            self.counter = int(f.readline())
            self.current = int(f.readline())

        with h5py.File(self.checkpoint_dir + os.sep + self.file_name + '_screens.h5', 'r') as hf:
            self.screens[0:(self.counter-1), ...] = hf[self.file_name][0:(self.counter-1)]

        with h5py.File(self.checkpoint_dir + os.sep + self.file_name + '_actions.h5', 'r') as hf:
            self.actions[0:(self.counter-1), ...] = hf[self.file_name][0:(self.counter-1)]

        with h5py.File(self.checkpoint_dir + os.sep + self.file_name + '_rewards.h5', 'r') as hf:
            self.rewards[0:(self.counter-1), ...] = hf[self.file_name][0:(self.counter-1)]

        with h5py.File(self.checkpoint_dir + os.sep + self.file_name + '_dones.h5', 'r') as hf:
            self.dones[0:(self.counter-1), ...] = hf[self.file_name][0:(self.counter-1)]

