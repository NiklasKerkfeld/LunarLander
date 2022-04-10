import numpy as np


class Memory:
    def __init__(self, buffer_size, batch_size, observation_space, action_space):
        """
        Memory Buffer vor Experience Replay.
        :param buffer_size: number of saves Replays
        :param batch_size: size of a returned batch
        :param observation_space: size of the observation_space
        :param action_space: number of posible actions
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.pointer = 0
        self.size = 0

        self.states = np.zeros(shape=(buffer_size, observation_space))
        self.actions = np.zeros(shape=(buffer_size, ), dtype=np.int32)
        self.rewards = np.zeros(shape=(buffer_size,))
        self.new_states = np.zeros(shape=(buffer_size, observation_space))
        self.dones = np.zeros(shape=(buffer_size,))

    @property
    def ready(self):
        return self.size > self.batch_size

    def append(self, state, action, reward, done, new_state):
        """
        add a Experience to the memory
        :param state: state at t
        :param action: action at t
        :param reward: reward for the action
        :param done: is simulation over
        :param new_state: state at t+1
        :return: None
        """

        assert state.shape == (self.observation_space, ), f"appended state hase wrong shape ({state.shape})"
        assert new_state.shape == (self.observation_space,), f"appended new_state hase wrong shape ({new_state.shape})"
        assert isinstance(action, int), f"appended action hase wrong type ({type(action)})"
        # assert reward == 1 or reward == -1, f"appended reward is not in [1, -1] ({reward})"
        assert isinstance(done, bool) or isinstance(done, np.bool_), f"appended done hase wrong type ({type(done)})"

        self.size = min(self.size + 1, self.buffer_size)
        self.pointer = (self.pointer + 1) % self.buffer_size
        self.states[self.pointer] = state
        self.actions[self.pointer] = action
        self.rewards[self.pointer] = reward
        self.new_states[self.pointer] = new_state
        self.dones[self.pointer] = done

    def batch(self):
        """
        :returns a random batch from memory
        :return: states, actions, rewards, dones, new_states
        """
        assert self.size >= self.batch_size, f"not enough datapoints in Memory have {self.size}"

        sample = np.random.randint(0, self.size, self.batch_size)

        return np.array(self.states[sample]), self.actions[sample], self.rewards[sample], self.dones[sample], np.array(self.new_states[sample])