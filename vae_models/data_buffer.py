import threading
import numpy as np

"""
the replay buffer here is basically from the openai baselines code

"""


class DataBuffer:
    def __init__(self, env_params, buffer_size):
        self.env_params = env_params
        self.size = buffer_size

        self.current_size = 0

        # create the buffer to store info
        self.buffer = np.empty([self.size, self.env_params['goal']])

        # thread lock
        self.lock = threading.Lock()

    # store the encountered goals
    def store_data(self, data_batch):
        batch_size = len(data_batch)
        with self.lock:
            idxs = self._get_storage_idx(inc=batch_size)

            for i, e in enumerate(data_batch):
                # store the informations
                self.buffer[idxs[i]] = e[i]

    # sample the data from the replay buffer
    def sample(self, batch_size):
        with self.lock:
            ids = np.random.choice(np.arange(self.current_size), size=batch_size)
            sampled_buffer = self.buffer[ids]

        return sampled_buffer

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_size + inc <= self.size:
            idx = np.arange(self.current_size, self.current_size + inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size + inc)
        if inc == 1:
            idx = [idx[0]]
        return idx
