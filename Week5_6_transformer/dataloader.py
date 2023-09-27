import numpy as np


class Dataloader:
    def __init__(self, x_data, batch_size=32, shuffle=True):
        self.x_data = x_data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(x_data)
        self.num_feature = x_data.shape[1]
        self.indices = np.arange(self.num_samples)  # -> indices su dung cho viec shuffle
        if (self.shuffle):
            np.random.shuffle(self.indices)
        self.current_id = 0  # su dung cho next trong iter

    def get_num_samples(self):
        return self.num_samples

    def get_num_feature(self):
        return self.num_feature

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_id + self.batch_size > self.num_samples:
            # raise StopIteration  # ket thuc vong lap
            self.current_id = 0
        batch_indices = self.indices[self.current_id: self.current_id + self.batch_size]
        x_batch = self.x_data[batch_indices]

        self.current_id = self.current_id + self.batch_size
        return x_batch

