class BatchedDataLoader:
    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < len(self.data):
            batch = self.data[self.index: self.index + self.batch_size]
            self.index += self.batch_size
            return batch
        else:
            raise StopIteration


# Tạo một đối tượng BatchedDataLoader với dữ liệu là danh sách các số nguyên từ 1 đến 10 và batch size là 3
data_loader = BatchedDataLoader(list(range(1, 11)), batch_size=3)

# Sử dụng vòng lặp for để duyệt qua các batch
for batch in data_loader:
    print(batch)