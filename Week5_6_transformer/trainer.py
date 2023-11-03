from tqdm.notebook import tqdm
import torch


class Trainer:
    def __init__(self, model, optimizer, criterion, epoch):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.epoch = epoch

    def update_lr(self):
        for param_group in self.optimizer.param_groups:
                param_group['lr'] *= 0.99

    def train(self, train_loader, args):
        epoch_bar = tqdm(range(self.epoch),
                         desc="Training",
                         position=0,)
        for epoch in epoch_bar:
            batch_bar = tqdm(enumerate(train_loader),
                             desc="Epoch: {}".format(str(epoch)),
                             position=1)
            for i, (x_batch, y_batch) in batch_bar:
                self.optimizer.zero_grad()
                preds = self.model(x_batch, torch.zeros((1, 1)))[0]
                # print(preds)
                # preds = torch.stack(preds)
                preds = preds.view(-1, preds.shape[-1])
                for id in range(y_batch.shape[0]):
                    y_batch[id] = torch.cat((y_batch[id][1:], torch.tensor([2])))
                y_batch = y_batch.view(-1)
                loss = self.criterion(preds, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=0.1)

                self.optimizer.step()
                batch_bar.write("Loss = {}".format(loss.item()))
                batch_bar.update()
                # print(self.model(torch.tensor([59, 49, 39, 29])))
            self.update_lr()  # cap nhat lr
            torch.save(self.model, args.pre_model_path)

    def val(self, test_loader):
        batch_bar = tqdm(enumerate(test_loader),
                         desc="Epoch: {}".format(str(self.epoch)),
                         position=1)
        for i, (x_batch, y_batch) in batch_bar:
            self.optimizer.zero_grad()
            preds = [self.model(x) for x in x_batch]
            preds = torch.stack(preds)

            preds = preds.view(-1, preds.shape[-1])
            y_batch = y_batch.view(-1)
            loss = self.criterion(preds, y_batch)
            # acc = (preds.argmax(dim=1) == y_batch).float().mean().cpu().item()
            batch_bar.write("Loss val = {}".format(loss.item()))
            batch_bar.update()

