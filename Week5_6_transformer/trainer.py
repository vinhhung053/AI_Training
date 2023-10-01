from tqdm.notebook import tqdm
import torch

class Trainer:
    def __init__(self, classifier, optimizer, criterion, epoch):
        self.classifier = classifier
        self.optimizer = optimizer
        self.criterion = criterion
        self.epoch = epoch
    def update_lr(self):
        for param_group in self.optimizer.param_groups:
                param_group['lr'] *= 0.95

    def train(self, train_loader):
        epoch_bar = tqdm(range(self.epoch),
                         desc="Training",
                         position=0,
                         total=2)
        for epoch in epoch_bar:
            batch_bar = tqdm(enumerate(train_loader),
                             desc="Epoch: {}".format(str(epoch)),
                             position=1)
            for i, (x_batch, y_batch) in batch_bar:

                self.optimizer.zero_grad()
                preds = [self.classifier(x) for x in x_batch]
                preds = torch.stack(preds)

                preds = preds.view(-1,preds.shape[-1])
                y_batch = y_batch.view(-1)
                # print(preds.shape)
                # print(y_batch.shape)

                loss = self.criterion(preds, y_batch)

                loss.backward(retain_graph=True)
                self.optimizer.step()
                acc = (preds.argmax(dim=1) == y_batch).float().mean().cpu().item()
                print(acc)
            self.update_lr(self.optimizer)  # cap nhat lr

    def val(self,test_loader):
        batch_bar = tqdm(enumerate(test_loader),
                         desc="Epoch: {}".format(str(self.epoch)),
                         position=1)
        for i, (x_batch, y_batch) in batch_bar:
            self.optimizer.zero_grad()
            preds = [self.classifier(x) for x in x_batch]
            preds = torch.stack(preds)

            preds = preds.view(-1, preds.shape[-1])
            y_batch = y_batch.view(-1)
            # print(preds.shape)
            # print(y_batch.shape)
            loss = self.criterion(preds, y_batch)
            acc = (preds.argmax(dim=1) == y_batch).float().mean().cpu().item()
            print(acc)

