import megengine as mge
import megengine.functional as F
import megengine.module as M
from megengine.data.dataset import Dataset
from megengine.data import DataLoader, RandomSampler, SequentialSampler
import megengine.optimizer as optim
from megengine.jit import trace

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class XORDataset(Dataset):
    def __init__(self, num_points):
        super().__init__()

        self.data = np.random.rand(num_points, 2).astype(np.float32) * 2 - 1
        self.label = (np.prod(self.data, axis=1) < 0).astype(np.int32)
        print(self.label)

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

def draw_dataset(dataset):
    xs = []
    ys = []
    for x, y in dataset:
        xs.append(x)
        ys.append(y)
    xs = np.stack(xs, 0)
    ys = np.stack(ys, 0)

    xs_0 = xs[ys==0]
    xs_1 = xs[ys==1]
    plt.plot(xs_0[:, 0], xs_0[:, 1], 'r.')
    plt.plot(xs_1[:, 0], xs_1[:, 1], 'b.')
    plt.show()

class SimpleNet(M.Module):
    def __init__(self, input_dim, features):
        super().__init__()
        layers = []
        out_dim = input_dim 
        num_layers = len(features)
        for i, feat in enumerate(features):
            layers.append(M.Linear(out_dim, feat))
            if i != num_layers - 1:
                layers.append(M.ReLU())
            out_dim = feat
        self.layers = M.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def build_model():
    return SimpleNet(input_dim=2, features=[8, 8, 2])


@trace
def train_func(data, label, *, opt, model):
    logits = model(data)
    loss = F.cross_entropy_with_softmax(logits, label)
    opt.backward(loss)
    return logits, loss

# 静态图比动态图快了很多
trace.enabled = True

if __name__ == '__main__':
    np.random.seed(39)
    train_dataset = XORDataset(30000)
    test_dataset = XORDataset(10000)

    # 这里为什么要传两次train_dataset
    train_sampler = RandomSampler(dataset=train_dataset, batch_size=32, drop_last=True)
    train_loader = DataLoader(train_dataset, sampler=train_sampler)

    test_sampler = SequentialSampler(dataset=test_dataset, batch_size=32, drop_last=False)
    test_loader = DataLoader(test_dataset, sampler=test_sampler)

    # draw_dataset(train_dataset)
    model = build_model()

    optimizer = optim.SGD(
        model.parameters(),
        lr=0.01,
    )

    data = mge.tensor()
    label = mge.tensor(dtype='int32')

    total_epochs = 10
    for epoch in range(total_epochs):
        total_loss = 0.0
        num_batch = len(train_loader)
        frequent = 100
        model.train()
        for step, (batch_data, batch_label) in enumerate(train_loader):
            data.set_value(batch_data) # 能直接将numpy.ndarray转换成mge.Tensor会更好
            label.set_value(batch_label)
            optimizer.zero_grad()
            logits, loss = train_func(data, label, opt=optimizer, model=model)
            optimizer.step()
            loss = loss.numpy().item()
            total_loss += loss
            if step % frequent == 0:
                print(f'[Epoch: {epoch}][{step}/{num_batch}]: {loss}')
        print('Epoch: {}, loss {}'.format(epoch, total_loss / num_batch))

        print('start testing...')
        model.eval()
        preds = []
        labels = []
        for step, (batch_data, batch_label) in tqdm(enumerate(test_loader)):
            data.set_value(batch_data) # 能直接将numpy.ndarray转换成mge.Tensor会更好
            labels.append(batch_label)
            logits = model(data)
            pred = np.argmax(logits.numpy(), axis=1)
            preds.append(pred)
        preds = np.concatenate(preds, 0)
        labels = np.concatenate(labels, 0)
        acc = (preds == labels).mean()
        print(f'Accuracy: {acc}')
