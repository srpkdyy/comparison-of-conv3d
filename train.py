import argparse
import accelerate
import torch
import torchmetrics
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms as T

from ezdl.datasets import GifReader, VideoReader

from models import ActionClassifier


def main(opt):
    ar = accelerate.Accelerator()
    device = ar.device

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,))
    ])
    #train_ds = GifReader(opt.path, 'train', transform)
    #valid_ds = GifReader(opt.path, 'validate', transform)
    train_ds = VideoReader(opt.path, 'train', size=64, frames=32, frameskip=4)
    valid_ds = VideoReader(opt.path, 'validate', size=64, frames=32, frameskip=4)

    kwargs = {
        'batch_size': opt.batch_size,
        'num_workers': 8,
        'pin_memory': True
    }
    trainloader = DataLoader(train_ds, shuffle=True, **kwargs)
    validloader = DataLoader(valid_ds, shuffle=False, **kwargs)

    block_types = ['res', 'legacy', 
    model = ActionClassifier('res', classes=600)

    criterion = nn.CrossEntropyLoss(reduction='sum')
    optimizer = optim.AdamW(model.parameters(), opt.lr)
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [opt.epoch//2, 4*opt.epoch//5], 0.1)

    trainloader, validloader, model, optimizer = ar.prepare(
        trainloader, validloader, model, optimizer
    )

    for epoch in tqdm(range(opt.epoch)):
        model.train()

        for data, label in trainloader:
            out = model(data)
            loss = criterion(out, label)

            optimizer.zero_grad()
            ar.backward(loss)
            optimizer.step()

            ar.print(f'Loss: {loss.item()}')

        model.eval()
        valid_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, label in validloader:
                out = model(data)
                preds = out.argmax(1)
                loss = criterion(out, label)

                preds, label, loss = ar.gather_for_metrics((preds, label, loss))
                crct = preds == label

                valid_loss += loss.mean().item()
                correct += crct.sum()
                total += crct.shape[0]
                
        ar.print(f'Valid Loss: {valid_loss}, Acc: {100*correct/total}({correct}/{total})')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('-e', '--epoch', type=int, default=10)
    parser.add_argument('-b', '--batch-size', type=int, default=128)
    parser.add_argument('-lr', type=float, default=1e-3)
    main(parser.parse_args())

