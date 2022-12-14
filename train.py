import argparse
import accelerate
import torch
import torchmetrics
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms as T

from ezdl.datasets import GifReader

from models import ActionClassifier


def main(opt):
    ar = accelerate.Accelerator()
    device = ar.device

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,))
    ])
    train_ds = GifReader(opt.path, 'train', transform)
    valid_ds = GifReader(opt.path, 'validate', transform)

    kwargs = {
        'batch_size': opt.batch_size,
        'num_workers': 8,
        'pin_memory': True
    }
    trainloader = DataLoader(train_ds, shuffle=True, **kwargs)
    validloader = DataLoader(valid_ds, shuffle=False, **kwargs)

    model = ActionClassifier('res', classes=600)

    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.AdamW(model.parameters(), opt.lr)
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [opt.epoch//2, 4*opt.epoch//5], 0.1)

    trainloader, validloader, model, optimizer = ar.prepare(
        trainloader, validloader, model, optimizer
    )

    if ar.is_main_process:
        metric = torchmetrics.Accuracy()

    for epoch in tqdm(range(opt.epoch)):
        model.train()

        for data, label in trainloader:
            out = model(data)
            loss = criterion(out, label).mean((1, 2, 3, 4)).sum()

            optimizer.zero_grad()
            ar.backward(loss)
            optimizer.step()

        model.eval()
        valid_loss = 0

        with torch.no_grad():
            for data, label in validloader:
                out = model(data)
                preds = out.argmax(1)
                loss = criterion(out, label).mean((1, 2, 3, 4)).sum()

                preds, label, loss = ar.gather_for_metrics(preds, label, loss)

                valid_loss += loss.item()
                if is_main_process:
                    metric.update(preds, label)

        ar.print(f'Loss: {valid_loss}, Acc: {metric.compute()}')

        if is_main_process:
            metric.reset()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('-e', '--epoch', type=int, default=10)
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-lr', type=float, default=1e-3)
    main(parser.parse_args())

