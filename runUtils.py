import torch
from torch import nn
import config

config.setup_seed()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

labelDict = {
    0: 'positive',
    1: 'negative',
    2: 'neutral'
}


def train(model, optimizer, train_loader, val_loader, mode):
    criterion = nn.CrossEntropyLoss()
    best_accuracy = 0.
    for e in range(config.epoch):
        for i, data in enumerate(train_loader):
            model.train()
            labels = data['tag'].to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            out = out.argmax(dim=1)
            accuracy = (out == labels).sum().item() / len(labels)
            print('epoch:', e + 1, 'step:', i + 1, 'loss:', loss.item(), 'train accuracy:', accuracy)
            if (i + 1) % 10 == 0:
                print('validation accuracy:', test(model, val_loader))
        accuracy = test(model, val_loader)
        print('epoch:', e + 1, 'validation accuracy:', accuracy)

        if accuracy >= best_accuracy:
            best_accuracy = accuracy
            torch.save(model, 'model_' + mode)
            print('saved the model')


def test(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            labels = data['tag'].to(device)
            out = model(data)
            out = out.argmax(dim=1)
            correct += (out == labels).sum().item()
            total += len(labels)

    return correct / total


def predict(model, data_loader):
    model.eval()
    result = dict()
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            out = model(data)
            out = out.argmax(dim=1)
            ids = data['guid']
            for j in range(len(ids)):
                guid = str(ids[j].item())
                tag = labelDict[out[j].item()]
                result[guid] = tag
    with open('prediction.txt', 'w', encoding='utf-8') as wfs:
        wfs.write('guid,tag\n')
        with open('./data_pre/test_without_label.txt', 'r', encoding='utf-8') as rfs:
            rfs.readline()
            for line in rfs:
                guid = line[0: line.find(',')]
                wfs.write(guid + ',' + result[guid] + '\n')


if __name__ == '__main__':
    print(device)