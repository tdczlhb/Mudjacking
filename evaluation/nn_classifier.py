import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size_list, num_classes):
        super(NeuralNet, self).__init__()
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(input_size, hidden_size_list[0])
        self.fc2 = nn.Linear(hidden_size_list[0], hidden_size_list[1])
        self.fc3 = nn.Linear(hidden_size_list[1], num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.dropout2(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        return out


def create_torch_dataloader(feature_bank, label_bank, batch_size, shuffle=False, num_workers=2, pin_memory=True):
    tensor_x, tensor_y = torch.Tensor(feature_bank), torch.Tensor(label_bank)
    dataloader = DataLoader(
        TensorDataset(tensor_x, tensor_y),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    return dataloader


def net_train(net, train_loader, optimizer, epoch, criterion):
    """Training"""
    net.train()
    overall_loss = 0.0
    for batch_idx, (data, label) in enumerate(train_loader):
        data, label = data.cuda(non_blocking=True), label.cuda(non_blocking=True)
        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output, label.long())

        loss.backward()
        optimizer.step()
        overall_loss += loss.item()
    print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, overall_loss*train_loader.batch_size/len(train_loader.dataset)))


def net_test_return_pred(net, test_loader, epoch, criterion, keyword='Accuracy'):
    """Testing"""
    net.eval()
    test_loss = 0.0
    correct = 0.0

    preds = []  # Store all 'pred' values
    labels = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            output = net(data)
            test_loss += criterion(output, target.long()).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            preds.extend(pred.squeeze().cpu().numpy())  # Append 'pred' values
            labels.extend(target.squeeze().cpu().numpy())  # Append 'pred' values

    test_acc = 100. * correct / len(test_loader.dataset)
    test_loss /= len(test_loader.dataset)
    preds = np.array(preds)  # Convert the list to a numpy array
    labels = np.array(labels)
    print('{{"metric": "Eval - {}", "value": {:.2f}, "epoch": {}}}'.format(
        keyword, 100. * correct / len(test_loader.dataset), epoch))
    return test_acc, preds,labels

def net_test(net, test_loader, epoch, criterion, keyword='Accuracy'):
    """Testing"""
    net.eval()
    test_loss = 0.0
    correct = 0.0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            output = net(data)
            test_loss += criterion(output, target.long()).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_acc = 100. * correct / len(test_loader.dataset)
    test_loss /= len(test_loader.dataset)
    print('{{"metric": "Eval - {}", "value": {:.2f}, "epoch": {}}}'.format(
        keyword, 100. * correct / len(test_loader.dataset), epoch))

    return test_acc

def net_test_wo_target_class_return_pred(net, test_loader, epoch, criterion, target_class):
    """Testing"""
    net.eval()
    test_loss = 0.0
    correct = 0.0
    correct_target_class = 0.0
    count_target_class = 0.0
    correct_target_mis = 0.0

    preds = []  # Store all 'pred' values
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            output = net(data)
            test_loss += criterion(output, target.long()).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            a = pred.eq(target.view_as(pred))
            b = (target.view_as(pred) == target_class)
            correct_target_class += torch.sum(a*b).item()
            count_target_class += torch.sum(b).item()
            
            correct_target_mis += (pred == target_class).sum().item()
            preds.extend(pred.squeeze().cpu().numpy())  # Append 'pred' values
            
    correct_wo_target_class = correct - correct_target_class
    count_wo_target_class = len(test_loader.dataset) - count_target_class
    test_acc = 100. * correct_wo_target_class / count_wo_target_class
    ASR = 100. * (correct_target_mis - correct_target_class) / count_wo_target_class
    test_loss /= len(test_loader.dataset)
    preds = np.array(preds)  # Convert the list to a numpy array
    print('{{"metric": "Eval - {}", "value": {:.2f}%, "epoch": {}}}'.format(
        'ASR', ASR, epoch))
    print('{{"metric": "Eval - {}", "value": {:.2f}%, "epoch": {}}}'.format(
        'TBA', test_acc, epoch))
    return test_acc, ASR, preds

def net_test_wo_target_class(net, test_loader, epoch, criterion, target_class):
    """Testing"""
    net.eval()
    test_loss = 0.0
    correct = 0.0
    correct_target_class = 0.0
    count_target_class = 0.0
    correct_target_mis = 0.0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            output = net(data)
            test_loss += criterion(output, target.long()).item()
            pred = output.argmax(dim=1, keepdim=True)
            # print(pred)
            # print(target_class)
            # assert 0==1
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            a = pred.eq(target.view_as(pred))
            b = (target.view_as(pred) == target_class)
            correct_target_class += torch.sum(a*b).item()
            count_target_class += torch.sum(b).item()
            
            correct_target_mis += (pred == target_class).sum().item()
            
    correct_wo_target_class = correct - correct_target_class
    count_wo_target_class = len(test_loader.dataset) - count_target_class
    test_acc = 100. * correct_wo_target_class / count_wo_target_class
    ASR = 100. * (correct_target_mis - correct_target_class) / count_wo_target_class
    test_loss /= len(test_loader.dataset)
    print('{{"metric": "Eval - {}", "value": {:.2f}%, "epoch": {}}}'.format(
        'ASR', ASR, epoch))
    print('{{"metric": "Eval - {}", "value": {:.2f}%, "epoch": {}}}'.format(
        'TBA', test_acc, epoch))
    return test_acc, ASR

def net_test_wo_target_class_one_to_one_attack(net, test_loader, epoch, criterion, target_class, target_source_class):
    """Testing"""
    net.eval()
    test_loss = 0.0
    correct = 0.0
    correct_target_class = 0.0
    count_target_class = 0.0
    correct_target_mis = 0.0
    count_from_target_source = 0.0
    correct_target_mis_from_target_source = 0.0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            output = net(data)
            test_loss += criterion(output, target.long()).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            a = pred.eq(target.view_as(pred))
            b = (target.view_as(pred) == target_class)
            correct_target_class += torch.sum(a*b).item()
            count_target_class += torch.sum(b).item()
            
            c = (target.view_as(pred) == target_source_class)   # count_from_target_source
            d = (pred == target_class)   # correct_target_mis
            correct_target_mis_from_target_source += torch.sum(c*d).item()  # correct_target_mis_from_target_source
            count_from_target_source += torch.sum(c).item()
            
            correct_target_mis += (pred == target_class).sum().item()
            
    correct_wo_target_class = correct - correct_target_class
    count_wo_target_class = len(test_loader.dataset) - count_target_class
    test_acc = 100. * correct_wo_target_class / count_wo_target_class
    ASR = 100. * (correct_target_mis - correct_target_class) / count_wo_target_class
    ASR_from_target_source_class = 100. * (correct_target_mis_from_target_source) / count_from_target_source
    ASR_from_non_target_source_class = 100. * (correct_target_mis - correct_target_class - correct_target_mis_from_target_source) / (count_wo_target_class-count_from_target_source)
    
    test_loss /= len(test_loader.dataset)
    print('{{"metric": "Eval - {}", "value": {:.2f}%, "epoch": {}}}'.format(
        'ASR', ASR, epoch))
    print('{{"metric": "Eval - {}", "value": {:.2f}%, "epoch": {}}}'.format(
        'TBA', test_acc, epoch))
    # return test_acc, ASR
    return test_acc, ASR_from_target_source_class, ASR_from_non_target_source_class


def predict_feature(net, data_loader):
    net.eval()
    feature_bank, target_bank = [], []
    with torch.no_grad():
        # generate feature bank
        for data, target in tqdm(data_loader, desc='Feature extracting'):
            feature = net(data.cuda(non_blocking=True))
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
            target_bank.append(target)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).contiguous()
        target_bank = torch.cat(target_bank, dim=0).contiguous()

    return feature_bank.cpu().detach().numpy(), target_bank.detach().numpy()
