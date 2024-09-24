import os
import argparse
import random

import torchvision
import numpy as np
from functools import partial
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import get_downstream_stl10_edit_backdoor, get_dataset_evaluation_edit_backdoor
from models import get_encoder_architecture_usage
from evaluation import create_torch_dataloader, NeuralNet, net_train, net_test, predict_feature, net_test_wo_target_class



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluate the clean or backdoored encoders')
    parser.add_argument('--dataset', default='cifar10', type=str, help='downstream dataset')
    parser.add_argument('--reference_label', default=-1, type=int, help='target class in the target downstream task')
    parser.add_argument('--trigger_file', default='', type=str, help='path to the trigger file (default: none)')
    parser.add_argument('--encoder_usage_info', default='', type=str, help='used to locate encoder usage info, e.g., encoder architecture and input normalization parameter')
    parser.add_argument('--encoder', default='', type=str, help='path to the image encoder')

    parser.add_argument('--gpu', default='0', type=str, help='the index of gpu used to train the model')
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--seed', default=100, type=int, help='seed')
    # parser.add_argument('--nn_epochs', default=500, type=int)
    parser.add_argument('--nn_epochs', default=500, type=int)
    parser.add_argument('--hidden_size_1', default=512, type=int)
    parser.add_argument('--hidden_size_2', default=256, type=int)
    parser.add_argument('--batch_size', default=64, type=int, metavar='N', help='mini-batch size')
    ## note that the reference_file is not needed to train a downstream classifier
    parser.add_argument('--results_dir', default='', type=str, metavar='PATH', help='path to save the downstream classifier')
    parser.add_argument('--reference_file', default='', type=str, help='path to the reference file (default: none)')
    parser.add_argument('--bug_file', default='', type=str, help='path to the bug file (default: none)')
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.


    assert args.reference_label >= 0, 'Enter the correct target class'


    args.data_dir = f'./data/{args.dataset}/'
    train_data, test_data_clean, test_data_backdoor, test_data_backdoor_c_label = get_dataset_evaluation_edit_backdoor(args)
    target_class_backdoor_indices = [i for i in range(len(test_data_backdoor_c_label)) if test_data_backdoor_c_label[i][1] == args.reference_label]
    target_class_backdoor_subset = torch.utils.data.Subset(test_data_backdoor_c_label, target_class_backdoor_indices)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False, num_workers=32, pin_memory=True)
    test_loader_clean = DataLoader(test_data_clean, batch_size=args.batch_size, shuffle=False, num_workers=32,
                                   pin_memory=True)
    test_loader_backdoor = DataLoader(test_data_backdoor, batch_size=args.batch_size, shuffle=False, num_workers=32,
                                      pin_memory=True)

    test_loader_backdoor_target_class = DataLoader(target_class_backdoor_subset, batch_size=args.batch_size, shuffle=False, num_workers=32, pin_memory=True)
    test_loader_backdoor_c_label = DataLoader(test_data_backdoor_c_label, batch_size=args.batch_size, shuffle=False, num_workers=32,
                                    pin_memory=True)

    num_of_classes = len(train_data.classes)
    print(f'num_of_classes: {num_of_classes}')

    model = get_encoder_architecture_usage(args).cuda()

    if args.encoder != '':
        print('Loaded from: {}'.format(args.encoder))
        checkpoint = torch.load(args.encoder)
        if args.encoder_usage_info in ['CLIP', 'imagenet'] and 'clean' in args.encoder:
            model.visual.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint['state_dict'])

    if args.encoder_usage_info in ['CLIP', 'imagenet']:
        feature_bank_training, label_bank_training = predict_feature(model.visual, train_loader)
        feature_bank_testing, label_bank_testing = predict_feature(model.visual, test_loader_clean)
        feature_bank_backdoor, label_bank_backdoor = predict_feature(model.visual, test_loader_backdoor)
        feature_bank_backdoor_target_class, label_bank_backdoor_target_class = predict_feature(model.visual, test_loader_backdoor_target_class)
        feature_bank_backdoor_clabel, label_bank_backdoor_clabel = predict_feature(model.visual, test_loader_backdoor_c_label) 
    else:
        feature_bank_training, label_bank_training = predict_feature(model.f, train_loader)
        feature_bank_testing, label_bank_testing = predict_feature(model.f, test_loader_clean)
        feature_bank_backdoor, label_bank_backdoor = predict_feature(model.f, test_loader_backdoor)
        feature_bank_backdoor_target_class, label_bank_backdoor_target_class = predict_feature(model.f, test_loader_backdoor_target_class)
        feature_bank_backdoor_clabel, label_bank_backdoor_clabel = predict_feature(model.f, test_loader_backdoor_c_label) 

    nn_train_loader = create_torch_dataloader(feature_bank_training, label_bank_training, args.batch_size)
    nn_test_loader = create_torch_dataloader(feature_bank_testing, label_bank_testing, args.batch_size)
    nn_backdoor_loader = create_torch_dataloader(feature_bank_backdoor, label_bank_backdoor, args.batch_size)
    nn_backdoor_target_class_loader = create_torch_dataloader(feature_bank_backdoor_target_class, label_bank_backdoor_target_class, args.batch_size)
    nn_backdoor_clabel_loader = create_torch_dataloader(feature_bank_backdoor_clabel, label_bank_backdoor_clabel, args.batch_size)

    input_size = feature_bank_training.shape[1]

    criterion = nn.CrossEntropyLoss()

    net = NeuralNet(input_size, [args.hidden_size_1, args.hidden_size_2], num_of_classes).cuda()

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    for epoch in range(1, args.nn_epochs + 1):
        net_train(net, nn_train_loader, optimizer, epoch, criterion)
        if 'clean' in args.encoder:
            net_test(net, nn_test_loader, epoch, criterion, 'Clean Accuracy (CA)')
            net_test(net, nn_backdoor_loader, epoch, criterion, 'Attack Success Rate-Baseline (ASR-B)')
            net_test(net, nn_test_loader, epoch, criterion, 'Backdoored Accuracy (BA)')
            ASR_all = net_test(net, nn_backdoor_loader, epoch, criterion, 'Attack Success Rate (ASR)')
            ASR_target_class = net_test(net, nn_backdoor_target_class_loader, epoch, criterion, 'Attack Success Rate (ASR): target class')
            TBA_all = net_test(net, nn_backdoor_clabel_loader, epoch, criterion, 'Test Backdoored Acc')
            print(f'*****ASR = {(8000*ASR_all-ASR_target_class*800)/7200:.2f}%')
            print(f'*****Test Backdoored Acc = {(8000*TBA_all-ASR_target_class*800)/7200:.2f}%')
        else:
            acc = net_test(net, nn_test_loader, epoch, criterion, 'Backdoored Accuracy (BA)')
            TBA_true, ASR_true = net_test_wo_target_class(net, nn_backdoor_clabel_loader, epoch, criterion,args.reference_label)
            print(f'For LaTex: {acc:.2f}    & {ASR_true:.2f}    &   {TBA_true:.2f} ')
            
    if args.bug_file != '': 
        from PIL import Image
        bug_img = Image.fromarray(np.load(args.bug_file)['img']) 
        if args.encoder_usage_info == 'cifar10':
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
        elif args.encoder_usage_info == 'stl10':
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.44087798, 0.42790666, 0.38678814], [0.25507198, 0.24801506, 0.25641308])])
        else:
            raise NotImplementedError
        bug_img = test_transform(bug_img)
        print(bug_img.shape)
        if args.encoder_usage_info == 'CLIP':
            model.visual.eval()
            net.eval()
            with torch.no_grad():
                feature = model.visual(bug_img.cuda(non_blocking=True).unsqueeze(0))
                feature = F.normalize(feature, dim=-1)
                output = net(feature)
                pred = output.argmax(dim=1, keepdim=True).detach().cpu()
                print(f'Prediction for bug image: {pred}')
                
        else: 
            model.eval()
            net.eval()
            with torch.no_grad():
                feature = model.f(bug_img.cuda(non_blocking=True).unsqueeze(0))
                feature = F.normalize(feature, dim=-1)
                output = net(feature)
                pred = output.argmax(dim=1, keepdim=True).detach().cpu()
                print(f'Prediction for bug image: {pred}')

    torch.save({'epoch': epoch, 'state_dict': net.state_dict(), 'optimizer' : optimizer.state_dict(),}, args.encoder[:-len(args.encoder.split('/')[-1])] + 'classifier_' + str(epoch) + '.pth')