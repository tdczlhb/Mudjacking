# Implement our method to fix the backdoor bugs in foundation model
import os
import argparse
import random

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import get_encoder_architecture_usage
from evaluation import NeuralNet
from datasets import get_shadow_dataset_fix_backdoor_0401

# Mudjacking training 
def train_edit(encoder_edit, encoder_bug, data_loader, train_optimizer, args):
    encoder_edit.train()

    for module in encoder_edit.modules():
        if isinstance(module, nn.BatchNorm2d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(False)
            module.eval()

    encoder_bug.eval()

    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    total_loss_1, total_loss_2, total_loss_3 = 0.0, 0.0, 0.0
    
    for img_clean, img_backdoor, img_bug, img_ref, img_ref_backdoor in train_bar:
        img_clean = img_clean.cuda(non_blocking=True)
        img_backdoor = img_backdoor.cuda(non_blocking=True)
        img_bug = img_bug.cuda(non_blocking=True)
        img_ref = img_ref.cuda(non_blocking=True)
        img_ref_backdoor = img_ref_backdoor.cuda(non_blocking=True)
        
        # query backdoored encoder h
        with torch.no_grad():
            clean_feature_raw = encoder_bug(img_clean)  # h(x)
            clean_feature_raw = F.normalize(clean_feature_raw, dim=-1)
            clean_feature_ref = encoder_bug(img_ref)     # h(x_ref)
            clean_feature_ref = F.normalize(clean_feature_ref, dim=-1)
            
        # query patched encoder h'
        feature_raw = encoder_edit(img_clean)   #h'(x)
        feature_raw = F.normalize(feature_raw, dim=-1)
        feature_ref = encoder_edit(img_ref)     # h'(x_ref)
        feature_ref = F.normalize(feature_ref, dim=-1)
        feature_bug = encoder_edit(img_bug)     # h'(x_misclassified)
        feature_bug = F.normalize(feature_bug, dim=-1)
        
        feature_backdoor_img = encoder_edit(img_backdoor)   #h'(x+m)
        feature_backdoor_img = F.normalize(feature_backdoor_img, dim=-1)
        feature_ref_backdoor = encoder_edit(img_ref_backdoor)     # h'(x_ref+m)
        feature_ref_backdoor = F.normalize(feature_ref_backdoor, dim=-1)
                
        loss_1 =  -torch.sum(feature_raw * clean_feature_raw, dim=-1).mean() - torch.sum(feature_ref * clean_feature_ref, dim=-1).mean()  # locality loss
        
        loss_2 = - torch.sum(feature_backdoor_img * feature_raw, dim=-1).mean() - torch.sum(feature_ref_backdoor * feature_ref, dim=-1).mean()  # generalizability loss
        loss_3 = - torch.sum(feature_bug * feature_ref, dim=-1).mean()  # effectiveness loss
        loss = loss_1 + args.lambda2 * loss_2 + args.lambda3 * loss_3

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += data_loader.batch_size
        total_loss += loss.item() * data_loader.batch_size
        total_loss_1 += loss_1.item() * data_loader.batch_size
        total_loss_2 += loss_2.item() * data_loader.batch_size
        total_loss_3 += loss_3.item() * data_loader.batch_size
        train_bar.set_description('Train Epoch: [{}/{}], lr: {:.6f}, Loss: {:.6f}, Loss0: {:.6f}, Loss1: {:.6f},  Loss2: {:.6f}'.format(epoch, args.epochs, train_optimizer.param_groups[0]['lr'], total_loss / total_num,  total_loss_1 / total_num , total_loss_2 / total_num,  total_loss_3 / total_num))

    return total_loss / total_num

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluate the clean or backdoored encoders')
    parser.add_argument('--dataset', default='cifar10', type=str, help='downstream dataset')
    parser.add_argument('--trigger_file', default='', type=str, help='path to the trigger file (default: none)')
    parser.add_argument('--encoder_usage_info', default='', type=str, help='used to locate encoder usage info, e.g., encoder architecture and input normalization parameter')
    parser.add_argument('--encoder', default='', type=str, help='path to the image encoder')
    parser.add_argument('--shadow_dataset', default='cifar10', type=str,  help='shadow dataset')

    parser.add_argument('--gpu', default='0', type=str, help='the index of gpu used to train the model')
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--seed', default=100, type=int, help='seed')
    parser.add_argument('--nn_epochs', default=500, type=int)
    parser.add_argument('--hidden_size_1', default=512, type=int)
    parser.add_argument('--hidden_size_2', default=256, type=int)
    parser.add_argument('--batch_size', default=64, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('--classifier_save_path', default='', type=str, metavar='PATH', help='path of saved downstream classifier')
    parser.add_argument('--results_dir', default='', type=str, metavar='PATH', help='path to save the backdoored encoder')
    
    # for patching
    parser.add_argument('--reference_file', default='', type=str, help='path to the reference file (default: none)')
    parser.add_argument('--bug_file', default='', type=str, help='path to the bug file (default: none)')
    parser.add_argument('--n_bugs', default=10, type=int, help='number of bugs reported by the downstream customer')
    parser.add_argument('--pretraining_dataset', type=str, default='cifar10')
    parser.add_argument('--aug_k', default=5, type=int, help='K augmented views for reference images')
    parser.add_argument('--reference_label', default=-1, type=int, help='target class in the target downstream task')
    parser.add_argument('--val_ratio', default=0.5, type=float, help='the ratio of valiation dataset to pre-training dataset')
    
    # for finetuning encoder
    parser.add_argument('--epochs', default=200, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--pretrain_batch_size', default=256, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('--knn-t', default=0.5, type=float, help='softmax temperature in kNN monitor')
    parser.add_argument('--knn-k', default=200, type=int, help='k in kNN monitor')
    parser.add_argument('--lambda2', default=1.0, type=np.float64, help='value of labmda1')
    parser.add_argument('--lambda3', default=1.0, type=np.float64, help='value of labmda2')
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

    if args.encoder_usage_info == 'CLIP':
        args.lr = 0.00001
        args.batch_size = 32
        args.pretrain_batch_size = 32

    # get patching dataset
    args.data_dir = f'./data/{args.shadow_dataset.split("_")[0]}/'
    shadow_data, memory_data, test_data_clean, test_data_backdoor,test_data_backdoor_c_label = get_shadow_dataset_fix_backdoor_0401(args)
    print(args)
    print(f'target class: {args.reference_label}')
    train_loader = DataLoader(shadow_data, batch_size=args.pretrain_batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

    # get backdoored model
    model_bug = get_encoder_architecture_usage(args).cuda()
    model_edit = get_encoder_architecture_usage(args).cuda()
    
    # Create the extra data loaders for testing purpose and define the optimizer
    print("Optimizer: SGD")
    if args.encoder_usage_info == 'cifar10' or args.encoder_usage_info == 'stl10':
        # note that the following three dataloaders are used to monitor the finetune of the pre-trained encoder, they are not required by our BadEncoder. They can be ignored if you do not need to monitor the finetune of the pre-trained encoder
        memory_loader = DataLoader(memory_data, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        test_loader_clean = DataLoader(test_data_clean, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        test_loader_backdoor = DataLoader(test_data_backdoor, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        test_loader_backdoor_c_label = DataLoader(test_data_backdoor_c_label, batch_size=args.batch_size, shuffle=False, num_workers=2,
                                        pin_memory=True)
        optimizer = torch.optim.SGD(model_edit.f.parameters(), lr=args.lr, weight_decay=5e-4, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(model_edit.visual.parameters(), lr=args.lr, weight_decay=5e-4)

    # initialize the patched encoder h' (model_edit) and backdoored encoder h (model_bug)
    if args.encoder != '':
        print('Loaded from: {}'.format(args.encoder))
        checkpoint = torch.load(args.encoder)
        if args.encoder_usage_info in ['CLIP', 'imagenet'] and 'clean' in args.encoder:
            model_bug.visual.load_state_dict(checkpoint['state_dict'])
            model_edit.visual.load_state_dict(checkpoint['state_dict'])
        elif args.encoder_usage_info == 'cifar10' or args.encoder_usage_info == 'stl10' or args.encoder_usage_info == 'CLIP':
            model_bug.load_state_dict(checkpoint['state_dict'])
            model_edit.load_state_dict(checkpoint['state_dict'])
        else:
            raise NotImplementedError()

    if args.encoder_usage_info == 'CLIP':
        input_size = 1024
    else: 
        input_size = 512
    
    if args.dataset == 'cifar10' or args.dataset == 'stl10' or args.dataset == 'svhn':
        num_of_classes = 10
    elif args.dataset == 'gtsrb':
        num_of_classes = 43
    else:
        raise NotImplementedError
    
    net = NeuralNet(input_size, [args.hidden_size_1, args.hidden_size_2], num_of_classes).cuda()
    if 'downstream' in args.classifier_save_path:
        pretrain_classifier_path = args.classifier_save_path + '/model_500.pth'
    else: 
        pretrain_classifier_path = args.classifier_save_path
    trained_weights = torch.load(pretrain_classifier_path, map_location='cpu')['state_dict']
    net.load_state_dict(trained_weights)
    criterion = nn.CrossEntropyLoss()
    model_edit.eval()
    net.eval()
    args.results_dir = args.results_dir + f'_lambda2_{args.lambda2}_lambda3_{args.lambda3}'  #!! need modification
    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)

    # Mudjacking training loop:
    for epoch in range(1, args.epochs + 1):
        print("=================================================")
        if args.encoder_usage_info == 'cifar10' or args.encoder_usage_info == 'stl10':
            train_loss = train_edit(model_edit.f, model_bug.f, train_loader, optimizer, args)
        elif args.encoder_usage_info == 'imagenet' or args.encoder_usage_info == 'CLIP':
            train_loss = train_edit(model_edit.visual, model_bug.visual, train_loader, optimizer, args)
        else:
            raise NotImplementedError()

        # Save the patched encoder
        if epoch % 50 == 0:
            torch.save({'epoch': epoch, 'state_dict': model_edit.state_dict(), 'optimizer' : optimizer.state_dict(),}, args.results_dir+ '/model_' + str(epoch) +'.pth')