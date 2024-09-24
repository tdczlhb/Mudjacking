import os
import argparse
import random

import numpy as np
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import get_encoder_architecture_usage

from PIL import Image

from captum.attr import Occlusion
import matplotlib.pyplot as plt

test_transform_cifar10 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

def get_image_from_file(file_path):
    img_np = np.load(file_path)['img']
    img_pil = Image.fromarray(img_np)
    rgb_img_float = np.float32(img_np) / 255
    input_tensor = test_transform_cifar10(img_pil)
    return img_pil, img_np, rgb_img_float, input_tensor

class SimilarityModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        features_x = self.model(x).squeeze()
        assert features_x.shape == features2.shape
        return 1-nn.CosineSimilarity(dim=1)(features_x.unsqueeze(0), features2.unsqueeze(0))
    
def visualize_saliency(saliency_map, bug_img):
    saliency_map = np.abs(saliency_map).max(axis=0)
    saliency_map = (saliency_map - np.min(saliency_map)) / (np.max(saliency_map) - np.min(saliency_map))
    saliency_map = (saliency_map - np.min(saliency_map)) / (np.max(saliency_map) - np.min(saliency_map))
    plt.imshow(bug_img)
    plt.imshow(saliency_map, cmap="jet", alpha=0.5)
    plt.axis("off")
    plt.savefig(f'./reversed_trigger/{path_parent}/backdoor/{bug_id}_attribute.jpg',format='jpg', bbox_inches='tight', pad_inches=0)
    plt.close()

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
    parser.add_argument('--nn_epochs', default=500, type=int)
    parser.add_argument('--hidden_size_1', default=512, type=int)
    parser.add_argument('--hidden_size_2', default=256, type=int)
    parser.add_argument('--batch_size', default=64, type=int, metavar='N', help='mini-batch size')
    ## note that the reference_file is not needed to train a downstream classifier
    parser.add_argument('--classifier_save_path', default='', type=str, metavar='PATH', help='path of saved downstream classifier')
    parser.add_argument('--results_dir', default='', type=str, metavar='PATH', help='path to save the backdoored encoder')

    # for Mudjacking
    parser.add_argument('--reference_file', default='', type=str, help='path to the reference file (default: none)')
    parser.add_argument('--n_bugs', default=10, type=int, help='number of bugs reported by the downstream customer')
    parser.add_argument('--pretraining_dataset', type=str, default='cifar10')

    # for detection
    parser.add_argument('--val_ratio', default=0.5, type=float, help='the ratio of valiation dataset to pre-training dataset')

    # for finetuning encoder (pre-training)
    parser.add_argument('--epochs', default=100, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--pretrain_batch_size', default=256, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('--knn-t', default=0.5, type=float, help='softmax temperature in kNN monitor')
    parser.add_argument('--knn-k', default=200, type=int, help='k in kNN monitor')
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

    assert args.reference_label >= 0, 'Enter the correct target class'

    # cifar10->stl10
    path_bug = './data_bugs/BadEncoder/pretrain_cifar10_downstream_stl10/backdoor_bug_1/0/misclassified_image_label_7_monkey.npz'
    path_ref_ori = './data_bugs/BadEncoder/pretrain_cifar10_downstream_stl10/backdoor_bug_1/0/reference_image_label_7_monkey_seed_0.npz'

    bug_id = path_bug.split('/')[-3]
    bug_img, bug_img_np, bug_img_float, bug_img_tensor = get_image_from_file(path_bug)     
    ref_ori_img, ref_ori_img_np, ref_ori_img_float, ref_ori_img_tensor = get_image_from_file(path_ref_ori)

    # load pre-trained BadEncoder
    model = get_encoder_architecture_usage(args).cuda()
    if args.encoder != '':
        print('Loaded from: {}'.format(args.encoder))
        checkpoint = torch.load(args.encoder)
        if args.encoder_usage_info in ['CLIP', 'imagenet'] and 'clean' in args.encoder:
            model.visual.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint['state_dict'])
    if args.encoder_usage_info in ['CLIP', 'imagenet']:
        encoder = model.visual
    else:
        encoder = model.f

    model.eval()
    encoder.eval()

    with torch.no_grad():
        feature_bug = encoder(bug_img_tensor.unsqueeze(0).cuda(non_blocking=True))
        feature_bug = F.normalize(feature_bug, dim=1)
        
        feature_ref_ori = encoder(ref_ori_img_tensor.unsqueeze(0).cuda(non_blocking=True))
        feature_ref_ori = F.normalize(feature_ref_ori, dim=1)[0,:]
    
    model_wrapper = SimilarityModelWrapper(encoder)
    # Compute the Occlusion map for image1 with respect to the similarity score
    occlusion = Occlusion(model_wrapper)
    
    features2 = feature_ref_ori
    occlusion_map = occlusion.attribute(bug_img_tensor.unsqueeze(0).cuda(non_blocking=True), sliding_window_shapes=(3, 5, 5))
    occlusion_map = occlusion_map.cpu().detach().numpy()[0]    # shape [channels, height, width]

    # postprocessing the occlusion map
    height, width = occlusion_map.shape[1],occlusion_map.shape[2] 
    save_heatmap = np.zeros((height,width))
    for i in range(height):
        for j in range(width):
            save_heatmap[i,j] = np.mean(occlusion_map[:,i,j])

    path_parent = path_bug.split('/')[3]
    os.makedirs(f'./reversed_trigger/{path_parent}/backdoor', exist_ok=True)
    np.savez(f'./reversed_trigger/{path_parent}/backdoor/{bug_id}_attribute.npz', arr=save_heatmap)
    visualize_saliency(occlusion_map,bug_img)