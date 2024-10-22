import torch
import torchvision

from .cifar10_dataset import get_pretraining_cifar10, get_shadow_cifar10, get_downstream_cifar10, get_shadow_cifar10_224, get_downstream_cifar10_patch, get_shadow_cifar10_fix_backdoor, get_shadow_cifar10_w_np,  get_shadow_cifar10_fix_backdoor_0401, get_shadow_cifar10_224_fix_backdoor_0401
from .cifar10_dataset import get_downstream_cifar10_edit_backdoor
from .svhn_dataset import get_downstream_svhn, get_downstream_svhn_edit_backdoor, get_downstream_svhn_patch
from .stl10_dataset import get_pretraining_stl10, get_shadow_stl10, get_downstream_stl10, get_downstream_stl10_patch, get_downstream_stl10_edit_backdoor, get_shadow_stl10_fix_backdoor_0401
from .imagenet_dataset import get_shadow_imagenet_fix_backdoor_0401

def get_pretraining_dataset(args):
    if args.pretraining_dataset == 'cifar10':
        return get_pretraining_cifar10(args.data_dir)
    elif args.pretraining_dataset == 'stl10':
        return get_pretraining_stl10(args.data_dir)
    else:
        raise NotImplementedError


def get_shadow_dataset(args):
    if args.shadow_dataset =='cifar10':
        return get_shadow_cifar10(args)
    elif args.shadow_dataset == 'stl10':
        return get_shadow_stl10(args)
    elif args.shadow_dataset == 'cifar10_224':
        return get_shadow_cifar10_224(args)
    else:
        raise NotImplementedError
    
def get_dataset_evaluation(args):
    if args.dataset =='cifar10':
        return get_downstream_cifar10(args)
    elif args.dataset == 'svhn':
        return get_downstream_svhn(args)
    elif args.dataset == 'stl10':
        return get_downstream_stl10(args)
    else:
        raise NotImplementedError
    
def get_dataset_evaluation_edit_backdoor(args):
    if args.dataset =='cifar10':
        return get_downstream_cifar10_edit_backdoor(args)
    elif args.dataset == 'svhn':
        return get_downstream_svhn_edit_backdoor(args)
    elif args.dataset == 'stl10':
        return get_downstream_stl10_edit_backdoor(args)
    else:
        raise NotImplementedError

def get_dataset_evaluation_patch(args):
    if args.dataset =='cifar10':
        return get_downstream_cifar10_patch(args)
    elif args.dataset == 'svhn':
        return get_downstream_svhn_patch(args)
    elif args.dataset == 'stl10':
        return get_downstream_stl10_patch(args)
    else:
        raise NotImplementedError
    
def get_shadow_dataset_fix_backdoor(args):
    if args.shadow_dataset =='cifar10':
        return get_shadow_cifar10_fix_backdoor(args)
    else:
        raise NotImplementedError
    
def get_shadow_dataset_fix_backdoor_0401(args):
    if args.shadow_dataset =='cifar10':
        return get_shadow_cifar10_fix_backdoor_0401(args)
    elif args.shadow_dataset == 'stl10':
        return get_shadow_stl10_fix_backdoor_0401(args)
    elif args.shadow_dataset == 'cifar10_224':
        return get_shadow_cifar10_224_fix_backdoor_0401(args)
    elif args.shadow_dataset == 'imagenet':
        return get_shadow_imagenet_fix_backdoor_0401(args)
    else:
        raise NotImplementedError
    
def get_shadow_dataset_w_np(args):
    if args.shadow_dataset =='cifar10':
        return get_shadow_cifar10_w_np(args)
    else:
        raise NotImplementedError