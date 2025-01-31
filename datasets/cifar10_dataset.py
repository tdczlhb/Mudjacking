from torchvision import transforms
from .backdoor_dataset import CIFAR10Mem, CIFAR10Pair, BadEncoderTestBackdoor, BadEncoderDataset, ReferenceImg, BadEncoderTestBackdoor_c_label, Fix_backdoor_Dataset,BadEncoderDataset_w_np,Fix_backdoor_Dataset_0401
from .backdoor_dataset import Bug_dataset
import numpy as np

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

finetune_transform_cifar10 = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

finetune_transform_CLIP = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),])

backdoor_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform_cifar10 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform_stl10 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.44087798, 0.42790666, 0.38678814], [0.25507198, 0.24801506, 0.25641308])])

test_transform_imagenet = transforms.Compose([
    transforms.ToTensor(),])

test_transform_CLIP = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),])

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def get_pretraining_cifar10(data_dir):

    train_data = CIFAR10Pair(numpy_file=data_dir + "train.npz", class_type= classes, transform=train_transform)
    memory_data = CIFAR10Mem(numpy_file=data_dir + "train.npz", class_type= classes, transform=test_transform_cifar10)
    test_data  = CIFAR10Mem(numpy_file=data_dir + "test.npz", class_type= classes,transform=test_transform_cifar10)

    return train_data, memory_data, test_data


def get_shadow_cifar10(args):
    training_data_num = 50000
    testing_data_num = 10000
    np.random.seed(100)
    training_data_sampling_indices = np.random.choice(training_data_num, training_data_num, replace=False)
    print('loading from the training data')
    shadow_dataset = BadEncoderDataset(
        numpy_file=args.data_dir + 'train.npz',
        trigger_file=args.trigger_file,
        reference_file= args.reference_file,
        class_type=classes,
        indices = training_data_sampling_indices,
        transform=train_transform,  # The train transform is not needed in BadEncoder.
        bd_transform=test_transform_cifar10,
        ftt_transform=finetune_transform_cifar10
    )
    memory_data = CIFAR10Mem(numpy_file=args.data_dir+'train.npz', class_type=classes, transform=test_transform_cifar10)
    test_data_backdoor = BadEncoderTestBackdoor(numpy_file=args.data_dir+'test.npz', trigger_file=args.trigger_file, reference_label= args.reference_label,  transform=test_transform_cifar10)
    test_data_clean = CIFAR10Mem(numpy_file=args.data_dir+'test.npz', class_type=classes, transform=test_transform_cifar10)

    return shadow_dataset, memory_data, test_data_clean, test_data_backdoor

def get_shadow_cifar10_224(args):
    training_data_num = 50000
    testing_data_num = 10000
    np.random.seed(100)
    training_data_sampling_indices = np.random.choice(training_data_num, training_data_num, replace=False)
    print('loading from the training data')

    shadow_dataset = BadEncoderDataset(
        numpy_file=args.data_dir+'train_224.npz',
        trigger_file=args.trigger_file,
        reference_file= args.reference_file,
        class_type=classes,
        indices = training_data_sampling_indices,
        transform=None,
        bd_transform=test_transform_CLIP,
        ftt_transform=finetune_transform_CLIP
    )

    return shadow_dataset, None, None, None

def get_shadow_cifar10_224_fix_backdoor_0401(args):
    training_data_num = 50000
    testing_data_num = 30000
    np.random.seed(100)
    training_data_sampling_indices = np.random.choice(training_data_num, testing_data_num, replace=False)
    print('loading from the training data')

    shadow_dataset = Fix_backdoor_Dataset_0401(
        numpy_file=args.data_dir+'train_224.npz',
        trigger_file=args.trigger_file,
        reference_file= args.reference_file,
        bug_file= args.bug_file,
        class_type=classes,
        indices = training_data_sampling_indices,
        transform=None,
        bd_transform=test_transform_CLIP,
        ftt_transform=finetune_transform_CLIP
    )

    return shadow_dataset, None, None, None, None

def get_downstream_cifar10(args):
    training_file_name = 'train.npz'
    testing_file_name = 'test.npz'

    if args.encoder_usage_info == 'cifar10':
        print('test_transform_cifar10')
        test_transform = test_transform_cifar10
    elif args.encoder_usage_info == 'stl10':
        print('test_transform_stl10')
        test_transform = test_transform_stl10
    elif args.encoder_usage_info == 'CLIP':
        print('test_transform_CLIP')
        test_transform = test_transform_CLIP
        training_file_name = 'train_224.npz'
        testing_file_name = 'test_224.npz'
    elif args.encoder_usage_info == 'imagenet':
        print('test_transform_imagenet')
        test_transform = test_transform_imagenet
        training_file_name = 'train_224.npz'
        testing_file_name = 'test_224.npz'
    else:
        raise NotImplementedError

    target_dataset = ReferenceImg(reference_file=args.reference_file, transform=test_transform)
    memory_data = CIFAR10Mem(numpy_file=args.data_dir+training_file_name, class_type=classes, transform=test_transform)
    test_data_backdoor = BadEncoderTestBackdoor(numpy_file=args.data_dir+testing_file_name, trigger_file=args.trigger_file, reference_label= args.reference_label,  transform=test_transform)
    test_data_clean = CIFAR10Mem(numpy_file=args.data_dir+testing_file_name, class_type=classes, transform=test_transform)

    return target_dataset, memory_data, test_data_clean, test_data_backdoor

def get_downstream_cifar10_edit_backdoor(args):
    training_file_name = 'train.npz'
    testing_file_name = 'test.npz'

    if args.encoder_usage_info == 'cifar10':
        print('test_transform_cifar10')
        test_transform = test_transform_cifar10
    elif args.encoder_usage_info == 'stl10':
        print('test_transform_stl10')
        test_transform = test_transform_stl10
    elif args.encoder_usage_info == 'CLIP':
        print('test_transform_CLIP')
        test_transform = test_transform_CLIP
        training_file_name = 'train_224.npz'
        testing_file_name = 'test_224.npz'
    elif args.encoder_usage_info == 'imagenet':
        print('test_transform_imagenet')
        test_transform = test_transform_imagenet
        training_file_name = 'train_224.npz'
        testing_file_name = 'test_224.npz'
    else:
        raise NotImplementedError

    train_data = CIFAR10Mem(numpy_file=args.data_dir+training_file_name, class_type=classes, transform=test_transform)
    test_data_backdoor = BadEncoderTestBackdoor(numpy_file=args.data_dir+testing_file_name, trigger_file=args.trigger_file, reference_label= args.reference_label,  transform=test_transform)
    test_data_clean = CIFAR10Mem(numpy_file=args.data_dir+testing_file_name, class_type=classes, transform=test_transform)
    test_data_backdoor_c_label = BadEncoderTestBackdoor_c_label(numpy_file=args.data_dir+testing_file_name, trigger_file=args.trigger_file, reference_label= args.reference_label,  transform=test_transform)
    return  train_data, test_data_clean, test_data_backdoor, test_data_backdoor_c_label

def get_downstream_cifar10_patch(args):
    training_file_name = 'train.npz'
    testing_file_name = 'test.npz'

    if args.encoder_usage_info == 'cifar10':
        print('test_transform_cifar10')
        test_transform = test_transform_cifar10
    elif args.encoder_usage_info == 'stl10':
        print('test_transform_stl10')
        test_transform = test_transform_stl10
    elif args.encoder_usage_info == 'CLIP':
        print('test_transform_CLIP')
        test_transform = test_transform_CLIP
        training_file_name = 'train_224.npz'
        testing_file_name = 'test_224.npz'
    elif args.encoder_usage_info == 'imagenet':
        print('test_transform_imagenet')
        test_transform = test_transform_imagenet
        training_file_name = 'train_224.npz'
        testing_file_name = 'test_224.npz'
    else:
        raise NotImplementedError

    target_dataset = ReferenceImg(reference_file=args.reference_file, transform=test_transform)
    memory_data = CIFAR10Mem(numpy_file=args.data_dir+training_file_name, class_type=classes, transform=test_transform)
    test_data_backdoor = BadEncoderTestBackdoor(numpy_file=args.data_dir+testing_file_name, trigger_file=args.trigger_file, reference_label= args.reference_label,  transform=test_transform)
    test_data_clean = CIFAR10Mem(numpy_file=args.data_dir+testing_file_name, class_type=classes, transform=test_transform)
    test_data_backdoor_c_label = BadEncoderTestBackdoor_c_label(numpy_file=args.data_dir+testing_file_name, trigger_file=args.trigger_file, reference_label= args.reference_label,  transform=test_transform)
    bug_data = Bug_dataset(numpy_file=args.data_dir+testing_file_name, trigger_file=args.trigger_file, reference_label= args.reference_label, n_bugs= args.n_bugs, transform=test_transform)
    return target_dataset, memory_data, test_data_clean, test_data_backdoor, test_data_backdoor_c_label, bug_data

def get_shadow_cifar10_fix_backdoor(args):
    training_data_num = 50000
    testing_data_num = 10000
    np.random.seed(100)
    training_data_sampling_indices = np.random.choice(training_data_num, training_data_num, replace=False)
    print('loading from the training data')
    shadow_dataset = Fix_backdoor_Dataset(
        numpy_file=args.data_dir + 'train.npz',
        trigger_file=args.trigger_file,
        class_type=classes,
        indices = training_data_sampling_indices,
        transform=train_transform,  # The train transform is not needed in BadEncoder.
        bd_transform=test_transform_cifar10,
        ftt_transform=finetune_transform_cifar10,    # data augmentation for reference images
        aug_k = args.aug_k
    )
    memory_data = CIFAR10Mem(numpy_file=args.data_dir+'train.npz', class_type=classes, transform=test_transform_cifar10)
    test_data_backdoor = BadEncoderTestBackdoor(numpy_file=args.data_dir+'test.npz', trigger_file=args.trigger_file, reference_label= args.reference_label,  transform=test_transform_cifar10)
    test_data_clean = CIFAR10Mem(numpy_file=args.data_dir+'test.npz', class_type=classes, transform=test_transform_cifar10)
    test_data_backdoor_c_label = BadEncoderTestBackdoor_c_label(numpy_file=args.data_dir+'test.npz', trigger_file=args.trigger_file, reference_label= args.reference_label,  transform=test_transform_cifar10)

    return shadow_dataset, memory_data, test_data_clean, test_data_backdoor, test_data_backdoor_c_label

def get_shadow_cifar10_fix_backdoor_0401(args):
    total_training_data_num = 50000
    if hasattr(args, 'valid_size'):
        training_data_num = int(total_training_data_num * args.valid_size)
    else:
        training_data_num = int(total_training_data_num * 1.0) # default setting
    testing_data_num = 10000
    np.random.seed(100)
    training_data_sampling_indices = np.random.choice(total_training_data_num, training_data_num, replace=False)
    print('loading from the training data')
    shadow_dataset = Fix_backdoor_Dataset_0401(
        numpy_file=args.data_dir + 'train.npz',
        reference_file= args.reference_file,
        bug_file= args.bug_file,
        trigger_file=args.trigger_file,
        class_type=classes,
        indices = training_data_sampling_indices,
        transform=train_transform,  # The train transform is not needed in BadEncoder.
        bd_transform=test_transform_cifar10,
        ftt_transform=finetune_transform_cifar10    # data augmentation for reference images
    )
    memory_data = CIFAR10Mem(numpy_file=args.data_dir+'train.npz', class_type=classes, transform=test_transform_cifar10)
    test_data_backdoor = BadEncoderTestBackdoor(numpy_file=args.data_dir+'test.npz', trigger_file=args.trigger_file, reference_label= args.reference_label,  transform=test_transform_cifar10)
    test_data_clean = CIFAR10Mem(numpy_file=args.data_dir+'test.npz', class_type=classes, transform=test_transform_cifar10)
    test_data_backdoor_c_label = BadEncoderTestBackdoor_c_label(numpy_file=args.data_dir+'test.npz', trigger_file=args.trigger_file, reference_label= args.reference_label,  transform=test_transform_cifar10)

    # return shadow_dataset, memory_data, test_data_clean, test_data_backdoor
    return shadow_dataset, memory_data, test_data_clean, test_data_backdoor, test_data_backdoor_c_label

def get_shadow_cifar10_w_np(args):
    total_training_data_num = 50000
    if hasattr(args, 'valid_size'):
        training_data_num = int(50000 * args.valid_size)
    else:
        training_data_num = 50000 
    testing_data_num = 10000
    np.random.seed(100)
    training_data_sampling_indices = np.random.choice(total_training_data_num, training_data_num, replace=False)
    print('loading from the training data')
    shadow_dataset = BadEncoderDataset_w_np(
        numpy_file=args.data_dir + 'train.npz',
        trigger_file=args.trigger_file,
        reference_file= args.reference_file,
        class_type=classes,
        indices = training_data_sampling_indices,
        transform=train_transform,  # The train transform is not needed in BadEncoder.
        bd_transform=test_transform_cifar10,
        ftt_transform=finetune_transform_cifar10
    )
    memory_data = CIFAR10Mem(numpy_file=args.data_dir+'train.npz', class_type=classes, transform=test_transform_cifar10)
    test_data_backdoor = BadEncoderTestBackdoor(numpy_file=args.data_dir+'test.npz', trigger_file=args.trigger_file, reference_label= args.reference_label,  transform=test_transform_cifar10)
    test_data_clean = CIFAR10Mem(numpy_file=args.data_dir+'test.npz', class_type=classes, transform=test_transform_cifar10)

    return shadow_dataset, memory_data, test_data_clean, test_data_backdoor