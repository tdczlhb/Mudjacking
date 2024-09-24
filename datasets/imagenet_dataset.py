from torchvision import transforms
from .backdoor_dataset import CIFAR10Mem, CIFAR10Pair, BadEncoderTestBackdoor, BadEncoderDataset, ReferenceImg, BadEncoderTestBackdoor_c_label, Bug_dataset
from .backdoor_dataset import Fix_backdoor_Dataset_0401, Fix_backdoor_Dataset_0401_imagenet
import numpy as np

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.44087798, 0.42790666, 0.38678814], [0.25507198, 0.24801506, 0.25641308])])

finetune_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.44087798, 0.42790666, 0.38678814], [0.25507198, 0.24801506, 0.25641308])])

test_transform_cifar10 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform_stl10 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.44087798, 0.42790666, 0.38678814], [0.25507198, 0.24801506, 0.25641308])])

backdoor_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.44087798, 0.42790666, 0.38678814], [0.25507198, 0.24801506, 0.25641308])])

test_transform_imagenet = transforms.Compose([
    transforms.ToTensor(),])

test_transform_CLIP = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),])

backdoor_transform_input_32_pre_cifar = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

finetune_transform_input_32_pre_cifar = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

classes = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']

def get_shadow_imagenet_fix_backdoor_0401(args):
    training_data_num = 1281167
    testing_data_num = 100000
    np.random.seed(100)
    training_data_sampling_indices = np.random.choice(training_data_num, testing_data_num, replace=False)
    print('loading from the training data')
    shadow_dataset = Fix_backdoor_Dataset_0401_imagenet(
        filepath= '/work/hl386/imagenet/train',
        reference_file= args.reference_file,
        bug_file= args.bug_file,
        trigger_file=args.trigger_file,
        class_type=classes,
        indices = training_data_sampling_indices,
        transform=train_transform,
        bd_transform=backdoor_transform,
        ftt_transform=finetune_transform
    )
    return shadow_dataset, None, None, None, None