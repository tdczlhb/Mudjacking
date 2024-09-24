
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from PIL import Image
import numpy as np
import torch
import random

import copy



class ReferenceImg(Dataset):

    def __init__(self, reference_file, transform=None):
        """
        Args:
            numpy_file (string): Path to the numpy file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.target_input_array = np.load(reference_file)

        self.data = self.target_input_array['x']
        self.targets = self.target_input_array['y']

        self.transform = transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)


class BadEncoderDataset(Dataset):
    def __init__(self, numpy_file, trigger_file, reference_file, indices, class_type, transform=None, bd_transform=None, ftt_transform=None):
        self.input_array = np.load(numpy_file)
        self.data = self.input_array['x']

        self.trigger_input_array = np.load(trigger_file)
        self.target_input_array = np.load(reference_file)

        self.trigger_patch_list = self.trigger_input_array['t']
        self.trigger_mask_list = self.trigger_input_array['tm']
        self.target_image_list = self.target_input_array['x']

        self.classes = class_type
        self.indices = indices
        self.transform = transform
        self.bd_transform = bd_transform
        self.ftt_transform = ftt_transform

    def __getitem__(self, index):
        img = self.data[self.indices[index]]
        img_copy = copy.deepcopy(img)
        backdoored_image = copy.deepcopy(img)
        img = Image.fromarray(img)
        '''original image'''
        if self.transform is not None:
            im_1 = self.transform(img)
        img_raw = self.bd_transform(img)
        '''generate backdoor image'''

        img_backdoor_list = []
        for i in range(len(self.target_image_list)):
            backdoored_image[:,:,:] = img_copy * self.trigger_mask_list[i] + self.trigger_patch_list[i][:]
            img_backdoor =self.bd_transform(Image.fromarray(backdoored_image))
            img_backdoor_list.append(img_backdoor)

        target_image_list_return, target_img_1_list_return = [], []
        for i in range(len(self.target_image_list)):
            target_img = Image.fromarray(self.target_image_list[i])
            target_image = self.bd_transform(target_img)
            target_img_1 = self.ftt_transform(target_img)
            target_image_list_return.append(target_image)
            target_img_1_list_return.append(target_img_1)

        return img_raw, img_backdoor_list, target_image_list_return, target_img_1_list_return
    def __len__(self):
        return len(self.indices)

class BadEncoderDataset_w_np(Dataset):

    def __init__(self, numpy_file, trigger_file, reference_file, indices, class_type, transform=None, bd_transform=None, ftt_transform=None):
        self.input_array = np.load(numpy_file)
        self.data = self.input_array['x']

        self.trigger_input_array = np.load(trigger_file)
        self.target_input_array = np.load(reference_file)

        self.trigger_patch_list = self.trigger_input_array['t']
        self.trigger_mask_list = self.trigger_input_array['tm']
        self.target_image_list = self.target_input_array['x']

        self.classes = class_type
        self.indices = indices
        self.transform = transform
        self.bd_transform = bd_transform
        self.ftt_transform = ftt_transform

    def __getitem__(self, index):
        img = self.data[self.indices[index]]
        img_copy = copy.deepcopy(img)
        backdoored_image = copy.deepcopy(img)
        img = Image.fromarray(img)
        '''original image'''
        if self.transform is not None:
            im_1 = self.transform(img)
        img_raw = self.bd_transform(img)
        '''generate backdoor image'''

        img_backdoor_list = []
        for i in range(len(self.target_image_list)):
            backdoored_image[:,:,:] = img_copy * self.trigger_mask_list[i] + self.trigger_patch_list[i][:]
            img_backdoor =self.bd_transform(Image.fromarray(backdoored_image))
            img_backdoor_list.append(img_backdoor)

        target_image_list_return, target_img_1_list_return = [], []
        for i in range(len(self.target_image_list)):
            target_img = Image.fromarray(self.target_image_list[i])
            target_image = self.bd_transform(target_img)
            target_img_1 = self.ftt_transform(target_img)
            target_image_list_return.append(target_image)
            target_img_1_list_return.append(target_img_1)

        return img_copy, img_raw, img_backdoor_list, target_image_list_return, target_img_1_list_return
    def __len__(self):
        return len(self.indices)

class BadEncoderTestBackdoor(Dataset):

    def __init__(self, numpy_file, trigger_file, reference_label, transform=None):
        """
        Args:
            numpy_file (string): Path to the numpy file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.input_array = np.load(numpy_file)
        self.data = self.input_array['x']
        self.targets = self.input_array['y']


        self.trigger_input_array = np.load(trigger_file)

        # self.trigger_patch_list = self.trigger_input_array['t']
        # self.trigger_mask_list = self.trigger_input_array['tm']
        if self.trigger_input_array['t'].shape[0] != 1:
            self.trigger_patch_list = np.expand_dims(self.trigger_input_array['t'],axis=0)
            self.trigger_mask_list = np.expand_dims(self.trigger_input_array['tm'],axis=0)# [1,32,32,3]
        else: 
            self.trigger_patch_list = self.trigger_input_array['t']
            self.trigger_mask_list = self.trigger_input_array['tm']

        self.target_class = reference_label

        self.test_transform = transform

    def __getitem__(self,index):
        img = copy.deepcopy(self.data[index])
        img[:] =img * self.trigger_mask_list[0] + self.trigger_patch_list[0][:]
        img_backdoor =self.test_transform(Image.fromarray(img))
        return img_backdoor, self.target_class


    def __len__(self):
        return self.data.shape[0]

class BadEncoderTestBackdoor_c_label(Dataset):

    def __init__(self, numpy_file, trigger_file, reference_label, transform=None):
        """
        Args:
            numpy_file (string): Path to the numpy file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.input_array = np.load(numpy_file)
        self.data = self.input_array['x']
        self.targets = self.input_array['y'][:,0].tolist()


        self.trigger_input_array = np.load(trigger_file)

        self.trigger_patch_list = self.trigger_input_array['t']
        self.trigger_mask_list = self.trigger_input_array['tm']
        if self.trigger_patch_list.shape[0] != 1:
            self.trigger_patch_list = np.expand_dims(self.trigger_patch_list, axis=0)
        if self.trigger_mask_list.shape[0] != 1:
            self.trigger_mask_list = np.expand_dims(self.trigger_mask_list, axis=0)

        self.target_class = reference_label

        self.test_transform = transform

    def __getitem__(self,index):
        img = copy.deepcopy(self.data[index])
        img[:] =img * self.trigger_mask_list[0] + self.trigger_patch_list[0][:]
        img_backdoor =self.test_transform(Image.fromarray(img))
        # label = copy.deepcopy(self.targets[index])
        label = self.targets[index]
        return img_backdoor, label


    def __len__(self):
        return self.data.shape[0]

class Bug_dataset(Dataset):
    def __init__(self, numpy_file, trigger_file, reference_label, n_bugs, transform=None):
        """
        Args:
            numpy_file (string): Path to the numpy file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        assert n_bugs > 0
        self.input_array = np.load(numpy_file)
        self.data = self.input_array['x'][:n_bugs]
        self.targets = self.input_array['y'][:n_bugs,0].tolist()


        self.trigger_input_array = np.load(trigger_file)

        self.trigger_patch_list = self.trigger_input_array['t']
        self.trigger_mask_list = self.trigger_input_array['tm']

        self.target_class = reference_label

        self.test_transform = transform

    def __getitem__(self,index):
        img = copy.deepcopy(self.data[index])
        img[:] =img * self.trigger_mask_list[0] + self.trigger_patch_list[0][:]
        img_backdoor =self.test_transform(Image.fromarray(img))
        # label = copy.deepcopy(self.targets[index])
        label = self.targets[index]
        return img_backdoor, label


    def __len__(self):
        return self.data.shape[0]

class Fix_backdoor_Dataset(Dataset):

    def __init__(self, numpy_file, trigger_file, indices, class_type, aug_k, transform=None, bd_transform=None, ftt_transform=None):
        self.input_array = np.load(numpy_file)
        self.data = self.input_array['x']
        
        self.trigger_input_array = np.load(trigger_file)
        if self.trigger_input_array['t'].shape[0] != 1:
            self.trigger_patch_list = np.expand_dims(self.trigger_input_array['t'],axis=0)
            self.trigger_mask_list = np.expand_dims(self.trigger_input_array['tm'],axis=0)# [1,32,32,3]
        else: 
            self.trigger_patch_list = self.trigger_input_array['t']
            self.trigger_mask_list = self.trigger_input_array['tm']
        for i in range(-11,-1):
            for j in range(-11,-1):
                if (self.trigger_mask_list[0][i,j,:] == [0,0,0]).all():
                    # print(f'before:{self.trigger_patch_list[0][i,j,:]}')
                    self.trigger_patch_list[0][i,j,:] = np.array([255,255,255])
                    # print(f'!!!after:{self.trigger_patch_list[0][i,j,:]}')
        self.classes = class_type
        self.indices = indices
        self.transform = transform
        self.bd_transform = bd_transform
        self.ftt_transform = ftt_transform
        self.aug_k = aug_k

    def __getitem__(self, index):
        img = self.data[self.indices[index]]
        img_copy = copy.deepcopy(img)   # [32,32,3]
        backdoored_image = copy.deepcopy(img)
        
        img = Image.fromarray(img)
        '''original image'''
        if self.transform is not None:
            im_1 = self.transform(img)
        img_raw = self.bd_transform(img)
        
        '''generate backdoor image'''
        img_backdoor_list = []
        for i in range(self.aug_k):
            backdoored_image[:,:,:] = img_copy * self.trigger_mask_list[0] + self.trigger_patch_list[0][:]
            img_backdoor =self.bd_transform(Image.fromarray(backdoored_image))
            img_backdoor_list.append(img_backdoor)

        img_list_aug = []
        for i in range(self.aug_k):
            img_ref_aug = self.ftt_transform(Image.fromarray(img_copy))
            img_list_aug.append(img_ref_aug)
        return img_raw, img_backdoor_list, img_list_aug
    
    def __len__(self):
        return len(self.indices)
    
class Fix_backdoor_Dataset_0401(Dataset):

    def __init__(self, numpy_file, trigger_file, bug_file, reference_file, indices, class_type, transform=None, bd_transform=None, ftt_transform=None):
        print(numpy_file)
        self.input_array = np.load(numpy_file)
        self.data = self.input_array['x']
        
        self.trigger_input_array = np.load(trigger_file)
        if self.trigger_input_array['t'].shape[0] != 1:
            self.trigger_patch_list = np.expand_dims(self.trigger_input_array['t'],axis=0)
            self.trigger_mask_list = np.expand_dims(self.trigger_input_array['tm'],axis=0)# [1,32,32,3]
        else: 
            self.trigger_patch_list = self.trigger_input_array['t']
            self.trigger_mask_list = self.trigger_input_array['tm']

        self.classes = class_type
        self.indices = indices
        self.transform = transform
        self.bd_transform = bd_transform
        self.ftt_transform = ftt_transform
        
        self.bug_input = Image.fromarray(np.load(bug_file)['img']) 
        self.ref_input = Image.fromarray(np.load(reference_file)['img']) 
        self.ref_input_np = np.load(reference_file)['img']

    def __getitem__(self, index):
        img = self.data[self.indices[index]]
        img_copy = copy.deepcopy(img)   # [32,32,3]
        backdoored_image = copy.deepcopy(img)
        
        img = Image.fromarray(img)
        '''original image'''
        if self.transform is not None:
            im_1 = self.transform(img)
        img_raw = self.ftt_transform(img)
        
        '''generate backdoor image'''
        # img_backdoor_list = []
        # for i in range(self.aug_k):
        # print(img_copy.shape, self.trigger_mask_list[0].shape)
        # print(self.trigger_patch_list[0][:].shape, backdoored_image[:,:,:].shape)
        # assert 0==1
        backdoored_image[:,:,:] = img_copy * self.trigger_mask_list[0] + self.trigger_patch_list[0][:]
        # img_backdoor =self.bd_transform(Image.fromarray(backdoored_image))
        img_backdoor =self.ftt_transform(Image.fromarray(backdoored_image))

        # img_backdoor_list.append(img_backdoor)
        img_bug = self.bug_input
        # img_bug = self.bd_transform(img_bug)
        img_bug = self.ftt_transform(img_bug)
        img_ref = self.ref_input
        # img_ref = self.bd_transform(img_ref)
        img_ref = self.ftt_transform(img_ref)
        
        img_ref_backdoor = copy.deepcopy(self.ref_input_np)
        img_ref_backdoor[:,:,:] = img_ref_backdoor * self.trigger_mask_list[0] + self.trigger_patch_list[0][:]
        img_ref_backdoor = self.ftt_transform(Image.fromarray(img_ref_backdoor))
        return img_raw, img_backdoor, img_bug, img_ref, img_ref_backdoor
    
    def __len__(self):
        return len(self.indices)
    
class Fix_backdoor_Dataset_0401_imagenet(Dataset):
    def __init__(self, filepath, trigger_file, bug_file, reference_file, indices, class_type, transform=None, bd_transform=None, ftt_transform=None):
        self.data_set = torchvision.datasets.ImageFolder(filepath)
        print("number of overall training examples")
        print(len(self.data_set))
        
        self.trigger_input_array = np.load(trigger_file)
        if self.trigger_input_array['t'].shape[0] != 1:
            self.trigger_patch_list = np.expand_dims(self.trigger_input_array['t'],axis=0)
            self.trigger_mask_list = np.expand_dims(self.trigger_input_array['tm'],axis=0)# [1,32,32,3]
        else: 
            self.trigger_patch_list = self.trigger_input_array['t']
            self.trigger_mask_list = self.trigger_input_array['tm']

        self.classes = class_type
        self.indices = indices
        self.transform = transform
        self.bd_transform = bd_transform
        self.ftt_transform = ftt_transform
        
        self.bug_input = Image.fromarray(np.load(bug_file)['img']) 
        self.ref_input = Image.fromarray(np.load(reference_file)['img']) 
        self.ref_input_np = np.load(reference_file)['img']

    def __getitem__(self, index):
        path, target = self.data_set.samples[self.indices[index]]
        img = self.data_set.loader(path)

        img = torchvision.transforms.Resize((224,224))(img)
        img = np.asarray(img).copy()
        img_copy = copy.deepcopy(img)
        backdoored_image = copy.deepcopy(img)
        
        img = Image.fromarray(img)
        '''original image'''
        if self.transform is not None:
            im_1 = self.transform(img)
        img_raw = self.ftt_transform(img)
        
        '''generate backdoor image'''
        # img_backdoor_list = []
        # for i in range(self.aug_k):
        # print(img_copy.shape, self.trigger_mask_list[0].shape)
        # print(self.trigger_patch_list[0][:].shape, backdoored_image[:,:,:].shape)
        # assert 0==1
        backdoored_image[:,:,:] = img_copy * self.trigger_mask_list[0] + self.trigger_patch_list[0][:]
        # img_backdoor =self.bd_transform(Image.fromarray(backdoored_image))
        img_backdoor =self.ftt_transform(Image.fromarray(backdoored_image))

        # img_backdoor_list.append(img_backdoor)
            
        img_bug = self.bug_input
        # img_bug = self.bd_transform(img_bug)
        img_bug = self.ftt_transform(img_bug)
        img_ref = self.ref_input
        # img_ref = self.bd_transform(img_ref)
        img_ref = self.ftt_transform(img_ref)
        
        img_ref_backdoor = copy.deepcopy(self.ref_input_np)
        img_ref_backdoor[:,:,:] = img_ref_backdoor * self.trigger_mask_list[0] + self.trigger_patch_list[0][:]
        img_ref_backdoor = self.ftt_transform(Image.fromarray(img_ref_backdoor))
        return img_raw, img_backdoor, img_bug, img_ref, img_ref_backdoor
    
    def __len__(self):
        return len(self.indices)

class CIFAR10CUSTOM(Dataset):

    def __init__(self, numpy_file, class_type, transform=None):
        """
        Args:
            numpy_file (string): Path to the numpy file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.input_array = np.load(numpy_file)
        self.data = self.input_array['x']
        self.targets = self.input_array['y'][:,0].tolist()
        self.classes = class_type
        self.transform = transform
    def __len__(self):
        return self.data.shape[0]

class CIFAR10Pair(CIFAR10CUSTOM):
    """CIFAR10 Dataset.
    """
    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            im_1 = self.transform(img)
            im_2 = self.transform(img)
        return im_1, im_2

class CIFAR10Mem(CIFAR10CUSTOM):
    """CIFAR10 Dataset.
    """
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        return img, target
