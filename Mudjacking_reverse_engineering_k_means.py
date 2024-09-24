# Implement our method
import os
import argparse
import random
import numpy as np
import torch
from sklearn.cluster import KMeans
from PIL import Image
import matplotlib.ticker as ticker

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate the clean or backdoored encoders')
    parser.add_argument('--dataset', default='cifar10', type=str, help='downstream dataset')
    parser.add_argument('--seed', default=100, type=int, help='seed')
    parser.add_argument('--results_dir', default='', type=str, metavar='PATH', help='path to save the backdoored encoder')
    parser.add_argument('--pretraining_dataset', type=str, default='cifar10')
    
    # for detection
    parser.add_argument('--val_ratio', default=0.5, type=float, help='the ratio of valiation dataset to pre-training dataset')
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    kmeans = KMeans(n_clusters=2)
    path_bug = './data_bugs/BadEncoder/pretrain_cifar10_downstream_stl10/backdoor_bug_1/0/misclassified_image_label_7_monkey.npz'
    path_ref_ori = './data_bugs/BadEncoder/pretrain_cifar10_downstream_stl10/backdoor_bug_1/0/reference_image_label_7_monkey_seed_0.npz'

    bug_id = path_bug.split('/')[-3]
    
    if 'CLIP' in path_bug:
        img_height = 224
    else:
        img_height = 32
    path_parent = path_bug.split('/')[3]
    # cifar->stl10 
    npz_attribute = f'./reversed_trigger/{path_parent}/backdoor/{bug_id}_attribute.npz'
    
    np_attribute = np.load(npz_attribute)['arr']
    
    kmeans.fit(np_attribute.reshape(-1, 1))
    labels = kmeans.labels_.reshape((-1,img_height))
    print(labels[10:,:])

    import matplotlib.pyplot as plt
    plt.clf()
    plt.hist(np_attribute.reshape(-1,1), bins=20)
    
    # Add x and y labels and title, and change font size
    plt.xlabel('Attribution scores (x $10^{-3}$)', fontsize=18)
    plt.ylabel('Frequency', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Format x-axis labels to two decimal places and power of 10
    plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    plt.gca().xaxis.set_minor_formatter('{:.2f}'.format)
    plt.gca().xaxis.get_offset_text().set_fontsize(14)
    plt.gca().ticklabel_format(axis='x', style='sci', scilimits=(-3,3))
    
    score = np_attribute.reshape(-1,1)
    labels = kmeans.labels_
    positive_list = []
    for i in range(len(score)):
        if labels[i] == 1:
            positive_list.append(score[i][0])
    min_pos = min(positive_list)
    plt.axvline(x=min_pos, color='r')
    # cifar->stl10  !!
    plt.savefig(f'./reversed_trigger/{path_parent}/backdoor/{bug_id}_k_means.pdf')

    labels = kmeans.labels_.reshape((-1,img_height))
    tm = np.zeros((labels.shape[0], labels.shape[1], 3)) # rgb trigger mask for image
    t = np.zeros((labels.shape[0], labels.shape[1], 3)) # rgb trigger for image
    bug_img = np.load(path_bug)['img']   # rgd misclassified image
    counts = np.bincount(labels.ravel())
    majority_element = np.argmax(counts)    # majority are benign pixels
    
    for i in range(labels.shape[0]):    # row   
        for j in range(labels.shape[1]):    # column
            if labels[i,j] == majority_element:
                tm[i,j,:] = 1   # no trigger here, 1 to original img
                t[i,j,:] = 0   # no trigger here
            else: 
                tm[i,j,:] = 0   # trigger here, 0 to original img
                t[i,j,:] = bug_img[i,j,:]   # segment from bug image here
    
    np.savez(f'./reversed_trigger/{path_parent}/triggers/{bug_id}_trigger.npz', t=t, tm=tm)
    Image.fromarray(t.astype(np.uint8)).save(f'./reversed_trigger/{path_parent}/triggers/{bug_id}_trigger.jpg')