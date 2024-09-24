import os

if not os.path.exists('./log/cifar10'):
    os.makedirs('./log/cifar10')

def run_eval(pretraining_dataset, downstream_dataset, key='clean'):
    cmd = f"python3 -u Mudjacking_reverse_engineering_k_means.py \
            --dataset {downstream_dataset} \
            --pretraining_dataset {pretraining_dataset} "
    os.system(cmd)

run_eval('cifar10', 'stl10')
