import os

if not os.path.exists('./log/cifar10'):
    os.makedirs('./log/cifar10')

def run_eval(gpu, encoder_usage_info, downstream_dataset, encoder, trigger, shadow_dataset, K, reference_label, lambda2=1.0, lambda3=1.0):
    if encoder_usage_info == 'cifar10' and downstream_dataset == 'stl10':
        backdoored_save_path = f'./output/{encoder_usage_info}/{downstream_dataset}_downstream_classifier_backdoored'
    else: 
        backdoored_save_path = encoder[:-len('/model_200.pth')] + '/classifier_500.pth'
    bug_id = trigger.split('/')[-1][-len('_trigger_seed_0.npz')-1]
    patched_save_path = f'./output/{encoder_usage_info}/{downstream_dataset}_patched_encoder_backdoor_bug_{bug_id}'
    print(f'patched_save_path:{patched_save_path}')
    log_path = patched_save_path+ f'_lambda2_{lambda2}_lambda3_{lambda3}'
    print(log_path)
    os.makedirs(log_path, exist_ok=True)
    log_path = log_path +'/log_patching.txt'
    
    if encoder_usage_info == 'cifar10':
        if downstream_dataset == 'stl10':
            bug_file = './data_bugs/BadEncoder/pretrain_cifar10_downstream_stl10/backdoor_bug_1/0/misclassified_image_label_7_monkey.npz'   # misclassified image
            reference_file = './data_bugs/BadEncoder/pretrain_cifar10_downstream_stl10/backdoor_bug_1/0/reference_image_label_7_monkey_seed_0.npz'  # reference image
    else:
        assert NotImplementedError

    cmd = f"nohup python3 -u Mudjacking_patch.py \
            --classifier_save_path {backdoored_save_path} \
            --results_dir {patched_save_path} \
            --dataset {downstream_dataset} \
            --trigger_file {trigger} \
            --encoder {encoder} \
            --encoder_usage_info {encoder_usage_info} \
            --gpu {gpu} \
            --pretraining_dataset cifar10 \
            --aug_k {K} \
            --shadow_dataset {shadow_dataset} \
            --reference_label {reference_label} \
            --reference_file {reference_file} \
            --bug_file {bug_file}\
            --lambda2 {lambda2} \
            --lambda3 {lambda3} \
            >{log_path} &"
    os.system(cmd)

run_eval(0, 'cifar10', 'stl10', './output/cifar10/stl10_backdoored_encoder/model_200.pth', './reversed_triggers/backdoor_bug_1_trigger_seed_0.npz','cifar10',1,9, 1,1)