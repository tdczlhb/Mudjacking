import os

if not os.path.exists('./log/cifar10'):
    os.makedirs('./log/cifar10')

def run_eval(gpu, encoder_usage_info, downstream_dataset, encoder, reference_label, trigger, reference_file, bug_file=None):
    log_path = encoder[:-len(encoder.split('/')[-1])] +'log_classifier.txt'
    
    if bug_file != None:
        cmd = f"nohup python3 -u training_downstream_classifier.py \
                --dataset {downstream_dataset} \
                --trigger_file {trigger} \
                --encoder {encoder} \
                --encoder_usage_info {encoder_usage_info} \
                --reference_label {reference_label} \
                --reference_file ./reference/{encoder_usage_info}/{reference_file}.npz \
                --bug_file {bug_file}\
                --gpu {gpu} \
                >{log_path} &"
    else: 
        cmd = f"nohup python3 -u training_downstream_classifier.py \
                --dataset {downstream_dataset} \
                --trigger_file {trigger} \
                --encoder {encoder} \
                --encoder_usage_info {encoder_usage_info} \
                --reference_label {reference_label} \
                --reference_file ./reference/{encoder_usage_info}/{reference_file}.npz \
                --gpu {gpu} \
                >{log_path} &"
    os.system(cmd)


run_eval(0, 'cifar10', 'stl10', 'output/cifar10/stl10_backdoored_encoder/model_200.pth', 9,  './trigger/trigger_pt_white_21_10_ap_replace.npz', 'truck','./data_bugs/BadEncoder/pretrain_cifar10_downstream_stl10/backdoor_bug_1/0/misclassified_image_label_7_monkey.npz')
run_eval(1, 'cifar10', 'stl10', 'output/cifar10/stl10_patched_encoder_backdoor_bug_1_lambda2_1.0_lambda3_1.0/model_200.pth', 9,  './trigger/trigger_pt_white_21_10_ap_replace.npz', 'truck','./data_bugs/BadEncoder/pretrain_cifar10_downstream_stl10/backdoor_bug_1/0/misclassified_image_label_7_monkey.npz')