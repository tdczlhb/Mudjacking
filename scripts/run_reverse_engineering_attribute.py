import os

if not os.path.exists('./log/cifar10'):
    os.makedirs('./log/cifar10')

def run_eval(gpu, encoder_usage_info, downstream_dataset, encoder, reference_label, trigger, reference_file, key='clean'):
    backdoored_save_path = f'./output/{encoder_usage_info}/{downstream_dataset}_downstream_classifier_backdoored'
    patched_save_path = f'./output/{encoder_usage_info}/{downstream_dataset}_downstream_classifier_patch_our_method'
    if not os.path.exists(patched_save_path):
        os.makedirs(patched_save_path)
    cmd = f"python3 -u Mudjacking_reverse_engineering_attribute.py \
            --classifier_save_path {backdoored_save_path} \
            --results_dir {patched_save_path}/ \
            --dataset {downstream_dataset} \
            --trigger_file {trigger} \
            --encoder {encoder} \
            --encoder_usage_info {encoder_usage_info} \
            --reference_label {reference_label} \
            --reference_file ./reference/{encoder_usage_info}/{reference_file}.npz \
            --gpu {gpu} \
            --pretraining_dataset cifar10 "

    os.system(cmd)

run_eval(0, 'cifar10', 'stl10', './output/cifar10/stl10_backdoored_encoder/model_200.pth', 9, './trigger/trigger_pt_white_21_10_ap_replace.npz', 'truck', 'backdoor')
