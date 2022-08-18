import function
import create_directory
import json
import os

if __name__=="__main__":
    with open(os.path.join(create_directory.vn_data_dir, 'label2idx.json'), 'r') as openfile:    
        label2idx = json.load(openfile)
    with open(os.path.join(create_directory.vn_data_dir, 'train_feature.json'), 'r') as openfile:    
        x_train = json.load(openfile)
    with open(os.path.join(create_directory.vn_data_dir, 'test_feature.json'), 'r') as openfile:    
        x_test = json.load(openfile)
    with open(os.path.join(create_directory.vn_data_dir, 'train_label_idx.json'), 'r') as openfile:    
        y_train = json.load(openfile)
    with open(os.path.join(create_directory.vn_data_dir, 'test_label_idx.json'), 'r') as openfile:    
        y_test = json.load(openfile)      
    
    threshold_list = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    recognition_result = function.test_threshold(threshold_list, x_train, x_test, y_train, label2idx, type='nearest')
    
    save_path = os.path.join(create_directory.result_dir, 'test_threshold')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    function.save_result(recognition_result, y_test, os.path.join(save_path, "nearest_candidate_threshold_test.csv"))