from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

def naive_neighbour_recognition(feat, train_labels, train_feats, label2idx, threshold):
    candidate_label = []
    candidate_distance = []
    for label, vector in zip(train_labels, train_feats):
        distance = euclidean_distances(np.array(feat).reshape(1, -1), np.array(vector).reshape(1, -1))[0][0]
        if distance<threshold:
            candidate_label.append(label)
            candidate_distance.append(distance)
    if len(candidate_label)==0:
        return label2idx["Unknown"]
    else:
        df_candidate=pd.DataFrame({'Label': candidate_label})
        df_candidate_count = pd.DataFrame(df_candidate["Label"].value_counts().to_dict().items(), columns=['Label','Count'])
        list_candidate = list(df_candidate_count[df_candidate_count['Count']==max(df_candidate_count['Count'])]['Label'])
        if len(list_candidate) == 1:
            return max(candidate_label, key = candidate_label.count)
        else:
            df_candidate_distance=pd.DataFrame({'Label': candidate_label, 'Distance': candidate_distance})
            best_candidate = df_candidate_distance[df_candidate_distance['Label'].isin(list_candidate)].groupby('Label').mean().to_dict()['Distance']
            return min(best_candidate, key = best_candidate.get)

def nearest_recognition(feat, train_labels, train_feats, label2idx, threshold):
    candidate_label = []
    candidate_distance = []
    for label, vector in zip(train_labels, train_feats):
        distance = euclidean_distances(np.array(feat).reshape(1, -1), np.array(vector).reshape(1, -1))[0][0]
        if distance<threshold:
            candidate_label.append(label)
            candidate_distance.append(distance)
    if len(candidate_label)==0:
        return label2idx["Unknown"]
    else:
        return candidate_label[candidate_distance.index(min(candidate_distance))]

def weight_neighbour_recognition(feat, train_labels, train_feats, label2idx, threshold):
    candidate_label = []
    candidate_distance = []
    for label, vector in zip(train_labels, train_feats):
        distance = euclidean_distances(np.array(feat).reshape(1, -1), np.array(vector).reshape(1, -1))[0][0]
        if distance<threshold:
            candidate_label.append(label)
            candidate_distance.append(distance)
    if len(candidate_label)==0:
        return label2idx["Unknown"]
    else:
        weight = {}
        for i in range(len(candidate_label)):
            can = candidate_label[i]
            dis = candidate_distance[i]
            if can not in weight:
                weight[can]=0
            weight[can]+=1/dis
        return max(weight, key = weight.get)

# Lấy kết quả dự đoán ứng với các threshold tương ứng
def test_threshold(threshold_list, x_train, x_test, y_train, label2idx, type):
    recognition_result = {}
    for i in threshold_list:
        print("Threshold: "+str(i))
        if type=="nearest":
            recognition_result[i] = [nearest_recognition(j, y_train, x_train, label2idx, i) for j in x_test]
        if type=="naive_neighbour":
            recognition_result[i] = [naive_neighbour_recognition(j, y_train, x_train, label2idx, i) for j in x_test]
        if type=="weight_neighbour":
            recognition_result[i] = [weight_neighbour_recognition(j, y_train, x_train, label2idx, i) for j in x_test]
    return recognition_result

def metric(true, pred, label2idx):
    fa = 0  # False accept
    wa = 0  # Wrong answer
    fr = 0  # False reject
    accept = 0
    reject = 0

    for (i, j) in zip(true, pred):
        # Hệ thống nhận diện khuôn mặt đó có trong database
        if j != label2idx["Unknown"]:
            accept+=1
            # Hệ thống nhận diện khuôn mặt Unknown thành khuôn mặt trong database
            if i == label2idx["Unknown"]:
                fa+=1
            else:
                # Hệ thống nhận diện nhầm khuôn mặt trong database
                if i!=j:
                    wa+=1
        else:
            reject+=1
            if i != label2idx["Unknown"]:
                fr+=1
    # Mong muốn giảm fa, wa
    return (fa, wa, fr, accept, reject)

# Save result
def save_result(result, y_test, save_path):
    thresh_list, fa_list, wa_list, fr_list, accept_list, reject_list, accuracy_list = [], [], [], [], [], [], []
    for threshold, pred in result.items():
        fa, wa, fr, accept, reject = metric(y_test, pred)
        acc = accuracy_score(y_test, pred)
        thresh_list.append(threshold)
        fa_list.append(fa)
        wa_list.append(wa)
        fr_list.append(fr)
        accept_list.append(accept)
        reject_list.append(reject)
        accuracy_list.append(acc)
    dict = {'threshold': thresh_list, 'accept': accept_list, 'fa': fa_list, 'wa': wa_list, 
            'reject': reject_list, 'fr': fr_list, 'accuracy': accuracy_list} 
    df = pd.DataFrame(dict)
    df.to_csv(save_path, index=False)