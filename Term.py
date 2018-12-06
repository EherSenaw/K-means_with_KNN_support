import numpy as np
import csv
import matplotlib.pyplot as plt
import random

################## 1. Prepare the data in the feature space.###################
cnt = 0
csv_lines = np.array([])
fd = open('iris_data.csv', 'r')
csv_reader = csv.reader(fd)
for line in csv_reader:
    if cnt==0:
        csv_lines = np.array(line)
        cnt +=1
    else:
        temp_line = np.array(line)
        csv_lines = np.vstack((csv_lines, temp_line))
fd.close()

iris_data = csv_lines
labels = np.array(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])

# Sets used for KNN (30% from full dataset)
knn_set = np.vstack((iris_data[:15], iris_data[50:65], iris_data[100:115]))
knn_set_label = knn_set[:, 4]
knn_set_features = knn_set[:, :4]
# np.savetxt('knn_set_label.csv', knn_set_label, fmt='%s', delimiter=',')
# np.savetxt('knn_set_features.csv', knn_set_features, fmt='%s', delimiter=',')

# Sets used for K-means training (30% from full dataset)
train_set = np.vstack((iris_data[15:30], iris_data[65:80], iris_data[115:130]))
train_set_y = train_set[:, 4]
train_set_x = train_set[:, :4]

# Sets used for K-means testing (40% from full dataset)
test_set = np.vstack((iris_data[30:50], iris_data[80:100], iris_data[130:150]))
test_set_y = test_set[:, 4]
test_set_x = test_set[:, :4]

###############################################################################

################################ 2. Do KNN ####################################

knn_set_train = knn_set[:10]
knn_train_x = knn_set[:, :4]
knn_train_y = knn_set[:, 4]

knn_set_test = knn_set[10:]
knn_test_x = knn_set_test[:, :4]
knn_test_y = knn_set_test[:, 4]


output_1 = np.array([0.0, 0.0, 0.0])
output_2 = np.array(['INIT'])
output_3 = np.array(['INIT(REAL_LABEL)'])
output_4 = np.array(['INIT(ESTIMATED_LABEL'])

cnt_output = 0
# Calculate distance of train_x
target_idx = 0
for target_x in knn_test_x:
    dist = []
    for t_x in knn_train_x:
        dist.append(np.linalg.norm(target_x.astype(float) - t_x.astype(float)))
    # K == 3 in KNN in this problem
    K = 3
    KNN = np.argpartition(dist, K)
    KNN_label_weighted = [0.0, 0.0, 0.0]
    for NN in KNN[:K]:
        idx = 0
        for label in labels:
            if (knn_train_y[NN] == label):
                if (dist[NN] != 0):
                    KNN_label_weighted[idx] += (1 / np.float_power(dist[NN], 2))
            else:
                idx += 1
    if (labels[np.argmax(KNN_label_weighted)] != knn_test_y[target_idx]):
        if (cnt_output == 0):
            output_1 = np.array(KNN_label_weighted)
            output_2 = np.array('F')
            output_3 = np.array(knn_test_y[target_idx]) # REAL_LABEL
            output_4 = np.array(labels[np.argmax(KNN_label_weighted)]) # ESIMATED_LABEL
            cnt_output += 1
        else:
            output_1 = np.vstack((output_1, KNN_label_weighted))
            output_2 = np.vstack((output_2, 'F'))
            output_3 = np.vstack((output_3, knn_test_y[target_idx])) # REAL_LABEL
            output_4 = np.vstack((output_4, labels[np.argmax(KNN_label_weighted)])) # ESIMATED_LABEL
    else:
        if (cnt_output == 0):
            output_1 = np.array(KNN_label_weighted)
            output_2 = np.array('.')
            output_3 = np.array(knn_test_y[target_idx]) # REAL_LABEL
            output_4 = np.array(labels[np.argmax(KNN_label_weighted)]) # ESIMATED_LABEL
            cnt_output += 1
        else:
            output_1 = np.vstack((output_1, KNN_label_weighted))
            output_2 = np.vstack((output_2, '.'))
            output_3 = np.vstack((output_3, knn_test_y[target_idx])) # REAL_LABEL
            output_4 = np.vstack((output_4, labels[np.argmax(KNN_label_weighted)])) # ESIMATED_LABEL
    target_idx += 1
'''
np.savetxt('KNN_label_weighted_1.csv', output_1, fmt='%s', delimiter=',')
np.savetxt('KNN_labels_1.csv', output_2, fmt='%s', delimiter=',')
np.savetxt('KNN_labels_real_1.csv', output_3, fmt='%s', delimiter=',')
np.savetxt('KNN_labels_est_1.csv', output_4, fmt='%s', delimiter=',')    
'''

###############################################################################

######################### 3. Do partial K-means ###############################
######################### 3-A. Get mean of each label ###############################

init_means = np.zeros((3,4))
cluster_cnt = np.zeros((3,1), dtype = int)
# knn set count
knn_set_cnt = 0

# For train_set
set_idx = 0
for t_set in knn_set_train:
    label_idx = 0
    for label in labels:
        if (t_set[4] == label):
            init_means[label_idx] = np.add(init_means[label_idx].astype(float), t_set[:4].astype(float))
            cluster_cnt[label_idx] += 1
        else:
            label_idx += 1
    set_idx += 1

knn_set_cnt += set_idx

# For test_set
set_idx = 0
for est_y in output_4:
    label_idx = 0
    for label in labels:
        if (est_y == label):
            init_means[label_idx] = np.add(init_means[label_idx].astype(float) , knn_test_x[set_idx].astype(float))
            cluster_cnt[label_idx] += 1
        else:
            label_idx += 1
    set_idx += 1

knn_set_cnt += set_idx

wow = 0
while wow < 20:
    # Get mean
    # Trial 1: Use KNN result for init mean
    if wow == 0:
        init_means = np.true_divide(init_means, cluster_cnt.astype(float))
    # Trial 2: Random mean
    else:
        init_means_random_numlist = range(0, 10)
        init_means_random_sample = random.sample(init_means_random_numlist, 3)
        print init_means_random_sample
        init_means = np.array([knn_train_x[init_means_random_sample[0]], knn_train_x[init_means_random_sample[1]], knn_train_x[init_means_random_sample[2]]]).astype(float)
    print init_means

    ##########ITERATE $$$$$$$$$$$$$$$$$$$$######################$$$$$$$$$$$
    t = 0
    dists_arr = []
    while t < 10:
        ###############################################################################
        ##################3-B. Get distance of unlabeled data #########################
        dist_k_means = np.array([])
        x_idx = 0
        for t_x in train_set_x:
            init_cnt = 0
            tmp_dist = np.array([])
            for m in init_means:
                if init_cnt == 0:
                    tmp_dist = np.array([np.linalg.norm(t_x.astype(float) - m.astype(float))])
                    init_cnt = 1
                else:
                    tmp_dist = np.vstack((tmp_dist, np.linalg.norm(t_x.astype(float) - m.astype(float))))
            tmp_dist = np.transpose(tmp_dist)

            if x_idx == 0:
                dist_k_means = tmp_dist
            else:
                dist_k_means = np.vstack((dist_k_means, tmp_dist))
            
            x_idx += 1
        '''
        np.savetxt('dist_k_means.csv', dist_k_means, fmt='%s', delimiter=',')
        '''
        ###############################################################################
        ###############3-C. Put unlabled data to proper cluster #######################
        cluster = []
        dists = 0.0
        for i in range(np.size(dist_k_means[0, :])):
            cluster.append([])
        predict_y = np.array([])
        for i in range(x_idx):
            min_label_idx = 0
            for j in range(np.size(dist_k_means[i, :])):
                if (j == 0):
                    min_label_idx = j
                else:
                    if (dist_k_means[i,j] < dist_k_means[i,min_label_idx]):
                        min_label_idx = j
            if (i == 0):
                predict_y = np.array(labels[min_label_idx])
            else:
                predict_y = np.vstack((predict_y, labels[min_label_idx]))
            dists = dists + dist_k_means[i, min_label_idx]
            cluster[min_label_idx].append(i)
        dists_arr.append(dists)
        # print cluster
        #np.savetxt('first_predict_y.csv', predict_y, fmt='%s', delimiter=',')
        np.savetxt('cluster_predict_y.csv', cluster, fmt='%s', delimiter=',')
        new_cluster_means = np.zeros((3,4))

        for i in range(np.size(cluster)):
            for j in cluster[i]:
                new_cluster_means[i] = np.add(new_cluster_means[i].astype(float), train_set_x[j].astype(float))
            if (np.size(cluster[i]) != 0):
                new_cluster_means[i] = np.true_divide(new_cluster_means[i], np.size(cluster[i]))
            else:
                new_cluster_means[i] = np.zeros((1,4))

        init_means = new_cluster_means
                
        ###############################################################################
        #########################3-E. Repeat until convergence.########################
        t = t + 1
        ###############################################################################
    if (wow == 0):
        fig = plt.figure()
        plt.plot(dists_arr)
        fig.savefig('loss_with_knn', bbox_inches='tight')
        plt.close(fig)
        print dists_arr[np.size(dists_arr)-1]
    else:
        fig = plt.figure()
        plt.plot(dists_arr)
        save_txt_name = []
        save_txt_name.append('loss_random_')
        save_txt_name.append(str(wow)) 
        save_txt_name.append('.png')
        save_txt_name = ''.join(save_txt_name)
        fig.savefig(save_txt_name, bbox_inches='tight')
        plt.close(fig)
        print dists_arr[np.size(dists_arr)-1]
    ##################################4. Do test ##################################
    ###############################################################################
    # 3. For each x, find its new cluster index
    C_test = []
    for i in range(np.size(cluster_cnt)):
        C_test.append([])

    test_idx = np.size(train_set[:, 0])
    for x in test_set_x:
        dist = []
        for p in range(np.size(cluster_cnt)):
            dist.append(0.0)
        for l in range(np.size(cluster_cnt)):
            dist[l] = np.linalg.norm( (x.astype(float))-(init_means[l].astype(float)) )
        idx_new = np.argmin(dist)
        #cluster_index.append(idx_new)
        C_test[idx_new].append(test_idx)
        test_idx += 1

    #np.savetxt('test_result.csv', C, fmt='%s', delimiter=',')
    print C_test

    # Save test_y as 'test_real_y.csv'
    #np.savetxt('test_real_y.csv', test_set_y, fmt='%s', delimiter=',')
    # Save estimated_y (clusters') as 'test_estimated_y.csv'
    estimated_y = []
    for i in range(np.size(cluster_cnt)):
        estimated_y.append([])
    threshold = np.size(train_set[:, 0])
    for i in range(np.size(cluster_cnt)):
        for j in C_test[i]:
            if (j < threshold):
                estimated_y[i].append(train_set_y[j])
            else:
                estimated_y[i].append(test_set_y[j-threshold])
    np.savetxt('test_estimated_y.csv', estimated_y, fmt='%s', delimiter=',')
    wow += 1
