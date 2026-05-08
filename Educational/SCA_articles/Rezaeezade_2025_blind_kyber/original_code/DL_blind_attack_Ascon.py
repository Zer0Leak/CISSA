import math
import os.path
import argparse

import numpy as np
np.set_printoptions(suppress=True)

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src.Phase_1_create_hypothetical_model_distribution import Ascon_Hypothetical_Distribution_Model
from src.Phase_2_1_poi_selection import cpa_m_and_y_poi_selection_Ascon
from src.Phase_2_VA_Slicing_labeling import LinearRegression_VarianceAnalysis_Ascon, Linge_label_methodology_Ascon
from src.Phase_3_maximum_likelihood import attack_blind_log_gpu_ascon
from src.utils import customized_accuracy_ascon, joint_hw_to_three_dim, load_ascon_blind,\
    Linge_Clustering_Ascon, NTGE_fn
from src.neural_networks import run_cnn, run_mlp

parser = argparse.ArgumentParser(description='Run DL_Joint, short version')
parser.add_argument('--model_idx', type=str, default=0, help='model ID')
parser.add_argument('--dropout', type=str, default=0, help='dropout')
parser.add_argument('--model_type', type=str, default=0, help='model_type')
parser.add_argument('--labeling_type', type=str, default=0, help='labeling_type')

# Parse arguments
args = parser.parse_args()
model_idx = int(args.model_idx)
dropout = args.dropout
if dropout == "True":
    dropout = True
else:
    dropout = False
model_type = args.model_type  # mlp, cnn
labeling_type = args.labeling_type  # ClavierLabel  #LingeLabel  #Mixture #RandomLabel

dataset = 'Ascon'  # Ascon
loss_type = "CCE"
poi_selection_mode = 'cpa'  # 'cpa', 'variance'
plot_cpa = False
mean = False

nb_attacks = 100
epochs = 50
num_poi = 50

root_data = f"/Datasets/{dataset}/"
root_results = f"/Results/{dataset}/"

print("dataset_root:", root_data)
if not os.path.exists(root_results):
    os.mkdir(root_results)

dataset_name = "ascon_cw_unprotected.trs"
byte = 0
num_bits = 8
joint = True
total_traces = 200000
nb_traces_training = 80000
nb_traces_attack = 10000
n_clusters = (num_bits + 1) * (num_bits + 1) * (num_bits + 1)
X, Y, P1, P2, correct_key = load_ascon_blind(root_data, dataset_name, total_traces, target=byte)
X_train = X[:nb_traces_training, :]
Y_train = Y[:nb_traces_training]
P1_train = P1[:nb_traces_training]
P2_train = P2[:nb_traces_training]
X_test = X[nb_traces_training:, :]
Y_test = Y[nb_traces_training:]
P1_test = P1[nb_traces_training:]
P2_test = P2[nb_traces_training:]
correct_key_test = correct_key

# Phase 1: Pre-computing theoretical joint distribution (for three targeted variable)
print("Computing the theoretical joint distribution for all the possible keys!")
theoretical_histogram_ascon = Ascon_Hypothetical_Distribution_Model(root_results, byte)

# Phase 2: Labeling traces to obtain empirical distribution. The goal of this phase is to acquire the
# empirical distribution

# Phase 2-1. PoIs Selection: To attain the empirical distribution, one first needs to identify a suitable PoI that
# represents HW(m1), another suitable PoI that represents HW(m2) and  another suitable PoI that represents HW(y)
if os.path.exists(root_results + f"{dataset}_poi_{num_poi}.npz"):
    data = np.load(root_results + f"{dataset}_poi_{num_poi}.npz", allow_pickle=True)
    poi_m1 = data["poi_m1"]
    poi_m2 = data["poi_m2"]
    poi_y = data["poi_y"]

else:
    poi_m1, poi_m2, poi_y = cpa_m_and_y_poi_selection_Ascon(X_train, P1_train, P2_train, Y_train, plot_cpa, root_results, num_poi)
    np.savez(root_results + f"{dataset}_poi_{num_poi}",
             poi_m1=poi_m1,
             poi_m2=poi_m2,
             poi_y=poi_y,

             )

print("-------------------")
print("Dataset: ", dataset, "poi_selection_mode: ", poi_selection_mode, "Number of PoI: ", len(poi_y))

# Specifying the variance of the noise according to [13] to be used in standalone Clavier VA labeling technique and in
# phase 3 for doing Maximum likelihood
var_noise = 0.00001
var_t = np.zeros(X_train.shape[1])
for i in range(len(var_t)):
    var_t[i] = np.var(X_train[:, i], dtype=np.float64)
    if var_t[i] != 0 and var_noise > var_t[i]:
        var_noise = var_t[i]
print("Var noise: ", var_noise)

# Generating the labels for the possible output classes (or clusters) when looking into all three targeted variables
# (m1, m2 and y) at the same time for training and test datasets
joint_hw_train = P1_train * (num_bits + 1) * (num_bits + 1) + P2_train * (num_bits + 1) + Y_train
joint_hw_test = P1_test * (num_bits + 1) * (num_bits + 1) + P2_test * (num_bits + 1) + Y_test

print("labeling_type is:", labeling_type)
if os.path.exists(root_results + f"{labeling_type}_empirical_hw_{dataset}_{poi_selection_mode}_{num_poi}.npy"):
    combined_hws = np.load(
        root_results + f"{labeling_type}_empirical_hw_{dataset}_{poi_selection_mode}_{num_poi}.npy")
    solo_hw_p1, solo_hw_p2, solo_hw_y = joint_hw_to_three_dim(combined_hws)

else:
    if labeling_type == 'ClavierLabel':
        solo_hw_p1, solo_hw_p2, solo_hw_y = LinearRegression_VarianceAnalysis_Ascon(X_train[:, poi_m1[0]],
                                                                                    X_train[:, poi_m2[0]],
                                                                                    X_train[:, poi_y[0]],
                                                                                    num_bits, var_noise=var_noise)

        combined_hws = solo_hw_p1 * (num_bits + 1) * (num_bits + 1) + solo_hw_p2 * (num_bits + 1) + solo_hw_y
        np.save((root_results + f"{labeling_type}_empirical_hw_{dataset}_{poi_selection_mode}_{num_poi}.npy"),
                combined_hws)

    elif labeling_type == 'LingeLabel':
        solo_hw_p1, solo_hw_p2, solo_hw_y = Linge_label_methodology_Ascon(X_train, poi_m1[0], poi_m2[0], poi_y[0],
                                                                          num_bits)
        combined_hws = solo_hw_p1 * (num_bits + 1) * (num_bits + 1) + solo_hw_p2 * (num_bits + 1) + solo_hw_y
        np.save((root_results + f"{labeling_type}_empirical_hw_{dataset}_{poi_selection_mode}_{num_poi}.npy"),
                combined_hws)

    elif labeling_type == 'Mixture':
        print("n_clusters: ", n_clusters)
        X_train_joint = np.hstack(([X_train[:, poi_m1], X_train[:, poi_m2], X_train[:, poi_y]]))
        # define the model
        model_joint = GaussianMixture(n_components=n_clusters, random_state=0)
        model_joint.fit(X_train_joint)

        # fit the model
        clusters_joint = model_joint.predict(X_train_joint)
        members_count_joint = np.bincount(clusters_joint)

        avg_y = np.zeros((n_clusters, num_poi))
        avg_m1 = np.zeros((n_clusters, num_poi))
        avg_m2 = np.zeros((n_clusters, num_poi))
        for cluster in range(n_clusters):
            cluster_members = X_train_joint[clusters_joint == cluster]
            avg_m1[cluster] = np.mean(cluster_members[:, 0:num_poi], axis=0)
            avg_m2[cluster] = np.mean(cluster_members[:, num_poi:2 * num_poi], axis=0)
            avg_y[cluster] = np.mean(cluster_members[:, 2 * num_poi:3 * num_poi], axis=0)

        empirical_hws = Linge_Clustering_Ascon(avg_m1, avg_m2, avg_y, mean)

        cluster_hw = np.zeros(X_train_joint.shape[0])
        for cluster in range(n_clusters):
            new_label = empirical_hws[cluster]
            for i in range(X_train.shape[0]):
                if clusters_joint[i] == cluster:
                    cluster_hw[i] = new_label
        np.save((root_results + f"{labeling_type}_empirical_hw_{dataset}_{poi_selection_mode}_{num_poi}.npy"), cluster_hw)
        combined_hws = cluster_hw
        solo_hw_p1, solo_hw_p2, solo_hw_y = joint_hw_to_three_dim(combined_hws)

print(f"Accuracy using {labeling_type} technique!")
labeling_total_acc, labeling_acc_m1, labeling_acc_m2, labeling_acc_y = \
    customized_accuracy_ascon(combined_hws, joint_hw_train)

scaler_std = StandardScaler()
X_train = scaler_std.fit_transform(X_train)
X_test = scaler_std.transform(X_test)

# replacing the loop for 100 models
if model_type == "cnn":
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

trained_model_root = root_results + f'{model_type}_{labeling_type}_{num_poi}_{epochs}/'
base_models = root_results + f"{model_type}_base_models"

if not os.path.exists(trained_model_root):
    os.mkdir(trained_model_root)
    os.mkdir(trained_model_root + "trained_models")

for loss_type in ["CCE"]:
    # Uncomment this part if you want to do the random search
    if loss_type == "CCE" and dropout == False and labeling_type == 'Mixture':
        generate_or_load = "generate"
        if not os.path.exists(base_models):
            os.mkdir(base_models)
    else:
        generate_or_load = "load"

    # comment this line if you want to do the random search
    # generate_or_load = "load_trained"

    print("DNN model type:", model_type)
    if model_type == "mlp":
        # Train random MLP
        jointed_predicted_hw, accuracy = run_mlp(X_train, X_test, combined_hws, joint_hw_train, joint_hw_test,
                                                 epochs, n_clusters, trained_model_root, base_models, model_idx,
                                                 dropout, loss_type, generate_or_load, dataset)


    elif model_type == "cnn":
        # Train random CNN
        jointed_predicted_hw, accuracy = run_cnn(X_train, X_test, combined_hws, joint_hw_train, joint_hw_test,
                                                 epochs, n_clusters, trained_model_root, base_models, model_idx,
                                                 dropout, loss_type, generate_or_load, dataset)

    else:
        print("Neural Network not supported")
        exit(-1)

    jointed_predicted_hw = np.asarray(jointed_predicted_hw)

    # preds0 = HW(m1), preds1 = HW(m2), preds2 = HW(y)
    preds0, preds1, preds2 = joint_hw_to_three_dim(jointed_predicted_hw)
    predicted_hw = np.array([preds0, preds1, preds2])

    # Phase 3: Comparing the empirical distribution with the theoretical joint distribution using maximum likelihood
    # criterion.
    sr_order = 1
    # Guessing Entropy parameter.
    ge_rounds = np.zeros((nb_attacks, nb_traces_attack))
    ge = np.zeros(nb_traces_attack)
    success_rate_sum = np.zeros(nb_traces_attack)
    for i in range(nb_attacks):
        # using test set to attack
        idx_trace = np.arange(predicted_hw.shape[1])
        np.random.shuffle(idx_trace)
        mle = attack_blind_log_gpu_ascon(predicted_hw[:, idx_trace[:nb_traces_attack]].T,
                                         theoretical_histogram_ascon, std=math.sqrt(var_noise))
        final_rank = np.zeros(nb_traces_attack)
        for j in range(nb_traces_attack):
            tmp_idx = np.argsort(mle[:, j])[::-1]
            final_rank[j] = np.float32(np.where(tmp_idx == correct_key)[0][0])
            if final_rank[j] <= sr_order:
                success_rate_sum[j] += 1
        ge_rounds[i] = final_rank
        ge += final_rank

    ge = ge / nb_attacks
    success_rate = success_rate_sum / nb_attacks
    print("GE: ", ge)

    NTGE = NTGE_fn(ge)
    print("NTGE:", NTGE)
    np.save(trained_model_root + f"{labeling_type}_{loss_type}_{dropout}_{model_idx}.npy",
            {"GE": ge, "GE_rounds": ge_rounds, "NTGE": NTGE, "success_rate": success_rate, "accuracy": accuracy})




