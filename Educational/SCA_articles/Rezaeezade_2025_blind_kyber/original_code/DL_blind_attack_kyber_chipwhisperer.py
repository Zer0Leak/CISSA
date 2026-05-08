import math
import os.path
import random
import argparse

import numpy as np
np.set_printoptions(suppress=True)

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src.Phase_1_create_hypothetical_model_distribution import AES_Hypothetical_Distribution_Model, \
    Kyber_Hypothetical_Distribution_Model
from src.Phase_2_1_poi_selection import cpa_m_and_y_poi_selection, variance_m_and_y_poi_selection
from src.Phase_2_VA_Slicing_labeling import LinearRegression_VarianceAnalysis, Linge_label_methodology
from src.Phase_3_maximum_likelihood import attack_blind_log_gpu
from src.utils import load_chipwhisperer_blind, NTGE_fn, customized_accuracy, joint_hw_to_two_dim, \
    load_kyber_blind, Linge_Clustering
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

dataset = 'Chipwhisperer'  # Chipwhisperer # Kyber
loss_type = "CCE"
poi_selection_mode = 'cpa'  # 'cpa', 'variance'
plot_cpa = False

nb_attacks = 100
epochs = 100
num_poi = 50
noise_std = 0.05

root_data = f"/Datasets/{dataset}/"
root_results = f"/Results/{dataset}/"


print("dataset_root:", root_data)
if not os.path.exists(root_results):
    os.mkdir(root_results)

if dataset == 'Chipwhisperer':
    byte = 0
    nb_traces_training = 8000
    nb_traces_attack = 1700
    # Number of bits in the targeted intermediate value (here s-box) and message (plaintext)
    num_bits = 8
    # Number of possible combinations for (HW(y), HW(m)) tuple
    n_clusters = (num_bits + 1) * (num_bits + 1)
    # Set "sync" parameter to True when attacking the synchronized and False when attacking the desynchronized traces in
    # ChipWhisperer dataset
    sync = True
    X, Y, P, correct_key = load_chipwhisperer_blind(root_data, byte, sync, noise_std)
    X_train = X[:nb_traces_training, :]
    Y_train = Y[:nb_traces_training]
    P_train = P[:nb_traces_training]
    X_test = X[nb_traces_training:, :]
    Y_test = Y[nb_traces_training:]
    P_test = P[nb_traces_training:]

elif dataset == "Kyber":
    dataset_name = "Reference-PPM"
    total_traces = 100000
    nb_traces_attack = 10000
    nb_traces_training = 80000
    # Number of bits in the targeted intermediate value (here u and s coefficients pair-pointwise multiplication) and
    # ciphertext (u)
    num_bits = 16
    # Number of possible combinations for (HW(y), HW(m)) tuple
    n_clusters = (num_bits + 1) * (num_bits + 1)
    # window sizes for resampling according to [31]
    window = 10
    # set this to True if you want to use resampling according to NOPOI approach in [31]
    use_reduced = True
    X, Y, P, correct_key = load_kyber_blind(root_data, dataset_name, total_traces, window, use_reduced)
    X_train = X[:nb_traces_training, :]
    Y_train = Y[:nb_traces_training]
    P_train = P[:nb_traces_training]
    X_test = X[nb_traces_training:, :]
    Y_test = Y[nb_traces_training:]
    P_test = P[nb_traces_training:]

# Phase 1: Pre-computing theoretical joint distribution
print("Computing the theoretical joint distribution for all the possible keys!")
if dataset == "Kyber":
    theoretical_histogram = Kyber_Hypothetical_Distribution_Model(root_results)
else:
    theoretical_histogram = AES_Hypothetical_Distribution_Model(root_results)

# Phase 2: Labeling traces to obtain empirical distribution. The goal of this phase is to acquire the
# empirical distribution

# Phase 2-1. PoIs Selection: To attain the empirical distribution, one first needs to identify a suitable PoI that
# represents HW(m) and another suitable PoI that represents HW(y)
if os.path.exists(root_results + f"{dataset}_{poi_selection_mode}_{num_poi}.npz"):
    data = np.load(root_results + f"{dataset}_{poi_selection_mode}_{num_poi}.npz", allow_pickle=True)
    poi_m = data["poi_m"]
    poi_y = data["poi_y"]

else:
    # We take #poi_m Points-of-Interest for HW(m) and another #poi_y Point-of-Interest for HW(y). These points are
    # selected as the top sample points with the highest correlation to HW(m) and HW(y), respectively.
    if poi_selection_mode == 'cpa':
        poi_m, poi_y = cpa_m_and_y_poi_selection(X_train, P_train, Y_train, plot_cpa, root_results, num_poi)
        np.savez(root_results + f"{dataset}_{poi_selection_mode}_{num_poi}",
                 poi_m=poi_m,
                 poi_y=poi_y,
                 )
    # [13] used the highest peak from the traces’ variances along with reasonable assumptions about the implementation
    # to locate the PoIs. However, based on our initial experiments, extracting the key was not successful using
    # this method.
    elif poi_selection_mode == "variance":
        poi_m, poi_y = variance_m_and_y_poi_selection(X_train, num_poi)
        np.savez(root_results + f"{dataset}_{poi_selection_mode}_{num_poi}",
                 poi_m=poi_m,
                 poi_y=poi_y,
                 )
    else:
        print("The poi_selection_mode is not supported!")
        exit(-1)
print("-------------------")
print("Dataset: ", dataset, "poi_selection_mode: ", poi_selection_mode, "Number of PoI: ", len(poi_m))

# Specifying the variance of the noise according to [13] to be used in standalone Clavier VA labeling technique and in
# phase 3 for doing Maximum likelihood
var_noise = 0.001
var_t = np.zeros(X_train.shape[1])
for i in range(len(var_t)):
    var_t[i] = np.var(X_train[:, i], dtype=np.float64)
    if var_t[i] != 0 and var_noise > var_t[i]:
        var_noise = var_t[i]
print("Var noise: ", var_noise)

# Generating the labels for the possible output classes (or clusters) when looking into both targeted variables
# (m and y) at the same time for training and test datasets
joint_hw_train = P_train * (num_bits + 1) + Y_train
joint_hw_test = P_test * (num_bits + 1) + Y_test

print("labeling_type is:", labeling_type)
if os.path.exists(root_results + f"{labeling_type}_empirical_hw_{dataset}_{poi_selection_mode}_{num_poi}.npy"):
    combined_hws = np.load(
        root_results + f"{labeling_type}_empirical_hw_{dataset}_{poi_selection_mode}_{num_poi}.npy")
    solo_hw_p, solo_hw_y = joint_hw_to_two_dim(combined_hws, num_bits + 1)

else:
    # Slicing labeling as used in [53]
    if labeling_type == 'LingeLabel':
        solo_hw_p, solo_hw_y = Linge_label_methodology(X_train, poi_m[0], poi_y[0], num_bits)
        combined_hws = solo_hw_p * (num_bits + 1) + solo_hw_y
        np.save((root_results + f"{labeling_type}_empirical_hw_{dataset}_{poi_selection_mode}_{num_poi}.npy"),
                combined_hws)
    # VA labeling as used in [13]
    elif labeling_type == 'ClavierLabel':
        solo_hw_p, solo_hw_y = LinearRegression_VarianceAnalysis(X_train[:, poi_m[0]],
                                                                 X_train[:, poi_y[0]],
                                                                 num_bits, variance_noise=var_noise)

        combined_hws = solo_hw_p * (num_bits + 1) + solo_hw_y
        np.save((root_results + f"{labeling_type}_empirical_hw_{dataset}_{poi_selection_mode}_{num_poi}.npy"),
                combined_hws)
    # Multi-point Cluster-based labeling in our work
    elif labeling_type == 'Mixture':
        print("n_clusters: ", n_clusters)
        X_train_joint = np.hstack(([X_train[:, poi_m], X_train[:, poi_y]]))

        # define the model
        model_joint = GaussianMixture(n_components=n_clusters)
        model_joint.fit(X_train_joint)

        # fit the model
        clusters_joint = model_joint.predict(X_train_joint)

        avg_y = np.zeros((n_clusters, num_poi))
        avg_m = np.zeros((n_clusters, num_poi))
        for cluster in range(n_clusters):
            cluster_members = X_train_joint[clusters_joint == cluster]
            avg_m[cluster] = np.mean(cluster_members[:, 0:num_poi], axis=0)
            avg_y[cluster] = np.mean(cluster_members[:, num_poi:2 * num_poi], axis=0)

        empirical_hws = Linge_Clustering(avg_y, avg_m, n_clusters, dataset)

        cluster_hw = np.zeros(X_train_joint.shape[0], dtype="uint8")
        for cluster in range(n_clusters):
            new_label = empirical_hws[cluster]
            for i in range(X_train.shape[0]):
                if clusters_joint[i] == cluster:
                    cluster_hw[i] = new_label
        np.save((root_results + f"{labeling_type}_empirical_hw_{dataset}_{poi_selection_mode}_{num_poi}.npy"),
                cluster_hw)
        combined_hws = cluster_hw
        solo_hw_p, solo_hw_y = joint_hw_to_two_dim(combined_hws, num_bits + 1)
    else:
        print("labeling_type is not supported!!")
        exit(-1)

print(f"Accuracy using {labeling_type} technique!")
labeling_total_acc, labeling_acc_m, labeling_acc_y = \
    customized_accuracy(combined_hws, joint_hw_train, num_bits + 1)

scaler_std = StandardScaler()
X_train = scaler_std.fit_transform(X_train)
X_test = scaler_std.transform(X_test)

# This part can be replaced with a loop for generating the pool of random models in random model search or run in
# as multiple jobs to speed up the random search if you have more than one gpu.
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
    # if loss_type == "CCE" and dropout == False and labeling_type == 'Mixture':
    #     generate_or_load = "generate"
    #     if not os.path.exists(base_models):
    #         os.mkdir(base_models)
    # else:
    #     generate_or_load = "load"

    # comment this line if you want to do the random search
    generate_or_load = "load_trained"

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
        print("Neural Network type is not supported")
        exit(-1)

    jointed_predicted_hw = np.asarray(jointed_predicted_hw)

    # preds0 = HW(m), preds1 = HW(y)
    preds0 = jointed_predicted_hw // (num_bits + 1)
    preds1 = jointed_predicted_hw % (num_bits + 1)
    predicted_hw = np.array([preds0, preds1])

    # Phase 3: Comparing the empirical distribution with the theoretical joint distribution using maximum likelihood
    # criterion.
    if dataset == "Kyber":
        sr_order = 10
    else:
        sr_order = 1
    # Guessing Entropy parameter.
    ge_rounds = np.zeros((nb_attacks, nb_traces_attack))
    ge = np.zeros(nb_traces_attack)
    success_rate_sum = np.zeros(nb_traces_attack)
    for i in range(nb_attacks):
        # using test set to attack
        idx_trace = np.arange(predicted_hw.shape[1])
        np.random.shuffle(idx_trace)
        mle = attack_blind_log_gpu(predicted_hw[:, idx_trace[:nb_traces_attack]].T, theoretical_histogram,
                                   std=math.sqrt(var_noise))
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
            {"GE": ge, "GE_rounds": ge_rounds, "NTGE": NTGE, "success_rate": success_rate, "accuracy": accuracy,
             "labeling_total_acc": labeling_total_acc, "labeling_acc_m": labeling_acc_m,
             "labeling_acc_y": labeling_acc_y})




