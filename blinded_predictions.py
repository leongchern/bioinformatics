"""
    COMPGI10 project

    blinded_predictions.py
    Approximate time to train and produce outputs: 6-8 minutes

    For this file, we remove all print strings and only print results relating to the blinded set
    i.e. print only the locations and their corresponding probability on sequences from the
    blinded set

"""
import itertools
import numpy as np
import pandas as pd
import sklearn.cross_validation
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn import metrics
from sklearn.metrics import confusion_matrix, roc_auc_score
from collections import OrderedDict as od

import matplotlib.pyplot as plt
import seaborn as sn


# ===============================
# load fasta data
# ===============================

def read_fasta(fp):
    name, seq = None, []
    for line in fp:
        line = line.rstrip()
        if line.startswith(">"):
            if name: yield (name, ''.join(seq))
            name, seq = line, []
        else:
            seq.append(line)
    if name: yield (name, ''.join(seq))


def print_statistics(y_true, y_hat):
    print('Accuracy:', metrics.accuracy_score(y_true, y_hat))
    print('Precision (macro):', metrics.precision_score(y_true, y_hat, labels=['1', '2', '3', '4'], average='macro'))
    print('Precision (micro):', metrics.precision_score(y_true, y_hat, labels=['1', '2', '3', '4'], average='micro'))
    print('Precision (weighted):',
          metrics.precision_score(y_true, y_hat, labels=['1', '2', '3', '4'], average='weighted'))
    print('Recall (macro):', metrics.recall_score(y_true, y_hat, labels=['1', '2', '3', '4'], average='macro'))
    print('Recall (micro):', metrics.recall_score(y_true, y_hat, labels=['1', '2', '3', '4'], average='micro'))
    print('Recall (weighted):', metrics.recall_score(y_true, y_hat, labels=['1', '2', '3', '4'], average='weighted'))
    print('F1 (macro):', metrics.f1_score(y_true, y_hat, labels=['1', '2', '3', '4'], average='macro'))
    print('F1 (micro):', metrics.f1_score(y_true, y_hat, labels=['1', '2', '3', '4'], average='micro'))
    print('F1 (weighted):', metrics.f1_score(y_true, y_hat, labels=['1', '2', '3', '4'], average='weighted'))


    
# dump data into numpy matrix
name_vec = []
seq_vec = []
loc_vec = []
blind_name_vec = []
blind_seq_vec = []

with open('../cyto.fasta') as fp:
    c = 0
    for name, seq in read_fasta(fp):
        loc_vec.append(1)
        name_vec.append(name)
        seq_vec.append(seq)
        c += 1
    print('no. of cyto sequences: ', c)
with open('../mito.fasta') as fp:
    c = 0
    for name, seq in read_fasta(fp):
        loc_vec.append(2)
        name_vec.append(name)
        seq_vec.append(seq)
        c += 1
    print('no. of mito sequences: ', c)
with open('../nucleus.fasta') as fp:
    c = 0
    for name, seq in read_fasta(fp):
        loc_vec.append(3)
        name_vec.append(name)
        seq_vec.append(seq)
        c += 1
    print('no. of nucleus sequences: ', c)
with open('../secreted.fasta') as fp:
    c = 0
    for name, seq in read_fasta(fp):
        loc_vec.append(4)
        name_vec.append(name)
        seq_vec.append(seq)
        c += 1
    print('no. of secreted sequences: ', c)

with open('../blind.fasta') as fp:
    c = 0
    for name, seq in read_fasta(fp):
        blind_name_vec.append(name)
        blind_seq_vec.append(seq)
        c += 1
    print('no. of blind sequences: ', c)



# # ===============================
# # analysis of sequence length
# # ===============================
# bins = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]
# bins2 = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
#
# for loc in ['cyto','mito','nucleus','secreted']:
#
#     analysis_seq = []
#     c = 0
#     with open('../' + loc + '.fasta') as fp:
#         for _, seq in read_fasta(fp):
#             analysis_seq.append(seq)
#             c += 1
#         print('no. of sequences: ', c)
#
#     print('\n\n==============================')
#     print(loc)
#     print('==============================')
#     analysis_len_vec = [len(i) for i in analysis_seq]
#     print(np.histogram(analysis_len_vec, bins))
#     print('mean', np.average(analysis_len_vec))
#     print('median', np.median(analysis_len_vec))
#     print('stdev', np.std(analysis_len_vec))
#
#     analysis_len_vec2 = [len(i) for i in analysis_seq if len(i) < 1000]
#     print(np.histogram(analysis_len_vec2, bins2))
#     print('mean', np.average(analysis_len_vec2))
#     print('median', np.median(analysis_len_vec2))
#     print('stdev', np.std(analysis_len_vec2))



# ===============================
# feature engineering: create count of all amino acid molecules in sequence
# ===============================
aminos = 'ARNDCQEGHILKMFPSTWYV'
aminos_vec = list(aminos)

# take out non-amino codes i.e. BUX
fasta_mat = np.array([loc_vec, seq_vec]).T
BUX_rows = [i for i in range(fasta_mat.shape[0]) if
            'B' in fasta_mat[i, 1] or 'U' in fasta_mat[i, 1] or 'X' in fasta_mat[i, 1]]


# take out all rows containing B, U or X
fasta_mat = np.delete(fasta_mat,BUX_rows,axis=0)
seq_vec = fasta_mat[:,-1]
total_letters = sum(len(i) for i in seq_vec)


# get amino freq in a seq for all seqs
aminos_freq_vec = []
for s in seq_vec:
    aminos_freq_vec.append([od((a, s.count(a)) for a in aminos_vec)])

aminos_mat = np.zeros([len(seq_vec), 20])
for c in range(len(aminos_freq_vec)):
    aminos_mat[c,:] = list(aminos_freq_vec[c][0].values())  # [seq_len x 20]


# ===============================
# feature engineering: front/back sequence patterns
# ===============================
# sum([1 for j in [len(s) for s in seq_vec] if j < 100])   # checker
pad_char = '0'
fb100_vec = []
for i, s in enumerate(seq_vec):
    fb100 = ''
    if len(s) < 100:
        fb100 = s + (100-len(s)) * pad_char
    else:
        fb100 = s[:50] + s[-50:]
    fb100_vec.append(fb100)

# work out the amino composition
fb100_aminos_vec = []
for s in fb100_vec:
    fb100_aminos_vec.append([od((a, s.count(a)) for a in aminos_vec)])

fb100_aminos_mat = np.zeros([len(seq_vec), 20])
for c in range(len(fb100_vec)):
    fb100_aminos_mat[c,:] = list(fb100_aminos_vec[c][0].values())  # [seq_len x 20]


# ===============================
# feature engineering: create dipeptide combos
# ===============================

dipeptides = [''.join(_) for _ in list(itertools.product(aminos_vec,repeat=2))]

# create dict of dipeptides for elemements in seq_vec
dipep_freq_vec = []
for s in seq_vec:
    s_dipep = [s[0+i:2+i] for i in range(len(s)-1)]
    dipep_freq_vec.append([od((dp, s_dipep.count(dp)) for dp in dipeptides)])

dipep_count = 0
for i in range(len(seq_vec)):
    dipep_count += sum(dipep_freq_vec[i][0].values())

dipep_freq_vec_normalized = []
for s in seq_vec:
    s_dipep = [s[0+i:2+i] for i in range(len(s)-1)]
    dipep_freq_vec_normalized.append([od((dp, s_dipep.count(dp)/dipep_count) for dp in dipeptides)])

dipep_mat = np.zeros([len(seq_vec), 400])
dipep_mat_normalized = np.zeros([len(seq_vec), 400])
for c in range(len(dipep_freq_vec)):
    dipep_mat[c,:] = list(dipep_freq_vec[c][0].values())    # [seq_len x 400]
    dipep_mat_normalized[c,:] = list(dipep_freq_vec_normalized[c][0].values())    # [seq_len x 400]




# ===============================
# feature engineering: monoisotropic mass
# ===============================

aminos_mass = {
    'A':  71.038,
    'C': 103.009,
    'D': 115.027,
    'E': 129.043,
    'F': 147.068,
    'G':  57.021,
    'H': 137.059,
    'I': 113.084,
    'K': 128.095,
    'L': 113.084,
    'M': 131.040,
    'N': 114.043,
    'P':  97.053,
    'Q': 128.059,
    'R': 156.101,
    'S':  87.032,
    'T': 101.048,
    'V':  99.068,
    'W': 186.079,
    'Y': 163.063
}

amino_mass_vec = []
for s in seq_vec:
    amino_mass_vec.append(sum([aminos_mass.get(c.upper()) for c in list(s)]))
amino_mass_vec = np.matrix(amino_mass_vec).T

# ===============================
# physicochemical properties
# ===============================
# All values in the following physicochemical lists are in the following order:
# Alanine, Arginine, Asparagine, Aspartic Acid, Cysteine, Glumine, Glutamic Acid, Glycine, Histidine, Isoleucine,
# Leucine, Lysine, Methionine, Phenylalanine, Proline, Serine, Threonine, Tryptophan, Tryosine, Valine

# refractivity
phy_re = [4.34, 26.66, 13.28, 12, 35.77, 17.56, 17.26, 0, 21.81, 19.06, 18.78,
          21.29, 21.64, 29.4, 10.93, 6.35, 11.01, 42.53, 31.53, 13.92]
dict_re = dict(zip(list(aminos), phy_re))

# flexibility
phy_fl = [0.357, 0.529, 0.463, 0.511, 0.346, 0.493, 0.497, 0.544, 0.323, 0.462, 0.365,
          0.466, 0.295, 0.314, 0.509, 0.507, 0.444, 0.305, 0.42, 0.386]
dict_fl = dict(zip(list(aminos), phy_fl))

# volume
phy_vo = [91.5, 202, 135.2, 124.5, 117.7, 161.1, 155.1, 66.4, 167.3, 168.8, 167.9,
          171.3, 170.8, 203.4, 129.3, 99.1, 122.1, 237.6, 203.6, 141.7]
dict_vo = dict(zip(list(aminos), phy_vo))

# transfer of free-electron
phy_fe = [-0.2,-0.12,0.08,-0.2,-0.45,0.16,-0.3,0,-0.12,-2.26,-2.46,
          -0.35,-1.47,-2.33,-0.98,-0.39,-0.52,-2.01,-2.24,-1.56]
dict_fe = dict(zip(list(aminos), phy_fe))

# electron-ion interactions
phy_ei = [0.0373,0.0959,0.0036,0.1263,0.0829,0.0761,0.0058,0.005,0.0242,
          0,0,0.0371,0.0823,0.0946,0.0198,0.0829,0.0941,0.0548,0.0516,0.0057]
dict_ei = dict(zip(list(aminos), phy_ei))

# hydrophilicity
phy_hi = [-0.5, 3, 0.2, 3, -1, 0.2, 3, 0, -0.5, -1.8, -1.8,
          3, -1.3, -2.5, 0, 0.3, -0.4, -3.4, -2.3, -1.5]
dict_hi = dict(zip(list(aminos), phy_hi))

# isoelectric
phy_is = [6, 10.76, 5.41, 2.77, 5.05, 5.65, 3.22, 5.97, 7.59, 6.02,
          5.98, 9.74, 5.74, 5.48, 6.3, 5.68, 5.66, 5.89, 5.66, 5.96]
dict_is = dict(zip(list(aminos), phy_is))

# hydrophobicity
phy_ho = [0.25, -1.76, -0.64, -0.72, 0.04, -0.69, -0.62, 0.16, -0.4, 0.73,
          0.53, -1.1, 0.26, 0.61, -0.07, -0.26, -0.18, 0.37, 0.02, 0.54]
dict_ho = dict(zip(list(aminos), phy_ho))

# polarity
phy_po = [0.0000, 52.000, 3.3800, 40.700, 1.4800, 3.5300, 49.910, 0.0000, 51.600, 0.1500,
          0.4500, 49.500, 1.4300, 0.3500, 1.5800, 1.6700, 1.6600, 2.1000, 1.6100, 0.1300]
dict_po = dict(zip(list(aminos), phy_po))

# pk of side chain
phy_pk = [0.0000,12.480,0.0000,3.6500,8.1800,0.0000,4.2500,0.0000,6.0000,0.0000,
          0.0000,10.530,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,10.700,0.0000]
dict_pk = dict(zip(list(aminos), phy_pk))

# construct vecs for various physicochemical properties
phy_re_vec = [[dict_re.get(c) for c in s] for s in seq_vec] # refractivity
phy_fl_vec = [[dict_fl.get(c) for c in s] for s in seq_vec] # flexibility
phy_vo_vec = [[dict_vo.get(c) for c in s] for s in seq_vec] # volume
phy_fe_vec = [[dict_fe.get(c) for c in s] for s in seq_vec] # trans of free-electron
phy_ei_vec = [[dict_ei.get(c) for c in s] for s in seq_vec] # electron-ion interactions
phy_hi_vec = [[dict_hi.get(c) for c in s] for s in seq_vec] # hydrophilicity
phy_is_vec = [[dict_is.get(c) for c in s] for s in seq_vec] # isoelectric
phy_ho_vec = [[dict_ho.get(c) for c in s] for s in seq_vec] # hydrophobicity
phy_po_vec = [[dict_po.get(c) for c in s] for s in seq_vec] # polarity
phy_pk_vec = [[dict_pk.get(c) for c in s] for s in seq_vec] # pk of side potential

# combine all physicochemical properties into a matrix
phy_mat = np.array([phy_re_vec, phy_fl_vec, phy_vo_vec, phy_fe_vec, phy_ei_vec,
                    phy_hi_vec, phy_is_vec, phy_ho_vec, phy_po_vec, phy_pk_vec]).T

phy_mat = np.array([phy_is_vec])

# ===============================
# calculate autocorrelations
# ===============================

acorrel_mat = np.zeros([len(seq_vec),10])

for k in range(10):
    for c, vec in enumerate(phy_mat[:,k]):
        tmp_sum = 0
        vec_len = len(vec)
        for p in range(vec_len-k):
            tmp_sum += vec[p] * vec[p+k]
        acorrel_mat[c, k-1] = (1/(vec_len-(k+1))) * tmp_sum     # [seq_len x 10]


# ===============================
# assembling feature matrix
# ===============================

# forming the combined feature vector
res = [i for i in fasta_mat[:,0]]
res_mat = np.matrix(res).T

feature_choice = 5
if feature_choice == 1:
    feature_matrix = np.hstack([res_mat, aminos_mat])
elif feature_choice == 2:
    feature_matrix = np.hstack([res_mat, aminos_mat, fb100_aminos_mat])
elif feature_choice == 3:
    feature_matrix = np.hstack([res_mat, aminos_mat, fb100_aminos_mat, amino_mass_vec])
elif feature_choice == 4:
    feature_matrix = np.hstack([res_mat, aminos_mat, fb100_aminos_mat, amino_mass_vec, dipep_mat])
elif feature_choice == 5:
    feature_matrix = np.hstack([res_mat, aminos_mat, fb100_aminos_mat, amino_mass_vec, dipep_mat, acorrel_mat])
else:
    feature_matrix = np.hstack([res_mat, aminos_mat, fb100_aminos_mat, amino_mass_vec, dipep_mat, acorrel_mat])

np.random.seed(8)
np.random.shuffle(feature_matrix)

# subsetting data sets ... we let sklearn perform the cross validation for us
train_x = feature_matrix[:8000, 1:].astype('float32')
train_y = feature_matrix[:8000, 0]
test_x = feature_matrix[8000:, 1:].astype('float32')
test_y = feature_matrix[8000:, 0]




train_y = train_y.tolist()
train_y = [i[0] for i in train_y]
test_y = test_y.tolist()
test_y = np.array([i[0] for i in test_y])

# ===============================
# K-NN
# ===============================
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(train_x, train_y)
yhat_knn = knn.predict(test_x)
# print('Accuracy on 3-NN:',metrics.accuracy_score(test_y, yhat_knn))
# print_statistics(test_y, yhat_knn)

# ===============================
# multiclass logistic classification
# ===============================

# lr = LogisticRegression().fit(train_x, train_y)
# yhat = lr.predict(test_x)
# sum(yhat == test_y)/len(test_y)

lrCV = LogisticRegressionCV(penalty='l2', cv=5).fit(train_x,train_y)
yhat_lr = lrCV.predict(test_x)
# print('Accuracy on Cross-Validated Log Reg:',metrics.accuracy_score(test_y, yhat_lr))


# prediction on blind set
blind_pred_matrix = lrCV.predict_proba(test_x[:10]) # change this later
blind_pred_prob = np.max(blind_pred_matrix, axis=1)
blind_pred_label = np.argmax(blind_pred_matrix, axis=1)

# # analytics part
# print_statistics(test_y, yhat_lr)
# metrics.classification_report(test_y, yhat_lr)

# ===============================
# multiclass support vector classification - SVC
# ===============================
# clf = sklearn.svm.SVC(decision_function_shape='ovo')
# clf.fit(train_x, train_y)
# yhat = clf.predict(test_x)
# print('Accuracy on Cross-Validated Log Reg:',metrics.accuracy_score(test_y, yhat))
#
# lin_clf = LinearSVC()
# lin_clf.fit(train_x, train_y)
# yhat = lin_clf.predict(test_x)
# print('Accuracy on Cross-Validated Log Reg:',metrics.accuracy_score(test_y, yhat))

# ===============================
# multiclass random forest
# ===============================
# # Optimum leaf size to use is 5

# rf_leaf_size = [1, 5, 10, 50, 100, 200, 500]
# for leaf_size in rf_leaf_size:
#     rf = RandomForestClassifier(n_estimators=500, oob_score=True,
#                                 max_features='auto', min_samples_leaf=5)
#     rf.fit(train_x, train_y)
#     yhat = rf.predict(test_x)
#     print('RF with leaf size', leaf_size, sum(yhat == test_y) / len(test_y))

rf = RandomForestClassifier(n_estimators=1000, oob_score=True,
                            max_features='auto', min_samples_leaf=5)
rf.fit(train_x, train_y)
yhat_rf = rf.predict(test_x)
# print('Accuracy on Random Forest:', metrics.accuracy_score(test_y, yhat_rf))

# # analytics for RF
# print_statistics(test_y, yhat_rf)


# ===============================
# voting classifer ensemble - 2 Classifiers (FINAL MODEL)
# ===============================

voting_ensemble2 = VotingClassifier(estimators=[
        ('lr', lrCV), ('rf', rf)], voting='soft', weights = [1,1])
voting_ensemble2 = voting_ensemble2.fit(train_x, train_y)
yhat_ve2 = voting_ensemble2.predict(test_x)

# # print statistics
# metrics.accuracy_score(test_y, yhat_ve2)
# print_statistics(test_y, yhat_ve2)
#
# confusion = confusion_matrix(test_y, yhat_ve2)
# plt.figure()
# sn.heatmap(confusion, annot=True, fmt='d', linewidths=.5,
#            xticklabels=['cyto', 'mito','nucleus','secreted'],
#            yticklabels=['cyto', 'mito','nucleus','secreted'])
# plt.title('Predicting Protein Locations using Voting Classifier Ensemble')
# plt.savefig('cm_VE2.png')
# plt.close('all')



# =========================================================================================================
# Running model on blinded samples
# =========================================================================================================

fasta_mat_blinded = np.array([blind_seq_vec]).T
# BUX_rows_blinded is empty since none of the sequnece contains 'B', 'U' or 'X'
BUX_rows_blinded = [i for i in range(20) if
                    'B' in blind_seq_vec[i] or
                    'U' in blind_seq_vec[i] or
                    'X' in blind_seq_vec[i]]

blind_seq_vec = fasta_mat_blinded[:,-1]
total_letters = sum(len(i) for i in blind_seq_vec)

# ===============================
# feature engineering: create count of all amino acid molecules in sequence
# ===============================
# get amino freq in a seq for all seqs
aminos_freq_vec = []
for s in blind_seq_vec:
    aminos_freq_vec.append([od((a, s.count(a)) for a in aminos_vec)])

aminos_mat = np.zeros([len(blind_seq_vec), 20])
for c in range(len(aminos_freq_vec)):
    aminos_mat[c,:] = list(aminos_freq_vec[c][0].values())  # [seq_len x 20]

# ===============================
# feature engineering: front/back sequence patterns
# ===============================
# sum([1 for j in [len(s) for s in blind_seq_vec] if j < 100])   # checker
pad_char = '0'
fb100_vec = []
for i, s in enumerate(blind_seq_vec):
    fb100 = ''
    if len(s) < 100:
        fb100 = s + (100-len(s)) * pad_char
    else:
        fb100 = s[:50] + s[-50:]
    fb100_vec.append(fb100)

# work out the amino composition
fb100_aminos_vec = []
for s in fb100_vec:
    fb100_aminos_vec.append([od((a, s.count(a)) for a in aminos_vec)])

fb100_aminos_mat = np.zeros([len(blind_seq_vec), 20])
for c in range(len(fb100_vec)):
    fb100_aminos_mat[c,:] = list(fb100_aminos_vec[c][0].values())  # [seq_len x 20]




# ===============================
# feature engineering: monoisotropic mass
# ===============================

amino_mass_vec = []
for s in blind_seq_vec:
    amino_mass_vec.append(sum([aminos_mass.get(c.upper()) for c in list(s)]))
amino_mass_vec = np.matrix(amino_mass_vec).T


# ===============================
# feature engineering: create dipeptide combos
# ===============================

# create dict of dipeptides for elemements in blind_seq_vec
dipep_freq_vec = []
for s in blind_seq_vec:
    s_dipep = [s[0+i:2+i] for i in range(len(s)-1)]
    dipep_freq_vec.append([od((dp, s_dipep.count(dp)) for dp in dipeptides)])

dipep_count = 0
for i in range(len(blind_seq_vec)):
    dipep_count += sum(dipep_freq_vec[i][0].values())

dipep_freq_vec_normalized = []
for s in blind_seq_vec:
    s_dipep = [s[0+i:2+i] for i in range(len(s)-1)]

dipep_mat = np.zeros([len(blind_seq_vec), 400])
for c in range(len(dipep_freq_vec)):
    dipep_mat[c,:] = list(dipep_freq_vec[c][0].values())    # [seq_len x 400]
    

# ===============================
# physicochemical properties
# ===============================
# construct vecs for various physicochemical properties
phy_re_vec = [[dict_re.get(c) for c in s] for s in blind_seq_vec] # refractivity
phy_fl_vec = [[dict_fl.get(c) for c in s] for s in blind_seq_vec] # flexibility
phy_vo_vec = [[dict_vo.get(c) for c in s] for s in blind_seq_vec] # volume
phy_fe_vec = [[dict_fe.get(c) for c in s] for s in blind_seq_vec] # trans of free-electron
phy_ei_vec = [[dict_ei.get(c) for c in s] for s in blind_seq_vec] # electron-ion interactions
phy_hi_vec = [[dict_hi.get(c) for c in s] for s in blind_seq_vec] # hydrophilicity
phy_is_vec = [[dict_is.get(c) for c in s] for s in blind_seq_vec] # isoelectric
phy_ho_vec = [[dict_ho.get(c) for c in s] for s in blind_seq_vec] # hydrophobicity
phy_po_vec = [[dict_po.get(c) for c in s] for s in blind_seq_vec] # polarity
phy_pk_vec = [[dict_pk.get(c) for c in s] for s in blind_seq_vec] # pk of side potential

# combine all physicochemical properties into a matrix
phy_mat = np.array([phy_re_vec, phy_fl_vec, phy_vo_vec, phy_fe_vec, phy_ei_vec,
                    phy_hi_vec, phy_is_vec, phy_ho_vec, phy_po_vec, phy_pk_vec]).T

phy_mat = np.array([phy_is_vec])

# ===============================
# calculate autocorrelations
# ===============================

acorrel_mat = np.zeros([len(blind_seq_vec),10])

for k in range(10):
    for c, vec in enumerate(phy_mat[:,k]):
        tmp_sum = 0
        vec_len = len(vec)
        for p in range(vec_len-k):
            tmp_sum += vec[p] * vec[p+k]
        acorrel_mat[c, k-1] = (1/(vec_len-(k+1))) * tmp_sum     # [seq_len x 10]

feature_matrix_blinded = np.hstack([aminos_mat, fb100_aminos_mat, amino_mass_vec, dipep_mat, acorrel_mat])
feature_matrix_blinded = feature_matrix_blinded.astype('float32')

loc_dict = dict({'1':'Cyto', '2':'Mito', '3':'Nucl', '4':'Secr'})
blinded_proba_mat = voting_ensemble2.predict_proba(feature_matrix_blinded)
blinded_proba_max = np.max(blinded_proba_mat,axis=1)
blinded_pred_loc_digits = voting_ensemble2.predict(feature_matrix_blinded)
blinded_pred_loc_vec = [loc_dict.get(i) for i in blinded_pred_loc_digits]
blinded_preds = list(zip(blind_name_vec, blinded_pred_loc_vec, blinded_proba_max))
print('\n===========================================')
print('protein location prediction')
print('===========================================')
for i in blinded_preds:
    print("{}{}{}{}{:.1f}{}".format(i[0][1:], ' ', i[1], ' Confidence ', i[2]*100,'%'))

# avereage prediction confidence
print('\n\n===========================================')
print('protein counts, location & avg confidence')
print('===========================================')
for loco in loc_dict.values():
    loco_count = sum([1 for i in blinded_preds if i[1] == loco])
    loco_avg_prob = np.mean([i[2] for i in blinded_preds if i[1] == loco])
    print(loco_count, loco, loco_avg_prob)




