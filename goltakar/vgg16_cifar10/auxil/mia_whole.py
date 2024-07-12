import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import numpy as np
import time
import copy
from torch.utils.data import TensorDataset
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import rotate as scipyrotate
from torchvision import datasets, transforms
import random
import matplotlib.pyplot as plt
import time
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
from sklearn import linear_model, model_selection
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
from sklearn import linear_model, model_selection
import torchvision.models as models
from sklearn.cluster import KMeans
import torch.nn.utils.prune as prune
import numpy as np
from typing import Tuple
from scipy import special
from sklearn import metrics
import tensorflow as tf
import tensorflow_privacy
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import membership_inference_attack as mia
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackInputData
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackResultsCollection
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackType
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import PrivacyMetric
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import PrivacyReportMetadata
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import SlicingSpec
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import privacy_report
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import membership_inference_attack as mia
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackInputData
import tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.plotting as plotting
import os



def evaluate_mia(directory, net, train_loader, test_loader,case='proposed'):
    net.to('cpu')
    for param in net.parameters():
        param.requires_grad = False

    logits_train = []
    logits_test = []
    labels_train = []
    labels_test = []

    for data in train_loader:
        inputs, labels = data
        logits = net(inputs)
        logits_train.append(logits)
        labels_train.append(labels)

    for data in test_loader:
        inputs, labels = data
        logits = net(inputs)
        logits_test.append(logits)
        labels_test.append(labels)

    logits_train = torch.cat(logits_train)
    logits_test = torch.cat(logits_test)
    labels_train = torch.cat(labels_train)
    labels_test = torch.cat(labels_test)

    #Apply softmax to get probabilities from logits
    prob_train = torch.softmax(logits_train, dim=1)
    prob_test = torch.softmax(logits_test, dim=1)


    #convert to numpy
    prob_train = prob_train.detach().numpy()
    prob_test = prob_test.detach().numpy()
    labels_train = labels_train.detach().numpy()
    labels_test = labels_test.detach().numpy()
    logits_train = logits_train.detach().numpy()
    logits_test = logits_test.detach().numpy()

    y_train = tf.keras.utils.to_categorical(labels_train, 10)
    y_test = tf.keras.utils.to_categorical(labels_test, 10)


    cce = tf.keras.backend.categorical_crossentropy
    constant = tf.keras.backend.constant

    loss_train = cce(constant(y_train), constant(prob_train), from_logits=False).numpy()
    loss_test = cce(constant(y_test), constant(prob_test), from_logits=False).numpy()

    print('\n')
    attacks_result = mia.run_attacks(
        AttackInputData(
            loss_train = loss_train,
            loss_test = loss_test),
            attack_types=(
                        # AttackType.MULTI_LAYERED_PERCEPTRON,
                        AttackType.LOGISTIC_REGRESSION,
                        ),
            )

    # Save the summary to a variable
    summary_text = attacks_result.summary()
    print(summary_text)
    

    file_path = os.path.join(directory,'mia_global_attack_'+str(case)+'.txt')
    # Save the summary to a text file
    with open(file_path, "w") as file:
        file.write(summary_text)

    # Plot the ROC curve of the best classifier
    fig = plotting.plot_roc_curve(
        attacks_result.get_result_with_max_auc().roc_curve)

    file_path = os.path.join(directory,'mia_global_attack_'+str(case)+'.png')
    fig.savefig(file_path)

    return attacks_result