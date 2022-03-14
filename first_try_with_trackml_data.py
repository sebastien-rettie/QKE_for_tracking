"""
Example of how quantum kernel estimation with a custom kernel can be used to classify data.
Edge data originates from TrackML, pre-processing by Tuysuz et al, adapted for QKE by Marcin Jastrzebski
"""

#general imports
import argparse
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
#qiskit
from qiskit import (Aer,IBMQ)
IBMQ.load_account()
IBMQ.providers()
provider = IBMQ.get_provider(group='open')
from qiskit.utils import QuantumInstance
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.kernels import QuantumKernel
#machine learning (external to qiskit)
from sklearn.svm import SVC
#my own library
from quantum_circuit import (param_U_gate,param_feature_map) #quantum_circuit isn't the most original and descriptive name, easy to mistake with qiskit's stuff


startTime = datetime.now()

#Parser is specific to the edge data provided. Events are numbered from 1000 to 1100. 
#They are divided into train and test, user needs to check whether given event is test or train.
parser = argparse.ArgumentParser(description='Load one (!) event for training and one for testing. Size decides how many edges from the event are used.')
add_arg = parser.add_argument

add_arg('--train_event_number', default=1000)
add_arg('--test_event_number', default=1001)
add_arg('--train_size', default= 1000)
add_arg('--test_size', default = 1000)
args = parser.parse_args()

train_event_number = args.train_event_number
test_event_number = args.test_event_number
train_size = int(args.train_size)
test_size = int(args.test_size)

train_event_features = np.load('TrackML_edges_data/ready_train/tuy_train_features'+str(train_event_number)+'.npy')
test_event_features = np.load('TrackML_edges_data/ready_test/tuy_test_features'+str(test_event_number)+'.npy')
train_event_labels = np.load('TrackML_edges_data/ready_train/tuy_train_labels'+str(train_event_number)+'.npy')
test_event_labels = np.load('TrackML_edges_data/ready_test/tuy_test_labels'+str(test_event_number)+'.npy')

if train_size < np.shape(train_event_features)[0]:
    train_features = train_event_features[:train_size]
    train_labels = train_event_labels[:train_size]
else:
    train_features = train_event_features
    train_labels = train_event_labels

if test_size < np.shape(test_event_features)[0]:
    test_features = test_event_features[:test_size]
    test_labels = test_event_labels[:test_size]
else:
    test_features = test_event_features
    test_labels = test_event_labels

#define what simulator we're using
statevector_backend = QuantumInstance(Aer.get_backend("statevector_simulator"))

#data dimension is 6 - two 3D points which make an edge
data_dim = 6

#one parameter per feature 
params = ParameterVector('phi',data_dim)
my_U = param_U_gate.U_flexible(data_dim,params)
my_map = param_feature_map.feature_map(data_dim,my_U)
my_kernel = QuantumKernel(feature_map=my_map,quantum_instance = statevector_backend) 

#train part
matrix_train = my_kernel.evaluate(x_vec=train_features)
svc = SVC(kernel="precomputed")
svc.fit(matrix_train, train_labels)

#test part 
matrix_test = my_kernel.evaluate(x_vec=test_features, y_vec=train_features)
score = svc.score(matrix_test, test_labels)

time_taken = datetime.now() - startTime

with open('kernel_estimation_times.txt','a') as file:
    file.write(f'Train_size: {train_size}, Test size: {test_size} \n')
    file.write(f'Time taken: {time_taken} \n')
    file.write(f'Kernel classification score: {score} \n')
    file.write('\n')


