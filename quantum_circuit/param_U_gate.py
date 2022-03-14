from qiskit import QuantumCircuit, QuantumRegister
import numpy as np

def U_flexible(nqubits,params,single_mapping=0,pair_mapping=1,interaction = 'ZZ'):
    """
    U gate defines the feature map circuit produced by feature_map
    Applies a series of rotations parametrised by input data.
    From Havlicek et. al.

    nqubits             -> int, number of qubits, should match features of input data
    params              -> ParameterVector object, each parameter corredponds to a feature in a data point
    single/pair_mapping -> choice of rotation coefficients in the circuit (check single_ and pair_map functions)
    interaction         -> 'ZZ' proposed by Havlicek, 'YY' proposed as a possible extension by Park

    """
    qbits = QuantumRegister(nqubits,'q')
    circuit = QuantumCircuit(qbits)

    #this little library of maps to choose from is loosely based on literature
    #might need updating

    #define some maps for single-qubit gates to choose from
    def single_map(param):
        #this used in Havlicek
        if single_mapping == 0:
            return param
        elif single_mapping == 1:
            return param*nqubits
        elif single_mapping == 2:
            return param*param
        elif single_mapping == 3:
            return param*param*param #note ** does not work for qiskit ParameterVector element objects
        elif single_mapping == 4:
            return param*param*param*param 


    #define some maps for two-qubit gates to choose from
    def pair_map(param1,param2):
        if pair_mapping == 0:
            return param1*param2
        #this used in Havlicek
        elif pair_mapping == 1:
            return (np.pi-param1)*(np.pi-param2)
        elif pair_mapping == 2:
            return (np.pi-(param1*param1))*(np.pi-(param2*param2))
        elif pair_mapping == 3:
            return(np.pi-(param1*param1*param1))*(np.pi-(param2*param2*param2))
        elif pair_mapping == 4:
            return(np.pi-(param1*param1*param1*param1))*(np.pi-(param2*param2*param2*param2))

    #use chosen single-qubit mapping to make a layer of single-qubit gates
    for component in range(nqubits):
        phi_j = single_map(params[component])
        circuit.rz(-2*phi_j,qbits[component])
        
    #use chosen two-qubit mapping to make a layer of 2-qubit gates
    for first_component in range(0,nqubits-1):
        for second_component in range(first_component+1,nqubits):
            #from Havlicek
            phi_ij = pair_map(params[first_component],params[second_component])
            if interaction == 'ZZ':
                circuit.cx(qbits[first_component],qbits[second_component])
                circuit.rz(-2*phi_ij,qbits[second_component])
                circuit.cx(qbits[0],qbits[component])
            #from Park - at the moment without hyperparameter
            if interaction == 'YY':
                circuit.rx(np.pi/2,first_component)
                circuit.rx(np.pi/2,second_component)
                circuit.cx(first_component,second_component)
                circuit.rz(-2*phi_ij,second_component)
                circuit.cx(first_component,second_component)
                circuit.rx(-np.pi/2,first_component)
                circuit.rx(-np.pi/2,second_component)
    return circuit