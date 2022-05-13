import numpy as np
from code_generator import generate_code, profiling_template, config_template
from expert_knowledge_metafile import a100_metafile, Heuristics
from cost_model import predict, train_model, collect_dataset
from profiler import profile_performance

# Number of autotuning roudns
NUM_AUTOTUNING_ROUNDS = 5
# Number of samples per round
NUM_SAMPLES = 4
# Number of Features
NUM_FEATURES = 11

SKIPPED_CONFIG = 0

CHECKED_CONFIGs = set({})

def sample_implementation_parameters(heuristics):
    parameters = heuristics.propose_parameters()
    while not heuristics.is_valid(parameters):
        parameters = heuristics.propose_parameters()
    return parameters

def sample_parameters_without_ML(heuristics, num_samples, num_features):
    global CHECKED_CONFIGs
    metafile_parameters = heuristics.get_feature_from_metafile()
    
    sampled_parameter = np.zeros((0,num_features))
    while sampled_parameter.shape[0] < num_samples:        
        implementation_parameters = sample_implementation_parameters(heuristics)
        implementation_parameters = heuristics.get_feature_from_parameters(implementation_parameters)
        parameter = implementation_parameters + metafile_parameters
        parameter_str = str(parameter)
        print("parameter_str: ", parameter_str)
        if parameter_str in CHECKED_CONFIGs:
            continue
        else:
            parameter = np.array(parameter)
            CHECKED_CONFIGs.add(parameter_str)
            sampled_parameter = np.vstack([sampled_parameter, parameter])
    return sampled_parameter

def sample_parameters_with_ML(heuristics, num_samples, num_features, model):
    metafile_parameters = heuristics.get_feature_from_metafile()
    
    CONFIG_THIS_ROUND = set({})

    sampled_parameter = np.zeros((0,num_features))
    while sampled_parameter.shape[0] < num_samples*10:
        implementation_parameters = sample_implementation_parameters(heuristics)
        implementation_parameters = heuristics.get_feature_from_parameters(implementation_parameters)
        parameter = implementation_parameters + metafile_parameters
        parameter_str = str(parameter)
        print("parameter_str: ", parameter_str)
        if parameter_str in CHECKED_CONFIGs or parameter_str in CONFIG_THIS_ROUND:
            continue
        else:
            CONFIG_THIS_ROUND.add(parameter_str)
            parameter = np.array(parameter)
            sampled_parameter = np.vstack([sampled_parameter, parameter])
    predicted_performance = []
    for i in range(10*num_samples):
        cur_parameter = sampled_parameter[i]
        cur_parameter = cur_parameter.reshape((1, num_features))
        perf = predict(model, cur_parameter)[0]
        predicted_performance.append(perf)
    
    predicted_performance = np.array(predicted_performance)
    top_idx = np.argsort(predicted_performance)[-num_samples:]
    selected_parameters = sampled_parameter[top_idx]

    print("selected_parameters.shape: ", selected_parameters.shape)
    for i in range(selected_parameters.shape[0]):
        parameter_str = str(selected_parameters[i])
        print("parameter_str: ", parameter_str, ", CHECKED_CONFIGs: ", CHECKED_CONFIGs)
        CHECKED_CONFIGs.add(parameter_str)
    return selected_parameters


def generate_code_and_profile(parameters, input_shape):
    # Generate configuration code
    generate_code(
        config_template,
        parameter=parameters,
        file_name="matmul_config.h",
        )
    
    # Generate profiling code
    generate_code(
        profiling_template,
        parameter=input_shape,
        file_name="cuda_profiler.cu",
    )

    latency = profile_performance()
    return latency


def autotuner(input_shape):
    global SKIPPED_CONFIG
    heuristics = Heuristics(a100_metafile)
    features = np.zeros((0, NUM_FEATURES))
    labels = np.array([])

    # Autotuning for NUM_AUTOTUNING_ROUNDS
    for round_idx in range(NUM_AUTOTUNING_ROUNDS):
        print("round: ", round_idx)
        if round_idx == 0:
            sampled_parameters = sample_parameters_without_ML(
                heuristics, 
                num_samples=NUM_SAMPLES,
                num_features=NUM_FEATURES,
                )
        else:
            sampled_parameters = sample_parameters_with_ML(
                heuristics,
                num_samples=NUM_SAMPLES,
                num_features=NUM_FEATURES,
                model=model,
            )
        profiled_latency = []
        for i in range(NUM_SAMPLES):
            parameter = sampled_parameters[i][:7]
            parameter = tuple(parameter)
            print(parameter)
            try:
                latency = generate_code_and_profile(parameter, input_shape)
                profiled_latency.append(latency)
            except:
                SKIPPED_CONFIG += 1
                print("SKIPPED_CONFIG: ", SKIPPED_CONFIG)
        
        # Append newly sampled (features, labels) to previous (features, labels)
        (features, labels) = collect_dataset(
            features=sampled_parameters,
            labels=profiled_latency,
            existing_feature=features,
            existing_label=labels,
            num_feature=NUM_FEATURES,
        )

        print("features: ", features, ", labels: ", labels)

        model = train_model(features, labels)
        array_labels = np.array(labels)
        top_idx = np.argsort(array_labels)[-1:]
        print("round: ", round_idx, ", best parameter:", features[top_idx], ", measured latency: ", array_labels[top_idx])

# [batch_size, length, dim_out, dim_y_out, dim_in] 
input_shape = (1, 128, 64, 64, 64)
autotuner(input_shape)
