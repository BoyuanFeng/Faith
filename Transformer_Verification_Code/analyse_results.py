import os
import numpy as np

times = {}
speed_up = {}
speed_deviation = {}
sample_involved = 1
for dataset in ["sst", "yelp"]:
    for layer_cnt in range(1,7,1):
        times[layer_cnt] = {}
        speed_up[layer_cnt] = {}
        speed_deviation[layer_cnt] = {}
        for kernel_type in ["tvm"]: #["pytorch", "faith"]:
            times[layer_cnt][kernel_type] = []
            log_file_name = "log_{}_model_{}_{}_no_hidden64_forward_2_1_{}.txt".format(kernel_type, dataset, layer_cnt, layer_cnt)
            time_file_name = "time_{}_model_{}_{}_no_hidden64_forward_2_1_{}.txt".format(kernel_type, dataset, layer_cnt, layer_cnt)
            os.system("grep 'Time' {} > {}".format(log_file_name, time_file_name))

            with open(time_file_name) as file:
                lines = file.readlines()
                sample_id = 0
                for line in lines:
                    # print(line.split(' ')[-2])
                    times[layer_cnt][kernel_type].append(float(line.split(' ')[-2]))
                    sample_id += 1
                    if sample_id +1 == sample_involved:
                        break
                # print(len(times[layer_cnt][kernel_type]))
        # speed_up[layer_cnt]["pytorch"] = np.mean(np.array(times[layer_cnt]["pytorch"])/np.array(times[layer_cnt]["pytorch"]))
        speed_up[layer_cnt]["tvm"] = np.mean(np.array(times[layer_cnt]["tvm"]))
        # speed_up[layer_cnt]["faith"] = np.mean(np.array(times[layer_cnt]["pytorch"])/np.array(times[layer_cnt]["faith"]))
        # speed_deviation[layer_cnt]["pytorch"] = np.std(np.array(times[layer_cnt]["pytorch"])/np.array(times[layer_cnt]["pytorch"]))
        # speed_deviation[layer_cnt]["tvm"] = np.std(np.array(times[layer_cnt]["pytorch"])/np.array(times[layer_cnt]["tvm"]))
        # speed_deviation[layer_cnt]["faith"] = np.std(np.array(times[layer_cnt]["pytorch"])/np.array(times[layer_cnt]["faith"]))

    print("A100, latency, {}".format(dataset))
    print("layer_cnt\ttvm")
    for layer_cnt in range(1,7,1):
        print("%d\t%f"%(layer_cnt, speed_up[layer_cnt]["tvm"]))

    # print("\n")
    # print("speed up deviation")
    # print("layer_cnt\tpytorch\tfaith")
    # for layer_cnt in range(1,7,1):
    #     print("%d\t%f\t%f"%(layer_cnt, speed_deviation[layer_cnt]["pytorch"], speed_deviation[layer_cnt]["faith"]))

