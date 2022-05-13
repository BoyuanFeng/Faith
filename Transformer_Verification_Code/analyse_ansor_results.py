import os
print("{}\t{}\t{}\t{}\t{}\t{}".format("dataset", "num_layer", "overall_time", "train_time", "measure_time", "other"))
print("sample = 1")
for dataset in ["sst", "yelp"]:
    for num_layer in [1,2]:
        brief_log = "log_ansor_model_{}_{}_no_hidden64_forward_2_1_{}.txt".format(dataset, num_layer, num_layer)
        detail_log = "log_ansor_model_{}_{}_autotuning.txt".format(dataset, num_layer)

        overall_time_log = "time_ansor_model_{}_{}_overall.txt".format(dataset, num_layer)
        measure_time_log = "time_ansor_model_{}_{}_measure.txt".format(dataset, num_layer)
        train_time_log = "time_ansor_model_{}_{}_train.txt".format(dataset, num_layer)

        os.system("grep 'Time' {} &> {}".format(brief_log, overall_time_log))
        os.system("grep 'Time elapsed for measurement' {} &> {}".format(detail_log, measure_time_log))
        os.system("grep 'Time elapsed for training' {} &> {}".format(detail_log, train_time_log))

        overall_time_file = open(overall_time_log)
        lines = overall_time_file.readlines()
        assert(len(lines) == 1)
        overall_time = float(lines[0].split(' ')[-2])

        measure_time_file = open(measure_time_log)
        lines = measure_time_file.readlines()
        measure_time = 0
        for line in lines:
            measure_time += float(lines[0].split(' ')[-2])

        train_time = 0
        train_time_file = open(train_time_log)
        lines = train_time_file.readlines()
        train_time = 0
        for line in lines:
            train_time += float(lines[0].split(' ')[-2])

        print("{}\t{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}".format(dataset, num_layer, overall_time, train_time, measure_time, overall_time-train_time-measure_time))

