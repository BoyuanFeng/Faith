# Copyright (c) 2020, Zhouxing shi <zhouxingshichn@gmail.com>
# Licenced under the BSD 2-Clause License.

"""
Computer certified bounds for main results
"""

import os, argparse, json
from multiprocessing import Pool

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default=None, required=True)
parser.add_argument("--model", type=str, nargs="*")
parser.add_argument("--p", type=str, nargs="*")
parser.add_argument("--method", type=str, nargs="*")
parser.add_argument("--perturbed_words", type=int, default=1)
parser.add_argument("--samples", type=int, default=10)
parser.add_argument("--suffix", type=str, default="")
parser.add_argument("--num_layers", type=int, default=1)
parser.add_argument("--current_flag", type=int, default=0) # 0 for pytorch, 1 for faith, 2 for tvm, 3 for ansor
args = parser.parse_args()

if len(args.suffix) > 0:
    args.suffix = "_" + args.suffix
if args.perturbed_words == 2:
    max_verify_length = 16
else:
    max_verify_length = 32    

res_all = []

def verify(model, method, p):
    if args.current_flag == 0:
        verifier_type="pytorch"
    elif args.current_flag == 1:
        verifier_type="faith"
    elif args.current_flag == 2:
        verifier_type="tvm"
    elif args.current_flag == 3:
        verifier_type="ansor"

    log = "log_{}_{}_{}_{}_{}_{}{}.txt".format(verifier_type, model, method, p, args.perturbed_words, args.num_layers, args.suffix)
    res = "res_{}_{}_{}_{}_{}_{}{}.json".format(verifier_type, model, method, p, args.perturbed_words, args.num_layers, args.suffix) 
    res_all.append(res)
    cmd = "python main.py --verify --num_layers={} --data={} --dir={} --method={} --p={} --current_flag={}\
            --max_verify_length={} --perturbed_words={} --samples={}\
            --log={} --res={}".format(
                args.num_layers, args.data, model, method, p, args.current_flag,
                max_verify_length, args.perturbed_words, args.samples,
                log, res
            )
    print(cmd)
    os.system(cmd)

for model in args.model:
    for method in args.method:
        for p in args.p:
            verify(model, method, p)

for res in res_all:
    print(res)
