# Copyright (c) 2020, Zhouxing shi <zhouxingshichn@gmail.com>
# Licenced under the BSD 2-Clause License.

import torch
import math, time, random, copy, os
from Verifiers import Verifier
from Verifiers.Bounds import Bounds
from Verifiers.utils import check

import tvm
from tvm.contrib import graph_executor

# can only accept one example in each batch
class VerifierForward(Verifier):
    def __init__(self, args, target, logger):
        super(VerifierForward, self).__init__(args, target, logger)
        self.ibp = args.method == "ibp"
        self.tvm_model_lut = {}
        self.ansor_model_lut = {}
        self.current_flag = args.current_flag # 0 for pytorch, 1 for faith, 2 for tvm, 3 for ansor
        if self.current_flag == 2:
            self.init_tvm_model_lut()
        # print("16: self.tvm_model_lut=", self.tvm_model_lut)

    def init_tvm_model_lut(self):
        dev = tvm.cuda(0)
        # file_names = ['deploy_matmul_2_256_1_1_256_256.so', 'deploy_matmul_256_256_1_1_256_256.so', 'deploy_matmul_256_256_1_15_256_256.so', 'deploy_matmul_256_256_1_16_256_256.so', 'deploy_matmul_256_256_1_18_256_256.so', 'deploy_matmul_256_256_1_20_256_256.so', 'deploy_matmul_256_256_1_29_256_256.so', 'deploy_matmul_256_256_1_7_256_256.so', 'deploy_matmul_256_512_1_15_256_512.so', 'deploy_matmul_256_512_1_16_256_512.so', 'deploy_matmul_256_512_1_18_256_512.so', 'deploy_matmul_256_512_1_20_256_512.so', 'deploy_matmul_256_512_1_29_256_512.so', 'deploy_matmul_256_512_1_7_256_512.so', 'deploy_matmul_512_256_1_15_256_256.so', 'deploy_matmul_512_256_1_16_256_256.so', 'deploy_matmul_512_256_1_18_256_256.so', 'deploy_matmul_512_256_1_20_256_256.so', 'deploy_matmul_512_256_1_29_256_256.so', 'deploy_matmul_512_256_1_7_256_256.so']
        file_names = ['deploy_matmul_128_128_1_128_128_128.so',  'deploy_matmul_256_512_1_16_256_512.so',  'deploy_matmul_512_256_1_29_256_256.so',\
            'deploy_matmul_128_128_1_16_128_128.so',   'deploy_matmul_256_512_1_17_256_512.so',  'deploy_matmul_512_256_1_31_256_256.so',\
            'deploy_matmul_128_128_1_2_128_128.so',    'deploy_matmul_256_512_1_18_256_512.so',  'deploy_matmul_512_256_1_32_256_256.so',\
            'deploy_matmul_128_128_1_32_128_128.so',   'deploy_matmul_256_512_1_20_256_512.so',  'deploy_matmul_512_256_1_7_256_256.so',\
            'deploy_matmul_128_128_1_4_128_128.so',    'deploy_matmul_256_512_1_21_256_512.so',  'deploy_matmul_512_256_1_8_256_256.so',\
            'deploy_matmul_128_128_1_64_128_128.so',   'deploy_matmul_256_512_1_22_256_512.so',  'deploy_matmul_512_256_1_9_256_256.so',\
            'deploy_matmul_128_128_1_8_128_128.so',    'deploy_matmul_256_512_1_23_256_512.so',  'deploy_matmul_512_512_1_16_512_512.so',\
            'deploy_matmul_256_256_1_10_256_256.so',   'deploy_matmul_256_512_1_24_256_512.so',  'deploy_matmul_512_64_1_12_64_64.so',\
            'deploy_matmul_256_256_1_11_256_256.so',   'deploy_matmul_256_512_1_25_256_512.so',  'deploy_matmul_512_64_1_16_64_64.so',\
            'deploy_matmul_256_256_1_13_256_256.so',   'deploy_matmul_256_512_1_27_256_512.so',  'deploy_matmul_512_64_1_17_64_64.so',\
            'deploy_matmul_256_256_1_14_256_256.so',   'deploy_matmul_256_512_1_28_256_512.so',  'deploy_matmul_512_64_1_18_64_64.so',\
            'deploy_matmul_256_256_1_15_256_256.so',   'deploy_matmul_256_512_1_29_256_512.so',  'deploy_matmul_512_64_1_20_64_64.so',\
            'deploy_matmul_256_256_1_16_256_256.so',   'deploy_matmul_256_512_1_31_256_512.so',  'deploy_matmul_512_64_1_22_64_64.so',\
            'deploy_matmul_256_256_1_17_256_256.so',   'deploy_matmul_256_512_1_32_256_512.so',  'deploy_matmul_512_64_1_27_64_64.so',\
            'deploy_matmul_256_256_1_18_256_256.so',   'deploy_matmul_256_512_1_7_256_512.so',   'deploy_matmul_512_64_1_8_64_64.so',\
            'deploy_matmul_256_256_1_20_256_256.so',   'deploy_matmul_256_512_1_8_256_512.so',   'deploy_matmul_640_640_1_16_640_640.so',\
            'deploy_matmul_256_256_1_21_256_256.so',   'deploy_matmul_256_512_1_9_256_512.so',   'deploy_matmul_64_512_1_12_64_512.so',\
            'deploy_matmul_256_256_1_22_256_256.so',   'deploy_matmul_384_384_1_16_384_384.so',  'deploy_matmul_64_512_1_16_64_512.so',\
            'deploy_matmul_256_256_1_23_256_256.so',   'deploy_matmul_512_256_1_10_256_256.so',  'deploy_matmul_64_512_1_17_64_512.so',\
            'deploy_matmul_256_256_1_24_256_256.so',   'deploy_matmul_512_256_1_11_256_256.so',  'deploy_matmul_64_512_1_18_64_512.so',\
            'deploy_matmul_256_256_1_25_256_256.so',   'deploy_matmul_512_256_1_13_256_256.so',  'deploy_matmul_64_512_1_20_64_512.so',\
            'deploy_matmul_256_256_1_27_256_256.so',   'deploy_matmul_512_256_1_14_256_256.so',  'deploy_matmul_64_512_1_22_64_512.so',\
            'deploy_matmul_256_256_1_28_256_256.so',   'deploy_matmul_512_256_1_15_256_256.so',  'deploy_matmul_64_512_1_27_64_512.so',\
            'deploy_matmul_256_256_1_29_256_256.so',   'deploy_matmul_512_256_1_16_256_256.so',  'deploy_matmul_64_512_1_8_64_512.so',\
            'deploy_matmul_256_256_1_31_256_256.so',   'deploy_matmul_512_256_1_17_256_256.so',  'deploy_matmul_64_64_1_12_64_64.so',\
            'deploy_matmul_256_256_1_32_256_256.so',   'deploy_matmul_512_256_1_18_256_256.so',  'deploy_matmul_64_64_1_16_64_64.so',\
            'deploy_matmul_256_256_1_7_256_256.so',    'deploy_matmul_512_256_1_20_256_256.so',  'deploy_matmul_64_64_1_17_64_64.so',\
            'deploy_matmul_256_256_1_8_256_256.so',    'deploy_matmul_512_256_1_21_256_256.so',  'deploy_matmul_64_64_1_18_64_64.so',\
            'deploy_matmul_256_256_1_9_256_256.so',    'deploy_matmul_512_256_1_22_256_256.so',  'deploy_matmul_64_64_1_20_64_64.so',\
            'deploy_matmul_256_512_1_10_256_512.so',   'deploy_matmul_512_256_1_23_256_256.so',  'deploy_matmul_64_64_1_22_64_64.so',\
            'deploy_matmul_256_512_1_11_256_512.so',   'deploy_matmul_512_256_1_24_256_256.so',  'deploy_matmul_64_64_1_27_64_64.so',\
            'deploy_matmul_256_512_1_13_256_512.so',   'deploy_matmul_512_256_1_25_256_256.so',  'deploy_matmul_64_64_1_8_64_64.so',\
            'deploy_matmul_256_512_1_14_256_512.so',   'deploy_matmul_512_256_1_27_256_256.so',\
            'deploy_matmul_256_512_1_15_256_512.so',   'deploy_matmul_512_256_1_28_256_256.so']
        for file_name in file_names:
            if os.path.exists(os.getcwd()+'/tvm_model/'+file_name):
                lib = tvm.runtime.load_module(os.getcwd()+'/tvm_model/'+file_name)
                m = graph_executor.GraphModule(lib["default"](dev))
                self.tvm_model_lut[file_name] = m
                print("tvm model %s loaded"%(file_name))
    
    def init_ansor_model_lut(self):
        dev = tvm.cuda(0)
        # file_names = ['deploy_matmul_256_256_1_20_256_256.so','deploy_matmul_256_256_1_23_256_256.so','deploy_matmul_256_256_1_27_2[56_256.so','deploy_matmul_256_256_1_9_256_256.so','deploy_matmul_256_512_1_20_256_512.so','deploy_matmul_256_[512_1_23_256_512.so','deploy_matmul_256_512_1_27_256_512.so','deploy_matmul_256_512_1_9_256_512.so','deploy_m[atmul_512_256_1_20_256_256.so','deploy_matmul_512_256_1_23_256_256.so','deploy_matmul_512_256_1_27_256_256.so[','deploy_matmul_512_256_1_9_256_256.so','deploy_matmul_512_64_1_20_64_64.[so','deploy_matmul_64_512_1_20_64_512.so','deploy_matmul_64_64_1_20_64_64.so'] #['deploy_matmul_256_256_1_20_256_256.so']
        file_names = ['deploy_matmul_128_128_1_128_128_128.so',  'deploy_matmul_256_256_1_7_256_256.so',   'deploy_matmul_512_256_1_18_256_256.so',\
            'deploy_matmul_128_128_1_16_128_128.so',   'deploy_matmul_256_256_1_9_256_256.so',   'deploy_matmul_512_256_1_20_256_256.so',\
            'deploy_matmul_128_128_1_2_128_128.so',    'deploy_matmul_256_512_1_15_256_512.so',  'deploy_matmul_512_256_1_23_256_256.so',\
            'deploy_matmul_128_128_1_32_128_128.so',   'deploy_matmul_256_512_1_16_256_512.so',  'deploy_matmul_512_256_1_27_256_256.so',\
            'deploy_matmul_128_128_1_4_128_128.so',    'deploy_matmul_256_512_1_18_256_512.so',  'deploy_matmul_512_256_1_29_256_256.so',\
            'deploy_matmul_128_128_1_64_128_128.so',   'deploy_matmul_256_512_1_20_256_512.so',  'deploy_matmul_512_256_1_7_256_256.so',\
            'deploy_matmul_128_128_1_8_128_128.so',    'deploy_matmul_256_512_1_23_256_512.so',  'deploy_matmul_512_256_1_9_256_256.so',\
            'deploy_matmul_256_256_1_15_256_256.so',   'deploy_matmul_256_512_1_27_256_512.so',  'deploy_matmul_512_512_1_16_512_512.so',\
            'deploy_matmul_256_256_1_16_256_256.so',   'deploy_matmul_256_512_1_29_256_512.so',  'deploy_matmul_512_64_1_20_64_64.so',\
            'deploy_matmul_256_256_1_18_256_256.so',   'deploy_matmul_256_512_1_7_256_512.so',   'deploy_matmul_640_640_1_16_640_640.so',\
            'deploy_matmul_256_256_1_20_256_256.so',   'deploy_matmul_256_512_1_9_256_512.so',   'deploy_matmul_64_512_1_20_64_512.so',\
            'deploy_matmul_256_256_1_23_256_256.so',   'deploy_matmul_384_384_1_16_384_384.so',  'deploy_matmul_64_64_1_16_64_64.so',\
            'deploy_matmul_256_256_1_27_256_256.so',   'deploy_matmul_512_256_1_15_256_256.so',  'deploy_matmul_64_64_1_20_64_64.so',\
            'deploy_matmul_256_256_1_29_256_256.so',   'deploy_matmul_512_256_1_16_256_256.so']
        for file_name in file_names:
            if os.path.exists(os.getcwd()+'/ansor_model/'+file_name):
                lib = tvm.runtime.load_module(os.getcwd()+'/ansor_model/'+file_name)
                m = graph_executor.GraphModule(lib["default"](dev))
                self.ansor_model_lut[file_name] = m
                print("ansor model %s loaded"%(file_name))

    def verify_safety(self, example, embeddings, index, eps):
        errorType = OSError if self.debug else AssertionError

        try:
            with torch.no_grad():
                bounds = self._bound_input(embeddings, index=index, eps=eps) # hard-coded yet

                check("embedding", bounds=bounds, std=self.std["embedding_output"], verbose=self.verbose)
            
                if self.verbose:
                    bounds.print("embedding")

                for i, layer in enumerate(self.encoding_layers):
                    attention_scores, attention_probs, bounds = self._bound_layer(bounds, layer)

                    check("layer %d attention_scores" % i, 
                        bounds=attention_scores, std=self.std["attention_scores"][i][0], verbose=self.verbose)
                    check("layer %d attention_probs" % i, 
                        bounds=attention_probs, std=self.std["attention_probs"][i][0], verbose=self.verbose)
                    check("layer %d" % i, bounds=bounds, std=self.std["encoded_layers"][i], verbose=self.verbose)
                    
                bounds = self._bound_pooling(bounds, self.pooler)
                check("pooled output", bounds=bounds, std=self.std["pooled_output"], verbose=self.verbose)

                safety = self._bound_classifier(bounds, self.classifier, example["label"])

                return safety
        except errorType as err: # for debug
            if self.verbose:
                print("Warning: failed assertion")
                print(err)
            print("Warning: failed assertion", eps)
            return False

    def _bound_input(self, embeddings, index, eps):
        length, dim = embeddings.shape[1], embeddings.shape[2]

        w = torch.zeros((length, dim * self.perturbed_words, dim)).to(self.device)
        b = embeddings[0]   
        lb, ub = b, b.clone()     
        
        if self.perturbed_words == 1:
            if self.ibp:
                lb[index], ub[index] = lb[index] - eps, ub[index] + eps
            else:
                w[index] = torch.eye(dim).to(self.device)
        else:
            if self.ibp:
                for i in range(self.perturbed_words):
                    lb[index[i]], ub[index[i]] = lb[index[i]] - eps, ub[index[i]] + eps
            else:
                for i in range(self.perturbed_words):
                    w[index[i], (dim * i):(dim * (i + 1)), :] = torch.eye(dim).to(self.device)
            
        lw = w.unsqueeze(0)
        uw = lw.clone()
        lb = lb.unsqueeze(0)
        ub = ub.unsqueeze(0)

        bounds = Bounds(self.args, self.p, eps, lw=lw, lb=lb, uw=uw, ub=ub)

        bounds = bounds.layer_norm(self.embeddings.LayerNorm, self.layer_norm)

        return bounds

    def _bound_layer(self, bounds_input, layer):
        start_time = time.time()

        # main self-attention
        attention_scores, attention_probs, attention = \
            self._bound_attention(bounds_input, layer.attention)

        attention = attention.dense(layer.attention.output.dense, self.tvm_model_lut, FLAG=self.current_flag)
        attention = attention.add(bounds_input).layer_norm(layer.attention.output.LayerNorm, self.layer_norm)
        del(bounds_input)

        if self.verbose:
            attention.print("after attention layernorm")
            attention.dense(layer.intermediate.dense, self.tvm_model_lut, FLAG=self.current_flag).print("intermediate pre-activation")
            print("dense norm", torch.norm(layer.intermediate.dense.weight, p=self.p))
            start_time = time.time()

        intermediate = attention.dense(layer.intermediate.dense, self.tvm_model_lut, FLAG=self.current_flag).act(self.hidden_act)

        if self.verbose:
            intermediate.print("intermediate")
            start_time = time.time()            

        dense = intermediate.dense(layer.output.dense, self.tvm_model_lut, FLAG=self.current_flag).add(attention)
        del(intermediate)
        del(attention)

        if self.verbose:
            print("dense norm", torch.norm(layer.output.dense.weight, p=self.p))
            dense.print("output pre layer norm")

        output = dense.layer_norm(layer.output.LayerNorm, self.layer_norm)

        if self.verbose:
            output.print("output")
            # print(" time", time.time() - start_time)
            start_time = time.time()            

        return attention_scores, attention_probs, output

    def _bound_attention(self, bounds_input, attention):
        num_attention_heads = attention.self.num_attention_heads
        attention_head_size = attention.self.attention_head_size

        query = bounds_input.dense(attention.self.query, self.tvm_model_lut, FLAG=self.current_flag)
        key = bounds_input.dense(attention.self.key, self.tvm_model_lut, FLAG=self.current_flag)
        value = bounds_input.dense(attention.self.value, self.tvm_model_lut, FLAG=self.current_flag)

        del(bounds_input)

        def transpose_for_scores(x):
            def transpose_w(x):
                return x\
                    .reshape(
                        x.shape[0], x.shape[1], x.shape[2], 
                        num_attention_heads, attention_head_size)\
                    .permute(0, 3, 1, 2, 4)\
                    .reshape(-1, x.shape[1], x.shape[2], attention_head_size)
            def transpose_b(x):
                return x\
                    .reshape(
                        x.shape[0], x.shape[1], num_attention_heads, attention_head_size)\
                    .permute(0, 2, 1, 3)\
                    .reshape(-1, x.shape[1], attention_head_size)
            x.lw = transpose_w(x.lw)
            x.uw = transpose_w(x.uw)
            x.lb = transpose_b(x.lb)
            x.ub = transpose_b(x.ub)
            x.update_shape()

        transpose_for_scores(query)
        transpose_for_scores(key)

        # TODO: no attention mask for now (doesn't matter for batch_size=1)
        attention_scores = query.dot_product(key, verbose=self.verbose)\
            .multiply(1. / math.sqrt(attention_head_size))        

        if self.verbose:
            attention_scores.print("attention score")

        del(query)
        del(key)
        attention_probs = attention_scores.softmax(verbose=self.verbose)

        if self.verbose:
            attention_probs.print("attention probs")

        transpose_for_scores(value)  

        context = attention_probs.context(value)

        if self.verbose:
            value.print("value")        
            context.print("context")

        def transpose_back(x):
            def transpose_w(x):
                return x.permute(1, 2, 0, 3).reshape(1, x.shape[1], x.shape[2], -1)
            def transpose_b(x):
                return x.permute(1, 0, 2).reshape(1, x.shape[1], -1)

            x.lw = transpose_w(x.lw)
            x.uw = transpose_w(x.uw)
            x.lb = transpose_b(x.lb)
            x.ub = transpose_b(x.ub)
            x.update_shape()
        
        transpose_back(context)

        return attention_scores, attention_probs, context
        
    def _bound_pooling(self, bounds, pooler):
        bounds = Bounds(
            self.args, bounds.p, bounds.eps,
            lw = bounds.lw[:, :1, :, :], lb = bounds.lb[:, :1, :],
            uw = bounds.uw[:, :1, :, :], ub = bounds.ub[:, :1, :]
        )
        if self.verbose:
            bounds.print("pooling before dense")

        bounds = bounds.dense(pooler.dense, self.tvm_model_lut, FLAG=self.current_flag)

        if self.verbose:
            bounds.print("pooling pre-activation")

        bounds = bounds.tanh()

        if self.verbose:
            bounds.print("pooling after activation")
        return bounds

    def _bound_classifier(self, bounds, classifier, label):
        classifier = copy.deepcopy(classifier)
        classifier.weight[0, :] -= classifier.weight[1, :]
        classifier.bias[0] -= classifier.bias[1]

        if self.verbose:
            bounds.print("before dense")
            print(torch.norm(classifier.weight[0, :]))
            print(torch.mean(torch.norm(bounds.lw, dim=-2)))
            print(torch.mean(torch.norm(bounds.dense(classifier, self.tvm_model_lut, FLAG=self.current_flag).lw, dim=-2)))

        bounds = bounds.dense(classifier, self.tvm_model_lut, FLAG=self.current_flag)
        
        if self.verbose:
            bounds.print("after dense")

        l, u = bounds.concretize()

        if self.verbose:
            print(l[0][0][0])
            print(u[0][0][0])

        if label == 0:
            safe = l[0][0][0] > 0
        else:
            safe = u[0][0][0] < 0

        if self.verbose:
            print("Safe" if safe else "Unsafe")

        return safe
