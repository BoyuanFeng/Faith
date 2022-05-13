from inspect import Parameter
from math import floor, log2
import random
from tabnanny import check

from numpy import block

v100_metafile = {
    # V100 has at most 96 KB shared memory per SM
    "MAX_SHMEM_SIZE": 96,
    # 1 thread has at most 256 registers. Otherwise there are register spilling
    "MAX_NUM_REGISTER_PER_THREAD": 256,
    "MAX_GLOBAL_MEM_SIZE" : 32, # Unit: GB
    "REGISTER_FILE_SIZE_PER_SM" : 256, # Unit: KB
}

a100_metafile = {
    # A100 has at most 164 KB shared memory per SM
    "MAX_SHMEM_SIZE": 164,
    # 1 thread has at most 256 registers. Otherwise there are register spilling
    "MAX_NUM_REGISTER_PER_THREAD": 256,
    "MAX_GLOBAL_MEM_SIZE" : 40, # Unit: GB
    "REGISTER_FILE_SIZE_PER_SM" : 256, # Unit: KB
}

# Heuristics helps to autotuning an operator implementation.
# It proposes parameters for autotuning and skip non-optimal parameters 
#   (e.g., parameters that may leads to register spilling).
# In Heuristics, we consider properties from both workload and hardware.
# While new workload requires manually writing new Heuristics, these Heuristics can be reused across
# a workload with different input sizes.
# We remind that, the same GEMM workload requires significantly different parameters under different 
#   input shapes.  
class Heuristics:
    # We only check a few heuristics such as tiling size should be power of 2.
    # We expect ML cost model to learn heuristics between parameters and GPU metafile.
    def __init__(self, metafile) -> None:
        self.metafile = metafile
        self.set_parameter_range()

    def check_shared_memory_size(self, shared_memory_size):
        return shared_memory_size < self.metafile["MAX_SHMEM_SIZE"]*1024 # Unit: Bytes

    # Check that register usage is within the register file size per SM.
    # We make a conservative assumption that there is only 1 block allocated per SM.
    def check_register_size(self, register_size):
        return register_size < self.metafile["REGISTER_FILE_SIZE_PER_SM"]*1024 # Unit Bytes

    def get_feature_from_metafile(self):
        res = []
        for key in self.metafile:
            res.append(self.metafile[key])
        return res

    def get_feature_from_parameters(self, parameters):
        res = []
        res += parameters["block_tiling_size"]
        res += parameters["warp_tiling_size"]
        res.append(parameters["num_stages"])
        return res

    # A heuristic for proposing a tiling size within the max_size. Should be power of 2 for efficiency.
    def propose_tiling_size(self, max_size):
        max_log = floor(log2(max_size))
        exponent = random.randint(1, max_log+1)
        return 1<<exponent

    def set_parameter_range(self):
        self.parameter_range = {
            "max_tiling_size" : 128,
            "max_stages" : 4
        }

    def propose_parameters(self):
        warp_tiling_size = []
        for i in range(3):
            v = self.propose_tiling_size(self.parameter_range["max_tiling_size"])
            warp_tiling_size.append(v)
        
        block_tiling_size = []
        for i in range(3):
            v = self.propose_tiling_size(self.parameter_range["max_tiling_size"])
            block_tiling_size.append(v)
        
        num_stages = random.randint(2, self.parameter_range["max_stages"])
        parameters = {
            "warp_tiling_size" : warp_tiling_size,
            "block_tiling_size" : block_tiling_size,
            "num_stages" : num_stages,
        } 
        return parameters

    # A heuristic for gemm only. Block tiles need to fit in shared memory.
    # Assuming there are three tiling sizes [b_m, b_n, b_k]
    # Assuming each element is float and accounts for 4 bytes
    def check_block_tiling_size_valid(self, tiling_size_list):
        assert len(tiling_size_list) == 3, "Warning: Too Little Block Tiling Sizes"
        bm = tiling_size_list[0]
        bn = tiling_size_list[1]
        bk = tiling_size_list[2]
        # Assuming each element is float and accounts for 4 bytes
        element_size = 4
        expected_shmem_size = element_size*(bm*bk + bn*bk) # Unit: Bytes
        if not self.check_shared_memory_size(expected_shmem_size):
            return False
        return True

    # A heuristic for gemm only. Warp tiles need to fit in registers
    # Assuming there are three tiling sizes [w_m, w_n, w_k]
    def check_warp_tiling_size_valid(self, tiling_size_list):
        assert len(tiling_size_list) == 3, "Warning: Too Little Warp Tiling Sizes"
        wm = tiling_size_list[0]
        wn = tiling_size_list[1]
        wk = tiling_size_list[2]
        # Assuming each element is float and accounts for 4 bytes
        element_size = 4
        expected_register_size = element_size * wm * wn
        if not self.check_register_size(expected_register_size):
            return False
        # wm & wn % 16 = 0 to match the requirement of ldmatrix
        if not(wm % 16 == 0 and wn % 16 == 0 and wk == 32):
            return False
        return True

    def check_harmony_block_warp_tiling_size(self, warp_tiling_size, block_tiling_size):
        if (warp_tiling_size[2] != block_tiling_size[2]):
            return False
        for i in range(3):
            if not (block_tiling_size[i] % warp_tiling_size[i] == 0):
                return False
        return True

    @staticmethod
    def check_too_many_predicate(num_iteration):
        if num_iteration <= 0:
            return False
        predicate_count = num_iteration
        predicate_byte_count = (predicate_count + 3) // 4
        predicate_word_count = (predicate_byte_count + 3) // 4
        return predicate_word_count <= 4

    
    def check_memory_load_iterations(self, warp_tiling_size, block_tiling_size):
        # Cutlass assume all the threads participate in loading the lhs and rhs operands
        # So at least one iteration is required for every thread
        bm = block_tiling_size[0]
        bn = block_tiling_size[1]
        bk = block_tiling_size[2]

        wm = warp_tiling_size[0]
        wn = warp_tiling_size[1]

        num_elements_per_access = (bm * bn / wm / wn) * 32 * 4
        lhs_elements_per_access = bk * bm
        rhs_elements_per_access = bk * bn

        lhs_iteration = lhs_elements_per_access // num_elements_per_access
        rhs_iteration = rhs_elements_per_access // num_elements_per_access

        if (lhs_elements_per_access % num_elements_per_access == 0 and rhs_elements_per_access % num_elements_per_access == 0 and Heuristics.check_too_many_predicate(lhs_iteration) and Heuristics.check_too_many_predicate(rhs_iteration)):
            return True
        else:
            return False

    def check_warp_count(self, warp_tiling_size, block_tiling_size):
        bm = block_tiling_size[0]
        bn = block_tiling_size[1]

        wm = warp_tiling_size[0]
        wn = warp_tiling_size[1]

        if (bm * bn / wm / wn > 64):
            return False
        return True

    def check_limited_tiling_size_difference(self, tiling_size):
        m = tiling_size[0]
        n = tiling_size[1]

        return (m == n) or (m == 2*n)

    # More heuristics can be added...
    def is_valid(self, parameters):
        return self.check_warp_tiling_size_valid(parameters["warp_tiling_size"]) \
                and self.check_block_tiling_size_valid(parameters["block_tiling_size"]) \
                and self.check_harmony_block_warp_tiling_size(parameters["warp_tiling_size"], parameters["block_tiling_size"]) \
                and self.check_memory_load_iterations(parameters["warp_tiling_size"], parameters["block_tiling_size"]) \
                and self.check_warp_count(parameters["warp_tiling_size"], parameters["block_tiling_size"]) \
                and self.check_limited_tiling_size_difference(parameters["warp_tiling_size"]) \
                and self.check_limited_tiling_size_difference(parameters["block_tiling_size"])
