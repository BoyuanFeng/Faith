import os
import subprocess

# Profile the performance
def profile_performance():
    os.chdir("scratch_space")
    os.system("nvcc -arch=sm_80 -I ../../cutlass/include -I ../../cutlass/tools/util/include -I ../../cutlass/examples/common -o cuda_profiler cuda_profiler.cu")
    proc = subprocess.Popen(["./cuda_profiler"], stdout=subprocess.PIPE)
    try:
        output = float(proc.communicate()[0])
    except:
        # If profiling the current configuration fails, we still need to change working directory to the parent directory
        os.chdir("../")
        assert(False, "Something goes wrong in profile_performance()...")
    # If profiling succeeds, we need to change working directory to the parent directory.
    os.chdir("../")
    return output

if __name__ == "__main__":
    res = profile_performance()
    print(res)