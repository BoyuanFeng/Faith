# CUDA_VISIBLE_DEVICES=0 python run_bounds.py --data=sst --model model_sst_1_no_hidden64  --p 2 --method forward --perturbed_words=1 --num_layers=1 --current_flag=0
# CUDA_VISIBLE_DEVICES=0 python run_bounds.py --data=sst --model model_sst_1_no_hidden64  --p 2 --method forward --perturbed_words=1 --num_layers=1 --current_flag=1
# CUDA_VISIBLE_DEVICES=0 python run_bounds.py --data=sst --model model_sst_1_no_hidden64  --p 2 --method forward --perturbed_words=1 --num_layers=1 --current_flag=2
CUDA_VISIBLE_DEVICES=0 python run_bounds.py --data=sst --model model_sst_1_no_hidden64  --p 2 --method forward --perturbed_words=1 --num_layers=1 --current_flag=3
# CUDA_VISIBLE_DEVICES=0 python run_bounds.py --data=sst --model model_sst_1_no_hidden64  --p 2 --method forward --perturbed_words=1 --num_layers=1 --current_flag=3
# CUDA_VISIBLE_DEVICES=0 python run_bounds.py --data=sst --model model_sst_1_no_hidden64  --p 2 --method forward --perturbed_words=1 --num_layers=1 --current_flag=3
# CUDA_VISIBLE_DEVICES=0 python run_bounds.py --data=sst --model model_sst_1_no_hidden64  --p 2 --method forward --perturbed_words=1 --num_layers=1 --current_flag=3
# CUDA_VISIBLE_DEVICES=0 python run_bounds.py --data=sst --model model_sst_1_no_hidden64  --p 2 --method forward --perturbed_words=1 --num_layers=1 --current_flag=3
# CUDA_VISIBLE_DEVICES=0 python run_bounds.py --data=sst --model model_sst_1_no_hidden64  --p 2 --method forward --perturbed_words=1 --num_layers=1 --current_flag=3

# CUDA_VISIBLE_DEVICES=1 python run_bounds.py --data=sst --model model_sst_2_no_hidden64  --p 2 --method forward --perturbed_words=1 --num_layers=2 --current_flag=0
# CUDA_VISIBLE_DEVICES=1 python run_bounds.py --data=sst --model model_sst_2_no_hidden64  --p 2 --method forward --perturbed_words=1 --num_layers=2 --current_flag=1
# CUDA_VISIBLE_DEVICES=1 python run_bounds.py --data=sst --model model_sst_2_no_hidden64  --p 2 --method forward --perturbed_words=1 --num_layers=2 --current_flag=2

# CUDA_VISIBLE_DEVICES=2 python run_bounds.py --data=sst --model model_sst_3_no_hidden64  --p 2 --method forward --perturbed_words=1 --num_layers=3 --current_flag=0
# CUDA_VISIBLE_DEVICES=2 python run_bounds.py --data=sst --model model_sst_3_no_hidden64  --p 2 --method forward --perturbed_words=1 --num_layers=3 --current_flag=1
# CUDA_VISIBLE_DEVICES=2 python run_bounds.py --data=sst --model model_sst_3_no_hidden64  --p 2 --method forward --perturbed_words=1 --num_layers=3 --current_flag=2

# CUDA_VISIBLE_DEVICES=3 python run_bounds.py --data=sst --model model_sst_4_no_hidden64  --p 2 --method forward --perturbed_words=1 --num_layers=4 --current_flag=0
# CUDA_VISIBLE_DEVICES=3 python run_bounds.py --data=sst --model model_sst_4_no_hidden64  --p 2 --method forward --perturbed_words=1 --num_layers=4 --current_flag=1
# CUDA_VISIBLE_DEVICES=3 python run_bounds.py --data=sst --model model_sst_4_no_hidden64  --p 2 --method forward --perturbed_words=1 --num_layers=4 --current_flag=2

# CUDA_VISIBLE_DEVICES=4 python run_bounds.py --data=sst --model model_sst_5_no_hidden64  --p 2 --method forward --perturbed_words=1 --num_layers=5 --current_flag=0
# CUDA_VISIBLE_DEVICES=4 python run_bounds.py --data=sst --model model_sst_5_no_hidden64  --p 2 --method forward --perturbed_words=1 --num_layers=5 --current_flag=1
# CUDA_VISIBLE_DEVICES=4 python run_bounds.py --data=sst --model model_sst_5_no_hidden64  --p 2 --method forward --perturbed_words=1 --num_layers=5 --current_flag=2

# CUDA_VISIBLE_DEVICES=5 python run_bounds.py --data=sst --model model_sst_6_no_hidden64  --p 2 --method forward --perturbed_words=1 --num_layers=6 --current_flag=0
# CUDA_VISIBLE_DEVICES=5 python run_bounds.py --data=sst --model model_sst_6_no_hidden64  --p 2 --method forward --perturbed_words=1 --num_layers=6 --current_flag=1
# CUDA_VISIBLE_DEVICES=5 python run_bounds.py --data=sst --model model_sst_6_no_hidden64  --p 2 --method forward --perturbed_words=1 --num_layers=6 --current_flag=2


# CUDA_VISIBLE_DEVICES=1 python run_bounds.py --data=sst --model model_sst_2_no_hidden64  --p 2 --method forward --perturbed_words=1 --num_layers=2
# CUDA_VISIBLE_DEVICES=2 python run_bounds.py --data=sst --model model_sst_3_no_hidden64  --p 2 --method forward --perturbed_words=1 --num_layers=3
# CUDA_VISIBLE_DEVICES=3 python run_bounds.py --data=sst --model model_sst_4_no_hidden64  --p 2 --method forward --perturbed_words=1 --num_layers=4
# CUDA_VISIBLE_DEVICES=4 python run_bounds.py --data=sst --model model_sst_5_no_hidden64  --p 2 --method forward --perturbed_words=1 --num_layers=5
# CUDA_VISIBLE_DEVICES=5 python run_bounds.py --data=sst --model model_sst_6_no_hidden64  --p 2 --method forward --perturbed_words=1 --num_layers=6 
# python run_bounds.py --data=sst --model model_sst_1_no_hidden64  --p 2 --method forward --perturbed_words=1 --num_layers=1 --kernel_type=2 --gpuid=0 &&
# python run_bounds.py --data=sst --model model_sst_2_no_hidden64  --p 2 --method forward --perturbed_words=1 --num_layers=2 --kernel_type=2 --gpuid=1 &&
# python run_bounds.py --data=sst --model model_sst_3_no_hidden64  --p 2 --method forward --perturbed_words=1 --num_layers=3 --kernel_type=2 --gpuid=2 &&
# python run_bounds.py --data=sst --model model_sst_4_no_hidden64  --p 2 --method forward --perturbed_words=1 --num_layers=4 --kernel_type=2 --gpuid=3 &&
# python run_bounds.py --data=sst --model model_sst_5_no_hidden64  --p 2 --method forward --perturbed_words=1 --num_layers=5 --kernel_type=2 --gpuid=4 &&
# python run_bounds.py --data=sst --model model_sst_6_no_hidden64  --p 2 --method forward --perturbed_words=1 --num_layers=6 --kernel_type=2 --gpuid=5

# python run_bounds.py --data=sst --model model_sst_9_no_hidden64  --p 2 --method forward --perturbed_words=1 --num_layers=9 --kernel_type=1 --gpuid=6 &&
