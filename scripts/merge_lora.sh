# CUDA_VISIBLE_DEVICES='0' accelerate launch --multi_gpu --main_process_port 29509 merge_lora/merge.py \
#     --original_model_id Qwen/Qwen2-VL-7B-Instruct or Qwen/Qwen2-VL-2B-Instruct \
#     --model_id the_model_path_after_the_first_stage_of_pre-training \
#     --save_path the_path_you_want_to_save

CUDA_VISIBLE_DEVICES='0' accelerate launch --multi_gpu --main_process_port 29509 merge_lora/merge.py \
    --original_model_id /mnt/disk2/yuanzm/weights/modelscope/Qwen2-VL-2B-Instruct/ \
    --model_id /mnt/disk2/yuanzm/weights/lamra/checkpoints/qwen2-vl-2b_LamRA_Ret_Pretrain2/ \
    --save_path /mnt/disk2/yuanzm/weights/lamra/checkpoints/LamRA-Ret-Pretrained-merged2

