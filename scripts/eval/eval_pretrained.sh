CUDA_VISIBLE_DEVICES='0' accelerate launch --multi_gpu --main_process_port 29510 eval/eval_zeroshot/eval_flickr.py \
    --image_data_path /mnt/disk2/yuanzm/dataset/lamra_data/flickr/images \
    --text_data_path /mnt/disk2/yuanzm/dataset/lamra_data/flickr/flickr_text.json \
    --original_model_id /mnt/disk2/yuanzm/weights/modelscope/Qwen2-VL-2B-Instruct/ \
    --model_id /home/yuanzm/LamRA/checkpoints/qwen2-vl-2b_LamRA_Ret_Pretrain2/
# /home/yuanzm/LamRA/checkpoints/qwen2-vl-2b_LamRA_Ret_Pretrain/checkpoint-7656/

# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_coco.py \
#     --image_data_path ./data/coco/coco_images \
#     --text_data_path ./data/coco/coco_text.json \
#     --original_model_id ./checkpoints/hf_models/Qwen2.5-VL-7B-Instruct \
#     --model_id ./checkpoints/qwen2.5-vl-7b_LamRA_Ret_Pretrain
