ORIGINAL_MODEL_ID="/mnt/disk2/yuanzm/weights/modelscope/Qwen2-VL-2B-Instruct/"
MODEL_ID="/mnt/disk2/yuanzm/weights/lamra/checkpoints/qwen2-vl-2b_LamRA-Ret_lrpro"

CUDA_VISIBLE_DEVICES='1' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_urban1k.py \
    --image_data_path /mnt/disk2/yuanzm/dataset/lamra_data/Urban1k/image \
    --text_data_path /mnt/disk2/yuanzm/dataset/lamra_data/Urban1k/caption \
    --batch_size 2 \
    --original_model_id $ORIGINAL_MODEL_ID \
    --model_id $MODEL_ID \
    --save_for_rerank

CUDA_VISIBLE_DEVICES='1' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_sharegpt4v.py \
    --image_data_path /mnt/disk2/yuanzm/dataset/lamra_data/sharegpt4v/val_data \
    --text_data_path /mnt/disk2/yuanzm/dataset/lamra_data/sharegpt4v/datas_for_validation.json \
    --batch_size 2 \
    --original_model_id $ORIGINAL_MODEL_ID \
    --model_id $MODEL_ID \
    --save_for_rerank

CUDA_VISIBLE_DEVICES='1' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_flickr.py \
    --image_data_path /mnt/disk2/yuanzm/dataset/lamra_data/flickr/images \
    --text_data_path /mnt/disk2/yuanzm/dataset/lamra_data/flickr/flickr_text.json \
    --batch_size 2 \
    --original_model_id $ORIGINAL_MODEL_ID \
    --model_id $MODEL_ID \
    --mode finetuned \
    --save_for_rerank

CUDA_VISIBLE_DEVICES='0' accelerate launch --multi_gpu --main_process_port 29518 eval/eval_zeroshot/eval_ccneg.py \
    --data_path /mnt/disk2/yuanzm/dataset/lamra_data/ccneg/ccneg_preprocessed.pt \
    --batch_size 2 \
    --original_model_id $ORIGINAL_MODEL_ID \
    --model_id $MODEL_ID \
    --save_for_rerank

CUDA_VISIBLE_DEVICES='1' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_genecis.py \
    --annotation_path /mnt/disk2/yuanzm/dataset/lamra_data/genecis/annotations/change_object.json \
    --image_path_prefix /mnt/disk2/yuanzm/dataset/lamra_data/genecis/val2017 \
    --data_type change_object \
    --batch_size 2 \
    --original_model_id $ORIGINAL_MODEL_ID \
    --model_id $MODEL_ID \
    --save_for_rerank

CUDA_VISIBLE_DEVICES='1' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_genecis.py \
    --annotation_path /mnt/disk2/yuanzm/dataset/lamra_data/genecis/annotations/focus_object.json \
    --image_path_prefix /mnt/disk2/yuanzm/dataset/lamra_data/genecis/val2017 \
    --data_type focus_object \
    --batch_size 2 \
    --original_model_id $ORIGINAL_MODEL_ID \
    --model_id $MODEL_ID \
    --save_for_rerank

CUDA_VISIBLE_DEVICES='1' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_genecis.py \
    --annotation_path /mnt/disk2/yuanzm/dataset/lamra_data/genecis/annotations/change_attribute.json \
    --image_path_prefix /mnt/disk2/yuanzm/dataset/lamra_data/genecis/vg/change_attribute \
    --data_type change_attribute \
    --batch_size 2 \
    --original_model_id $ORIGINAL_MODEL_ID \
    --model_id $MODEL_ID \
    --save_for_rerank

CUDA_VISIBLE_DEVICES='1' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_genecis.py \
    --annotation_path /mnt/disk2/yuanzm/dataset/lamra_data/genecis/annotations/focus_attribute.json \
    --image_path_prefix /mnt/disk2/yuanzm/dataset/lamra_data/genecis/vg/focus_attribute \
    --data_type focus_attribute \
    --batch_size 2 \
    --original_model_id $ORIGINAL_MODEL_ID \
    --model_id $MODEL_ID \
    --save_for_rerank

CUDA_VISIBLE_DEVICES='1' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_circo.py \
    --annotation_path_prefix /mnt/disk2/yuanzm/dataset/lamra_data/circo/annotations \
    --image_path_prefix /mnt/disk2/yuanzm/dataset/lamra_data/circo/images/unlabeled2017 \
    --split test \
    --batch_size 2 \
    --original_model_id $ORIGINAL_MODEL_ID \
    --model_id $MODEL_ID \
    --save_for_rerank

CUDA_VISIBLE_DEVICES='1' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_vist.py \
    --data_path /mnt/disk2/yuanzm/dataset/lamra_data/vist/sis/val.story-in-sequence.json \
    --image_path_prefix /mnt/disk2/yuanzm/dataset/lamra_data/vist/images/val \
    --batch_size 2 \
    --original_model_id $ORIGINAL_MODEL_ID \
    --model_id $MODEL_ID \
    --save_for_rerank
    
CUDA_VISIBLE_DEVICES='1' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_visdial.py \
    --data_path /mnt/disk2/yuanzm/dataset/lamra_data/visdial/visdial_1.0_val.json \
    --image_path_prefix /mnt/disk2/yuanzm/dataset/lamra_data/visdial/VisualDialog_val2018 \
    --batch_size 2 \
    --original_model_id $ORIGINAL_MODEL_ID \
    --model_id $MODEL_ID \
    --save_for_rerank

CUDA_VISIBLE_DEVICES='1' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_multiturn_fashion.py \
    --query_data_path /mnt/disk2/yuanzm/dataset/lamra_data/multiturnfashion/data/all.val.json \
    --cand_data_path /mnt/disk2/yuanzm/dataset/lamra_data/multiturnfashion/image_splits/split.all.val.json \
    --image_path_prefix /mnt/disk2/yuanzm/dataset/lamra_data/M-BEIR/mbeir_images/fashioniq_images \
    --batch_size 2 \
    --category all \
    --original_model_id $ORIGINAL_MODEL_ID \
    --model_id $MODEL_ID \
    --save_for_rerank

CUDA_VISIBLE_DEVICES='1' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_sugar_crepe.py \
    --annotation_path_prefix /mnt/disk2/yuanzm/dataset/lamra_data/sugar-crepe/data \
    --data_type add_att \
    --image_path_prefix /mnt/disk2/yuanzm/dataset/lamra_data/sugar-crepe/images/val2017 \
    --batch_size 2 \
    --original_model_id $ORIGINAL_MODEL_ID \
    --model_id $MODEL_ID \
    --save_for_rerank

CUDA_VISIBLE_DEVICES='1' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_sugar_crepe.py \
    --annotation_path_prefix /mnt/disk2/yuanzm/dataset/lamra_data/sugar-crepe/data \
    --data_type add_obj \
    --image_path_prefix /mnt/disk2/yuanzm/dataset/lamra_data/sugar-crepe/images/val2017 \
    --batch_size 2 \
    --original_model_id $ORIGINAL_MODEL_ID \
    --model_id $MODEL_ID \
    --save_for_rerank

CUDA_VISIBLE_DEVICES='1' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_sugar_crepe.py \
    --annotation_path_prefix /mnt/disk2/yuanzm/dataset/lamra_data/sugar-crepe/data \
    --data_type replace_att \
    --image_path_prefix /mnt/disk2/yuanzm/dataset/lamra_data/sugar-crepe/images/val2017 \
    --batch_size 2 \
    --original_model_id $ORIGINAL_MODEL_ID \
    --model_id $MODEL_ID \
    --save_for_rerank

CUDA_VISIBLE_DEVICES='1' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_sugar_crepe.py \
    --annotation_path_prefix /mnt/disk2/yuanzm/dataset/lamra_data/sugar-crepe/data \
    --data_type replace_obj \
    --image_path_prefix /mnt/disk2/yuanzm/dataset/lamra_data/sugar-crepe/images/val2017 \
    --batch_size 2 \
    --original_model_id $ORIGINAL_MODEL_ID \
    --model_id $MODEL_ID \
    --save_for_rerank

CUDA_VISIBLE_DEVICES='1' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_sugar_crepe.py \
    --annotation_path_prefix /mnt/disk2/yuanzm/dataset/lamra_data/sugar-crepe/data \
    --data_type replace_rel \
    --image_path_prefix /mnt/disk2/yuanzm/dataset/lamra_data/sugar-crepe/images/val2017 \
    --batch_size 2 \
    --original_model_id $ORIGINAL_MODEL_ID \
    --model_id $MODEL_ID \
    --save_for_rerank

CUDA_VISIBLE_DEVICES='1' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_sugar_crepe.py \
    --annotation_path_prefix /mnt/disk2/yuanzm/dataset/lamra_data/sugar-crepe/data \
    --data_type swap_att \
    --image_path_prefix /mnt/disk2/yuanzm/dataset/lamra_data/sugar-crepe/images/val2017 \
    --batch_size 2 \
    --original_model_id $ORIGINAL_MODEL_ID \
    --model_id $MODEL_ID \
    --save_for_rerank

CUDA_VISIBLE_DEVICES='1' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_zeroshot/eval_sugar_crepe.py \
    --annotation_path_prefix /mnt/disk2/yuanzm/dataset/lamra_data/sugar-crepe/data \
    --data_type swap_obj \
    --image_path_prefix /mnt/disk2/yuanzm/dataset/lamra_data/sugar-crepe/images/val2017 \
    --batch_size 2 \
    --original_model_id $ORIGINAL_MODEL_ID \
    --model_id $MODEL_ID \
    --save_for_rerank

