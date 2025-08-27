import json
import os 
from transformers import AutoProcessor
import sys 
current_file_path = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(current_file_path, "../../")
sys.path.append(module_path)
from models.qwen2_vl import Qwen2VLRetForConditionalGeneration
import torch 
import argparse
from dataset.datasets_coco import CocoDataset
from collators.eval_collator import EvalDataCollator
from torch.utils.data import DataLoader 
import torch.nn.functional as F 
from accelerate import Accelerator
import accelerate


def eval(args):
    original_model_id = args.original_model_id
    model_id = args.model_id 
    # model = Qwen2VLRetForConditionalGeneration.from_pretrained(
    #     model_id, 
    #     torch_dtype=torch.bfloat16, 
    #     low_cpu_mem_usage=True, 
    #     device_map="auto",
    # )
    # 使用新的参数加载模型
    from eval.eval_zeroshot.util import load_mlp_parameters
    from models.qwen2_vl_finetune import Qwen2VLRetFinetuneForConditionalGeneration
    model = Qwen2VLRetFinetuneForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True, 
        device_map="auto",
    )
    load_mlp_parameters(model, os.path.join(model_id, "mlp.pth"))

    # processor is not changed so we still load from the original model repo
    processor = AutoProcessor.from_pretrained(original_model_id)

    tokenizer = processor.tokenizer 

    def add_embed_token(tokenizer, model, emb_token="<emb>"):
        emb_tokens = [emb_token]
        num_new_tokens = tokenizer.add_tokens(emb_tokens)
        assert len(emb_tokens) == num_new_tokens

        model.resize_token_embeddings(len(tokenizer))

        emb_token_ids = tokenizer.convert_tokens_to_ids(emb_tokens)
        model.config.emb_token_ids = emb_token_ids

    add_embed_token(tokenizer, model)

    query_dataset = CocoDataset(
        image_data_path=args.image_data_path, 
        text_data_path=args.text_data_path,
        type='image',
    )

    cand_dataset = CocoDataset(
        image_data_path=args.image_data_path, 
        text_data_path=args.text_data_path,
        type='text',
    )

    query_data_collator = EvalDataCollator(tokenizer=tokenizer, processor=processor)
    cand_data_collator = EvalDataCollator(tokenizer=tokenizer, processor=processor)
    
    query_dataloader = DataLoader(query_dataset, batch_size=8, num_workers=8, shuffle=False, collate_fn=query_data_collator)
    candidate_dataloader = DataLoader(cand_dataset, batch_size=16, num_workers=8, shuffle=False, collate_fn=cand_data_collator)

    accelerator = Accelerator(mixed_precision='bf16')
    device = accelerator.device 
    is_main_process = accelerator.is_main_process

    model.eval()

    def tensors_to_device(data, device, dtype=model.dtype):
        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                if key == 'pixel_values':
                    data[key] = data[key].to(device).to(dtype)
                else:
                    data[key] = data[key].to(device)
        return data 

    query_features = []
    query_ids = []
    candidate_features = []
    candidate_ids = []

    from tqdm import tqdm 
    with torch.no_grad():
        query_dataloader, candidate_dataloader, model = accelerator.prepare(query_dataloader, candidate_dataloader, model)

        for batch in tqdm(query_dataloader, disable=not is_main_process):
            batch = tensors_to_device(batch, device)
            query_embed, batch_query_ids = model(**batch, inference=True)
            query_embed = F.normalize(query_embed, dim=-1)
            query_embed = accelerator.gather_for_metrics(query_embed)
            batch_query_ids = accelerate.utils.gather_object(batch_query_ids)[:len(query_embed)]
            query_ids.extend(batch_query_ids)
            query_features.append(query_embed)

        for batch in tqdm(candidate_dataloader, disable=not is_main_process):
            batch = tensors_to_device(batch, device)
            candidate_embed, batch_candidate_ids = model(**batch, inference=True)
            candidate_embed = F.normalize(candidate_embed, dim=-1)
            candidate_embed = accelerator.gather_for_metrics(candidate_embed)
            batch_candidate_ids = accelerator.gather_for_metrics(batch_candidate_ids)[:len(candidate_embed)]
            candidate_ids.extend(batch_candidate_ids)
            candidate_features.append(candidate_embed)

    query_features = torch.cat(query_features, dim=0)
    candidate_features = torch.cat(candidate_features, dim=0)

    
    if is_main_process:
        # Adjust the order according to ids 
        import numpy as np 
        query_ids = np.array(query_ids)
        sorted_query_indices = np.argsort(query_ids)
        image_features = query_features[sorted_query_indices]
        candidate_ids = np.array(candidate_ids)
        sorted_candidate_indices = np.argsort(candidate_ids)
        text_features = candidate_features[sorted_candidate_indices]

        texts_image_index = [i // 5 for i in range(image_features.shape[0]*5)]

        assert text_features.isnan().sum().item() == 0, 'nan in retrieve emb'
        assert image_features.isnan().sum().item() == 0, 'nan in images emb'

        # get the score for each text and image pair
        scores  = text_features @ image_features.t() 


        positive_pairs = torch.zeros_like(scores, dtype=bool)
        positive_pairs[torch.arange(len(scores)), texts_image_index] = True
        metrics = {}
        recall_k_list = [1, 5, 10]
        batch_size = 64
        for recall_k in recall_k_list:
            metrics[f"image_retrieval_recall@{recall_k}"] = (batchify(recall_at_k, scores, positive_pairs, batch_size, device, k=recall_k)>0).float().mean().item()
            metrics[f"text_retrieval_recall@{recall_k}"] = (batchify(recall_at_k, scores.T, positive_pairs.T, batch_size, device, k=recall_k)>0).float().mean().item()

        print(metrics)


def recall_at_k(scores, positive_pairs, k):
    """
    Compute the recall at k for each sample
    :param scores: compability score between  text and image embeddings (nb texts, nb images)
    :param k: number of images to consider per text, for retrieval
    :param positive_pairs: boolean matrix of positive pairs (nb texts, nb images)
    :return: recall at k averaged over all texts
    """
    nb_texts, nb_images = scores.shape
    # for each text, sort according to image scores in decreasing order
    topk_indices = torch.topk(scores, k, dim=1)[1]
    # compute number of positives for each text
    nb_positive = positive_pairs.sum(dim=1)
    # nb_texts, k, nb_images
    topk_indices_onehot = torch.nn.functional.one_hot(topk_indices, num_classes=nb_images)
    # compute number of true positives
    positive_pairs_reshaped = positive_pairs.view(nb_texts, 1, nb_images)
    # a true positive means a positive among the topk
    nb_true_positive = (topk_indices_onehot * positive_pairs_reshaped).sum(dim=(1,2))
    # compute recall at k
    recall_at_k = (nb_true_positive / nb_positive)
    return recall_at_k

def batchify(func, X, Y, batch_size, device, *args, **kwargs):
    results = []
    for start in range(0, len(X), batch_size):
        end = start + batch_size
        x = X[start:end].to(device)
        y = Y[start:end].to(device)
        result = func(x, y, *args, **kwargs).cpu()
        results.append(result)
    return torch.cat(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_data_path', type=str)
    parser.add_argument('--text_data_path', type=str)
    parser.add_argument('--original_model_id', type=str)
    parser.add_argument('--model_id', type=str)

    args = parser.parse_args()
    eval(args)