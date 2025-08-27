from typing import Dict, Sequence
import torch
from .base import BaseDataCollator
from .qwen2_vision_process import process_vision_info
from transformers import PreTrainedTokenizer, AutoProcessor
from typing import Dict, Sequence, Optional
import pdb
# from qwen_vl_utils import process_vision_info

def determine_mode(dataset_type):
    if dataset_type == 'share':
        return "text->image"
    if dataset_type == 'flickr':
        return "text->image"
    if dataset_type == 'urban':
        return "text->image"
    if dataset_type == 'ccneg':
        return "image->text"
    if dataset_type == 'sugar':
        return "image->text"
    if dataset_type == 'visd':
        return "text->image"
    if dataset_type == 'vist':
        return "image+text->image"
    if dataset_type == 'mtfiq':
        return "image+text->image"
    if dataset_type == 'circo':
        return "image+text->image"
    if dataset_type == 'gene':
        return "image+text->image"

class EvalDataCollator(BaseDataCollator):
    @property
    def PAD_TOKEN_ID(self) -> int:
        return self.tokenizer.pad_token_id
    
    def __init__(
        self, 
        tokenizer: Optional[PreTrainedTokenizer] = None,
        processor: Optional[AutoProcessor] = None,
        mask_question_tokens: bool = True, 
        dataset_type = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.processor = processor
        self.mask_question_tokens = mask_question_tokens
        self.dataset_type = dataset_type

    def __call__(self, messages: Sequence[Dict]) -> Dict[str, torch.Tensor]:

        new_messages = []
        ids = []

        for item in messages:
            new_messages.append(item[0])
            ids.append(item[1])

        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in new_messages
        ]
        image_inputs, video_inputs = process_vision_info(new_messages)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        input_ids = inputs['input_ids']
        labels = input_ids.clone()
        labels[labels == self.PAD_TOKEN_ID] = self.IGNORE_TOKEN_ID

        if 'attention_mask' in inputs:
            attention_mask = inputs['attention_mask']
        else:
            attention_mask = None 
        if 'pixel_values' in inputs:
            pixel_values = inputs['pixel_values']
        else:
            pixel_values = None 
        if 'image_grid_thw' in inputs:
            image_grid_thw = inputs['image_grid_thw']
        else:
            image_grid_thw = None 
        if 'pixel_values_videos' in inputs:
            pixel_values_videos = inputs['pixel_values_videos']
        else:
            pixel_values_videos = None 
        if 'video_grid_thw' in inputs:
            video_grid_thw = inputs['video_grid_thw']
        else:
            video_grid_thw = None 
        
        has_hard_negative = False 

        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            labels=labels,
            has_hard_negative=has_hard_negative,
            ids=ids,
            retrieve_mode=determine_mode(self.dataset_type),
            processor=self.processor
        )