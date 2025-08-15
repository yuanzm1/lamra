import torch
import torch.nn.functional as F
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import pdb

model = Qwen2VLForConditionalGeneration.from_pretrained("code-kunkun/LamRA-Ret", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="auto").cuda()
processor = AutoProcessor.from_pretrained("code-kunkun/LamRA-Ret")
tokenizer = processor.tokenizer 
# pdb.set_trace()

def add_embed_token(tokenizer, model, emb_token="<emb>"):
    emb_tokens = [emb_token]
    num_new_tokens = tokenizer.add_tokens(emb_tokens)
    assert len(emb_tokens) == num_new_tokens

    model.resize_token_embeddings(len(tokenizer))

    emb_token_ids = tokenizer.convert_tokens_to_ids(emb_tokens)
    model.config.emb_token_ids = emb_token_ids

def get_embed_feature(hidden_states, input_ids, embed_index):
    embed_indices = torch.argmax((input_ids == embed_index).int(), dim=1)
    embed_features = hidden_states[torch.arange(len(embed_indices)), embed_indices - 1]
    return embed_features

def qwen2vl_process(messages, processor):
    texts = [
      processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
      for msg in messages
    ]
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    return inputs


add_embed_token(tokenizer, model)

image_message = [
  {
      "role": "user",
      "content": [
          {"type": "image", "image": "./demo.jpeg"},
          {"type": "text", "text": "Find an image caption describing the following everyday image."},
          {"type": "text", "text": f"\nSummarize above image and sentence in one word: "}
      ],
  },
  {
      "role": "assistant", "content": [{"type": "text", "text": f"<emb>."}]
  },
]

text_message1 = [
  {
      "role": "user",
      "content": [
          {"type": "text", "text": "a dog and a woman are playing on the bench.\nSummarize above sentence in one word: "},
      ],
  },
  {
      "role": "assistant", "content": [{"type": "text", "text": f"<emb>."}]
  },
]

text_message2 = [
  {
      "role": "user",
      "content": [
          {"type": "text", "text": "a dog.\nSummarize above sentence in one word: "},
      ],
  },
  {
      "role": "assistant", "content": [{"type": "text", "text": f"<emb>."}]
  },
]

image_messages = [image_message]
text_messages = [text_message1, text_message2]

image_inputs = qwen2vl_process(image_messages, processor)
text_inputs = qwen2vl_process(text_messages, processor)

with torch.no_grad():
    text_hidden_state = model(**text_inputs, output_hidden_states=True, return_dict=True).hidden_states[-1]
    text_embeds = get_embed_feature(text_hidden_state, text_inputs['input_ids'], model.config.emb_token_ids[0])
    pdb.set_trace()
    image_hidden_state = model(**image_inputs, output_hidden_states=True, return_dict=True).hidden_states[-1]
    image_embeds = get_embed_feature(image_hidden_state, image_inputs['input_ids'], model.config.emb_token_ids[0])

    text_embeds = F.normalize(text_embeds, dim=-1)
    image_embeds = F.normalize(image_embeds, dim=-1)

print(image_embeds @ text_embeds.t())


# ############################################################
# # 单样本处理（进一步减少显存压力）
# def process_single_message(message):
#     inputs = qwen2vl_process([message], processor)  # 单次处理1个样本
#     with torch.no_grad():
#         hidden_state = model(** inputs, output_hidden_states=True, return_dict=True).hidden_states[-1]
#         embed = get_embed_feature(hidden_state, inputs['input_ids'], model.config.emb_token_ids[0])
#     return embed

# # 分别处理图像和文本（单样本逐个处理）
# image_embeds = process_single_message(image_message[0])
# text_embeds1 = process_single_message(text_message1[0])
# text_embeds2 = process_single_message(text_message2[0])

# # 合并文本嵌入并归一化
# text_embeds = torch.cat([text_embeds1, text_embeds2], dim=0)
# text_embeds = F.normalize(text_embeds, dim=-1)
# image_embeds = F.normalize(image_embeds, dim=-1)

# # 计算相似度（确保张量在同一设备）
# if image_embeds.device != text_embeds.device:
#     text_embeds = text_embeds.to(image_embeds.device)
# print(image_embeds @ text_embeds.t())