from typing import Tuple, Optional, List, Union 
import torch 
from transformers.utils import logging

logger = logging.get_logger(__name__)

from peft import PrefixTuningConfig, get_peft_model
from transformers import AutoProcessor, AutoModel, AutoModelForCausalLM, Qwen2VLForConditionalGeneration, PreTrainedTokenizer
from torch import nn 
import torch.distributed as dist
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLCausalLMOutputWithPast
import torch.nn.functional as F
import pdb

class RetrievalPromptGenerator(nn.Module):
    def __init__(self, hidden_size, prompt_length=10):
        super().__init__()
        self.hidden_size = hidden_size
        self.prompt_length = prompt_length
        
        # 8种检索模式的专属嵌入层
        self.mode_embeddings = nn.Embedding(
            num_embeddings=8,  # 8种检索模式
            embedding_dim=hidden_size * prompt_length
        )
        
        # 初始化嵌入参数
        nn.init.normal_(self.mode_embeddings.weight, mean=0.0, std=0.02)
        
        # 模式映射：文本描述 -> 索引
        self.mode_mapping = {
            "image+text->image": 0,
            "image+text->text": 1,
            "text->image+text": 2,
            "image->image+text": 3,
            "text->image": 4,
            "text->text": 5,
            "image->text": 6,
            "image->image": 7
        }

    def determine_mode(self, query, candidate):
        """根据key是否存在判断模态（没有的模态不包含对应key）"""
        # 判断query的模态：存在'txt' key表示有文本，存在'image' key表示有图像
        query_has_text = "txt" in query
        query_has_image = "image" in query
        # 判断candidate的模态：存在'txt' key表示有文本，存在'image' key表示有图像
        cand_has_text = "txt" in candidate
        cand_has_image = "image" in candidate
        
        # 8种模式精确匹配
        if query_has_text and query_has_image:
            if cand_has_image and not cand_has_text:
                return "image+text->image"
            elif cand_has_text and not cand_has_image:
                return "image+text->text"
        if query_has_text and not query_has_image:
            if cand_has_image and cand_has_text:
                return "text->image+text"
            elif cand_has_image and not cand_has_text:
                return "text->image"
            elif cand_has_text and not cand_has_image:
                return "text->text"
        if query_has_image and not query_has_text:
            if cand_has_image and cand_has_text:
                return "image->image+text"
            elif cand_has_text and not cand_has_image:
                return "image->text"
            elif cand_has_image and not cand_has_text:
                return "image->image"
        # 默认模式：文本到图文（应对异常情况）
        return "text->image+text"

    def forward(self, mode, batch_size=1):
        """生成对应检索模式的prompt_embed"""
        # 1. 确定检索模式
        mode_idx = self.mode_mapping[mode]
        # 2. 获取该模式对应的可学习prompt参数
        mode_embed = self.mode_embeddings(
            torch.tensor([mode_idx], device=self.mode_embeddings.weight.device)
        )
        # 3. 重塑为固定长度的prompt嵌入
        prompt_embed = mode_embed.view(1, self.prompt_length, self.hidden_size)
        # 4. 扩展到批次大小
        prompt_embed = prompt_embed.repeat(batch_size, 1, 1)
        return prompt_embed, mode_embed

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp=0.07):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp

class Qwen2VLRetFinetuneForConditionalGeneration(Qwen2VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        # peft_config = PrefixTuningConfig(
        #     task_type="CAUSAL_LM",
        #     num_virtual_tokens=5,
        # )
        # self.model = get_peft_model(self.model, peft_config)

        # # 定义MLP：输入为倒数第二层隐藏维度，输出维度与原特征一致（确保后续计算兼容）
        # self.modify_mlp = nn.Sequential(
        #     nn.Linear(config.hidden_size, config.hidden_size),  # 第一层线性变换
        #     nn.GELU(),  # 激活函数
        #     nn.Linear(config.hidden_size, config.hidden_size)   # 输出维度与原特征匹配
        # )
        # # 初始化MLP权重
        # for m in self.modify_mlp.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.xavier_uniform_(m.weight)
        #         nn.init.zeros_(m.bias)

        # 新增：可学习的prompt
        hidden_size = config.hidden_size
        self.prompt_length = 1  # 可学习prompt的长度（可根据需求调整）2 10
        self.modify_generate=RetrievalPromptGenerator(hidden_size=hidden_size, prompt_length=self.prompt_length)
        self.modify_prompt_embeddings = nn.Embedding(
            self.prompt_length,  # prompt token数量
            config.hidden_size   # 嵌入维度，与模型隐藏层维度一致
        )
        self.modify_prompt_mapping = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * self.prompt_length),  # 扩展维度
            nn.GELU(),
            nn.Unflatten(1, (self.prompt_length, hidden_size))  # 拆分维度
        )
        # # 初始化prompt嵌入（可选：用正态分布初始化）
        # nn.init.normal_(self.modify_prompt_embeddings.weight, mean=0.0, std=config.initializer_range)
                
        # # 新增：记录训练轮次，用于控制伪标签更新频率
        # self.register_buffer("current_iter", torch.tensor(0, dtype=torch.long))
        # # 新增：用于缓存历史相似度，稳定伪标签生成
        # self.register_buffer("prev_similarities", None)

    def get_features(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.get_dtype())
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw).to(inputs_embeds.device)
                image_mask = input_ids == self.config.image_token_id
                if self.training:
                    inputs_embeds = inputs_embeds.clone()
                inputs_embeds[image_mask] = image_embeds
            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw).to(inputs_embeds.device)
                video_mask = input_ids == self.config.video_token_id
                inputs_embeds[video_mask] = video_embeds
            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        output_hidden_states = True
        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        
        # # 输出倒数第二层的hidden_states
        # hidden_states = outputs[1][-2]
        # hidden_states = self.modify_mlp(hidden_states)
        
        embed_index = self.config.emb_token_ids[0]
        embed_indices = torch.argmax((labels == embed_index).int(), dim=1) 
        embed_features = hidden_states[torch.arange(len(embed_indices)), embed_indices - 1] # (batch_size, embed_dim)
        return embed_features 
    
    def insert_prompt_and_adjust_mask(self, batch_embed, task_prompt_embed, input_ids, attention_mask, processor):
        """
        插入任务特异prompt嵌入，并同步调整attention_mask
        
        参数:
            batch_embed: 原始输入嵌入，形状[batch_size, seq_len, hidden_size]
            task_prompt_embed: 任务特异prompt嵌入，形状[batch_size, prompt_length, hidden_size]
            input_ids: 原始input_ids（用于定位插入位置），形状[batch_size, seq_len]
            attention_mask: 原始注意力掩码，形状[batch_size, seq_len]
            processor: 处理器（含tokenizer）
        返回:
            modified_batch_embed: 插入后的嵌入，[batch_size, seq_len + prompt_len, hidden_size]
            modified_attention_mask: 调整后的掩码，[batch_size, seq_len + prompt_len]
        """
        # new_input_ids = processor(text="This is a retrieve task.")['input_ids'].to(input_ids.device)
        # new_task_prompt_embed = self.model.embed_tokens(new_input_ids).repeat(task_prompt_embed.shape[0],1,1)
        # task_prompt_embed = new_task_prompt_embed
        
        batch_size = batch_embed.size(0)
        prompt_length = task_prompt_embed.size(1)
        hidden_size = batch_embed.size(2)
        # 1. 验证形状兼容性
        assert task_prompt_embed.size(0) == batch_size, "prompt批次大小不匹配"
        assert task_prompt_embed.size(2) == hidden_size, "prompt隐藏维度不匹配"
        assert attention_mask.size() == input_ids.size(), "原始掩码与input_ids形状不匹配"
        
        # 2. 解析特殊标记ID（用于定位插入位置）
        im_end_token = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        user_start_token = processor.tokenizer.convert_tokens_to_ids("user")
        system_start_token = processor.tokenizer.convert_tokens_to_ids("system")
        
        modified_embeds = []
        modified_masks = []
        
        for b in range(batch_size):
            # 当前样本的输入、嵌入和掩码
            current_ids = input_ids[b]
            current_embed = batch_embed[b]
            current_mask = attention_mask[b]
            # 3. 定位插入位置：用户消息块的<|im_end|>之前
            # 查找<|im_end|>的位置（从后往前找最后一个用户消息块）
            im_end_positions = (current_ids == im_end_token).nonzero().squeeze(dim=1).tolist()
            if not isinstance(im_end_positions, list):
                im_end_positions = [im_end_positions]
            
            insert_pos = (current_ids == system_start_token).nonzero().squeeze(dim=1).tolist()[0]  # 默认插入位置（若未找到用户消息块）
            insert_pos += 2
                
            # 4. 拆分并插入嵌入
            part1_embed = current_embed[:insert_pos]
            part2_embed = current_embed[insert_pos:]
            inserted_embed = torch.cat([part1_embed, task_prompt_embed[b], part2_embed], dim=0)
            # 5. 同步调整attention_mask
            # 拆分原始掩码，插入prompt对应的掩码（全1，因为prompt是有效信息）
            part1_mask = current_mask[:insert_pos]
            part2_mask = current_mask[insert_pos:]
            prompt_mask = torch.ones(prompt_length, device=current_mask.device, dtype=current_mask.dtype)
            inserted_mask = torch.cat([part1_mask, prompt_mask, part2_mask], dim=0)
            modified_embeds.append(inserted_embed)
            modified_masks.append(inserted_mask)
        
        # 6. 合并批次
        modified_batch_embed = torch.stack(modified_embeds, dim=0)
        modified_attention_mask = torch.stack(modified_masks, dim=0)
        
        return modified_batch_embed, modified_attention_mask

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        inference=False,
        has_hard_negative=False,
        qids=None,
        dids=None,
        ids=None,
        retrieve_mode=None,
        processor=None,
        # 新增：控制伪标签生成的参数
        pseudo_label_top1_ratio=0.1,   # 顶部10%样本赋予最高相关性
        pseudo_label_top2_ratio=0.3,   # 顶部10%-30%样本赋予中等相关性
        pseudo_label_smoothing=0.1,    # 伪标签平滑系数，减少噪声影响
        update_pseudo_labels_every=100   # 每100个iter更新一次伪标签缓存
    ) -> Union[Tuple, Qwen2VLCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        >>> model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # 根据不同mode生成不同的prompt
        batch_size = input_ids.size(0) if input_ids is not None else inputs_embeds.size(0)
        all_embeds, all_modpro = [], []
        # 为每个测试用例生成embed（单样本）
        if not inference:
            for case in retrieve_mode:
                mode = self.modify_generate.determine_mode(case["query"], case["pos_cand"])
                embed, mode_pro= self.modify_generate(mode, batch_size=1)
                all_embeds.append(embed)  # 每个embed形状: [1, prompt_length, hidden_size]
                all_modpro.append(mode_pro)
            # 拼接所有embed：test_case长度即batch_size
            all_embeds = all_embeds + all_embeds.copy()
            prompt_embeds = torch.cat(all_embeds, dim=0)  # 拼接后形状: [batch_size, prompt_length, hidden_size
            modpro = torch.cat(all_modpro, dim=0) # [batch_size, hidden_size]
        else:
            for _ in range(batch_size):
                embed, _ = self.modify_generate(retrieve_mode, batch_size=1)
                all_embeds.append(embed)  # 每个embed形状: [1, prompt_length, hidden_size]
            prompt_embeds = torch.cat(all_embeds, dim=0)  # 拼接后形状: [batch_size, prompt_length, hidden_size

        # # -------------------------- 新增：生成基于所有有效token的固定长度prompt_embeds --------------------------
        # batch_size = input_ids.size(0) if input_ids is not None else inputs_embeds.size(0)
        # device = input_ids.device if input_ids is not None else inputs_embeds.device
        # hidden_size = self.config.hidden_size
        # # 1. 提取每个样本的有效token并编码
        # valid_embeds_list = []
        # for i in range(batch_size):
        #     # 获取有效token的掩码和id
        #     mask = attention_mask[i] != 0
        #     valid_ids = input_ids[i][mask].unsqueeze(0)  # 形状: [1, valid_len]
        #     if valid_ids.size(1) == 0:
        #         # 无有效token时用零向量填充
        #         valid_embeds = torch.zeros(1, hidden_size, device=device)
        #     else:
        #         # 用模型嵌入层编码所有有效token
        #         valid_embeds = self.model.embed_tokens(valid_ids)  # [1, valid_len, hidden_size]
        #         # 2. 聚合有效token特征（保留全部信息）
        #         # 策略：均值池化 + 最大池化融合（兼顾全局信息和显著特征）
        #         mean_pooled = torch.mean(valid_embeds, dim=1)  # [1, hidden_size]
        #         max_pooled = torch.max(valid_embeds, dim=1)[0]  # [1, hidden_size]
        #         valid_embeds = (mean_pooled + max_pooled) / 2  # 融合两种池化结果
        #     valid_embeds_list.append(valid_embeds)
        # # 3. 聚合为batch级特征
        # batch_valid_embeds = torch.cat(valid_embeds_list, dim=0)  # [batch_size, hidden_size]
        # prompt_embeds = self.modify_prompt_mapping(batch_valid_embeds)  # [batch_size, prompt_length, hidden_size]

        # # 插入一个固定的prompt
        # # 新增：生成可学习prompt的嵌入（形状：[batch_size, prompt_length, hidden_size]）
        # batch_size = input_ids.size(0) if input_ids is not None else inputs_embeds.size(0)
        # for i in range(batch_size):
        #     mask = attention_mask[i] != 0
        #     prompt_ids = input_ids[mask]
        #     prompt_ids = torch.arange(self.prompt_length, device=self.device).repeat(batch_size, 1)  # [batch_size, prompt_length]
        # prompt_embeds = self.modify_prompt_embeddings(prompt_ids)  # [batch_size, prompt_length, hidden_size]
        # import pdb; pdb.set_trace() 

        # set mini_batch to 32
        mini_batch_size = 32
        input_ids_list = torch.split(input_ids, mini_batch_size)
        attention_mask_list = torch.split(attention_mask, mini_batch_size)
        prompt_embeds_list = torch.split(prompt_embeds, mini_batch_size)
        if image_grid_thw is not None:
            cumsum_pixel_values = torch.cumsum(image_grid_thw[:, 1] * image_grid_thw[:, 2], dim=-1) 
            zero_tensor = torch.tensor([0], device=cumsum_pixel_values.device) # be convinient for extracting batch_pixel_values
            cumsum_pixel_values = torch.cat((zero_tensor, cumsum_pixel_values))
            image_nums = 0
        
        all_hidden_states = []
        attentions = []

        for i in range(len(input_ids_list)):
            if inputs_embeds is None:
                batch_inputs_embeds = self.model.embed_tokens(input_ids_list[i])
                if pixel_values is not None:
                    image_mask = input_ids_list[i] == self.config.image_token_id
                    current_image_num = torch.sum(torch.any(image_mask, dim=-1)).cpu().item()
                    if current_image_num != 0:
                        batch_pixel_values = pixel_values[cumsum_pixel_values[image_nums] : cumsum_pixel_values[image_nums + current_image_num]]
                        batch_pixel_values = batch_pixel_values.type(self.visual.get_dtype())
                        batch_image_embeds = self.visual(batch_pixel_values, grid_thw=image_grid_thw[image_nums:image_nums + current_image_num]).to(batch_inputs_embeds.device)
                        image_nums = image_nums + current_image_num
                        if self.training:
                            batch_inputs_embeds = batch_inputs_embeds.clone()
                        batch_inputs_embeds[image_mask] = batch_image_embeds
                # if pixel_values is not None:
                #     pixel_values = pixel_values.type(self.visual.get_dtype())
                #     batch_image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw).to(batch_inputs_embeds.device)
                #     image_mask = input_ids_list[i] == self.config.image_token_id
                #     if self.training:
                #         batch_inputs_embeds = batch_inputs_embeds.clone()
                #     batch_inputs_embeds[image_mask] = batch_image_embeds
                if pixel_values_videos is not None:
                    pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
                    video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw).to(inputs_embeds.device)
                    video_mask = input_ids == self.config.video_token_id
                    inputs_embeds[video_mask] = video_embeds
                # if attention_mask is not None:
                #     batch_attention_mask = attention_mask_list[i].to(batch_inputs_embeds.device)
                    
                # # 3. 在所有token之前插入可学习prompt（新增逻辑）
                # # 获取当前迷你批次的prompt嵌入
                # start_idx = i * mini_batch_size
                # end_idx = min((i+1)*mini_batch_size, batch_size)
                # current_prompt_embeds = prompt_embeds[start_idx:end_idx]  # [mini_batch, prompt_length, hidden_size]
                # # 将prompt插入到输入嵌入前面
                # batch_inputs_embeds = torch.cat([current_prompt_embeds, batch_inputs_embeds], dim=1)
                # if attention_mask is not None:
                #     # 生成prompt部分的注意力掩码（全1）
                #     prompt_mask = torch.ones(
                #         (batch_inputs_embeds.size(0), self.prompt_length),
                #         device=batch_inputs_embeds.device,
                #         dtype=attention_mask_list[i].dtype
                #     )
                #     batch_attention_mask = attention_mask_list[i].to(batch_inputs_embeds.device)
                #     batch_attention_mask = torch.cat([prompt_mask, batch_attention_mask], dim=1)
                # else:
                #     # 若没有原始掩码，生成全1掩码
                #     batch_attention_mask = torch.ones(
                #         (batch_inputs_embeds.size(0), batch_inputs_embeds.size(1)),
                #         device=batch_inputs_embeds.device,
                #         dtype=torch.float32
                #     )
                
                # pdb.set_trace()
                # 找到特定的位置(system, user)插入prompt并调整掩码
                batch_inputs_embeds, batch_attention_mask = self.insert_prompt_and_adjust_mask(
                    batch_embed=batch_inputs_embeds,
                    task_prompt_embed=prompt_embeds_list[i],
                    input_ids=input_ids_list[i],
                    attention_mask=attention_mask_list[i],
                    processor=processor
                )
            
            output_attentions = True
            outputs = self.model(
                input_ids=None,
                position_ids=position_ids,
                attention_mask=batch_attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=batch_inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            
            valid_mask = batch_attention_mask[i].bool()
            valid_indices = torch.where(valid_mask)[0]
            # 步骤1: 保留有效token作为Query（行）
            layer_avg_attentions = []
            for layer_attention in outputs['attentions']:
                # layer_attention 形状: (batch_size, num_heads, seq_len, seq_len)
                avg_attention = torch.mean(layer_attention, dim=1)  # 结果形状: (batch_size, seq_len, seq_len)
                layer_avg_attentions.append(avg_attention)
            attention_valid_query = torch.stack(layer_avg_attentions, dim=0)[:,:,valid_indices, :]  # (valid_seq_len, seq_len)
            attention_valid = attention_valid_query[..., valid_indices]
            attention_valid = attention_valid[-1]
            
            hidden_states = outputs[0]
            all_hidden_states.append(hidden_states)
            attentions.append(attention_valid)

        hidden_states = torch.cat(all_hidden_states)
        attentions = torch.cat(attentions)

        if has_hard_negative:
            batch_size = len(hidden_states) // 3
        elif not inference:
            batch_size = len(hidden_states) // 2
        elif inference:
            batch_size = len(hidden_states)

        if inference:
            assert batch_size == len(hidden_states)

        embed_index = self.config.emb_token_ids[0]
        embed_indices = torch.argmax((labels == embed_index).int(), dim=1) + self.prompt_length
        embed_features = hidden_states[torch.arange(len(embed_indices)), embed_indices - 1] # (batch_size, embed_dim)

        if inference:
            if ids is not None:
                return embed_features, ids 
            elif qids is not None or dids is not None:
                return embed_features, qids, dids 
            return embed_features 
        if has_hard_negative:
            embed1, embed2, embed3 = embed_features[:batch_size], embed_features[batch_size:2*batch_size], embed_features[2*batch_size:]
        else:
            embed1, embed2 = embed_features[:batch_size], embed_features[batch_size:]
        loss_fct = nn.CrossEntropyLoss()

        if dist.is_initialized():
            if has_hard_negative:
                embed3_list = [torch.zeros_like(embed3) for _ in range(dist.get_world_size())]
                dist.all_gather(tensor_list=embed3_list, tensor=embed3.contiguous())
                embed3_list[dist.get_rank()] = embed3 
                embed3 = torch.cat(embed3_list, 0)
            
            # Dummy vectors for allgather
            embed1_list = [torch.zeros_like(embed1) for _ in range(dist.get_world_size())]
            embed2_list = [torch.zeros_like(embed2) for _ in range(dist.get_world_size())]
            # Allgather
            dist.all_gather(tensor_list=embed1_list, tensor=embed1.contiguous())
            dist.all_gather(tensor_list=embed2_list, tensor=embed2.contiguous())

            # Since allgather results do not have gradients, we replace the
            # current process's corresponding embeddings with original tensors
            embed1_list[dist.get_rank()] = embed1
            embed2_list[dist.get_rank()] = embed2
            # Get full batch embeddings: (bs x N, hidden)
            embed1 = torch.cat(embed1_list, 0)
            embed2 = torch.cat(embed2_list, 0)

        sim = Similarity(temp=0.05)

        # add normalization
        embed1 = F.normalize(embed1, dim=-1)
        embed2 = F.normalize(embed2, dim=-1)

        cos_sim = sim(embed1.unsqueeze(1), embed2.unsqueeze(0))

        if has_hard_negative:
            embed1_embed3_cos = sim(embed1.unsqueeze(1), embed3.unsqueeze(0))
            cos_sim = torch.cat([cos_sim, embed1_embed3_cos], 1)
        
        nce_labels = torch.arange(cos_sim.size(0)).long().to(cos_sim.device)
        
        norms = torch.norm(modpro, dim=1, keepdim=True)  # 保持维度以便广播
        modpro_normalized = modpro / norms  # 形状仍为 [10, 1024]
        modpro_sim = torch.matmul(modpro_normalized, modpro_normalized.T)
        # with torch.no_grad():
        cos_sim *= modpro_sim 

        loss = loss_fct(cos_sim, nce_labels)
        
        return SequenceClassifierOutput(loss=loss)