"""
@inproceedings{jha2024clap4clip,
  title={{CLAP4CLIP}: Continual Learning with Probabilistic Finetuning for Vision-Language Models},
  author={Saurav Jha and Dong Gong and Lina Yao},
  booktitle={Thirty-eighth Conference on Neural Information Processing Systems},
  year={2024},
  url={https://arxiv.org/pdf/2403.19137}
}
Code Reference:
https://github.com/srvCodes/clap4clip
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
from torch.utils.data import Dataset, DataLoader

import numpy as np

from .finetune import Finetune
from .backbone.clip import load, tokenize
from tqdm import tqdm
from copy import deepcopy
import pickle
import random

import os
import errno
from PIL import Image

# for data transform -- todo: should be merge into data part
import torchvision.transforms as transforms
from torch.utils.data import Dataset # for cifar dataset definition
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    from PIL import Image
    BICUBIC = Image.BICUBIC
    
from ..utils.clap4clip_utils import build_cosine_scheduler, freeze_parameters, init_weights, accuracy

class BufferDataset(Dataset):
    def __init__(self, images, labels, mode, data_root, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.mode = mode
        self.data_root = data_root
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        img = Image.open(os.path.join(self.data_root, self.mode, img_path)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

def freeze_parameters(m, requires_grad=False):
    if m is None:
        return

    if isinstance(m, torch.nn.Parameter):
        m.requires_grad = requires_grad
    else:
        for p in m.parameters():
            p.requires_grad = requires_grad

class Adapter(nn.Module):
    def __init__(self, in_dim, out_dim, sigma=False, layer_num=1):
        super().__init__()
        
        self.fc = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.sigma = sigma
        
        # 强制初始化
        self._force_init_weights()
        
        # print(f"Adapter created - sigma: {self.sigma}")
        self._debug_weights()
        
    def _force_init_weights(self):
        """强制重新初始化权重"""
        with torch.no_grad():
            if self.sigma:
                # Sigma适配器：使用非常小的权重
                self.fc[0].weight.fill_(0.001)  # 几乎为0的权重
                self.fc[0].bias.fill_(-3.0)     # 较大的负偏置
                # print("Sigma adapter: 强制初始化为小权重")
            else:
                # Mu适配器：标准小权重初始化
                nn.init.xavier_uniform_(self.fc[0].weight, gain=0.01)
                nn.init.zeros_(self.fc[0].bias)
                # print("Mu adapter: 标准初始化")
    
    def _debug_weights(self):
        """调试权重信息"""
        weight = self.fc[0].weight
        bias = self.fc[0].bias
        # print(f"  权重形状: {weight.shape}")
        # print(f"  权重范围: {weight.min().item():.6f} to {weight.max().item():.6f}")
        # print(f"  偏置范围: {bias.min().item():.6f} to {bias.max().item():.6f}")
        # print(f"  权重是否包含NaN: {torch.isnan(weight).any()}")
        # print(f"  偏置是否包含NaN: {torch.isnan(bias).any()}")

    def forward(self, x):
        # print(f"\n=== Adapter Forward (sigma={self.sigma}) ===")
        # print(f"输入x形状: {x.shape}")
        # print(f"输入x范围: {x.min().item():.4f} to {x.max().item():.4f}")
        # print(f"输入x是否包含NaN: {torch.isnan(x).any()}")
        # print(f"输入x是否包含Inf: {torch.isinf(x).any()}")
        
        # 检查权重状态
        weight = self.fc[0].weight
        bias = self.fc[0].bias
        
        # print(f"当前权重范围: {weight.min().item():.6f} to {weight.max().item():.6f}")
        # print(f"当前偏置范围: {bias.min().item():.6f} to {bias.max().item():.6f}")
        # print(f"权重包含NaN: {torch.isnan(weight).any()}")
        # print(f"偏置包含NaN: {torch.isnan(bias).any()}")
        
        # 如果权重已经是NaN，强制重新初始化
        if torch.isnan(weight).any() or torch.isnan(bias).any():
            print("检测到权重NaN，强制重新初始化")
            self._force_init_weights()
        
        # 如果输入有问题，清理输入
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("输入包含NaN/Inf，进行清理")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
            print(f"清理后输入范围: {x.min().item():.4f} to {x.max().item():.4f}")
        
        # 计算线性变换
        try:
            fc_output = self.fc(x)
            # print(f"线性层输出范围: {fc_output.min().item():.4f} to {fc_output.max().item():.4f}")
            # print(f"线性层输出包含NaN: {torch.isnan(fc_output).any()}")
        except Exception as e:
            print(f"线性层计算出错: {e}")
            # 创建安全的输出
            fc_output = torch.zeros_like(x) if not self.sigma else torch.full_like(x, -2.0)
        
        # 检查线性层输出
        if torch.isnan(fc_output).any() or torch.isinf(fc_output).any():
            print("线性层输出包含NaN/Inf，使用安全默认值")
            if self.sigma:
                fc_output = torch.full_like(x, -2.0)  # 对sigma使用负值
            else:
                fc_output = torch.zeros_like(x)       # 对mu使用零值
            print(f"安全默认值范围: {fc_output.min().item():.4f} to {fc_output.max().item():.4f}")
        
        if self.sigma:
            # print("应用sigma变换...")
            # 限制范围，防止softplus溢出
            fc_output_clamped = torch.clamp(fc_output, min=-10.0, max=5.0)
            # print(f"限制后范围: {fc_output_clamped.min().item():.4f} to {fc_output_clamped.max().item():.4f}")
            
            try:
                softplus_output = F.softplus(fc_output_clamped)
                # print(f"softplus输出范围: {softplus_output.min().item():.4f} to {softplus_output.max().item():.4f}")
                # print(f"softplus输出包含NaN: {torch.isnan(softplus_output).any()}")
                
                if torch.isnan(softplus_output).any():
                    print("softplus产生NaN，使用备用方案")
                    result = torch.full_like(softplus_output, 0.1)
                else:
                    result = softplus_output * 0.999 + 0.001
                    
            except Exception as e:
                print(f"softplus计算出错: {e}")
                result = torch.full_like(fc_output_clamped, 0.1)
            
            # print(f"最终sigma输出范围: {result.min().item():.4f} to {result.max().item():.4f}")
            return result
        else:
            # print(f"最终mu输出范围: {fc_output.min().item():.4f} to {fc_output.max().item():.4f}")
            return fc_output

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, x, tokenized_prompts):
        x = x + self.positional_embedding.type(self.dtype)  # position_embeding可训练
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection  # @ and
        return x


class CLIP(nn.Module):
    def __init__(self, kwargs, class_names, clip_model, vga,
                 mu_adapters=None, sigma_adapters=None, task_tokens=None,
                 task_to_cls_num=None, prompt_templates=None, previous_components=None,
                 task_to_distribution=None, mu_global_adapter=None, sigma_global_adapter=None,
                 global_vga=None, cur_task_idx=0):
        super().__init__()
        self.cur_task_idx = cur_task_idx
        self.n_class = len(class_names)
        self.kwargs = kwargs
        self.cur_task_idx = cur_task_idx
        # self.args = args
        # text encoder
        self.text_encoder = TextEncoder(clip_model)
        if torch.cuda.device_count() > 1:
            self.text_encoder = nn.DataParallel(self.text_encoder, device_ids=self.kwargs["default_gpu"])

        self.current_class_names = class_names
        # prompt learner
        ctx_dim = clip_model.ln_final.weight.shape[0]
        dtype = clip_model.dtype
        if previous_components is not None:
            self.unpack_prev_components(previous_components)

        # image encoder
        self.image_encoder = clip_model.visual
        self.vga = vga
        self.vga_global = global_vga
        self.logit_scale = clip_model.logit_scale

        self.mu_adapters = mu_adapters
        self.sigma_adapters = sigma_adapters
        self.mu_global_adapter = mu_global_adapter
        self.sigma_global_adapter = sigma_global_adapter

        self.forward_times = self.kwargs["forward_times"]
        self.forward_times_global = self.kwargs["forward_times_global"]

        self.task_tokens = task_tokens
        self.task_to_cls_num = task_to_cls_num
        self.prompt_templates = prompt_templates
        self.pretrained_text_encoder = clip_model.encode_text
        self.prior_text_features()
        self.class_to_task_mapping = {}  # for faster indexing to get task ids
        self.classwise_centroids = {}
        self.task_to_distribution = task_to_distribution
        self.init_new_heads()

    def init_new_heads(self):
        def get_new_task_embed(var=False):
            if var:
                new_class_embeds = self.frozen_text_features_individual.var(1)
            else:
                new_class_embeds = self.frozen_text_features_individual.mean(1)
            layer_embeds = new_class_embeds.t() @ new_class_embeds
            # layer_embeds = layer_embeds / layer_embeds.norm()
            layer_embeds = layer_embeds / layer_embeds.shape[0]
            return layer_embeds

        def init_with_task_embed(module, var=False):
            layer_embeds = get_new_task_embed(var=var)
            for m in module.fc.children():
                if isinstance(m, torch.nn.Linear):
                    m.weight.copy_(layer_embeds)

        with torch.no_grad():
            init_with_task_embed(self.mu_adapters[-1])
            init_with_task_embed(self.sigma_adapters[-1], var=True)

    def unpack_prev_components(self, previous_components):
        previous_mu, previous_sigma, previous_task_tokens, previous_vga, previous_mu_global_adapter, previous_sigma_global_adapter = previous_components
        self.previous_mu_adapters = previous_mu
        self.previous_sigma_adapters = previous_sigma
        self.previous_task_tokens = previous_task_tokens
        self.previous_vga = previous_vga
        self.previous_mu_global_adapter, self.previous_sigma_global_adapter = previous_mu_global_adapter, previous_sigma_global_adapter

    @torch.no_grad()
    def prior_text_features(self):
        prompts = [[temp.format(c.replace("_", " ")) for temp in self.prompt_templates] for c in
                   self.current_class_names]
        text_features_, text_features_per_prompt = [], []
        for per_cls_prompts in prompts:
            per_cls_prompt_embs = tokenize(per_cls_prompts).cuda(device=self.kwargs["default_gpu"])  # t_c(p)
            text_features = self.pretrained_text_encoder(per_cls_prompt_embs)  # g(t_c(p))
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # g(t_c(p))
            text_features_per_prompt.append(text_features)
            text_features = text_features.mean(dim=0)
            text_features = text_features / text_features.norm()
            text_features_.append(text_features)
        self.frozen_text_features = torch.stack(text_features_, dim=0)
        self.frozen_text_features_individual = torch.stack(text_features_per_prompt, dim=0)

    def get_variational_adapter_features(self, x, i=None, distill=False, global_adapter=False):
        if global_adapter:
            mu_adapter = self.previous_mu_global_adapter if distill else self.mu_global_adapter
            sigma_adapter = self.previous_sigma_global_adapter if distill else self.sigma_global_adapter
        else:
            mu_adapter = self.previous_mu_adapters[i] if distill else self.mu_adapters[i]
            sigma_adapter = self.previous_sigma_adapters[i] if distill else self.sigma_adapters[i]
        mu = mu_adapter(x)
        sigma = sigma_adapter(x)
        dist = Normal(mu, sigma)
        return dist

    def get_prior_from_memory(self, x_for_prior, text_features, task_num):
        with torch.no_grad():
            n_class = self.n_class
            image_features = self.image_encoder(x_for_prior.to(text_features.device).type(self.dtype))
            image_features = (image_features / image_features.norm(dim=-1, keepdim=True)).detach()
        vga_features = self.vga(text_features.clone().unsqueeze(0), image_features.unsqueeze(0)).squeeze(0)
        text_featues_ = vga_features + text_features
        pdist = self.get_variational_adapter_features(text_featues_, task_num if self.kwargs["expandable_adapter"] else 0)
        return pdist

    def get_prior_dist(self, image_features=None, text_features=None, batch_labels=None, task_num=None,
                       task_specific_labels=None, task_token=None, use_np_prior=False, global_adapter=False,
                       tgt_mask=None):
        if not use_np_prior:
            return Normal(torch.zeros_like(text_features), torch.ones_like(text_features))

        import math
        def get_context_indices(bs, labels, task_specific_labels=None, context_size=0.67):
            if task_specific_labels is None:
                # m = random.randint(math.ceil(0.3 * bs), math.ceil(0.8 * bs))
                m = math.ceil(context_size * bs)
                context_indices = torch.randperm(labels.size(0)).to(labels.device)[:m]
                # context_indices = get_context_by_labels(labels, m)
            else:
                context_indices = []
                for label in task_specific_labels:
                    idx = (labels == label).nonzero(as_tuple=True)[0]
                    context_indices.append(idx)
                context_indices = torch.cat(context_indices)
                if context_indices.shape[0] == labels.shape[0]:
                    context_indices = get_context_indices(bs, labels)
            return context_indices

        context_indices = get_context_indices(image_features.size(0), batch_labels,
                                              task_specific_labels if task_num > 0 else None,
                                              context_size=self.kwargs["context_size"])
        if len(context_indices) == 0:
            # no task-specific data points so resort to standard normal prior
            return Normal(torch.zeros_like(text_features), torch.ones_like(text_features))
        else:
            image_features = image_features[context_indices]
            nquery = text_features.size(0)
            query = torch.cat([text_features.unsqueeze(0), task_token],
                              1) if task_token is not None else text_features.unsqueeze(0)
            vga_features = self.vga(query, image_features.unsqueeze(0), tgt_mask=tgt_mask).squeeze(0)
            text_features_ = vga_features[:nquery] + text_features
            if task_token is not None:
                text_features_ = text_features_ + vga_features[-1]
            pdist = self.get_variational_adapter_features(text_features_,
                                                          task_num if self.kwargs["expandable_adapter"] else 0,
                                                          global_adapter=global_adapter)

        return pdist

    @staticmethod
    def get_contrastive_matrix(text_feats, image_feats, logit_scale=None):
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        if logit_scale is not None:
            image_feats = image_feats.clone() * logit_scale
        contrastive_matrix = image_feats @ text_feats.t()  # 16 x 16 matrix
        return contrastive_matrix

    def get_attention_mask(self, attn_shape, nb_task_tokens, original_query_num):
        """Mask so that task tokens don't interact together.

        Given two task tokens (t1, t2) and three patch tokens (p1, p2, p3), the
        attention matrix is:

        t1-t1 t1-t2 t1-p1 t1-p2 t1-p3
        t2-t1 t2-t2 t2-p1 t2-p2 t2-p3

        So that the mask (True values are deleted) should be:

        False True False False False
        True False False False False
        """
        mask = torch.zeros(attn_shape, dtype=torch.bool).cuda(device=self.kwargs["default_gpu"])
        if self.kwargs["expandable_tokens"]:
            for i in range(nb_task_tokens):
                mask[original_query_num + i, original_query_num:original_query_num + i] = True
                mask[original_query_num + i, original_query_num + i + 1:original_query_num + nb_task_tokens] = True

        start_cls_idx, end_cls_idx = 0, 0
        for i in range(nb_task_tokens):
            start_cls_idx = end_cls_idx
            end_cls_idx += self.task_to_cls_num[i]
            curr_class_indices = np.arange(start_cls_idx, end_cls_idx)
            for cls in curr_class_indices:
                mask[cls][:start_cls_idx] = True
                mask[cls][end_cls_idx:] = True
                if self.kwargs["expandable_tokens"]:
                    mask[cls][original_query_num + i] = False
            if self.kwargs["expandable_tokens"]:
                mask[original_query_num + i, :start_cls_idx] = True
                mask[original_query_num + i, end_cls_idx:original_query_num] = True
        return mask

    def get_avg_inter_adapter_distance(self, per_task_samples):
        pairwise_distances = []
        # per_task_samples = per_task_samples / per_task_samples.norm(dim=-1, keepdim=True)
        for i in range(per_task_samples.shape[0]):
            for j in range(i, per_task_samples.shape[0]):
                cos = ((per_task_samples[i] * per_task_samples[j]) / (
                            per_task_samples[i].shape[0] * per_task_samples[j].shape[1])).sum()
                pairwise_distances.append(1 - cos.item())
        avg_distance = np.mean(pairwise_distances)
        return avg_distance

    def forward(self, image, labels=None, test=False, finetuning=False, return_mean=True, for_prior=None):
        
                # print(f"=== 调试信息 ===")
        # print(f"输入图像形状: {image.shape}")
        # print(f"输入图像数据类型: {image.dtype}")
        # print(f"输入图像设备: {image.device}")
        
        # HACK: 确保输入格式正确
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
            # print(f"添加批次维度后: {image.shape}")
        ############以上是调试信息##########    
        
        if image.shape[-1] != 224 or image.shape[-2] != 224:
            # print(f"调整图像尺寸从 {image.shape[-2:]} 到 (224, 224)")
            image = torch.nn.functional.interpolate(
                image, 
                size=(224, 224), 
                mode='bilinear', 
                align_corners=False
            )
            # print(f"调整后图像形状: {image.shape}")       
        
        
        with torch.no_grad():
            image_features = self.image_encoder(image.type(self.dtype))
            image_features_normed = image_features / image_features.norm(dim=-1, keepdim=True)
            image_features = image_features_normed.detach()
            image_features_normed = image_features_normed.detach()
            
            
                # 验证任务索引和相关属性
        # print(f"当前任务索引: {self.cur_task_idx}")
        # print(f"任务到类别数映射: {self.task_to_cls_num}")
        # print(f"总类别数: {self.n_class}")

        # 安全地计算 prev_cls_num
        if self.cur_task_idx in self.task_to_cls_num:
            prev_cls_num = self.n_class - self.task_to_cls_num[self.cur_task_idx]
            # print(f"前一任务类别数: {prev_cls_num}")
        else:
            prev_cls_num = 0
            print(f"警告：任务 {self.cur_task_idx} 不在映射中，使用默认值 0")

        n_class = self.n_class
        prev_cls_num = self.n_class - self.task_to_cls_num[self.cur_task_idx]  # DONE: self.cur_task_idx, should be maintained while training, not realize yet
        logit_scale = self.logit_scale.exp()
        if test:
            with torch.no_grad():
                text_features = self.frozen_text_features
                context = image_features_normed.clone()  # torch.cat([image_features.unsqueeze(0), self.task_token_two[-1]], 1)
                n_query = text_features.shape[0]
                query = text_features.clone().unsqueeze(0)
                if self.kwargs["expandable_tokens"]:
                    query = torch.cat([query] + [token for token in self.task_tokens], 1)
                attn_mask = self.get_attention_mask((query.shape[1], query.shape[1]), self.cur_task_idx + 1,
                                                    text_features.shape[0])
                if self.kwargs["use_vga"]:
                    vga_features = self.vga(query, context.unsqueeze(0), tgt_mask=attn_mask).squeeze(0)

                rsamples_g = None
                if self.kwargs["hierarchical"]:
                    # vga_features_global = self.vga(query, context.unsqueeze(0)).squeeze(0)
                    global_input_features = vga_features[:n_query] if self.kwargs["use_vga"] else text_features
                    global_input_features = global_input_features + text_features
                    qdist_g = self.get_variational_adapter_features(global_input_features, global_adapter=True)
                    rsamples_g = qdist_g.rsample([self.forward_times_global])

                logits = []
                samplewise_text_feats = []
                start_cls_idx, end_cls_idx = 0, 0
                for i in range(self.cur_task_idx + 1):  # todo: all self.cur_task_idx maybe equal to self.cur_task_idx. to be check
                    start_cls_idx = end_cls_idx
                    end_cls_idx += self.task_to_cls_num[i]
                    text_features_relevant = text_features[start_cls_idx:end_cls_idx].clone()
                    text_features_ = text_features_relevant
                    if self.kwargs["use_vga"]:
                        text_features_ = text_features_ + vga_features[start_cls_idx:end_cls_idx]
                    if self.kwargs["expandable_tokens"]:
                        text_features_ = text_features_ + vga_features[n_query + i]

                    if self.kwargs["hierarchical"]:
                        text_features_ = text_features_.unsqueeze(0).expand(self.forward_times_global, -1,
                                                                            -1) + rsamples_g[:,
                                                                                  start_cls_idx:end_cls_idx, :]
                    qdist = self.get_variational_adapter_features(text_features_,
                                                                  i if self.kwargs["expandable_adapter"] else 0)
                    rsamples = qdist.rsample([self.forward_times])

                    text_features_ = text_features_.unsqueeze(0).expand(self.forward_times, -1, -1,
                                                                        -1) if self.kwargs["hierarchical"] else text_features_.unsqueeze(
                        0).expand(self.forward_times, -1, -1)
                    if self.kwargs["hierarchical"]:
                        rsamples = rsamples.flatten(0, 1)
                        text_features_ = text_features_.flatten(0, 1)
                    text_features_ = rsamples + text_features_

                    logits_ = logit_scale * image_features_normed @ text_features_.permute(0, 2, 1)

                    logits.append(logits_)
                    if self.kwargs["compute_ram"]:
                        samplewise_text_feats.append(text_features_relevant)
                # logits = torch.stack(logits, 0).sum(0)
                logits = torch.cat(logits, -1)
                logits = logits.detach()
            if self.kwargs["compute_ram"]:
                visual_feats = image_features_normed
                samplewise_text_feats = torch.cat(samplewise_text_feats, 0)
                samplewise_text_feats = samplewise_text_feats / samplewise_text_feats.norm(dim=-1, keepdim=True)
                samplewise_text_feats = samplewise_text_feats[labels]
                return logits, (visual_feats.detach().cpu(), samplewise_text_feats.detach().cpu())
            if return_mean:
                return logits.mean(0), (None, None)
            else:
                return logits, (None, None)

        else:

            text_features = self.frozen_text_features
            logits = []
            kl_losses = []
            prior_matching_losses = []
            start_cls_idx, end_cls_idx = 0, 0
            context = image_features_normed.clone()
            n_query = text_features.shape[0]
            query = text_features.clone().unsqueeze(0)
            if self.kwargs["expandable_tokens"]:
                query = torch.cat([query] + [token for token in self.task_tokens], 1)
            attn_mask = self.get_attention_mask((query.shape[1], query.shape[1]), self.cur_task_idx + 1,
                                                text_features.shape[0])
            if self.kwargs["use_vga"]:
                vga_features_all = self.vga(query, context.unsqueeze(0), tgt_mask=attn_mask).squeeze(0)

            rsamples_g = None
            if self.kwargs["hierarchical"]:
                # vga_features_global = self.vga(query, context.unsqueeze(0)).squeeze(0)
                global_input_features = vga_features_all[:n_query] if self.kwargs["use_vga"] else text_features
                global_input_features = global_input_features + text_features
                pdist_g = self.get_prior_dist(context, global_input_features, labels, self.cur_task_idx + 1,
                                              None,
                                              None,
                                              use_np_prior=self.kwargs["use_np_prior"] if not finetuning else False,
                                              global_adapter=True
                                              )
                qdist_g = self.get_variational_adapter_features(global_input_features, global_adapter=True)
                # pdist_g = self.get_prior_dist(text_features=global_input_features, use_np_prior=False)
                prior_matching_losses.append(kl_divergence(qdist_g, pdist_g).mean(0).sum() * 0.001)
                rsamples_g = qdist_g.rsample([self.forward_times_global])
                if self.kwargs["lasp"] and self.kwargs["beta"] > 0:
                    prior_text_features = self.frozen_text_features_individual.clone()
                    sims = torch.stack([prior_text_features @ rsamples_g[r].t() for r in range(rsamples_g.shape[0])], 0)
                    sims = sims.mean(2).mean(0)
                    kl_losses.append(F.cross_entropy(sims, torch.arange(sims.size(0)).cuda(
                        device=self.kwargs["default_gpu"])) * self.kwargs["beta"])

            if self.kwargs["distill"] and self.cur_task_idx > 0 and self.kwargs["alpha"] > 0:
                with torch.no_grad():
                    prev_task_text_features = text_features[:-self.task_to_cls_num[self.cur_task_idx]].clone()
                    n_query_prev = prev_task_text_features.shape[0]
                    prev_vga_query = prev_task_text_features.unsqueeze(0)
                    if self.kwargs["expandable_tokens"]:
                        prev_vga_query = torch.cat([prev_vga_query] + [token for token in self.previous_task_tokens], 1)
                    prev_attn_mask = self.get_attention_mask((prev_vga_query.shape[1], prev_vga_query.shape[1]),
                                                             self.cur_task_idx, prev_task_text_features.shape[0])
                    prev_vga_features_all = self.previous_vga(prev_vga_query, context.unsqueeze(0),
                                                              tgt_mask=prev_attn_mask).squeeze(0).detach()
                    prev_global_input_features = prev_vga_features_all[:n_query_prev] + prev_task_text_features
                    qdist_g_prev = self.get_variational_adapter_features(prev_global_input_features, distill=True,
                                                                         global_adapter=True)
                    prev_loc = qdist_g_prev.loc.detach()
                kl_losses.append(F.mse_loss(prev_loc, qdist_g.loc[:prev_loc.shape[0]]) * 0.3)

            per_sample_text_feats = []
            taskwise_means = []

            for i in range(self.cur_task_idx + 1):

                start_cls_idx = end_cls_idx
                end_cls_idx += self.task_to_cls_num[i]
                if start_cls_idx not in self.class_to_task_mapping:
                    # update class to task mapping for faster indexing of task id based on class label id
                    self.class_to_task_mapping.update(
                        dict(zip(np.arange(start_cls_idx, end_cls_idx), [i] * (end_cls_idx - start_cls_idx))))

                text_features_relevant = text_features.clone()[start_cls_idx:end_cls_idx]
                if self.kwargs["use_vga"]:
                    vga_features = vga_features_all[start_cls_idx:end_cls_idx]
                    if self.kwargs["expandable_tokens"]:
                        vga_features = vga_features + vga_features_all[n_query + i]
                    text_features_ = text_features_relevant + vga_features
                else:
                    text_features_ = text_features_relevant

                if self.kwargs["hierarchical"]:
                    text_features_ = text_features_.unsqueeze(0).expand(self.forward_times_global, -1, -1) + rsamples_g[
                                                                                                             :,
                                                                                                             start_cls_idx:end_cls_idx,
                                                                                                             :]
                qdist = self.get_variational_adapter_features(text_features_, i if self.kwargs["expandable_adapter"] else 0)
                rsamples = qdist.rsample([self.forward_times])

                text_features_ = text_features_.unsqueeze(0).expand(self.forward_times, -1, -1,
                                                                    -1) if self.kwargs["hierarchical"] else text_features_.unsqueeze(
                    0).expand(self.forward_times, -1, -1)
                if self.kwargs["hierarchical"]:
                    rsamples = rsamples.flatten(0, 1)
                    text_features_ = text_features_.flatten(0, 1)
                text_features_ = rsamples + text_features_

                taskwise_means.append(rsamples.mean(0))
                if self.kwargs["lasp"] and self.kwargs["beta"] > 0 and (finetuning or (not finetuning and self.cur_task_idx == i)):
                    prior_text_features = self.frozen_text_features_individual.clone()[start_cls_idx:end_cls_idx]
                    sims = torch.stack([prior_text_features @ rsamples[r].t() for r in range(rsamples.shape[0])], 0)
                    sims = sims.mean(2).mean(0)
                    kl_losses.append(F.cross_entropy(sims, torch.arange(sims.size(0)).cuda(
                        device=self.kwargs["default_gpu"])) * self.kwargs["beta"])
                logits_ = (logit_scale * image_features_normed @ text_features_.permute(0, 2, 1))
                if finetuning or (not finetuning and self.cur_task_idx == i):
                    if self.kwargs["frozen_prior"]:
                        prior_text_features = self.frozen_text_features_individual.clone()[start_cls_idx:end_cls_idx]
                        pdist = self.get_variational_adapter_features(prior_text_features.mean(1),
                                                                      i if self.kwargs["expandable_adapter"] else 0)
                    else:
                        pdist = self.get_prior_dist(context, text_features_relevant, labels, i,
                                                    None,
                                                    self.task_tokens[i] if self.kwargs["expandable_tokens"] else None,
                                                    use_np_prior=self.kwargs["use_np_prior"],
                                                    # if not finetuning else False,
                                                    tgt_mask=attn_mask
                                                    )
                    prior_matching_losses.append(kl_divergence(qdist, pdist).mean(0).sum() * 0.001)

                logits.append(logits_)
                if (self.kwargs["get_interclass_dist"] and self.cur_task_idx == 9 and finetuning) or (
                        self.kwargs["get_adapter_distances"] and self.cur_task_idx > 0):
                    with torch.no_grad():
                        per_sample_text_feats.append(rsamples.clone().detach().mean(0))

            if self.kwargs["ortho_loss"] and self.cur_task_idx >= 0:
                taskwise_means = torch.cat(taskwise_means)
                # taskwise_means = taskwise_means / taskwise_means.norm(dim=-1, keepdim=True)
                sims = taskwise_means @ taskwise_means.t()
                kl_losses.append(
                    F.cross_entropy(sims, torch.arange(sims.size(0)).cuda(device=self.kwargs["default_gpu"])) * 5)

            logits = torch.cat(logits, -1)

            kl_loss = sum(kl_losses) if len(kl_losses) else 0.
            prior_matching_loss = sum(prior_matching_losses)
            # prior_matching_loss = prior_matching_loss * 0.01 #if not finetuning else prior_matching_loss * 0.1

            avg_cos_distance = None
            if self.kwargs["get_adapter_distances"] and self.cur_task_idx > 0:
                with torch.no_grad():
                    per_sample_text_feats_ = torch.stack(per_sample_text_feats, 0)
                    avg_cos_distance = self.get_avg_inter_adapter_distance(per_sample_text_feats_)

            if self.kwargs["get_interclass_dist"] and self.cur_task_idx == 9 and finetuning:
                with torch.no_grad():
                    per_sample_text_feats_ = torch.cat(per_sample_text_feats, 0)
                    for label in np.arange(per_sample_text_feats_.shape[0]):
                        if label not in self.classwise_centroids:
                            self.classwise_centroids[label] = per_sample_text_feats_[label].unsqueeze(0)
                        else:
                            self.classwise_centroids[label] = torch.cat(
                                [self.classwise_centroids[label], per_sample_text_feats_[label].unsqueeze(0)], 0)

            return logits, (kl_loss, prior_matching_loss, avg_cos_distance)

    def finetuning(self, logits, logits_prior):
        student_conf = -torch.logsumexp(logits, dim=-1)
        teacher_conf = -torch.logsumexp(logits_prior, dim=-1)
        # if confidence > 1, it means student has a higher energy in which case the instance should be distilled using teacher logits
        confidence_ratio = student_conf / teacher_conf
        mask = confidence_ratio > 1
        student_dist = F.log_softmax(logits[mask], dim=-1)
        teacher_dist = F.softmax(logits_prior[mask], dim=-1)
        # kld = -1. * (student_dist * teacher_dist).sum(-1).mean()#.unsqueeze(0).expand(student_dist.shape[0], -1, -1)).sum(-1).mean()
        kld = nn.KLDivLoss(reduction='batchmean')(student_dist, teacher_dist)  # .sum(-1).mean()
        return kld * 0.1

    def get_naive_distillation_loss(self, curr_model_logits, image_feats, image_feats_normed, prev_cls_num):
        # from the BiC paper (Large scale incremental learning)
        with torch.no_grad():
            prev_model_logits = self.forward_prev_model(image_feats, image_feats_normed)
            prev_model_logits = prev_model_logits.detach()

        kl_loss = nn.KLDivLoss(reduction='none')(F.log_softmax(curr_model_logits[:, :, :prev_cls_num], dim=-1),
                                                 F.softmax(prev_model_logits, dim=-1)).sum(-1).mean()
        lamb = prev_cls_num / self.n_class
        return kl_loss * lamb

    @torch.no_grad()
    def set_classifier(self):
        pass

    @property  # 变成属性
    def dtype(self):
        return self.image_encoder.conv1.weight.dtype  # return int/float


class CLAP4CLIP(Finetune):
    def __init__(self, backbone, feat_dim, num_class, **kwargs):
        super().__init__(backbone, feat_dim, num_class, **kwargs)
        # kwargs
        self.kwargs = kwargs
        
        # Optimization parameters
        # fixed: 这是参数里的默认学习率
        self.lr = kwargs.get('lr', 0.001)*kwargs.get('train_batch',32)/20
        self.wd = kwargs.get('wd', 0.0)# default value for weight decay
        self.epochs = kwargs.get('epochs', 5)  # default epochs
        self.train_batch = kwargs.get('train_batch',32)  # default train batch size
        self.test_batch = kwargs.get('test_batch', 32)  # default test batch size

        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # model
        clip_model = backbone  # maybe
        clip_model.eval()
        if self.kwargs["use_float32"]:
            clip_model.float()
        self.clip_model = clip_model  # not equal to self.model, which is defined in self.init_model()
        ctx_dim = self.clip_model.ln_final.weight.shape[0]
        # print("ctx_dim in self.clip_model in ClClipVar:",ctx_dim)  # 512

        self.kwargs["lr"] = kwargs["lr"] * self.kwargs["train_batch_size"] / 20
        self.current_class_names = []

        decoder_layer = torch.nn.TransformerDecoderLayer(d_model=ctx_dim, nhead=1, activation='gelu',
                                                         batch_first=True).cuda(device=self.kwargs["default_gpu"]).type(
            self.clip_model.dtype)
        self.vga = torch.nn.TransformerDecoder(decoder_layer, 1) if kwargs["use_vga"] else None

        self.get_variational_adapters(ctx_dim)
        self.vga_global = None
        if self.kwargs["hierarchical"]:
            self.get_variational_adapters(ctx_dim, global_adapter=True)

        self.init_task_tokens(ctx_dim)

        self.task_to_cls_num = {}
        self.task_to_distribution = {}

        # for distillation
        self.previous_mu_adapters, self.previous_mu_global_adapter = None, None
        self.previous_sigma_adapters, self.previous_sigma_global_adapter = None, None
        self.previous_task_tokens = None
        self.previous_vga = None

        # fixed: task 索引的初始化
        self.cur_task_idx = 0

        # directories
        self.save_path = kwargs["save_path"]
        self.ckpt_path = self.kwargs["ckpt_path"] 
        self.checkpoint = self.kwargs["checkpoint"]
        
        # fixed: 应用数据变换 todo: 为什么变换之后准确率骤降成20%？
        self.train_transforms_list = [
        transforms.Resize(224, interpolation=BICUBIC),
        transforms.CenterCrop(224),
        lambda image: image.convert("RGB"),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),(0.26862954,0.26130258, 0.27577711)),
        ]
    
        self.test_transforms_list = [
        transforms.Resize(224, interpolation=BICUBIC),
        transforms.CenterCrop(224),
        lambda image: image.convert("RGB"),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),(0.26862954,0.26130258, 0.27577711)),
        ]
        
        self.train_transform = transforms.Compose(self.train_transforms_list)
        self.test_transform = transforms.Compose(self.test_transforms_list)
        
        def mkdir_p(path):  # todo: this func should not be placed here, and should be written in utils files...
            '''make dir if not exist'''
            try:
                os.makedirs(path)
            except OSError as exc:  # Python >2.5
                if exc.errno == errno.EEXIST and os.path.isdir(path):
                    pass
                else:
                    raise

        if not os.path.isdir(self.kwargs["ckpt_path"]):
            mkdir_p(self.kwargs["checkpoint"])
        if not os.path.isdir(self.kwargs["save_path"]):
            mkdir_p(self.kwargs["save_path"])
        np.save(self.kwargs["checkpoint"] + "/seed.npy", self.kwargs["seed"])
        
        # fixed: 没这个定义跑不了, args改成kwargs
    def init_task_tokens(self, ctx_dim):
        task_token = torch.zeros((1, 1,  ctx_dim), dtype=self.clip_model.dtype, requires_grad=True).cuda(device=self.kwargs["default_gpu"]) 
        nn.init.normal_(task_token, std=.02)
        self.task_tokens =  nn.ParameterList([nn.Parameter(task_token)]) if self.kwargs["expandable_tokens"] else None

    def observe(self, data):
        # todo: inherit all the logic from Finetune, or overwrite?
        # DONE: 接口对齐
        # 确保任务索引同步
        if hasattr(self, 'model') and self.model is not None:
            if self.model.cur_task_idx != self.cur_task_idx:
                self.model.cur_task_idx = self.cur_task_idx
                print(f"Synced model cur_task_idx to {self.cur_task_idx}")
        
        x, y = data['image'], data['label']
        x = x.to(self.device)
        y = y.to(self.device)
        # x = torch.stack([self.train_transform(transforms.ToPILImage()(x[i])) for i in range(x.shape[0])])  # fixed: 应用数据变换
        output, (kl_loss, prior_matching_loss, _) = self.model(x.cuda(device=self.kwargs["default_gpu"]), y)  # todo: check correct or not
        if self.kwargs["variational"]:
            targets = y.unsqueeze(0).expand(output.shape[0], -1).contiguous().view(-1)
            output = output.view(-1, output.shape[-1])
        else:
            targets = y

        loss = F.cross_entropy(output, targets) + kl_loss + prior_matching_loss
        # print("output shape:", output.shape)
        pred = torch.argmax(output, dim=1)  # DONE: ok! output shape: torch.Size([640, 10])
        # fixed: pred和y的维度不匹配！pred shape: torch.Size([640]), y shape: torch.Size([32]) 改用target: torch.Size([640])
        top1_acc = accuracy(output, targets, topk=(1,))[0]  # 计算top-1准确率  # DONE: dim ok??? 
        # print(f"Predictions: {pred}, Targets: {targets}, Accuracy: {acc / x.size(0)}, Loss: {loss.item()}")  # for debug
        # DONE: accuracy的计算方法不对。参考clap4clip/classifier/evaluator.py和clap4clip/utils/eval.py
        # print("Observe loss:",loss)
        acc = top1_acc.item() / 100.0
        # print("Observe Accuracy:", acc) 
        return pred, acc, loss

    def inference(self, data):
        # todo: inherit all the logic from Finetune, or overwrite?
        x, y = data['image'], data['label']
        x = x.to(self.device)
        y = y.to(self.device)
        # x = torch.stack([self.test_transform(transforms.ToPILImage()(x[i])) for i in range(x.shape[0])])# fixed: 应用数据变换

        logits, _ = self.model(x.cuda(device=self.kwargs["default_gpu"]), y, test=True, return_mean=False)
        
        # DONE: 从logits到accuracy的计算方法不对。参考clap4clip/classifier/evaluator.py和clap4clip/utils/eval.py
        if logits.dim() == 3:
            # 变分：平均所有采样, 参考evaluator.py
            logits = logits.mean(0)  # [batch_size, num_classes]
        
        # print(f"预测结果: {pred}")
        # print(f"真实标签: {y}")
        pred = torch.argmax(logits, dim=1)  # !!!
        acc = (pred == y).float().mean().item()
        # print("Inference Accuracy:", acc/ x.size(0), "or? ", acc)  # 打印准确率
        return pred, acc # fixed: 返回的应该是正确率
    
    def before_task(self, task_idx, buffer, train_loader, test_loaders):
        
        self.update_task_idx(task_idx)# fixed: 任务索引更新
        # DONE: check ok? 
        self.task_to_cls_num[task_idx] = len(train_loader.dataset.get_class_names())  # todo: why dataset has this attr--class_names
        self.current_class_names += train_loader.dataset.get_class_names()# fixed : dataset has no attribute named 'class_names'
        # self.prompt_templates = train_loader.dataset.prompt_templates  # DONE: 复杂。需要自己搓。不属于kwargs
        self.prompt_templates = ["a photo of a {}."] # fixed: 相当于 cifar100 的 sigle templates, 还有 ensemble 方法没有实现
        # DONE: 如何利用 clap4clip 对 cifar100 自定义的类别顺序？好像不用，被注释掉了。
        self.cur_task_idx = task_idx

        if len(train_loader.dataset)< self.kwargs["train_batch_size"]:
            real_img_bsz = len(train_loader.dataset)  
            self.kwargs["lr"] = self.kwargs["lr"] * real_img_bsz / self.kwargs["train_batch_size"]
            
        per_epoch_steps = len(train_loader)

        self.init_model(class_names=self.current_class_names, cur_task_idx=self.cur_task_idx, per_epoch_steps=per_epoch_steps, prompt_templates=self.prompt_templates)

        if self.model.vga is not None:
            self.model.vga.train()
            
        # todo: memory
        # if task_idx > 0:
        #     with open(self.save_path + "memory_"+str(task_idx)+".pickle", "rb") as f:
        #         buf = pickle.load(f)
        #         buffer.images = list(buf["images"])
        #         buffer.labels = list(buf["labels"])
        
    def get_current_task_class_indexes(self, task_idx):
        """计算当前任务的类别索引"""
        start_idx = 0
        for i in range(task_idx):
            start_idx += self.task_to_cls_num[i]
        
        end_idx = start_idx + self.task_to_cls_num[task_idx]
        cur_cls_indexes = list(range(start_idx, end_idx))
        
        print(f"Task {task_idx} class indexes: {cur_cls_indexes}")
        return cur_cls_indexes

    def after_task(self, task_idx, buffer, train_loader, test_loaders):
        print("+++++ after_task +++++")
        # check ok?
        self.model.eval()
        self.model.set_classifier()
        cur_cls_indexes = self.get_current_task_class_indexes(task_idx)

        # 统计 inter_adapter_distances、class centroids 等
        # if self.args.get_adapter_distances:
        #     self.compute_adapter_distances()
        
        if task_idx > 0:
        #     trsf = transforms.Compose([
        #     transforms.Resize(224, interpolation=BICUBIC, antialias=True),
        #     transforms.CenterCrop(224),
        #     lambda image: image.convert("RGB"),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
        #                        (0.26862954, 0.26130258, 0.27577711)),
        # ])
        #     buffer.update(self.model, train_loader, trsf, task_idx, self.kwargs["total_cls_num"], cur_cls_indexes, self.device)  # ???
        #     buffer.reduce_old_data(task_idx, self.kwargs["total_cls_num"])
        #     with open(self.kwargs["save_path"] + "memory_"+str(task_idx)+".pickle", "wb") as f:
        #         pickle.dump({"images": buffer.images, "labels": buffer.lables}, f)
            if self.kwargs["finetune"] and buffer is not None:
                def seed_worker(worker_id):
                    worker_seed = torch.initial_seed() % 2 ** 32
                    np.random.seed(worker_seed)
                    random.seed(worker_seed)

                g = torch.Generator()
                g.manual_seed(0)
                
                buffer_transform = transforms.Compose([
                    transforms.ToTensor(),  
                ])

                memory_loader = DataLoader(BufferDataset(images=buffer.images, labels=buffer.labels,transform=buffer_transform, mode='train', data_root=self.kwargs["data_root"]),
                                           batch_size=buffer.batch_size, shuffle=True,num_workers=8, worker_init_fn=seed_worker,generator=g)
                self.finetuning(memory_loader)

        self.model.eval()
        self.model.set_classifier() 
        if self.kwargs["distill"]:
            self.preserve_copy_for_distillation()
            
    def expand_adapter(self):
        ctx_dim = self.clip_model.ln_final.weight.shape[0]
        dtype = self.clip_model.dtype
        new_mu = Adapter(ctx_dim, ctx_dim).cuda(device=self.kwargs["default_gpu"]).type(dtype)
        new_sigma = Adapter(ctx_dim, ctx_dim, sigma=True).cuda(device=self.kwargs["default_gpu"]).type(dtype)
        
        print
        
        
        self.mu_adapters.append(new_mu)
        self.sigma_adapters.append(new_sigma)
        self.mu_adapters[:-1].eval()
        self.sigma_adapters[:-1].eval()
        freeze_parameters(self.mu_adapters[:-1], requires_grad=False)
        freeze_parameters(self.sigma_adapters[:-1], requires_grad=False)
        freeze_parameters(self.mu_adapters[-1], requires_grad=True)
        freeze_parameters(self.sigma_adapters[-1], requires_grad=True)
        
    def expand_task_token_list(self):
        new_task_token = deepcopy(self.task_tokens[-1])
        nn.init.trunc_normal_(new_task_token, std=.02)
        self.task_tokens.append(new_task_token)
        freeze_parameters(self.task_tokens[:-1], requires_grad=False)
        freeze_parameters(self.task_tokens[-1], requires_grad=True)
        
    def expand_prompts(self):
        ctx_vectors = deepcopy(self.ctx[-1])
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx.append(ctx_vectors)
        freeze_parameters(self.ctx[:-1], requires_grad=False)
        freeze_parameters(self.ctx[-1], requires_grad=True)


    def get_parameters(self, config):
        # DONE? see logic in paper
        #return filter(lambda p: p.requires_grad, self.model.parameters())
        
        train_parameters = []
        
        # fixed: 遍历所有参数，只选择需要梯度的参数
        trainable_params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
        
        if trainable_params:
            train_parameters.append({"params": trainable_params})
        
        return train_parameters
    
    def init_model(self, class_names, cur_task_idx, per_epoch_steps, prompt_templates=None):
        # DONE: 逻辑要改? 要接收cur_task_idx 
        if cur_task_idx > 0:  # current task idx.
            freeze_parameters(self.vga, requires_grad=True)
            if self.kwargs["expandable_tokens"]:
                self.expand_task_token_list()
            if self.kwargs["expandable_adapter"]:
                self.expand_adapter()
            if self.kwargs["expandable_prompt"]:
                self.expand_prompts()

        clip_model = deepcopy(self.clip_model)

        prev_model_components = (
                                 self.previous_mu_adapters, self.previous_sigma_adapters, 
                                 self.previous_task_tokens, self.previous_vga, 
                                 self.previous_mu_global_adapter, self.previous_sigma_global_adapter )
        self.model = CLIP(kwargs=self.kwargs, class_names=class_names, clip_model=clip_model, vga=self.vga,
                          mu_adapters=self.mu_adapters, sigma_adapters=self.sigma_adapters,
                          task_tokens=self.task_tokens, task_to_cls_num = self.task_to_cls_num,
                          prompt_templates=prompt_templates, previous_components=prev_model_components,
                          task_to_distribution=self.task_to_distribution,
                          mu_global_adapter=self.mu_global_adapter if self.kwargs["hierarchical"] else None,
                          sigma_global_adapter=self.sigma_global_adapter if self.kwargs["hierarchical"] else None,
                           global_vga=self.vga_global,
                           cur_task_idx=cur_task_idx # to maintain current task index
                          )  # diy CLIP
        self.model.eval()
        if self.kwargs["use_grad_checkpoint"]:
            try:
                self.model.text_encoder.transformer.use_gradient_checkpoint = True 
            except:
                self.model.text_encoder.module.transformer.use_gradient_checkpoint = True


        self.build_optimizer(per_epoch_steps, lr=self.lr, warmup=True)
        
    def update_task_idx(self, task_idx):
        """更新当前任务索引"""
        self.cur_task_idx = task_idx
        if hasattr(self, 'model') and self.model is not None:
            self.model.cur_task_idx = task_idx
            print(f"Updated cur_task_idx to {task_idx}")
    
    def finetuning(self, memory_loader):
        # todo: use buffer for finetuning instead of memory_loader
        self.unfreeze_for_finetuning()
        self.cur_iter_idx = 0
        if len(memory_loader.dataset)< self.kwargs["train_batch_size"]:
            real_img_bsz = len(memory_loader.dataset)
            self.lr = self.lr * real_img_bsz / self.kwargs["train_batch_size"] 
        else:
            real_img_bsz = self.kwargs["train_batch_size"]
            
        per_epoch_steps=len(memory_loader)
        inter_adapter_distances = []
        self.build_optimizer(per_epoch_steps=per_epoch_steps, lr=self.lr/10., warmup=False, finetune=True)
        if self.model.vga is not None:
            self.model.vga.eval()

        for epoch in tqdm(range(self.kwargs["finetune_epochs"])): 
            for idx, (x, y) in tqdm(enumerate(memory_loader), total=len(memory_loader), desc = 'Finetuning'):

                cur_iter_idx = epoch*per_epoch_steps+idx
                self.cur_iter_idx = cur_iter_idx
                self.scheduler.step(cur_iter_idx)
                # fixed: transform y from tuple to tensor
                y = torch.tensor(list(y), dtype=torch.long)
                x = x.to(self.device)
                y = y.to(self.device)

                output, (kl_loss, prior_matching_loss, inter_adapter_distance) = self.model(x.cuda(device=self.kwargs["default_gpu"]), y, finetuning=True)
                # pdb.set_trace()
                y = y.cuda(device=self.kwargs["default_gpu"])
                # pdb.set_trace()
                loss = 0.
                if self.kwargs["variational"]:
                    targets = y.unsqueeze(0).expand(output.shape[0], -1).contiguous().view(-1)
                    output = output.view(-1, output.shape[-1])
                else:
                    targets = y 
                loss = loss + F.cross_entropy(output, targets) + kl_loss + prior_matching_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if inter_adapter_distance is not None and (epoch == self.epochs-1):
                        inter_adapter_distances.append(inter_adapter_distance)
        if self.cur_task_idx == 9 and self.kwargs["get_interclass_dist"]:
            with torch.no_grad():
                self.compute_class_centroids()
        if len(inter_adapter_distances):
                print(f"Average inter-adapter distance: {np.mean(inter_adapter_distance)}")

        if self.cur_task_idx > 0 and self.kwargs["expandable_tokens"]:
            self.epoch_log()
    
    # fixed: 没有这个方法跑不起来，但是可以照搬吗？        
    def build_optimizer(self, per_epoch_steps, lr, warmup=False, finetune=False):
        for name, param in self.model.named_parameters():
            if "vga" not in name and "task_token" not in name and "adapter" not in name:
                param.requires_grad_(False)
            
        # double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)

        print(f"\nParameters to be updated: {sorted(enabled)}\n")

        param_dict = [{'params': [p for p in self.model.parameters() if p.requires_grad]}]

        self.optimizer = torch.optim.SGD(param_dict, lr=lr, weight_decay=self.wd)
        total_step=self.epochs*per_epoch_steps if not finetune else self.kwargs["finetune_epochs"]*per_epoch_steps
        warmup_steps = int(0.3 * total_step) if warmup else 0
        self.scheduler = build_cosine_scheduler(
            self.optimizer,
            lr=lr,
            total_step=total_step,
            lr_warmup_step=warmup_steps
            )
        

    
    @torch.no_grad()
    def preserve_copy_for_distillation(self):
        self.model.eval()
        self.previous_mu_adapters = deepcopy(self.model.mu_adapters)
        self.previous_sigma_adapters = deepcopy(self.model.sigma_adapters)
        self.previous_task_tokens = deepcopy(self.model.task_tokens)
        self.previous_vga = deepcopy(self.model.vga)
        if self.kwargs["hierarchical"]:
            self.previous_mu_global_adapter = deepcopy(self.model.mu_global_adapter)
            self.previous_sigma_global_adapter = deepcopy(self.model.sigma_global_adapter)
            freeze_parameters(self.previous_mu_global_adapter, requires_grad=False)
            freeze_parameters(self.previous_sigma_global_adapter, requires_grad=False)
        freeze_parameters(self.previous_mu_adapters, requires_grad=False)
        freeze_parameters(self.previous_sigma_adapters, requires_grad=False)
        freeze_parameters(self.previous_task_tokens, requires_grad=False)
        freeze_parameters(self.previous_vga, requires_grad=False)
        
        
    def get_variational_adapters(self, ctx_dim, global_adapter=False):
        if not global_adapter:
            self.mu_adapters = nn.ModuleList([Adapter(ctx_dim, ctx_dim).cuda(device=self.kwargs["default_gpu"]).type(self.clip_model.dtype)])
            self.sigma_adapters = nn.ModuleList([Adapter(ctx_dim, ctx_dim, sigma=True).cuda(device=self.kwargs["default_gpu"]).type(self.clip_model.dtype)])
            self.mu_adapter_deter = None
        else:
            self.mu_global_adapter = Adapter(ctx_dim, ctx_dim).cuda(device=self.kwargs["default_gpu"]).type(self.clip_model.dtype)
            self.sigma_global_adapter = Adapter(ctx_dim, ctx_dim, sigma=True).cuda(device=self.kwargs["default_gpu"]).type(self.clip_model.dtype)

    def unfreeze_for_finetuning(self, requires_grad=True):
        freeze_parameters(self.vga, requires_grad=False)
        freeze_parameters(self.mu_adapters[:-1], requires_grad=requires_grad)
        freeze_parameters(self.sigma_adapters[:-1], requires_grad=requires_grad)
        if self.kwargs["expandable_tokens"]:
            freeze_parameters(self.task_tokens[:-1], requires_grad=requires_grad)
        if requires_grad:
            self.mu_adapters[:-1].train()
            self.sigma_adapters[:-1].train()
            
    @torch.no_grad()
    def compute_class_centroids(self):
        class_embeddings = {}
        for cls,  class_embedding in self.model.classwise_centroids.items():
            class_embeddings[cls] = class_embedding.mean(0)
        class_embeddings = dict(sorted(class_embeddings.items()))
        class_embeddings = torch.stack(list(class_embeddings.values()))
        class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
        pairwise_cosine_dists = class_embeddings @ class_embeddings.t()
        pairwise_cosine_dists = pairwise_cosine_dists.cpu()
        torch.save(pairwise_cosine_dists, "3.pt")
        
    
    @torch.no_grad()
    def epoch_log(self):  # todo: use log in LibContinual is enough...?
        """Write here whatever you want to log on the internal state of the model."""
        log = {}

        # Compute mean distance between class tokens
        mean_dist, min_dist, max_dist = [], float('inf'), 0.
        for i in range(len(self.task_tokens)):
            for j in range(i + 1, len(self.task_tokens)):
                dist = torch.norm(self.task_tokens[i] - self.task_tokens[j], p=2).item()
                mean_dist.append(dist)

                min_dist = min(dist, min_dist)
                max_dist = max(dist, max_dist)

        if len(mean_dist) > 0:
            mean_dist = sum(mean_dist) / len(mean_dist)
        else:
            mean_dist = 0.
            min_dist = 0.

        assert min_dist <= mean_dist <= max_dist, (min_dist, mean_dist, max_dist)
        log['token_mean_dist'] = round(mean_dist, 5)
        log['token_min_dist'] = round(min_dist, 5)
        log['token_max_dist'] = round(max_dist, 5)
        print(f"\n{log}")
# ************* cifar 100 Dataset specific code *************

class cifar100(Dataset):
    # base_folder = 'cifar-100-python'
    train_list=[
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list=[
        ['test', 'foef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta={
        'filename':'meta',
        'key': 'fine_label_names',
        'md5':'7973b15100ade9c7d40fb424638fde48'
    }

    templates=[
        'a photo of a {}.',
        'a blurry photo of a {}.',
        'a black and white photo of a {}.',
        'a low contrast photo of a {}.',
        'a high contrast photo of a {}.',
        'a bad photo of a {}.',
        'a good photo of a {}.',
        'a photo of a small {}.',
        'a photo of a big {}.',
        'a photo of the {}.',
        'a blurry photo of the {}.',
        'a black and white photo of the {}.',
        'a low contrast photo of the {}.',
        'a high contrast photo of the {}.',
        'a bad photo of the {}.',
        'a good photo of the {}.',
        'a photo of the small {}.',
        'a photo of the big {}.',
    ]


    def __init__(self, root, transform=None, train=True):
        self.root = root
        self.train = train
        self.transform = transform
        self.base_folder = 'cifar-100-python'

        if self.train:
            downloaded_list = self.train_list 
        else:
            downloaded_list = self.test_list

        self.data =[]
        self.targets =[]
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))# convert to HWC

        self._load_meta()

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta[ 'key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img,target,int(index)

    def __len__(self):
        return len(self.data)

    def prompts(self, mode='single'):
        if mode =='single':
            prompts = [[self.templates[0].format(label)] for label in self.classes]
            return prompts
        elif mode == 'ensemble':
            prompts = [[template.format(label) for template in self.templates] for label in self.classes]
            return prompts

    def get_labels(self):
        return np.array(self.targets)

    def get_classes(self):
        return self.classes