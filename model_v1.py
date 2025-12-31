import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims = []):
        super().__init__()

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.GELU())
        
        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            layers.append(nn.GELU())
        
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


class MoeLayerAllLayer(nn.Module):
    def __init__(self, input_dim, num_experts, hidden_dims, output_dim):
        super().__init__()
        self.num_experts = num_experts
        #门控网络
        self.gate = nn.Linear(input_dim, num_experts)
        #专家网络
        self.experts = nn.ModuleList([
            Expert(input_dim, output_dim, hidden_dims) for _ in range(num_experts)
        ])
        
    
    def forward(self, x):
        #计算门控网络输出的概率
        logits = self.gate(x)
        probs = torch.softmax(logits, dim=-1)  #(batch_size, num_experts)

        #所有专家都被激活
        expert_outputs = []
        for expert in self.experts:
            expert_out = expert(x)  #(batch_size, output_dim)
            expert_outputs.append(expert_out.unsqueeze(1)) #(batch_size, 1, output_dim)
        
        expert_outputs = torch.cat(expert_outputs, dim=1)  #(batch_size, num_experts, output_dim)

        #计算所有专家输出的加权和
        probs = probs.unsqueeze(-1) #(batch_size, num_experts, 1)
        output = torch.sum(expert_outputs * probs, dim=1) #(batch_size, output_dim)

        return output


class MoeLayerTopPLayer(nn.Module):
    def __init__(self, input_dim, num_experts, top_p, hidden_dims, output_dim):
        super().__init__()
        self.num_experts = num_experts
        self.top_p = top_p
        #门控网络
        self.gate = nn.Linear(input_dim, num_experts)
        #专家网络
        self.experts = nn.ModuleList([
            Expert(input_dim, output_dim, hidden_dims) for _ in range(num_experts)
        ])
    
    def forward(self, x):
        # 1. 门控网络
        logits = self.gate(x)
        probs = torch.softmax(logits, dim=-1)  # (batch_size, num_experts)
        
        # 2. 计算所有专家的输出 (为了正确的 Diversity Loss，这里需要计算所有专家)
        expert_outputs_list = []
        for expert in self.experts:
            expert_outputs_list.append(expert(x).unsqueeze(1))
        
        # (batch_size, num_experts, output_dim)
        all_expert_outputs = torch.cat(expert_outputs_list, dim=1)

        # 3. 计算损失 (使用原始概率和所有专家的输出来计算)
        routing_loss = self.calculate_routing_loss(probs)
        diversity_loss = self.calculate_diversity_loss(all_expert_outputs, probs)

        # 4. Top-P 采样与 Mask (只用于最终输出)
        # expert_mask: (batch_size, num_experts)
        expert_mask = self.top_sample(probs, self.top_p)
        

        # 5. 计算加权输出
        # 重新归一化权重 (只对被选中的专家)
        mask_probs = probs * expert_mask.float()
        sum_weights = mask_probs.sum(dim=-1, keepdim=True) + 1e-8
        normalized_mask_probs = mask_probs / sum_weights 
        
        # 加权求和: (batch_size, num_experts, 1) * (batch_size, num_experts, output_dim)
        # 此时未被选中的位是0，不会影响结果
        output = torch.sum(normalized_mask_probs.unsqueeze(-1) * all_expert_outputs, dim=1)

        return output, routing_loss, diversity_loss
    
    def top_sample(self, probs, top_p):
        #按概率从大到小排序，并拿到在原始序列中的索引
        probs_sorted, indices = torch.sort(probs, dim=-1, descending=True)
        #计算每个i位置的前缀累加概率和，选择p个专家
        prob_accum = torch.cumsum(probs_sorted, dim=-1)
        mask = prob_accum <= top_p
        #防止top_p过小或者top_1概率太高，保证至少要选择概率最高的top_1专家
        mask[:, 0] = True #现在的mask是相对于排好序的概率的

        #将mask映射回原始序列中
        _, original_indices = torch.sort(indices, dim=-1)
        mask_original = torch.gather(mask, -1, original_indices)

        return mask_original #(batch_size, num_experts)
    
    #输入的概率是所有专家的概率，不是top-p专家归一化后的
    def calculate_routing_loss(self, probs):
        #加上1e-8防止Log中出现0
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        return entropy.mean()  # 鼓励专家专业化
    
    #计算专家差异性损失
    def calculate_diversity_loss(self, expert_outputs, probs, eps=1e-6):
        batch_size, num_experts, output_dim = expert_outputs.shape

        # 1. 计算每个专家的平均关注度 (Mean Attention) -> \bar{g}_i
        # (num_experts, 1)
        mean_probs = probs.mean(dim=0, keepdim=True).T 

        # 2. 计算每个专家的“代表性方向”向量 -> \tilde{e}_i
        # 修正：先在 batch 维度求平均，得到专家的平均输出向量
        # (num_experts, output_dim)
        expert_mean_raw = expert_outputs.mean(dim=0) 
        
        # 再进行 L2 归一化，确保它是单位向量 (Direction)
        expert_norm = torch.norm(expert_mean_raw, p=2, dim=1, keepdim=True) + eps
        expert_directions = expert_mean_raw / expert_norm

        # 3. 构建矩阵 V
        V = mean_probs * expert_directions

        # 4. 计算 Gram 矩阵: G = V * V^T
        # (num_experts, num_experts)
        gram = torch.matmul(V, V.T)

        # 5. 计算 Log Determinant
        # 添加 Identity 噪声保证数值稳定性
        gram = gram + eps * torch.eye(num_experts, device=gram.device)
        
        # 这里直接最大化 log_det 相当于最大化体积
        log_det = torch.logdet(gram)
        
        # 因为我们要最大化差异(体积)，也就是最小化 -LogDet
        diversity_loss = -log_det
        
        return diversity_loss


class Model(nn.Module):
    def __init__(self, input_sizes = [[8, 8], [16, 16], [32, 32]], output_dim = 162, hidden_dims = [[]],  num_experts = 8, top_p = 0.7):
        super().__init__()
        self.num_experts = num_experts
        self.top_p = top_p
        self.output_dim = output_dim
        # 全部激活
        input_dim = 0
        for i in range(len(input_sizes)):
            input_dim += input_sizes[i][0] * input_sizes[i][1]
        self.moe_layer1 = MoeLayerAllLayer(
            input_dim=input_dim,
            output_dim=128,
            num_experts=self.num_experts,
            hidden_dims=hidden_dims[0]
        )
        #top-p策略激活
        self.moe_layer2 = MoeLayerTopPLayer(
            input_dim=128,
            output_dim=self.output_dim,
            num_experts=self.num_experts,
            hidden_dims=hidden_dims[1],
            top_p=top_p
        )

    def forward(self, x_8, x_16, x_32):
        
        flatten_x_8, flatten_x_16, flatten_x_32 = torch.flatten(x_8, start_dim=1), torch.flatten(x_16, start_dim=1), torch.flatten(x_32, start_dim=1)
        input = torch.cat([flatten_x_8, flatten_x_16, flatten_x_32], dim=-1)

        output = self.moe_layer1(input)
        output = F.gelu(output)
        output = F.dropout(output, 0.1)
        output, routing_loss, diversity_loss = self.moe_layer2(output) #模型输出形状为(bathc_size,  output_size[0] * output_size[1])

        return output, routing_loss, diversity_loss


