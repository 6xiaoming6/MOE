import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

#每个专家网络就是简单的MLP input_size  --> hidden_size --> output_size
class Expert(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim = 1024, dropout=0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class MoeLayerAllLayer(nn.Module):
    def __init__(self, input_dim, num_experts, hidden_dim, output_dim):
        super().__init__()
        self.num_experts = num_experts
        #门控网络
        self.gate = nn.Linear(input_dim, num_experts)
        #专家网络
        self.experts = nn.ModuleList([
            Expert(input_dim, output_dim, hidden_dim) for _ in range(num_experts)
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

class MoeLayerTopKLayer(nn.Module):
    def __init__(self, input_dim, num_experts, hidden_dim, output_dim, top_k=2,
                 norm_topk_prob=True, dropout=0.1):
        """
        基于 Top-K 路由和 Switch Transformer 负载均衡损失的 MOE 层
        """
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm_topk_prob = norm_topk_prob

        # 门控网络
        self.gate = nn.Linear(input_dim, num_experts, bias=False)

        #专家网络
        self.experts = nn.ModuleList([
            Expert(input_dim, output_dim, hidden_dim) for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: 输入张量 (batch_size, sequence_length, input_dim) 或 (batch_size, input_dim)
        """
        original_shape = x.shape
        # 展平 batch 和 sequence 维度以便统一处理 -> (total_tokens, input_dim)
        if x.dim() == 3:
            x = x.view(-1, self.input_dim)

        batch_size_flat = x.shape[0]

        # 1. 门控网络计算路由 Logits
        router_logits = self.gate(x)  # (total_tokens, num_experts)

        # 2. 计算路由概率 (Softmax)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)

        # 3. Top-K 选择
        # selected_experts: (total_tokens, top_k)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)

        # 4. (可选) 对 top-k 权重进行归一化
        if self.norm_topk_prob:
            routing_weights = routing_weights / (routing_weights.sum(dim=-1, keepdim=True) + 1e-6)

        # 转换回输入数据类型
        routing_weights = routing_weights.to(x.dtype)

        # 5. 初始化输出
        final_output = torch.zeros(
            (batch_size_flat, self.output_dim),
            dtype=x.dtype,
            device=x.device
        )

        # 6. 创建专家掩码用于分发
        # one_hot: (total_tokens, top_k, num_experts)
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts)
        # permute -> (num_experts, top_k, total_tokens) 以便按专家循环
        expert_mask_flat = expert_mask.permute(2, 1, 0)

        # 7. 稀疏计算：遍历每个专家
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]

            # 查找分配给当前专家的 token
            # idx: 在 top-k 中的排名索引 (0..k-1)
            # top_x: token 在 batch 中的索引
            idx, top_x = torch.where(expert_mask_flat[expert_idx])

            if top_x.numel() > 0:
                # 获取对应的输入 token
                current_state = x[top_x]

                # 专家前向传播
                current_hidden_states = expert_layer(current_state)

                # 乘以路由权重: routing_weights[top_x, idx]
                weights = routing_weights[top_x, idx].unsqueeze(-1)  # (num_tokens, 1)
                current_hidden_states = current_hidden_states * weights

                # 累加到最终输出
                # 注意：多个专家可能处理同一个 token (因为是 Top-K)，所以是 add
                final_output.index_add_(0, top_x, current_hidden_states.to(x.dtype))

        # 8. 还原形状
        if len(original_shape) == 3:
            final_output = final_output.view(original_shape[0], original_shape[1], self.output_dim)

        # 9. 计算负载均衡损失 (Auxiliary Loss)
        aux_loss = self._compute_load_balancing_loss(router_logits, selected_experts)

        return final_output, aux_loss

    def _compute_load_balancing_loss(self, gate_logits, selected_experts):
        """
        计算 Switch Transformer 风格的负载均衡损失
        """
        # gate_logits: (total_tokens, num_experts)
        # selected_experts: (total_tokens, top_k)

        total_tokens = gate_logits.shape[0]

        # 1. 计算每个专家的路由概率
        routing_probs = F.softmax(gate_logits, dim=-1)  # (total_tokens, num_experts)

        # 2. 创建专家选择掩码
        # selected_experts: (total_tokens, top_k)
        expert_mask = torch.zeros_like(routing_probs)  # (total_tokens, num_experts)

        # 将选中的专家位置标记为1
        for i in range(self.top_k):
            expert_mask.scatter_(1, selected_experts[:, i:i + 1], 1)

        # 3. 计算每个专家的路由概率平均值
        # (num_experts,)
        router_prob_per_expert = routing_probs.mean(dim=0)

        # 4. 计算每个专家被选中的频率
        # (num_experts,)
        tokens_per_expert = expert_mask.sum(dim=0) / (total_tokens * self.top_k)

        # 5. 计算负载均衡损失
        # Switch Transformer 公式: loss = num_experts * sum(f_i * p_i)
        dot_product = torch.sum(router_prob_per_expert * tokens_per_expert)
        aux_loss = self.num_experts * dot_product

        return aux_loss

class Model(nn.Module):
    def __init__(self, input_sizes = [[8, 8], [16, 16], [32, 32]], output_dim = 162, hidden_dim = 1024,  num_experts = 8, top_k = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.output_dim = output_dim
        # 全部激活
        input_dim = 0
        for i in range(len(input_sizes)):
            input_dim += input_sizes[i][0] * input_sizes[i][1]
        self.moe_layer1 = MoeLayerAllLayer(
            input_dim=input_dim,
            output_dim=128,
            num_experts=self.num_experts,
            hidden_dim=hidden_dim
        )
        #top-p策略激活
        self.moe_layer2 = MoeLayerTopKLayer(
            input_dim=128,
            output_dim=self.output_dim,
            num_experts=self.num_experts,
            hidden_dim=hidden_dim,
            top_k=top_k
        )

    def forward(self, x_8, x_16, x_32):
        
        flatten_x_8, flatten_x_16, flatten_x_32 = torch.flatten(x_8, start_dim=1), torch.flatten(x_16, start_dim=1), torch.flatten(x_32, start_dim=1)
        input = torch.cat([flatten_x_8, flatten_x_16, flatten_x_32], dim=-1)

        output = self.moe_layer1(input)
        output = F.gelu(output)
        output = F.dropout(output, 0.1)
        output, load_balance_loss = self.moe_layer2(output) #模型输出形状为(bathc_size,  output_size[0] * output_size[1])

        return output, load_balance_loss


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32

    net = Model().to(device)

    x8 = torch.randn(batch_size, 8, 8).to(device)
    x16 = torch.randn(batch_size, 16, 16).to(device)
    x32 = torch.randn(batch_size, 32, 32).to(device)

    y, loss = net(x8, x16, x32)
    print(y.shape)
    print(loss)

