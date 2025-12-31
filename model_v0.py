import torch
import torch.nn as nn

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
        #计算门控网络输出的概率
        logits = self.gate(x)
        probs = torch.softmax(logits, dim=-1)  #(batch_size, num_experts)
        #根据动态路由输出的概率计算路由损失
        routing_loss = self.calculate_routing_loss(probs)
        #expert_mask表示每个样本top-p策略选择的模型
        expert_mask  = self.top_sample(probs, self.top_p)
        print(f"选择了{expert_mask.float().sum().item()}个专家，分别为:{expert_mask},概率为:{probs.detach()}")

        # 只计算被选中的专家
        expert_outputs = []
        active_experts = []
        for i, expert in enumerate(self.experts):
            # 检查每个样本是否选择了当前专家
            batch_mask = expert_mask[:, i]  # (batch_size,)
            if batch_mask.any():  # 如果有样本选择了这个专家
                expert_out = expert(x)  # 计算所有样本的输出
                # 只保留被选中样本的输出，其他置零
                expert_out = expert_out * batch_mask.unsqueeze(-1).float()
                expert_outputs.append(expert_out.unsqueeze(1))
                active_experts.append(i)
            else:
                # 如果没有样本选择这个专家，设置为零
                expert_out = torch.zeros_like(expert(x))
                expert_outputs.append(expert_out.unsqueeze(1))
        
        #只有被选中的专家才有输出值，没选中的为0
        expert_outputs = torch.cat(expert_outputs, dim=1)  # (batch_size, num_experts, output_dim)
        #拿到被选中的专家的权重重新归一化
        mask_probs = probs * expert_mask.float()
        sum_weights = mask_probs.sum(dim=-1, keepdim=True)
        normalized_mask_probs = mask_probs / sum_weights  # (batch_size, num_experts)

        #加权求和被选择的专家的输出
        normalized_mask_probs = normalized_mask_probs.unsqueeze(-1)  # (batch_size, num_experts, 1)
        output = torch.sum(expert_outputs * normalized_mask_probs, dim=1)  # (batch_size, output_dim)

        return output, routing_loss
    
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

        mean_probs = probs.mean(dim=0, keepdim=True)  # (1, num_experts)
        mean_probs = mean_probs / (mean_probs.sum() + eps)

        expert_norm = expert_outputs / (expert_outputs.norm(dim=-1, keepdim=True) + eps)

        expert_mean = expert_norm.mean(dim=0)  # (num_experts, output_dim)

        V = mean_probs.T * expert_mean  # (num_experts, output_dim)
        

        gram = torch.matmul(V, V.T)  # (num_experts, num_experts)

        gram = gram + eps * torch.eye(num_experts, device=gram.device)
        det = torch.det(gram)
        log_det = torch.log(det + eps)

        diversity_loss = -log_det
        return diversity_loss
    

class Model(nn.Module):
    def __init__(self, input_sizes = [[8, 8], [16, 16], [32, 32]], output_size = [32, 32], hidden_dims = [[]],  num_experts = 8, top_p = 0.7):
        super().__init__()
        self.num_experts = num_experts
        self.top_p = top_p
        self.output_size = output_size
        self.W = nn.Parameter(torch.randn(*input_sizes[-1]))
        # 全部激活
        input_dim = 0
        for i in range(len(input_sizes)):
            input_dim += input_sizes[i][0] * input_sizes[i][1]
        self.moe_layer1 = MoeLayerAllLayer(
            input_dim=input_dim,
            output_dim=512,
            num_experts=self.num_experts,
            hidden_dims=hidden_dims[0]
        )
        #top-p策略激活
        output_dim = output_size[0] * output_size[1]
        self.moe_layer2 = MoeLayerTopPLayer(
            input_dim=512,
            output_dim=output_dim,
            num_experts=self.num_experts,
            hidden_dims=hidden_dims[1],
            top_p=top_p
        )

        self.net = nn.Sequential(
            self.moe_layer1,
            nn.GELU(),
            nn.Dropout(0.2),
            self.moe_layer2
        )

    def forward(self, x_8, x_16, x_32, mask):

        x_32 = x_32 + torch.matmul(mask, self.W)
        
        flatten_x_8, flatten_x_16, flatten_x_32 = torch.flatten(x_8, start_dim=1), torch.flatten(x_16, start_dim=1), torch.flatten(x_32, start_dim=1)
        input = torch.cat([flatten_x_8, flatten_x_16, flatten_x_32], dim=-1)

        output, routing_loss = self.net(input) #模型输出形状为(bathc_size,  output_size[0] * output_size[1])
        return output.reshape(-1, *self.output_size), routing_loss


if __name__ == '__main__':
    batch_size = 2
    input_sizes = [[8, 8], [16, 16], [32, 32]]
    output_size = [32, 32]
    top_p = 0.8
    num_experts = 8
    hidden_dims = [[512, 1024],[512, 512, 1024]]

    net = Model(input_sizes, output_size, hidden_dims,num_experts, top_p)
    x_8, x_16, x_32 = torch.randn((batch_size, *input_sizes[0])), torch.randn((batch_size, *input_sizes[1])), torch.randn((batch_size, *input_sizes[2]))
    mask = torch.randn((batch_size, *input_sizes[2]))
    output, routing_loss = net(x_8, x_16, x_32, mask)

    print(output.shape)


