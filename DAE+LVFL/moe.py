import torch
import torch.nn as nn

class MOE(nn.Module):
    def __init__(self, input_dim, num_experts=2, hidden_dim=512):
        super(MOE, self).__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([nn.Linear(input_dim, hidden_dim) for _ in range(num_experts)])
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        gate_output = self.gate(x)  # (batch_size, num_experts)
        gate_output = torch.softmax(gate_output, dim=-1)  

        expert_outputs = [expert(x).unsqueeze(1) for expert in self.experts]  # (batch_size, 1, hidden_dim)
        expert_outputs = torch.cat(expert_outputs, dim=1)  # (batch_size, num_experts, hidden_dim)

        weighted_output = torch.sum(expert_outputs * gate_output.unsqueeze(2), dim=1)  # (batch_size, hidden_dim)

        return weighted_output
