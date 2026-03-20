import torch
import torch.nn as nn
import torch.nn.functional as F

def layer_wise_beam_search(model, start_token_id, beam_sizes, max_steps=3):
    """
    逐层动态 Beam Search (针对 RQ-VAE 语义 ID 生成)
    
    :param model: 序列推荐模型
    :param start_token_id: BOS (Begin of Sequence) 的 ID
    :param beam_sizes: 逐层的 Beam Size 数组，例如 [128, 256, 512]
    :param max_steps: 解码的层数 (默认 3 层 Codebook)
    :return: 最终生成的 top 候选序列及其得分
    """
    # 🌟 修复：Mac CPU 绝对安全的设备获取方式
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    
    # 1. 初始化候选池和得分
    # candidates 初始 shape: [1, 1] (即 1条路径，长度为1的起始Token)
    candidates = torch.tensor([[start_token_id]], dtype=torch.long, device=device)
    scores = torch.tensor([0.0], device=device)
    
    for step in range(max_steps):
        # 读取当前层规定的最大分支数
        current_beam_size = beam_sizes[step]
        
        # 2. 前向传播：模型吐出各个词的预测 Logits
        # logits shape: [num_paths, seq_len, vocab_size]
        logits = model(candidates)
        
        # 提取最后一个时间步的输出
        next_token_logits = logits[:, -1, :] 
        
        # 计算对数概率，防止连乘导致数值下溢
        log_probs = F.log_softmax(next_token_logits, dim=-1) 
        
        # 3. 累加得分 (广播机制)
        # scores.unsqueeze(1) shape: [num_paths, 1] -> 加上当前步的概率
        candidate_scores = scores.unsqueeze(1) + log_probs 
        
        # 4. 扁平化，准备做全局排序
        flat_scores = candidate_scores.view(-1)
        
        # 防御机制：实际候选数量不能超过 flat_scores 的总大小
        k = min(current_beam_size, flat_scores.size(0))
        
        # 5. 截取全局分数最高的 k 个分支
        top_scores, top_indices = torch.topk(flat_scores, k)
        
        # 6. 核心黑魔法：一维索引还原到二维 (来自哪个旧路径，选中了哪个新词)
        vocab_size = log_probs.size(-1)
        path_indices = top_indices // vocab_size  # 除法取商：找到它继承自哪条旧路径
        token_indices = top_indices % vocab_size  # 取余数：找到选中的新词是啥
        
        # 7. 拼接并更新池子
        best_paths = candidates[path_indices]
        new_tokens = token_indices.unsqueeze(1)
        candidates = torch.cat([best_paths, new_tokens], dim=1) # 拼上新词
        
        # 更新当前的全局得分
        scores = top_scores
        
        print(f"✅ Step {step+1}: 限制最大 Beam Size = {current_beam_size}")
        print(f"   -> 截取了全局 Top-{k} 个分支，当前候选池 Tensor Shape: {candidates.shape}")
        
    return candidates, scores

# ================= 测试执行 =================
class MockModel(nn.Module):
    def __init__(self, vocab_size=256):
        super().__init__()
        self.vocab_size = vocab_size
        # 🌟 修复：随便给个空壳参数，骗过 device 检测，避免 StopIteration
        self.dummy_param = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch, seq_len = x.shape
        # 返回完全随机的张量，模拟模型的预测输出，保持 device 同步
        return torch.randn(batch, seq_len, self.vocab_size, device=x.device)

if __name__ == "__main__":
    # 配置你的 RQ-VAE 参数
    VOCAB_SIZE = 256  # 对应 Codebook Size，这里必须跟上面 Mock 里的词表大小一致
    BOS_TOKEN = 0     # 假设 0 是 起始 Token
    BEAM_SIZES = [128, 256, 512] # 你的层级 Beam 大小
    
    # 实例化假模型
    mock_model = MockModel(vocab_size=VOCAB_SIZE)
    
    print("🚀 开始运行逐层动态 Beam Search (Mac CPU 测试版)...\n")
    final_candidates, final_scores = layer_wise_beam_search(
        model=mock_model, 
        start_token_id=BOS_TOKEN, 
        beam_sizes=BEAM_SIZES,
        max_steps=len(BEAM_SIZES)
    )
    
    print(f"\n🎉 搜索结束！")
    print(f"最终生成的候选商品序列数量: {final_candidates.size(0)} 条")
    print(f"每个序列的长度 (1 个 BOS + 3 层语义 ID): {final_candidates.size(1)} 个 Token")
    print("\n看一眼分数排第一的序列长什么样：")
    print(final_candidates[0].tolist(), f"得分: {final_scores[0].item():.4f}")

    import pdb
    pdb.set_trace()
    