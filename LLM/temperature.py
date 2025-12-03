import numpy as np

def apply_temperature(logits, temperature):
    # 1. 避免除以零的错误，通常设置一个极小值
    if temperature <= 0:
        # 如果温度为0，直接采用"Argmax"，即只选分数最大的那个（贪婪采样）
        probabilities = np.zeros_like(logits)
        probabilities[np.argmax(logits)] = 1.0
        return probabilities
        
    # 2. 核心步骤：将 Logits 除以温度
    # 温度越低，scaled_logits 的数值差距越大
    # 温度越高，scaled_logits 的数值差距越小
    scaled_logits = logits / temperature
    
    # 3. 计算 Softmax (为了数值稳定性，通常会减去最大值，这里省略)
    exp_logits = np.exp(scaled_logits)
    probabilities = exp_logits / np.sum(exp_logits)
    
    return probabilities

# 模拟数据
logits = np.array([2.0, 1.0, 0.1]) 

# 打印不同温度下的概率分布
print("T=1.0:", apply_temperature(logits, 1.0)) # 正常分布
print("T=0.1:", apply_temperature(logits, 0.1)) # 几乎全在第一个词
print("T=5.0:", apply_temperature(logits, 5.0)) # 概率非常平均