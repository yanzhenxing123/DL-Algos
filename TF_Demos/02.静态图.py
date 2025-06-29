import tensorflow as tf


def self_attention(inputs, d_model, d_k, d_v):
    """手动实现 Self-Attention
    Args:
        inputs: [batch_size, seq_len, d_model]
        d_model: 输入维度
        d_k: Key 和 Query 的维度
        d_v: Value 的维度
    Returns:
        output: [batch_size, seq_len, d_v]
        attention_weights: [batch_size, seq_len, seq_len]
    """
    batch_size, seq_len, _ = tf.shape(inputs)

    # 初始化权重矩阵
    W_q = tf.Variable(tf.random.normal([d_model, d_k]))  # Query 权重
    W_k = tf.Variable(tf.random.normal([d_model, d_k]))  # Key 权重
    W_v = tf.Variable(tf.random.normal([d_model, d_v]))  # Value 权重

    # 计算 Q, K, V
    Q = tf.matmul(inputs, W_q)  # [batch_size, seq_len, d_k]
    K = tf.matmul(inputs, W_k)  # [batch_size, seq_len, d_k]
    V = tf.matmul(inputs, W_v)  # [batch_size, seq_len, d_v]

    # 计算注意力分数
    scores = tf.matmul(Q, K, transpose_b=True)  # [batch_size, seq_len, seq_len]
    scores = scores / tf.sqrt(tf.cast(d_k, tf.float32))  # 缩放

    # 计算注意力权重
    attention_weights = tf.nn.softmax(scores, axis=-1)  # [batch_size, seq_len, seq_len]

    # 计算输出
    output = tf.matmul(attention_weights, V)  # [batch_size, seq_len, d_v]

    return output, attention_weights


if __name__ == '__main__':
    inputs = tf.random.normal([5, 10, 64])
    d_model = 64
    d_k = 128
    d_v = 256
    output, attention_weights = self_attention(inputs, d_model, d_k, d_v)
    print(output)


