import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q, K, V, W_q, W_k, W_v, W_o, num_heads):
    batch_size, seq_len, d_model = Q.shape
    d_k = d_model // num_heads

    Q_proj = np.dot(Q, W_q)
    K_proj = np.dot(K, W_k)
    V_proj = np.dot(V, W_v)

    Q_proj = Q_proj.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    K_proj = K_proj.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    V_proj = V_proj.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)

    scores = np.matmul(Q_proj, K_proj.transpose(0, 1, 3, 2)) / np.sqrt(d_k)

    attn_weights = softmax(scores, axis=-1)

    head_output = np.matmul(attn_weights, V_proj)

    head_output = head_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)

    output = np.dot(head_output, W_o)

    return output