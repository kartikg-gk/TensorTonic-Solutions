import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
  
    pe = np.zeros((seq_length, d_model))

    pos = np.arange(seq_length).reshape(-1, 1)

    _2i = np.arange(0, d_model, 2)

    denominator = np.power(10000, _2i / d_model)

 
    pe[:, 0::2] = np.sin(pos / denominator)
    pe[:, 1::2] = np.cos(pos / denominator)

    return pe