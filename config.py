import torch

HIDDEN_SIZE = 8
NUM_EPOCHS = 1000
BATCH_SIZE = 8
LEARNING_RATE = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"