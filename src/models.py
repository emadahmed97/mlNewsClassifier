import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class FinetunedLLM(nn.Module):
    def __init__(self, llm, dropout_p)