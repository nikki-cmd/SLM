import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random 

class TinyStoriesDataset(Dataset):
    def __init__(self, file_path: str, tokenizer, block_size: int = 256, max_samples: int = None, shuffle: bool = False):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.pad_id = tokenizer.special_tokens["<PAD>"]
        self.eos_id = tokenizer.special_tokens["<EOS>"]
        
        #load all strings but not tokenize them
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        #optional: shuffle and bound them
        if max_samples and len(lines) > max_samples:
            if shuffle:
                random.seed(42)
                random.shuffle(lines)
            lines = lines[:max_samples]
        
        #delete empty and too short lines
        self.lines = [line.strip() for line in lines if len(line.strip()) > 0]
        
    def __len__(self):
        return len(self.lines)
    
    def __getitem__(self, idx):
        text = self.lines[idx]
        
        #encode text and add <EOS> in the end
        tokens = self.tokenizer.encode(text)
        tokens.append(self.eos_id) # learning model to predict end
        
        #cut up to block_size + 1(for x and y to be block_size)
        if len(tokens) > self.block_size + 1:
            tokens = tokens[:self.block_size + 1]
        
        else:
            tokens += [self.pad_id] * (self.block_size + 1 - len(tokens))
            
        #input: all except last one. Return all except first one
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        
        return x, y