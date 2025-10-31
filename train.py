import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from model import TransformerDecoderLM
from dataset import TinyStoriesDataset
import time
import os
import random
import json
from tokenizer import BPETokenizer
import tokenizer

config = {
    "vocab_size": None,          # будет установлено из токенизатора
    "d_model": 256,
    "n_layers": 4,
    "n_heads": 8,
    "block_size": 256,
    "batch_size": 32,
    "learning_rate": 3e-4,
    "weight_decay": 0.1,
    "max_epochs": 10,
    "grad_clip": 1.0,
    "eval_interval": 200,        # каждые N шагов — валидация и генерация
    "save_dir": "checkpoints",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42
}

#load tokenizer
def load_tokenizer(path: str):
    tok = tokenizer.BPETokenizer.load_tokenizer(path=path)
    return tok

tokenizer = load_tokenizer("bpe_tokenizer.json")
config["vocab_size"] = tokenizer.vocab_size
pad_id = tokenizer.special_tokens["<PAD>"]

#datasets and loaders
train_set = TinyStoriesDataset(
    file_path="datasets/TinyStories-train.txt",
    tokenizer=tokenizer,
    block_size=config["block_size"],
    max_samples=100000,
    shuffle=True
)

val_set = TinyStoriesDataset(
    file_path="datasets/TinyStories-valid.txt",
    tokenizer=tokenizer,
    block_size=config["block_size"],
    max_samples=5000
)

train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_set, batch_size=config["batch_size"], shuffle=False)

#model, optimizer, loss
model = TransformerDecoderLM(
    vocab_size=config["vocab_size"],
    d_model=config["d_model"],
    n_layers=config["n_heads"],
    max_seq_len=config["block_size"],
    dropout=0.1
).to(config["device"])

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config["learning_rate"],
    weight_decay=config["weight_decay"]
)

criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

@torch.no_grad()
def estimate_loss():
    model.eval()
    losses = {"train": 0.0, "val": 0.0}
    num_batches = {"train": min(20, len(train_loader)), "val": min(20, len(val_loader))}
    
    for split in ["train", "val"]:
        loader = train_loader if split == "train" else val_loader
        total_loss = 0.0
        
        for i, (x, y) in enumerate(loader):
            if i >= num_batches[split]:
                break
            
            x, y = x.to(config["device"]), y.to(config["device"])
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()
        losses[split] = total_loss / num_batches[split]
    model.train()
    return losses

@torch.no_grad()
def generate_text(prompt: str = "Once upon a time", max_new_tokens: int = 100):
    model.eval()
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens], dtype=torch.long).to(config["device"])
    
    for _ in range(max_new_tokens):
        #cut upto block_size
        if input_ids.size(1) > config["block_size"]:
            input_ids = input_ids[:, -config["block_size"]:]
            
        logits = model(input_ids)
        next_token_logits = logits[:, -1, :]
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=1)
        
        #stop on <EOS>
        if next_token.item() == tokenizer.special_tokens.get("<EOS>", -1):
            break
    
    generated = tokenizer.decode(input_ids[0].tolist())
    model.train()
    
    return generated

os.makedirs(config["save_dir"], exist_ok=True)

model.train()
step=0
best_val_loss = float('inf')

for epock in range(config["max_epochs"]):
    for x, y in train_loader:
        x, y = x.to(config["device"]), y.to(config["device"])
        
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
        optimizer.step()
        
        if step % config["eval_interval"] == 0:
            losses = estimate_loss()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            
            sample = generate_text("Once upon a time")
            print(f"Sample generation:\n{sample}\n{'-'*50}")
        
            #save the best model
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                checkpoint_path = os.path.join(config["save_dir"], "best_model.pth")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'step': step,
                    'val_loss': best_val_loss,
                    'config': config
                }, checkpoint_path)
                print(f"✅ Saved best model to {checkpoint_path}")
        
        step += 1
        
print("Training finished")