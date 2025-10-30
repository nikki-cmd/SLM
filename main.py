import tokenizer
import dataset
from torch.utils.data import DataLoader
from model import TransformerDecoderLM

if __name__ == "__main__":
    tok = tokenizer.BPETokenizer.load_tokenizer(path="bpe_tokenizer.json")
    '''new_tokenizer.train(
        file_path="datasets/TinyStories-train.txt",
        vocab_size=2000,
        max_lines=10000
    )
    new_tokenizer.save("bpe_tokenizer.json")
    
    test_text = "Never gonna give you up. Never gonna let you down"
    print("Original:", test_text)
    print("Tokens:    ", new_tokenizer.tokenize(test_text))
    print("IDs:       ", new_tokenizer.encode(test_text))
    print("Decoded:   ", new_tokenizer.decode(new_tokenizer.encode(test_text)))'''
    
    train_set = dataset.TinyStoriesDataset(
        file_path="datasets/TinyStories-train.txt",
        tokenizer=tok,
        block_size=256,
        max_samples=50000, 
        shuffle=True    
    )
    
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=2)
    
    x, y = next(iter(train_loader))
    print("Input shape:", x.shape)   # [32, 256]
    print("Target shape:", y.shape) # [32, 256]
    print("Sample input tokens:", x[0][:10])
    print("Sample decoded:", tok.decode(x[0].tolist()))
    
    
    model = TransformerDecoderLM(
    vocab_size=2000,
    d_model=256,
    n_layers=4,
    n_heads=8,
    max_seq_len=256,
    dropout=0.1
)
    x_batch, _ = next(iter(train_loader))
    logits = model(x_batch)
    print("Logits shape:", logits.shape)
    