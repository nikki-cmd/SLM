import tokenizer

if __name__ == "__main__":
    new_tokenizer = tokenizer.BPETokenizer()
    new_tokenizer.train(
        file_path="datasets/TinyStories-train.txt",
        vocab_size=2000,
        max_lines=10000
    )
    new_tokenizer.save("bpe_tokenizer.json")
    
    test_text = "Never gonna give you up. Never gonna let you down"
    print("Original:", test_text)
    print("Tokens:    ", new_tokenizer.tokenize(test_text))
    print("IDs:       ", new_tokenizer.encode(test_text))
    print("Decoded:   ", new_tokenizer.decode(new_tokenizer.encode(test_text)))