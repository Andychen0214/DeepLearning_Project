import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import os
import time

BATCH_SIZE = 4096    
EMBED_SIZE = 128    
NUM_LAYERS_BASE = 2 
HIDDEN_SIZE_BASE = 512 
LEARNING_RATE = 0.001 
EPOCHS = 20       
CLIP_MAX_NORM = 5
NUM_WORKERS = 8

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def load_text(train_file, valid_file):

    try:
        with open(train_file, 'r', encoding='utf-8') as f: train_text = f.read()
        with open(valid_file, 'r', encoding='utf-8') as f: valid_text = f.read()
    except FileNotFoundError:
        exit()
        
    
    all_text = train_text + valid_text
    chars = tuple(sorted(set(all_text)))
    int2char = dict(enumerate(chars))
    char2int = {ch: ii for ii, ch in int2char.items()}
    vocab_size = len(chars)
    
    train_encoded = np.array([char2int[ch] for ch in train_text])
    valid_encoded = np.array([char2int[ch] for ch in valid_text])
    
    return train_encoded, valid_encoded, vocab_size, int2char, char2int

class CharDataset(Dataset):
    def __init__(self, encoded_data, seq_length):
        self.encoded_data = encoded_data
        self.seq_length = seq_length
        self.num_sequences = len(encoded_data) - seq_length

    def __len__(self): return self.num_sequences

    def __getitem__(self, index):
        return torch.tensor(self.encoded_data[index:index+self.seq_length], dtype=torch.long), \
               torch.tensor(self.encoded_data[index+self.seq_length], dtype=torch.long)

def get_data_loaders(encoded_data, seq_length):
    dataset = CharDataset(encoded_data, seq_length)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, 
        num_workers=NUM_WORKERS, pin_memory=True
    )
    return loader


class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        self.lstm = nn.LSTM(
            input_size=embed_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=0.2 
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        embeds = self.embedding(x)
        
        output, hidden = self.lstm(embeds, hidden)
        
        output = output[:, -1, :]
        out = self.fc(output)
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.num_layers, batch_size, self.hidden_size).zero_().to(device),
                  weight.new(self.num_layers, batch_size, self.hidden_size).zero_().to(device))
        return hidden

def run_experiment(exp_name, vocab_size, train_loader, val_loader, 
                   embed_size, hidden_size, num_layers, 
                   learning_rate, epochs):
    
    model = SimpleLSTM(vocab_size, embed_size, hidden_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = torch.cuda.amp.GradScaler()
    

    print(f"  Config: H={hidden_size}, L={num_layers}, SeqLen={train_loader.dataset.seq_length}")
    start_time = time.time()
    
    history = { 'train_loss': [], 'val_loss': [], 'train_error': [], 'val_error': [] }
    
    for e in range(epochs):
        model.train()
        h = model.init_hidden(BATCH_SIZE)
        
        batch_losses = []
        
        for inputs, targets in train_loader:
            inputs = inputs.to(device) 
            targets = targets.to(device)
            # ---------------------------
            
            h = tuple([each.data for each in h]) 
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                output, h = model(inputs, h)
                loss = criterion(output, targets)
                
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CLIP_MAX_NORM)
            scaler.step(optimizer)
            scaler.update()
            
            batch_losses.append(loss.item())

        history['train_loss'].append(np.mean(batch_losses))
        history['val_loss'].append(0) 

        model.eval()
        batch_val_losses = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                val_h = model.init_hidden(BATCH_SIZE) 
                with torch.cuda.amp.autocast():
                    output, _ = model(inputs, val_h)
                    val_loss = criterion(output, targets)
                batch_val_losses.append(val_loss.item())

        history['val_loss'][-1] = np.mean(batch_val_losses) 

        if (e + 1) % 5 == 0 or e == 0 or e == epochs - 1:
            print(f'  Epoch {e+1:02d}/{epochs} | Val Loss: {history["val_loss"][-1]:.4f}')
              
    end_time = time.time()
    print(f"--- [實驗結束: {exp_name}] (耗時: {end_time - start_time:.2f} 秒) ---")
    return history


def plot_comparison(results_dict, title, filename_prefix):
    plt.figure(figsize=(10, 6))
    
    for experiment_name, history in results_dict.items():
        plt.plot(history['val_loss'], label=experiment_name, marker='o', markersize=4)
    
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.legend()
    plt.grid(True)
    
    safe_title = title.replace(" ", "_").replace("=", "").replace("/", "")
    filename = f'{filename_prefix}_{safe_title}.png'
    plt.savefig(filename)
    plt.show()
if __name__ == "__main__":
    

    try:
        train_data, valid_data, vocab_size, int2char, char2int = load_text(
            'shakespeare_train.txt', 'shakespeare_valid.txt'
        )
    except FileNotFoundError as e:
        print(e)
        exit()
    SEQ_LEN_BASE = 100
    train_loader_base = get_data_loaders(train_data, SEQ_LEN_BASE)
    val_loader_base = get_data_loaders(valid_data, SEQ_LEN_BASE)

    print("\n" + "="*50)
    print("="*50)

    hidden_size_results = {}
    H_SIZES = [256, HIDDEN_SIZE_BASE, 1024] 
    
    for h_size in H_SIZES:
        exp_name = f'H={h_size}'
        history = run_experiment(
            exp_name=exp_name,
            vocab_size=vocab_size, train_loader=train_loader_base, val_loader=val_loader_base,
            embed_size=EMBED_SIZE, hidden_size=h_size, num_layers=NUM_LAYERS_BASE,
            learning_rate=LEARNING_RATE, epochs=EPOCHS
        )
        hidden_size_results[exp_name] = history

    plot_comparison(hidden_size_results, title="LSTM Impact of Hidden Size (SeqLen=100)", filename_prefix="lstm_h_size_comp")
    print("\n" + "="*50)
    print("="*50)
    
    seq_len_results = {}
    S_LENS = [50, SEQ_LEN_BASE, 150] 
    H_SIZE_BASE_EXP = HIDDEN_SIZE_BASE 

    for s_len in S_LENS:
        exp_name = f'S={s_len}'

        if s_len == SEQ_LEN_BASE:
            baseline_history = hidden_size_results.get(f'H={H_SIZE_BASE_EXP}') 
            if baseline_history:
                seq_len_results[exp_name] = baseline_history
                continue
        
        train_loader_curr = get_data_loaders(train_data, s_len)
        val_loader_curr = get_data_loaders(valid_data, s_len)
        
        history = run_experiment(
            exp_name=exp_name,
            vocab_size=vocab_size, train_loader=train_loader_curr, val_loader=val_loader_curr,
            embed_size=EMBED_SIZE, hidden_size=H_SIZE_BASE_EXP, num_layers=NUM_LAYERS_BASE,
            learning_rate=LEARNING_RATE, epochs=EPOCHS
        )
        seq_len_results[exp_name] = history

    plot_comparison(seq_len_results, title="LSTM Impact of Sequence Length (Hidden=512)", filename_prefix="lstm_s_len_comp")

    print("\n" + "="*50)
    print("所有 LSTM 實驗 (Part 4-3) 完成！")
    print("請分析 lstm_h_size_comp.png 和 lstm_s_len_comp.png")
    print("="*50)