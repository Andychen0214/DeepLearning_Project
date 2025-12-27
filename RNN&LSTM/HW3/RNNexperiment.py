import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import os
import time

BATCH_SIZE = 4096    
EMBED_SIZE_BASE = 128 
NUM_LAYERS_BASE = 2  
HIDDEN_SIZE_BASE = 256 
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
        print(f"錯誤：找不到 {train_file} 或 {valid_file}")
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

class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_size) 
        
        self.rnn = nn.RNN(
            input_size=embed_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        embeds = self.embedding(x)
        output, hidden = self.rnn(embeds, hidden)
        
        output = output[:, -1, :]
        out = self.fc(output)
        return out, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

def run_experiment(exp_name, vocab_size, train_loader, val_loader, 
                   embed_size, hidden_size, num_layers, 
                   learning_rate, epochs):
    model = SimpleRNN(vocab_size, embed_size, hidden_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = torch.cuda.amp.GradScaler()

    start_time = time.time()
    
    history = { 'train_loss': [], 'val_loss': [], 'train_error': [], 'val_error': [] }
    
    for e in range(epochs):
        model.train()
        h = model.init_hidden(BATCH_SIZE)
        
        batch_losses = []
        
        for inputs, targets in train_loader:
            inputs = inputs.to(device) 
            targets = targets.to(device)
            
            h = h.detach() 
            
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
    
    BASELINE_HISTORY = None
    print("\n" + "="*50)

    hidden_size_results = {}
    H_SIZES = [128, HIDDEN_SIZE_BASE, 512] 
    
    for h_size in H_SIZES:
        exp_name = f'H={h_size}'
        history = run_experiment(
            exp_name=exp_name,
            vocab_size=vocab_size, train_loader=train_loader_base, val_loader=val_loader_base,
            embed_size=EMBED_SIZE_BASE, hidden_size=h_size, num_layers=NUM_LAYERS_BASE,
            learning_rate=LEARNING_RATE, epochs=EPOCHS
        )
        hidden_size_results[exp_name] = history
        if h_size == HIDDEN_SIZE_BASE:
            BASELINE_HISTORY = history 

    plot_comparison(hidden_size_results, title="RNN Impact of Hidden Size (E=128, S=100)", filename_prefix="rnn_h_size_comp")
    print("\n" + "="*50)
 
    
    embed_size_results = {}
    E_SIZES = [64, EMBED_SIZE_BASE, 256] 

    for e_size in E_SIZES:
        exp_name = f'E={e_size}'
        if e_size == EMBED_SIZE_BASE:
            embed_size_results[exp_name] = BASELINE_HISTORY
            continue
        
        history = run_experiment(
            exp_name=exp_name,
            vocab_size=vocab_size, train_loader=train_loader_base, val_loader=val_loader_base,
            embed_size=e_size, hidden_size=HIDDEN_SIZE_BASE, num_layers=NUM_LAYERS_BASE,
            learning_rate=LEARNING_RATE, epochs=EPOCHS
        )
        embed_size_results[exp_name] = history

    plot_comparison(embed_size_results, title="RNN Impact of Embedding Size (H=256, S=100)", filename_prefix="rnn_e_size_comp")

    print("\n" + "="*50)
    
    seq_len_results = {}
    S_LENS = [50, SEQ_LEN_BASE, 150] 

    for s_len in S_LENS:
        exp_name = f'S={s_len}'
        
        if s_len == SEQ_LEN_BASE:
            seq_len_results[exp_name] = BASELINE_HISTORY
            continue
    
        train_loader_curr = get_data_loaders(train_data, s_len)
        val_loader_curr = get_data_loaders(valid_data, s_len)
        
        history = run_experiment(
            exp_name=exp_name,
            vocab_size=vocab_size, train_loader=train_loader_curr, val_loader=val_loader_curr,
            embed_size=EMBED_SIZE_BASE, hidden_size=HIDDEN_SIZE_BASE, num_layers=NUM_LAYERS_BASE,
            learning_rate=LEARNING_RATE, epochs=EPOCHS
        )
        seq_len_results[exp_name] = history

    plot_comparison(seq_len_results, title="RNN Impact of Sequence Length (H=256, E=128)", filename_prefix="rnn_s_len_comp")

    print("\n" + "="*50)
 