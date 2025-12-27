import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import os
import time

SEQ_LENGTH = 100    
BATCH_SIZE = 4096
HIDDEN_SIZE = 512
EMBED_SIZE = 128    
NUM_LAYERS = 2      
LEARNING_RATE = 0.001 
EPOCHS = 30         
CLIP_MAX_NORM = 5
NUM_WORKERS = 8      

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_text(train_file, valid_file):
    if not os.path.exists(train_file) or not os.path.exists(valid_file):
        raise FileNotFoundError("找不到檔案")
    with open(train_file, 'r', encoding='utf-8') as f: train_text = f.read()
    with open(valid_file, 'r', encoding='utf-8') as f: valid_text = f.read()

    
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

try:
    train_data, valid_data, vocab_size, int2char, char2int = load_text('shakespeare_train.txt', 'shakespeare_valid.txt')
    train_dataset = CharDataset(train_data, SEQ_LENGTH)
    valid_dataset = CharDataset(valid_data, SEQ_LENGTH)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True, num_workers=NUM_WORKERS, pin_memory=True)
except FileNotFoundError as e: print(e); exit()


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

model_lstm = SimpleLSTM(vocab_size, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_lstm.parameters(), lr=LEARNING_RATE)


def generate_text(model, start_text="To be ", size=200, top_k=5):
    model.eval()
    start_chars = [char2int.get(ch, 0) for ch in start_text] 
    input_tensor = torch.tensor(start_chars).unsqueeze(0).to(device)
    
    h = model.init_hidden(1)
    
    for i in range(input_tensor.shape[1]):
        char_input = input_tensor[:, i:i+1]
        _, h = model(char_input, h) 
        
    generated_text = list(start_text)
    char_input = input_tensor[:, -1:]
    
    with torch.no_grad():
        for _ in range(size):
            output, h = model(char_input, h)
            probs = torch.softmax(output, dim=1).squeeze()
            top_k_probs, top_k_indices = torch.topk(probs, k=top_k)
            top_k_probs = top_k_probs / torch.sum(top_k_probs)
            sampled_idx = torch.multinomial(top_k_probs, 1).item()
            final_idx = top_k_indices[sampled_idx].item()
            
            generated_text.append(int2char[final_idx])
            char_input = torch.tensor([[final_idx]]).to(device)
            
    return "".join(generated_text)

total_start_time = time.time()
history = {'train_loss': [], 'val_loss': [], 'train_error': [], 'val_error': []}
scaler = torch.cuda.amp.GradScaler()

for e in range(EPOCHS):
    epoch_start_time = time.time()
    model_lstm.train()
    h = model_lstm.init_hidden(BATCH_SIZE) 
    
    batch_losses = []
    correct_train = 0
    total_train = 0
    
    num_batches = len(train_loader)
    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        h = tuple([each.data for each in h])
        
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output, h = model_lstm(inputs, h)
            loss = criterion(output, targets)
            
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model_lstm.parameters(), max_norm=CLIP_MAX_NORM)
        scaler.step(optimizer)
        scaler.update()
        
        batch_losses.append(loss.item())
        _, predicted = torch.max(output.data, 1)
        total_train += targets.size(0)
        correct_train += (predicted == targets).sum().item()
        
        if (i + 1) % 200 == 0:
            print(f'  Epoch {e+1:02d}/{EPOCHS} | Batch {i+1:05d}/{num_batches} | Loss: {loss.item():.4f}')
            
    history['train_loss'].append(np.mean(batch_losses))
    history['train_error'].append(1.0 - (correct_train / total_train))

    model_lstm.eval()
    val_h = model_lstm.init_hidden(BATCH_SIZE)
    batch_val_losses = []
    correct_val = 0
    total_val = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            val_h = model_lstm.init_hidden(BATCH_SIZE) 

            with torch.cuda.amp.autocast():
                output, val_h = model_lstm(inputs, val_h)
                val_loss = criterion(output, targets)
            batch_val_losses.append(val_loss.item())
            _, predicted = torch.max(output.data, 1)
            total_val += targets.size(0)
            correct_val += (predicted == targets).sum().item()

    history['val_loss'].append(np.mean(batch_val_losses))
    history['val_error'].append(1.0 - (correct_val / total_val))

    print(f'Epoch {e+1:02d}/{EPOCHS} | Train Loss: {history["train_loss"][-1]:.4f} | Val Loss: {history["val_loss"][-1]:.4f} | '
          f'Train Err: {history["train_error"][-1]:.4f} | Val Err: {history["val_error"][-1]:.4f} | '
          f'Time: {(time.time() - epoch_start_time)/60:.2f} min')

    if (e+1) % 5 == 0:
        print(f"\n--- [Epoch {e+1} LSTM 生成] ---")
        print(generate_text(model_lstm, start_text="ROMEO: ", size=300))
        print("-------------------------------\n")
        torch.save(model_lstm.state_dict(), 'lstm_checkpoint.pth')

print(f"LSTM 訓練完成！總時間: {(time.time() - total_start_time)/3600:.2f} 小時")
torch.save(model_lstm.state_dict(), 'lstm_final_model.pth')

# 繪圖
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.title('LSTM Learning Curve')
plt.legend(); plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(history['train_error'], label='Train Error')
plt.plot(history['val_error'], label='Val Error')
plt.title('LSTM Error Rate')
plt.legend(); plt.grid(True)
plt.savefig('lstm_training_curves.png')