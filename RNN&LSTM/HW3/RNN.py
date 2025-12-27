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
print(f"Using device: {device}")



def load_text(train_file, valid_file):
    if not os.path.exists(train_file) or not os.path.exists(valid_file):
        raise FileNotFoundError(
            f"錯誤"
        )
        
    with open(train_file, 'r', encoding='utf-8') as f:
        train_text = f.read()
    with open(valid_file, 'r', encoding='utf-8') as f:
        valid_text = f.read()

    all_text = train_text + valid_text
    chars = tuple(sorted(set(all_text)))
    
    int2char = dict(enumerate(chars))
    char2int = {ch: ii for ii, ch in int2char.items()}
    vocab_size = len(chars)
    
    print(f"資料集總字元數: {len(all_text)}")
    print(f"字典大小 (Vocab size): {vocab_size}")
    
    train_encoded = np.array([char2int[ch] for ch in train_text])
    valid_encoded = np.array([char2int[ch] for ch in valid_text])
    
    return train_encoded, valid_encoded, vocab_size, int2char, char2int

class CharDataset(Dataset):
    def __init__(self, encoded_data, seq_length):
        self.encoded_data = encoded_data
        self.seq_length = seq_length
        self.num_sequences = len(encoded_data) - seq_length

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, index):
        seq_in = self.encoded_data[index : index + self.seq_length]
        seq_out = self.encoded_data[index + self.seq_length]
        return torch.tensor(seq_in, dtype=torch.long), torch.tensor(seq_out, dtype=torch.long)

try:
    train_data, valid_data, vocab_size, int2char, char2int = load_text(
        'shakespeare_train.txt', 'shakespeare_valid.txt'
    )
    train_dataset = CharDataset(train_data, SEQ_LENGTH)
    valid_dataset = CharDataset(valid_data, SEQ_LENGTH)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        drop_last=True,
        num_workers=NUM_WORKERS, 
        pin_memory=True  
    )
    val_loader = DataLoader(
        valid_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        drop_last=True,
        num_workers=NUM_WORKERS, 
        pin_memory=True           
    )
    
    print(f"訓練集序列數: {len(train_dataset)}")
    print(f"驗證集序列數: {len(valid_dataset)}")
except FileNotFoundError as e:
    print(e)
    exit()


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

model_rnn = SimpleRNN(vocab_size, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(device)
print("\n--- RNN 模型架構 ---")
print(model_rnn)
print("---------------------\n")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_rnn.parameters(), lr=LEARNING_RATE)


def generate_text(model, start_text="To be ", size=200, top_k=5):
    model.eval() 
    
    start_chars = []
    for ch in start_text:
        if ch in char2int:
            start_chars.append(char2int[ch])
            
    if not start_chars:
        return start_text 
        
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
            sampled_char_index_in_topk = torch.multinomial(top_k_probs, 1).item()
            sampled_char_index = top_k_indices[sampled_char_index_in_topk].item()
            
            new_char = int2char[sampled_char_index]
            generated_text.append(new_char)
            
            char_input = torch.tensor([[sampled_char_index]]).to(device)
            
    return "".join(generated_text)


total_start_time = time.time() 

history = {
    'train_loss': [], 'val_loss': [], 'train_error': [], 'val_error': []
}


scaler = torch.cuda.amp.GradScaler()

for e in range(EPOCHS):
    epoch_start_time = time.time()
    
    model_rnn.train()
    h = model_rnn.init_hidden(BATCH_SIZE)
    
    batch_losses = []
    correct_train = 0
    total_train = 0
    
    num_batches = len(train_loader)
    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        h = h.detach()
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            output, h = model_rnn(inputs, h)
            loss = criterion(output, targets)
        

        scaler.scale(loss).backward()
        
        torch.nn.utils.clip_grad_norm_(model_rnn.parameters(), max_norm=CLIP_MAX_NORM)
        
        scaler.step(optimizer)
        scaler.update()
        
        batch_losses.append(loss.item())
        
        _, predicted = torch.max(output.data, 1)
        total_train += targets.size(0)
        correct_train += (predicted == targets).sum().item()
        
        if (i + 1) % 500 == 0:
            print(f'  Epoch {e+1:02d}/{EPOCHS} | Batch {i+1:05d}/{num_batches} | '
                  f'Current Loss: {loss.item():.4f}')
            
    history['train_loss'].append(np.mean(batch_losses))
    history['train_error'].append(1.0 - (correct_train / total_train))
    model_rnn.eval()
    val_h = model_rnn.init_hidden(BATCH_SIZE)
    batch_val_losses = []
    correct_val = 0
    total_val = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            val_h = val_h.detach()
            
            with torch.cuda.amp.autocast():
                output, val_h = model_rnn(inputs, val_h)
                val_loss = criterion(output, targets)
            
            batch_val_losses.append(val_loss.item())
            
            _, predicted = torch.max(output.data, 1)
            total_val += targets.size(0)
            correct_val += (predicted == targets).sum().item()

    history['val_loss'].append(np.mean(batch_val_losses))
    history['val_error'].append(1.0 - (correct_val / total_val))

    epoch_end_time = time.time()
    
    print(f'Epoch {e+1:02d}/{EPOCHS} | '
          f'Train Loss: {history["train_loss"][-1]:.4f} | '
          f'Val Loss: {history["val_loss"][-1]:.4f} | '
          f'Train Error: {history["train_error"][-1]:.4f} | '
          f'Val Error: {history["val_error"][-1]:.4f} | '
          f'Time: {(epoch_end_time - epoch_start_time)/60:.2f} min') 

    if (e+1) % 5 == 0:
        print(f"\n--- [Epoch {e+1} 文字生成範例] ---")
        generated_text = generate_text(model_rnn, start_text="QUEEN: ", size=300)
        print(generated_text)

    torch.save(model_rnn.state_dict(), 'rnn_checkpoint.pth')

print("RNN完成！")
total_end_time = time.time()
print(f"總訓練時間: {(total_end_time - total_start_time)/3600:.2f} 小時") 

torch.save(model_rnn.state_dict(), 'rnn_final_model.pth')


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('RNN: Learning Curve (Loss vs. Epoch) - FULL DATASET')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history['train_error'], label='Training Error Rate')
plt.plot(history['val_error'], label='Validation Error Rate')
plt.title('RNN: Error Rate vs. Epoch - FULL DATASET')
plt.xlabel('Epoch')
plt.ylabel('Error Rate')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('rnn_training_curves_FULL.png')
plt.show()
