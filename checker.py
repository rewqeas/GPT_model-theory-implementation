import torch
from GPT_model.model import gpt_model
from torch.amp import autocast, GradScaler


scaler = GradScaler(device = 'cuda')
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

model = gpt_model(GPT_CONFIG_124M).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr = 0.0001)
accumlation_steps = 2
# def test_batch_size(batch_size):
#     try:
#         torch.cuda.reset_peak_memory_stats()
        
#         #simulate full training step
#         optimizer.zero_grad()
#         for acc_step in range(accumlation_steps):
#             dummy_input = torch.randint(0,50247, (batch_size, 256)).to(device)
#             dummy_target = torch.randint(0,50247, (batch_size, 256)).to(device)

#             with autocast(device_type = 'cuda'):
#                 logits = model(dummy_input)
#                 loss = torch.nn.functional.cross_entropy(logits.flatten(0,1), dummy_target.flatten())

#                 loss = loss / accumlation_steps # after accumulation gradients basically are added with each other, so we normalize it by scaler factor acucmulate steps
#             scaler.scale(loss).backward()
        
#         #update only after accumulation steps
#         scaler.unscale_(optimizer)
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)
#         scaler.step(optimizer)
#         # optimizer.step()
#         scaler.update()

#         used = torch.cuda.memory_allocated() / 1024**3
#         peak = torch.cuda.max_memory_allocated() / 1024**3
#         print(f"batch={batch_size:<4} | current VRAM: {used:.2f}GB | peak VRAM: {peak:.2f}GB | ✅ OK")
        
#     except torch.cuda.OutOfMemoryError:
#         print(f"batch={batch_size:<4} | ❌ OOM — too large")
    
#     finally:
#         torch.cuda.empty_cache()

# if __name__ == '__main__':
#     for batch in [4,8, 16, 32]:
#         test_batch_size(batch)

def test_batch_size(batch_size):
    try:
        torch.cuda.reset_peak_memory_stats()
        
        #simulate full training step
        optimizer.zero_grad()
        
        dummy_input = torch.randint(0,50247, (batch_size, 256)).to(device)
        dummy_target = torch.randint(0,50247, (batch_size, 256)).to(device)

        with autocast(device_type = 'cuda'):
            logits = model(dummy_input)
            loss = torch.nn.functional.cross_entropy(logits.flatten(0,1), dummy_target.flatten())

 # after accumulation gradients basically are added with each other, so we normalize it by scaler factor acucmulate steps
        loss.backward()
        
        #update only after accumulation steps
        optimizer.step()
        # scaler.update()

        used = torch.cuda.memory_allocated() / 1024**3
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"batch={batch_size:<4} | current VRAM: {used:.2f}GB | peak VRAM: {peak:.2f}GB | ✅ OK")
        
    except torch.cuda.OutOfMemoryError:
        print(f"batch={batch_size:<4} | ❌ OOM — too large")
    
    finally:
        torch.cuda.empty_cache()

if __name__ == '__main__':
    for batch in [4,8, 16, 32]:
        test_batch_size(batch)
