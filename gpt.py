from GPT_model.model import gpt_model,generate
from tokenizer.tokenizer_gpt import GPTDataset, create_dataloader
import torch
import tiktoken
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from datasets import load_dataset
from torch.amp import autocast, GradScaler
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator



########
#model config
########
GPT_CONFIG_124M = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 256,  # Context length
    "emb_dim": 768,          # Embedding dimension
    "n_heads": 12,           # Number of attention heads
    "n_layers": 12,          # Number of layers
    "drop_rate": 0.1,        # Dropout rate
    "qkv_bias": False        # Query-Key-Value bias
}
####
#helper functions
####
def text_to_token_ids(text, tokenizer):
    token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    token_ids = torch.tensor(token_ids).unsqueeze(0)
    return token_ids

def token_ids_to_text(token_ids, tokenizer):
    squeezed_text = token_ids.squeeze(0).tolist()
    decoded = tokenizer.decode(squeezed_text)
    return decoded

###################################
##creating train_Dataloader and val_dataloader
##################################
tokenizer = tiktoken.get_encoding('gpt2')

with open("wikitext2_train.txt","r", encoding='utf-8') as f:
    train_text = f.read()

with open("wikitext2_val.txt","r", encoding='utf-8') as f:
    val_text = f.read()

with open("wikitext2_test.txt","r", encoding='utf-8') as f:
    test_text = f.read()

# ✅ tokenize once here, not inside workers
train_token_ids = tokenizer.encode(train_text, allowed_special={"<|endoftext|>"})
val_token_ids   = tokenizer.encode(val_text,   allowed_special={"<|endoftext|>"})
test_token_ids  = tokenizer.encode(test_text,  allowed_special={"<|endoftext|>"})


#train dataloader
train_dataloader = create_dataloader(
    token_ids = train_token_ids,
    batch=4,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    shuffle=True,
    drop_last=True,
    num_workers=4,
    pin_memory=True
)

# #val dataloader
val_dataloader = create_dataloader(
    token_ids = val_token_ids,
    batch=4,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    shuffle=False,
    drop_last=False,
    num_workers=4,
    pin_memory=True
)
#test dataloader
test_dataloader = create_dataloader(
    token_ids = test_token_ids,
    batch=4,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    shuffle=False,
    drop_last=False,
    num_workers=4,
    pin_memory=True
)

##########################
#loss functions
##########################


def calculate_loss(input_batch, target_batch, g_model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = g_model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0,1), target_batch.flatten(0))
    return loss


def calc_loss_loader(dataloader, g_model, device, num_batches = None):
    total_loss = 0

    if len(dataloader) == 0:
        return float('nan')
    
    elif num_batches == None:
        num_batches = len(dataloader)

    else:
        num_batches = min(num_batches, len(dataloader))
    
    for i, (input_batch, target_batch) in enumerate(dataloader):
        if i >= num_batches:
            break
            
        loss = calculate_loss(input_batch,target_batch, g_model, device)
        total_loss += loss.item()

    return total_loss / num_batches

torch.manual_seed(123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# g_model = gpt_model(GPT_CONFIG_124M)
# g_model = g_model.to(device)

# with torch.no_grad():
#     train_loss = calc_loss_loader(train_dataloader, g_model,device)
#     val_loss = calc_loss_loader(val_dataloader, g_model,device)

# print(f"training loss:{train_loss}")
# print(f"val loss:{val_loss}")


##########################
#training the model
##########################

def train_model(model, train_loader, val_loader, optimizer,device,n_epochs,
                 eval_freq,eval_iter,start_content,tokenizer, 
                 warmup_steps, initial_lr = 3e-05, min_lr = 1e-6,
                   temperature = 0.0, top_k = None, accumulation_steps = 2):
    
    train_losses, val_losses, track_token_seen, track_lrs = [],[],[],[]
    token_seen, global_steps = 0,-1

    peak_lr = optimizer.param_groups[0]['lr']
    total_training_steps = len(train_loader) * n_epochs
    lr_increment = (peak_lr - initial_lr) / warmup_steps
    scaler = GradScaler(device='cuda')
    
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        for i,(input_batch, target_batch) in enumerate(train_loader):

            # optimizer.zero_grad()
            global_steps += 1
            if global_steps < warmup_steps:
                lr = initial_lr + global_steps * lr_increment

            else:
                progress = (global_steps - warmup_steps) / (total_training_steps - warmup_steps)
                lr = min_lr + 0.5*(peak_lr - min_lr)*(1 + math.cos(math.pi * progress))


            for param in optimizer.param_groups:
                 param['lr'] = lr
            
            track_lrs.append(optimizer.param_groups[0]['lr'])

            with autocast(device_type = 'cuda'):
                loss = calculate_loss(input_batch, target_batch, model, device)
                loss = loss / accumulation_steps

            scaler.scale(loss).backward() # here is where gradients get calculated, so accumulation of gradient is here

            if (i + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            token_seen += input_batch.numel()

            if global_steps % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model,train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss),val_losses.append(val_loss)
                track_token_seen.append(token_seen)

                print(f"epoch:{epoch}, train_loss: {train_loss:.3f}")
                print(f"epoch:{epoch}, val_loss: {val_loss:.3f}")

        # in case if the last batch is even (0 index based, gradient accumulates at every odd indexes)
        if (i + 1) % accumulation_steps != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

    generate_and_print_sample(model,start_content,device,tokenizer,temperature,top_k)
    return train_losses, val_losses, track_token_seen


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, eval_iter)
    
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, start_content, device, tokenizer, temperature, top_k):
    model.eval()
    context_length = model.pos_emb.weight.shape[0]
    encoded_text = text_to_token_ids(start_content, tokenizer).to(device)
    
    with torch.no_grad():
        generated_tokens = generate(
            idx = encoded_text,
            model=model,
            context_length=context_length,
            max_new_token=50,
            device = device,
            temperature=temperature,
            top_k=top_k
        )

    decoded_text = token_ids_to_text(generated_tokens, tokenizer)
    print(decoded_text.replace("\n", " "))

    model.train()

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
        fig, ax1 = plt.subplots(figsize=(5, 3))
        ax1.plot(epochs_seen, train_losses, label="Training loss")
        ax1.plot(
            epochs_seen, val_losses, linestyle="-.", label="Validation loss"
        )
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss")
        ax1.legend(loc="upper right")
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax2 = ax1.twiny()                   #1
        ax2.plot(tokens_seen, train_losses, alpha=0)     #2
        ax2.set_xlabel("Tokens seen")
        fig.tight_layout()
        plt.show()

"""
model, train_loader, val_loader, optimizer,device,n_epochs,
                 eval_freq,eval_iter,start_content,tokenizer, 
                 warmup_steps, initial_lr = 3e-05, min_lr = 1e-6,
                   temperature = 0.0, top_k = None
"""

if __name__ == "__main__":
    model_gpt = gpt_model(GPT_CONFIG_124M)
    model_gpt = model_gpt.to(device)
    n_epochs = 3
    optimizer = torch.optim.AdamW(model_gpt.parameters(), lr=0.0001, weight_decay=0.1)
    start_content = "Every effort moves you"
    train_losses, val_losses, tokens_seen = train_model(
        model_gpt, train_dataloader, val_dataloader, optimizer,device, n_epochs,
        eval_freq=200, eval_iter=20,start_content=start_content,tokenizer=tokenizer,
        warmup_steps=100, temperature=4, top_k=3)
   
    torch.save({
        "model_state_dict":model_gpt.state_dict(),
        "optimizer_state_dict":optimizer.state_dict(),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "tokens_seen": tokens_seen,

    }, "model_and_optimizer.pth")
    

    epochs_tensor = torch.linspace(0, n_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
        
    test_loss = calc_loss_loader(test_dataloader, model_gpt, device)
    perplexity = torch.exp(torch.tensor(test_loss))
    print(f"test loss:{test_loss:.3f}")
    print(f"perplexity:{perplexity.item():.3f}")

# print(device)
# print(torch.cuda.is_available())
