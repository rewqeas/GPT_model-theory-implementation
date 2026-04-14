import torch
import torch.nn as nn
import tiktoken


########
#multihead attention
########
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, num_heads, dropout, qkv_bias = False):
        super().__init__()
        assert d_out % num_heads == 0, "this must be true"

        self.head_dim = d_out // num_heads
        self.d_out = d_out
        self.num_heads = num_heads
        
        self.W_query = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal = 1)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_out, d_out)


    def forward(self, x):
        b, n_tok, emb_dim = x.shape
        q = self.W_query(x) 
        k = self.W_key(x) 
        v = self.W_value(x)


        q = q.view(b, n_tok, self.num_heads, self.head_dim).transpose(1,2)  
        k = k.view(b, n_tok, self.num_heads, self.head_dim).transpose(1,2)  
        v = v.view(b, n_tok, self.num_heads, self.head_dim).transpose(1,2) 

        attn_score = q @ k.transpose(2,3)
        attn_score.masked_fill_(self.mask.bool()[:n_tok, :n_tok], -torch.inf)

        attn_weight = torch.softmax(attn_score / k.shape[-1] ** (0.5), dim=-1)
        attn_weight = self.dropout(attn_weight)

        context_vec = attn_weight @ v
        context_vec = context_vec.transpose(1,2)
        context_vec = context_vec.contiguous().view(b, n_tok, self.d_out)

        return self.out_proj(context_vec)

########
#layer Norm
########
class LayerNorm(nn.Module):

    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))


    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        var = x.var(dim = -1, unbiased = False, keepdim = True)
        out = (x - mean) / (var + self.eps) ** (0.5)

        return self.scale * out + self.shift

########
#Gelu
########   
class GELU(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self,x):
        ang = ((2/torch.pi) ** 0.5) * (x + 0.044715 * x ** 3)
        out = 0.5 * x * (1 + torch.tanh(ang))
        return out

########
#feed forward Network
########  
class FeedForwardNetwork(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(cfg['emb_dim'], 4 * cfg['emb_dim']),
            GELU(),
            nn.Linear(4 * cfg['emb_dim'], cfg['emb_dim']),
        )

    def forward(self, x):
        return self.seq(x)

class FeedForwardNetwork_my_ver(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(cfg['emb_dim'], 4 * cfg['emb_dim']),
            GELU(),
            nn.Dropout(cfg['drop_rate']),
            nn.Linear(4 * cfg['emb_dim'], cfg['emb_dim']),
        )

    def forward(self, x):
        return self.seq(x)


class TransformerBlock(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.layer_norm1 = LayerNorm(cfg['emb_dim'])
        self.mha = MultiHeadAttention(cfg['emb_dim'],cfg['emb_dim'],cfg['context_length'],cfg['n_heads'],cfg['drop_rate'])
        self.dropout1 = nn.Dropout(cfg['drop_rate'])
        self.dropout2 = nn.Dropout(cfg['drop_rate'])    
        self.layer_norm2 = LayerNorm(cfg['emb_dim'])
        self.ffn = FeedForwardNetwork(cfg)

    def forward(self,x):
        x = x + self.dropout1(self.mha(self.layer_norm1(x)))
        x = x + self.dropout2(self.ffn(self.layer_norm2(x)))
        return x


class gpt_model(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        self.pos_emb = nn.Embedding(cfg['context_length'], cfg['emb_dim'])

        self.dropout = nn.Dropout(cfg['drop_rate'])
        self.trfs = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg['n_layers'])]
        )
        self.final_norm = LayerNorm(cfg['emb_dim'])
        self.out_layer = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias = False)

    def forward(self,x):
        b, seq_len = x.shape
        tok_embds = self.tok_emb(x)
        pos_embds = self.pos_emb(torch.arange(seq_len, device=x.device))

        vec = tok_embds + pos_embds
        context_vec = self.trfs(self.dropout(vec))
        logits = self.out_layer(self.final_norm(context_vec))
        return logits

# tokenizer = tiktoken.get_encoding('gpt2')
# batch = []

# torch.manual_seed(123)
# def generate_text(idx,model, context_length, max_new_token):

#   for _ in range(max_new_token):
#     idx_cond = idx[:, -context_length:]
#     with torch.no_grad():
#       logits = model(idx_cond)#no backprop here

#     logits = logits[:,-1,:] # taking the last vec generated
#     probs = torch.softmax(logits, dim = -1)
#     idx_next = torch.argmax(probs, dim = -1, keepdim = True)
#     idx = torch.cat((idx, idx_next), dim = 1)

#   return idx


def generate(idx, model, context_length, max_new_token,device,temperature, top_k):

    for _ in range(max_new_token):
        idx_cond = idx[:,-context_length:]
        with torch.no_grad():
            logits = model(idx_cond)
        
        logits = logits[:,-1,:]
        
        if top_k is not None:
            top_logits, top_pos = torch.topk(logits, top_k)
            logits = torch.where(
                condition=logits < top_logits[:,-1].unsqueeze(1),
                input=torch.tensor(float('-inf')).to(device),
                other=logits
            )

        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim = -1)
            idx_next = torch.multinomial(probs, num_samples=1)

        else:
            probs = torch.softmax(logits, dim = -1)
            idx_next = torch.argmax(probs, dim = -1, keepdim=True)
        
        idx = torch.cat((idx, idx_next), dim = -1)

    return idx
            


# model = gpt_model(GPT_CONFIG_124M)
# start_context = "Hello, I am"
# encoded = tokenizer.encode(start_context)
# # encoded
# encoded_tensor = torch.tensor(encoded).unsqueeze(0)
# print(encoded_tensor)



# model.eval()
# out = generate_text(
#     encoded_tensor, model, GPT_CONFIG_124M['context_length'], 6
# )
# print(out)
# print(len(out[0]))

# decoded = tokenizer.decode(out.squeeze(0).tolist())
# print(decoded)