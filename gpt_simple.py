import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken

# 1. 하이퍼파라미터 설정
class GPTConfig:
    block_size = 1024
    vocab_size = 100277 # [중요] cl100k_base 토크나이저 크기
    n_layer = 12
    n_head = 12
    n_embd = 768
    dropout = 0.1
    batch_size = 12
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = GPTConfig()

# ... (중간 부품들 Head, MultiHeadAttention, FeedForward, Block은 기존과 동일) ...

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5 
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        out = wei @ v 
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config.n_embd, config.n_head) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd) 
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) 
        pos_emb = self.position_embedding_table(torch.arange(T, device=config.device)) 
        x = tok_emb + pos_emb 
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) 

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] 
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# --- 실행 및 테스트 (수정됨) ---
# 이 부분이 없으면 다른 파일에서 import 할 때마다 아래 코드가 실행되어 에러가 납니다.
if __name__ == "__main__":
    # 1. 모델 준비
    model = GPT().to(config.device)
    
    # [수정] 토크나이저를 'gpt2'에서 'cl100k_base'로 변경!
    # 이유: GPTConfig의 vocab_size가 100277이라서 gpt2(50257)로는 감당 불가능
    enc = tiktoken.get_encoding("cl100k_base") 

    print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # 2. 테스트용 입력
    input_text = "Hello, I am a robot."
    tokens = enc.encode(input_text)
    print(f"\n tokens : {tokens}")
    
    tokens_tensor = torch.tensor(tokens, dtype=torch.long, device=config.device).unsqueeze(0) 

    # 3. 생성 테스트
    model.eval()
    generated_tokens = model.generate(tokens_tensor, max_new_tokens=20)
    
    # 이제 cl100k_base를 쓰므로 에러가 나지 않습니다.
    decoded_text = enc.decode(generated_tokens[0].tolist())

    print(f"\n입력: {input_text}")
    print(f"생성 결과(학습 전): {decoded_text}")