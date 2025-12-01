import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken

# 1. 하이퍼파라미터 설정
class GPTConfig:
    block_size = 32      # 한 번에 볼 문맥의 길이 (Context Window)
    batch_size = 4       # 배치의 크기
    n_embd = 64          # 임베딩 차원 (벡터 크기)
    n_head = 4           # 어텐션 헤드 개수
    n_layer = 4          # 트랜스포머 블록(레이어) 개수
    dropout = 0.0        # 드롭아웃 비율
    vocab_size = 50257   # GPT-2 기준 vocab size (tiktoken 사용 시)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = GPTConfig()

# 2. 간단한 Causal Self-Attention (GPT의 핵심: 미래 정보 가리기)
class Head(nn.Module):
    """ 하나의 Self-Attention Head """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        # 미래의 토큰을 보지 못하게 하는 마스크 (Lower Triangular Matrix)
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        
        # 어텐션 스코어 계산 (Scaled Dot-Product Attention)
        wei = q @ k.transpose(-2, -1) * C**-0.5 
        
        # [중요] 마스킹: 현재보다 미래의 위치는 -inf로 채워서 확률을 0으로 만듦
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        
        v = self.value(x)
        out = wei @ v 
        return out

class MultiHeadAttention(nn.Module):
    """ 여러 개의 Head를 병렬로 실행 """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    """ 토큰 별로 정보를 섞어주는 단순한 MLP """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)

# 3. 트랜스포머 블록
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size) # Self-Attention
        self.ffwd = FeedForward(n_embd)                 # Feed-Forward
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Residual Connection (x + ...) 적용
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# 4. 전체 GPT 모델
class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        # 토큰 임베딩 테이블
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        # 포지셔널 임베딩 테이블 (위치 정보)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        # 트랜스포머 블록들
        self.blocks = nn.Sequential(*[Block(config.n_embd, config.n_head) for _ in range(config.n_layer)])
        # 최종 LayerNorm
        self.ln_f = nn.LayerNorm(config.n_embd) 
        # 최종 출력 헤드 (단어 예측)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # 1. 임베딩 계산
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=config.device)) # (T, C)
        x = tok_emb + pos_emb # 토큰 정보 + 위치 정보
        
        # 2. 블록 통과
        x = self.blocks(x)
        x = self.ln_f(x)
        
        # 3. 로짓(Logits) 계산
        logits = self.lm_head(x) # (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    # 문장 생성 함수
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # 들어온 문맥을 block_size만큼 자르기 (너무 길면 에러남)
            idx_cond = idx[:, -config.block_size:]
            # 예측
            logits, _ = self(idx_cond)
            # 마지막 토큰에 대한 예측값만 가져오기
            logits = logits[:, -1, :] 
            probs = F.softmax(logits, dim=-1)
            # 확률 분포에 따라 다음 토큰 샘플링
            idx_next = torch.multinomial(probs, num_samples=1)
            # 정답을 현재 시퀀스에 붙이기
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# --- 실행 및 테스트 ---

# 1. 모델과 토크나이저 준비
model = GPT().to(config.device)
enc = tiktoken.get_encoding("gpt2") # GPT-2/3용 토크나이저

print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

# 2. 테스트용 입력 (안녕하세요! 같은 느낌의 영어)
input_text = "Hello, I am a robot."
tokens = enc.encode(input_text)
tokens_tensor = torch.tensor(tokens, dtype=torch.long, device=config.device).unsqueeze(0) # (1, T)

# 3. 생성 테스트 (학습 전이라 헛소리 출력함)
model.eval()
generated_tokens = model.generate(tokens_tensor, max_new_tokens=20)
decoded_text = enc.decode(generated_tokens[0].tolist())

print(f"\n입력: {input_text}")
print(f"생성 결과(학습 전): {decoded_text}")