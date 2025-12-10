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
"""
Head → MultiHeadAttention, FeedForward → Block → GPT
작은 부품(Head)을 만들어서 → 묶고(Block) → 최종 조립(GPT)하는 과정
"""
"""
Head (가장 작은 부품)
역할: 눈알 하나입니다.

실제로 수학 계산(행렬 곱하기)을 하는 녀석입니다. 여기서 "과거만 볼 수 있게 가리는(Masking)" 작업을 합니다.
"""
# 2. 간단한 Causal Self-Attention (GPT의 핵심: 미래 정보 가리기)
class Head(nn.Module):
    """
      하나의 Self-Attention Head 
**'느금마' (토큰 1)**가 기계에 들어갑니다.

Q: "나랑 어울리는 애 누구?" (느금마의 질문)

K: "#사람 #대상" (느금마의 이름표)

V: "엄마 정보" (느금마의 내용물)

**'만수무강' (토큰 2)**이 똑같은 기계에 들어갑니다.

Q: "누가 오래 살아?" (만수무강의 질문) → 질문 내용이 다름!

K: "#상태 #축복" (만수무강의 이름표) → 이름표가 다름!

V: "오래 산다는 뜻" (만수무강의 내용물) → 내용물이 다름!
    """
    def __init__(self, head_size):
        super().__init__()
        # 토큰 하나당 Q 백터,K 백터,V 백터 
        # "나는 이런 특성을 가진 단어야." (다른 단어들이 가진 꼬리표)
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        # "나랑 관련된 단어가 누구니?" (현재 단어가 던지는 질문)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        # "나의 실제 의미는 이거야." (결과값으로 줄 정보)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        # 미래의 토큰을 보지 못하게 하는 마스크 (Lower Triangular Matrix)
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))

    def forward(self, x):
        """
1. B, T, C가 뭐냐? (입력 데이터의 신상정보)
x.shape는 지금 들어온 데이터 덩어리의 크기를 말합니다.

B (Batch Size, 배치 크기): "한 번에 몇 문장 처리해?"

예: 4문장을 동시에 공부 중이면 B=4.

T (Time step, 문장 길이): "한 문장에 단어가 몇 개야?"

예: "느금마 만수무강"이면 단어가 2개니까 T=2.

C (Channel, 정보의 깊이): "단어 하나를 숫자 몇 개로 표현해?"

예: 단어 하나를 숫자 64개로 자세히 설명하고 있으면 C=64. (n_embd)

즉, x는 (4개 문장, 각 2단어, 단어당 64개 숫자)로 된 거대한 숫자 덩어리입니다.
        """
        B, T, C = x.shape
        """
2. k = self.key(x)와 q = self.query(x)는 뭐하는 짓?
여기서 아까 말한 **"변환 기계"**가 작동합니다. **C(64)**만큼 뚱뚱했던 정보를 **head_size(16)**만큼 압축해서 **특수한 목적(Q, K)**으로 바꿉니다.

입력 (x): 그냥 "만수무강"이라는 일반적인 정보 (크기: 64)

⬇ self.key (기계) 통과 ⬇

출력 (k): "만수무강"의 이름표(Key) 정보 (크기: 16)

결과적으로 모양이 이렇게 바뀝니다:

전: (B, T, 64) → (문장 4개, 길이 2, 뚱뚱한 일반 정보)

후: (B, T, 16) → (문장 4개, 길이 2, 압축된 이름표 정보)
        """
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        
        # 어텐션 스코어 계산 (Scaled Dot-Product Attention)
        wei = q @ k.transpose(-2, -1) * C**-0.5 
        
        # [중요] 마스킹: 현재보다 미래의 위치는 -inf로 채워서 확률을 0으로 만듦
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        """
        점수를 백분율(%)로 바꾸기
아까 계산한 점수(wei)는 10점, 500점, -99점 등 제멋대로입니다. 이걸 **확률(총합 100%)**로 싹 정리합니다.

전: A단어(100점), B단어(10점)

후(Softmax): A단어(90%), B단어(10%)
        """
        wei = F.softmax(wei, dim=-1)
        
        v = self.value(x)
        out = wei @ v 
        return out

"""
MultiHeadAttention (부품 조립 1)
역할: 눈알 여러 개 묶음입니다.

눈이 하나면 불안하니까, Head를 4개(설정값) 만들어서 붙여놓은 껍데기입니다.
"""
class MultiHeadAttention(nn.Module):
    """ 여러 개의 Head를 병렬로 실행 
Head들: 각자 흩어져서 조사함 (개인 플레이)

Concat: 조사한 거 책상 위에 모아둠

self.proj: "자, 다들 모여봐. 이거 무슨 뜻인지 결론 내자." (팀 미팅 & 보고서 작성)
    """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

"""
FeedForward (부품 조립 2)
역할: 계산기입니다.

눈으로 본 정보를 가지고 머리를 굴리는 곳입니다. 단순한 신경망(MLP)입니다.
정보를 4배로 뻥튀기해서 자세히 본 다음(Linear), 필요 없는 건 버리고(ReLU), 다시 원래대로 압축(Linear)해서 돌려주는 계산기

이 "뻥튀기 후 압축" 과정이 사실 딥러닝에서 지능이 생기는 핵심 마법
"""
class FeedForward(nn.Module):
    """ 토큰 별로 정보를 섞어주는 단순한 MLP """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            # 원래 정보량(n_embd)을 4배(4 * n_embd)로 뻥튀기해서 아주 자세하게 늘어놓습니다. (예: 4차원을 16차원으로 늘렸다는얘기) 
            nn.Linear(n_embd, 4 * n_embd),
            # 중요한 정보(양수)는 살리고, 필요 없거나 방해되는 정보(음수)는 과감하게 0으로 지워버립니다. 
            nn.ReLU(),
            # 4배로 늘렸던 정보를 다시 원래 크기로 압축해서 다음 단계로 넘겨줍니다.
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)

# 3. 트랜스포머 블록
"""
Block (중간 조립)
역할: 지능 한 층입니다.

위에서 만든 MultiHeadAttention(눈)과 FeedForward(머리)를 한 세트로 묶습니다. 이 블록을 4층, 12층, 96층 쌓으면 GPT가 됩니다.
"""
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size) # Self-Attention. (질문) 주변 친구들(다른 단어들)에게 관련 정보를 물어봅니다.
        self.ffwd = FeedForward(n_embd)                 # Feed-Forward. (숙고) 아까 배운 내용을 혼자 곰곰이 생각해서 결론을 냅니다. (뻥튀기 후 압축)
        self.ln1 = nn.LayerNorm(n_embd) # (심호흡) 일단 흥분한 상태를 가라앉히고(정규화) 차분해집니다.
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Residual Connection (x + ...) 적용
        # 친구들과 대화 (Attention). x = x + ... (기록) 내 노트에 친구들이 알려준 내용을 추가합니다. (내용이 풍부해짐)
        x = x + self.sa(self.ln1(x))
        # 혼자 생각 정리 (FeedForward). x = x + ... (기록) 깨달은 내용을 내 노트에 다시 추가합니다.
        x = x + self.ffwd(self.ln2(x))
        return x

# 4. 전체 GPT 모델
"""
GPT (최종 완제품)
역할: 로봇 본체입니다.

입력을 받아서 -> Block들을 통과시키고 -> 최종 결과를 내뱉는 전체 과정을 지휘합니다.
"""
class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        # 토큰 임베딩 테이블. (숫자 ID → vector정보 덩어리 변환기).
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        # 포지셔널 임베딩 테이블 (위치 정보). (위치 번호표). 단어한테 **"너는 문장의 몇 번째 순서야"**라는 **위치 정보(명찰)**를 붙여주는 부품입니다.
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        # 트랜스포머 블록들. (생각하는 뇌). 여기가 핵심입니다. 아까 만든 Block(어텐션 + 피드포워드)을 n_layer(여기선 4개)만큼 층층이 쌓습니다.
        self.blocks = nn.Sequential(*[Block(config.n_embd, config.n_head) for _ in range(config.n_layer)])
        # 최종 LayerNorm. (최종 정리 정돈). 뇌(Block)를 거치면서 데이터 값들이 너무 커지거나 들쑥날쑥해졌을 수 있습니다. 마지막으로 결과를 내보내기 전에 데이터를 깔끔하게 표준화(정규화) 시켜주는 필터입니다.
        self.ln_f = nn.LayerNorm(config.n_embd) 
        # 최종 출력 헤드 (정답 발표기).  n_embd(64개짜리 숫자 덩어리)를 다시 vocab_size(50257개 단어장) 크기로 쫙 펼쳐서, "어떤 단어가 올 확률이 가장 높은지" 점수를 매깁니다.
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # 1. 임베딩 계산
        """
상황: 지금 idx라는 숫자(단어 번호)들이 들어왔습니다.

tok_emb: 숫자 번호를 **의미를 가진 벡터(숫자 뭉치)**로 바꿉니다. (예: 3번 -> [0.1, -0.5, ...]) "이건 사과라는 뜻이야."

pos_emb: 위치 정보를 더해줍니다. "이건 문장의 맨 첫 번째 단어야."

x = ...: 이 두 정보를 더해서 **"첫 번째 자리에 있는 사과"**라는 최종 정보를 만듭니다.
        """
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=config.device)) # (T, C)
        x = tok_emb + pos_emb # 토큰 정보 + 위치 정보
        
        # 2. 뇌 가동 (Blocks). "배운 걸 토대로 고민해 보자"
        """
self.blocks(x): 아까 만든 **4개의 지능 층(Layer)**을 통과시킵니다.

과거의 단어들을 참고하고(Attention), 혼자 생각해서(FeedForward) 정보를 점점 구체화합니다.

self.ln_f(x): 4번이나 고민하느라 데이터 값들이 너무 튀었을 수 있으니, 마지막으로 차분하게 **정돈(정규화)**합니다.
        """
        x = self.blocks(x)
        x = self.ln_f(x)
        
        # 3. 로짓(Logits) 계산
        """
지금 x는 64개(n_embd)의 숫자로 된 추상적인 생각 덩어리입니다.

이걸 다시 50,257개(vocab_size) 단어장 크기로 확 늘립니다.

결과(logits): 50,257개의 단어 각각에 대해 **"이게 정답일 점수"**를 매긴 채점표입니다. (점수가 높을수록 그 단어가 나올 확률이 높음)
        """
        logits = self.lm_head(x) # (B, T, vocab_size)

        """
        (중요) 채점 시간. 상황: 로봇이 예측한 값(logits)과 실제 정답(targets)이 같이 들어왔습니다.
        """
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