# GPT 코드 완전 해설: 인공지능의 뇌를 해부하다

이 문서는 `gpt_sample.ipynb` 파일의 코드를 **한 줄 한 줄** 아주 상세하게 설명한 문서입니다.
인공지능이나 코딩을 전혀 모르는 사람도 이해할 수 있도록 비유를 들었고, 컴퓨터 공학적인 원리도 함께 설명했습니다.

---

## 1. 라이브러리 설치 및 불러오기

요리를 하려면 도구(칼, 냄비)와 재료가 필요하듯, 코딩에서도 다른 사람들이 만들어둔 도구(라이브러리)를 가져와야 합니다.

```python
!pip install tiktoken
```
- **설명**: `tiktoken`이라는 도구를 설치합니다.
- **원리**: 컴퓨터는 '사과'라는 글자를 이해하지 못합니다. 숫자로 바꿔줘야 하죠. `tiktoken`은 글자를 숫자(토큰)로 바꿔주는 번역기 역할을 합니다. AI에게 말을 가르치기 위한 필수 번역기를 설치하는 과정입니다.
- **Input**: `pip install` 명령어
- **Output**: `tiktoken` 라이브러리 설치 완료 메시지

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```
- **설명**: `torch`(파이토치)는 페이스북에서 만든 인공지능 전용 레고 블록 세트입니다.
    - `torch`: 기본 블록들.
    - `torch.nn`: 인공지능의 뇌세포(뉴런)를 만드는 부품들.
    - `torch.nn.functional`: 뇌세포를 연결하고 활성화시키는 함수들(접착제).
- **원리**: 딥러닝 프레임워크를 불러옵니다. 여기서 'Tensor(텐서)'라는 개념이 중요한데, 텐서는 '숫자들을 담는 다차원 박스'입니다.
- **Input**: 라이브러리 이름
- **Output**: 해당 라이브러리를 사용할 준비 완료

```python
import tiktoken
```
- **설명**: 아까 설치한 번역기(`tiktoken`)를 이제 꺼내서 씁니다.

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```
- **설명**: "그래픽카드(GPU)가 있으면 그걸 쓰고, 없으면 그냥 CPU 써라"라고 정하는 코드입니다.
- **원리**: 인공지능 공부는 단순 계산을 수억 번 해야 합니다. CPU는 똑똑한 박사님 1명이라면, GPU는 초등학생 1000명입니다. 단순 계산은 초등학생 1000명이 훨씬 빠릅니다. 이를 **병렬 연산**이라고 합니다.
- **Input**: 시스템의 하드웨어 상태 확인
- **Output**: `device` 변수에 `'cuda'`(GPU) 또는 `'cpu'`가 저장됨.

---

## 2. GPT 설정 (설계도 그리기)

로봇을 만들기 전에 크기는 얼마로 할지, 부품은 몇 개 쓸지 정하는 단계입니다.

```python
class GPTConfig:
    block_size = 256
    vocab_size = 100277
    n_layer = 12
    n_head = 8
    n_embd = 512
    dropout = 0.1
    batch_size = 16
```
- **설명**: GPT 모델의 스펙(Specification)을 정의합니다.
    - `block_size = 256`: **한 번에 읽을 수 있는 글자 수**. 책을 읽을 때 한 번에 256 단어씩 본다는 뜻입니다. (Context Window)
    - `vocab_size = 100277`: **알고 있는 단어의 개수**. 이 모델은 약 10만 개의 단어 사전을 가집니다.
    - `n_layer = 12`: **뇌의 깊이**. 생각하는 단계가 12단계로 이루어져 있습니다. 층이 깊을수록 복잡한 생각을 할 수 있습니다. (Layer Depth)
    - `n_head = 8`: **눈의 개수**. 한 문장을 볼 때, 8개의 다른 관점으로 동시에 쳐다봅니다. (멀티헤드 어텐션)
    - `n_embd = 512`: **단어의 의미 크기**. '사과'라는 단어를 512개의 숫자로 표현해서 아주 디테일한 의미를 담습니다. (Embedding Dimension)
- **원리**: 하이퍼파라미터(Hyperparameter) 설정입니다. 모델의 성능과 용량을 결정짓는 중요한 숫자들입니다.

---

## 3. 모델의 핵심 부품: 어텐션 (Head)

여기가 GPT의 심장입니다. **"무엇이 중요한지 집중(Attention)하는 능력"**을 구현합니다.

```python
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
```
- **설명**: 어텐션을 하기 위한 3가지 도구를 만듭니다.
    - **Query (질문)**: "내가 지금 뭘 찾고 있지?" (예: '철수'라는 단어를 볼 때 '주어인가?'라고 묻는 것)
    - **Key (답변 후보)**: "나는 이런 특성을 가졌어." (예: '가다'라는 단어가 '나는 동사야'라고 들고 있는 팻말)
    - **Value (내용)**: "나의 실제 의미는 이거야." (책의 실제 내용)
- **원리**: **Self-Attention(자가 주의 메커니즘)**의 핵심입니다. 모든 단어가 서로에게 질문(Query)을 던지고, 답변(Key)이 잘 맞는 단어끼리 정보를 교환(Value)합니다.
- **Input**: 설정된 임베딩 크기 (`config.n_embd`)
- **Output**: 3개의 선형 변환 레이어 (Linear Layer) 생성

```python
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))
```
- **설명**: **"미래를 보지 못하게 가리는 가림막"**을 만듭니다.
- **원리**: GPT는 다음 단어를 맞히는 게임을 합니다. 뒤에 나올 정답을 미리 보면 안 되겠죠? 그래서 행렬의 윗부분(미래)을 지워버리는 마스크(Mask)를 만듭니다. 이를 **Causal Masking**이라고 합니다.
- **Input**: `block_size` (256)
- **Output**: 0과 1로 이루어진 삼각형 모양의 행렬 `tril`

```python
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
```
- **설명**: 입력 데이터 `x`가 들어오면, 질문(q)과 답변 후보(k)를 만들어냅니다.
- **Input**: `x` (데이터). 크기는 `[배치크기(B), 문장길이(T), 의미크기(C)]`
- **Output**: `k`, `q` (의미 공간에서 변환된 벡터들)

```python
        wei = q @ k.transpose(-2, -1) * C**-0.5
```
- **설명**: 질문(q)과 답변 후보(k)를 비교합니다(행렬 곱셈 `@`).
- **원리**: 이 값이 클수록 두 단어가 **"관련이 깊다"**는 뜻입니다. `C**-0.5`를 곱하는 건 숫자가 너무 커지지 않게 조절(Scaling)하는 것입니다. 이를 **Scaled Dot-Product Attention**이라고 합니다.
- **Input**: 질문 벡터 `q`, 답변 벡터 `k`
- **Output**: `wei` (단어들 간의 관계 점수표)

```python
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
```
- **설명**: 미래의 단어들과의 관계 점수를 `-무한대`로 만들어버립니다. 즉, **"미래는 쳐다보지 마!"**라고 눈을 가리는 과정입니다.
- **Input**: 관계 점수표 `wei`
- **Output**: 미래가 가려진 점수표

```python
        wei = F.softmax(wei, dim=-1)
```
- **설명**: 점수를 확률(0~1 사이, 합치면 1)로 바꿉니다.
- **원리**: `-무한대`였던 미래의 값들은 확률 0이 되어 사라집니다. 이제 현재 단어가 과거의 어떤 단어에 얼만큼 집중(Attention)해야 하는지 비율이 나옵니다.
- **Input**: 관계 점수표
- **Output**: **Attention Weights** (집중 확률)

```python
        v = self.value(x)
        out = wei @ v
```
- **설명**: 계산된 집중 확률(`wei`)에 따라 실제 정보(`v`)를 섞습니다.
- **예시**: "나는 사과를 먹었다"에서 '먹었다'는 '사과'와 관계가 깊으므로, '사과'의 정보를 많이 가져오게 됩니다.
- **Input**: 집중 확률 `wei`, 정보 벡터 `v`
- **Output**: `out` (문맥을 고려해서 섞인 새로운 단어 정보)

---

## 4. 멀티 헤드 어텐션 (MultiHeadAttention)

눈이 하나면 불안하죠? 여러 개의 눈으로 동시에 봅니다.

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
```
- **설명**: 위에서 만든 `Head`(눈)를 `num_heads`(8개)만큼 만듭니다.
- **원리**: 어떤 눈은 문법을 보고, 어떤 눈은 의미를 보고, 어떤 눈은 앞뒤 관계를 봅니다. 다양한 관점을 종합하기 위함입니다.

```python
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
```
- **설명**: 8개의 눈이 본 결과물을 하나로 이어 붙입니다(`cat`). 교차 검증을 해서 더 정확한 의미를 파악합니다.
- **Input**: 데이터 `x`
- **Output**: 8가지 관점이 합쳐진 풍부한 정보

---

## 5. 피드 포워드 (FeedForward) - 생각 정리하기

정보를 모았으니 이제 혼자 곰곰이 생각하며 정리하는 시간입니다.

```python
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )
```
- **설명**: 정보를 4배로 뻥튀기했다가 다시 원래대로 줄이면서 내용을 정제합니다. 심사숙고하는 과정입니다.
    - `ReLU()`: 의미 없는(음수) 정보는 0으로 지워버리는 필터 역할(활성화 함수)을 합니다.
- **원리**: 사람의 뇌세포가 신호를 주고받을 때, 일정 자극 이상이어야 전달되는 원리를 흉내 낸 것입니다.
- **Input**: 어텐션을 거친 정보
- **Output**: 한층 더 깊이 생각하고 정제된 정보 (차원은 원래대로 돌아옴)

---

## 6. 블록 (Block) - 생각의 단계 (Transformer Block)

위의 과정(어텐션 + 생각 정리)을 하나의 세트(Block)로 묶습니다.

```python
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        # ... (생략: 어텐션과 피드포워드 준비) ...
    
    def forward(self, x, verbose=False, layer_idx=0):
        # 1. 어텐션
        x = x + self.sa(self.ln1(x))
        # 2. 피드포워드
        x = x + self.ffwd(self.ln2(x))
```
- **설명**:
    1. `self.sa(...)`: 어텐션(주변 살피기)을 수행합니다.
    2. `x + ...`: **잔차 연결(Residual Connection)**. 원래 있던 정보(`x`)를 까먹지 않으려고 더해줍니다. "선생님 설명(어텐션)도 듣지만, 내 원래 생각(x)도 중요해"라는 뜻입니다. 이게 없으면 층이 깊어질수록 멍청해집니다.
    3. `self.ln1(...)`: **레이어 정규화(LayerNorm)**. 정보들의 크기를 고르게 맞춰주어 공부(학습)가 잘 되게 돕습니다.
    4. `self.ffwd(...)`: 피드포워드(생각 정리)를 수행합니다.
- **원리**: 이 블록을 12개 쌓으면 12단계의 깊은 사고를 할 수 있는 GPT가 됩니다.

---

## 7. GPT 전체 모델 (The Body)

이제 모든 부품을 조립해서 몸체를 완성합니다.

```python
class GPT(nn.Module):
    def __init__(self):
        # ...
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
```
- **설명**:
    - `token_embedding`: **단어 사전**. 단어 번호를 의미 벡터로 바꿉니다. (예: 1번 -> 사과 느낌 벡터)
    - `position_embedding`: **위치 정보**. "나는 첫 번째 단어", "나는 두 번째 단어"라는 위치표를 붙여줍니다. (어텐션은 위치를 모르기 때문에 이게 꼭 필요합니다)

```python
        self.blocks = nn.ModuleList([Block(config.n_embd, config.n_head) for _ in range(config.n_layer)])
```
- **설명**: 아까 만든 생각 블록을 12층으로 쌓아 올립니다. (Deep Learning의 'Deep'이 여기서 나옵니다. 깊이가 깊을수록 똑똑합니다)

```python
    def forward(self, idx, targets=None, verbose=False):
        # ...
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=config.device))
        x = tok_emb + pos_emb
```
- **설명**: 단어의 의미와 위치 정보를 더해서(`+`) 시작합니다.
- **Input**: 단어 번호들 `idx` (예: `[10, 24, 55]`)
- **Output**: 의미가 담긴 벡터 `x`

```python
        for i, block in enumerate(self.blocks):
            x = block(x, verbose=verbose, layer_idx=i)
```
- **설명**: 12개의 생각 단계를 차례대로 통과합니다. 갈수록 정보는 추상적이고 고차원적으로 변합니다 12번 고민하는 것입니다.

```python
        x = self.ln_f(x)
        logits = self.lm_head(x)
```
- **설명**:
    - `ln_f`: 마지막으로 옷매무새를 다듬습니다(정규화).
    - `lm_head`: 10만 개(`vocab_size`)의 단어 중 다음에 올 단어가 무엇일지 점수(`logits`)를 매깁니다.
- **Output**: `logits` (다음 단어 예측 점수표)

---

## 8. 학습 (Training) - 공부시키기

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
```
- **설명**: **AdamW**라는 유능한 과외 선생님을 모셔옵니다. 틀린 문제(`loss`)를 보고 뇌세포를 어떻게 고쳐야 할지(`gradient`) 최적의 방법으로 알려줍니다.

```python
    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
        logits, loss = model(xb, yb)
```
- **설명**: `autocast`는 **"혼합 정밀도(Mixed Precision)"** 기술을 씁니다. 중요하지 않은 계산은 대충(16비트) 계산해서 속도를 빠르게 하고 메모리를 아낍니다.
- **Input**: 문제 `xb`, 정답 `yb`
- **Output**: 예측값 `logits`, 오답률 `loss`

```python
    scaler.scale(loss).backward()
    scaler.step(optimizer)
```
- **설명**: 공부하는 핵심 과정입니다.
    1. `model(xb, yb)`: 문제를 풀어보고 정답이랑 맞춰봅니다.
    2. `backward()`: **역전파(Backpropagation)**. 틀린 이유를 찾아서 머리끝부터 발끝까지 "너 때문에 틀렸어!"라고 책임을 묻고 미분값을 구합니다.
    3. `step()`: 책임을 물은 만큼 뇌세포(파라미터)를 조금씩 수정합니다. 이 과정을 수천 번 반복하면 모델이 점점 똑똑해집니다.

---

## 9. 텍스트 생성 (Inference)

```python
def generate(model, start_text, max_new_tokens=50, temperature=1.0):
    # ...
    idx_next = torch.multinomial(probs, num_samples=1)
    idx = torch.cat((idx, idx_next), dim=1)
```
- **설명**:
    1. 모델이 예측한 확률(`probs`)을 봅니다. (예: 사과 다음엔 '맛있다' 확률 80%, '아프다' 20%)
    2. `multinomial`: 룰렛을 돌려서 다음 단어 하나를 뽑습니다. 확률이 높을수록 뽑힐 가능성이 큽니다.
    3. `cat`: 뽑은 단어를 현재 문장 뒤에 이어 붙입니다.
    4. 이 과정을 반복하면 문장이 완성됩니다.

---

## 마치며

이 코드는 **ChatGPT의 조상**격인 GPT-2 모델의 방식을 그대로 따르고 있습니다.
1. 글자를 숫자로 바꾸고 (**Tokenization**)
2. 숫자를 의미 벡터로 바꾸고 (**Embedding**)
3. 단어들끼리 관계를 파악하고 (**Attention**)
4. 여러 층을 거치며 추론하고 (**Deep Layer**)
5. 다음 단어를 예측하는 방식 (**Next Token Prediction**)

이것이 현재 세상을 놀라게 하는 거대언어모델(LLM)의 모든 비밀입니다.
