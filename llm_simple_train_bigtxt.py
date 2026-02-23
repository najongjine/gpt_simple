import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import os
import time
from datasets import load_dataset # [추가] Hugging Face 데이터셋 로드용

# gpt_simple.py 파일에서 모델 가져오기
from gpt_simple import GPT, GPTConfig
from huggingface_hub import login

token = os.getenv("HUGGING_FACE_TOKEN")

# --- 설정 ---
max_iters = 1000
eval_interval = 100
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"사용 장치: {device}")

# --- 1. 데이터 준비 (Hugging Face 데이터셋 사용) ---
print("Hugging Face에서 데이터셋을 불러옵니다... (시간이 조금 걸릴 수 있습니다)")
print("Hugging Face 로그인 중...")
try:
    login(token=HF_TOKEN)
    print("로그인 성공!")
except Exception as e:
    print(f"로그인 실패: {e}")
    print("토큰을 다시 확인해주세요.")
    exit()

# [변경] Hugging Face 데이터셋 로드
# 스트리밍 방식(streaming=True)을 쓰지 않으면 데이터를 한 번에 다운로드합니다.
try:
    ds = load_dataset("Yomm1927/aihub-ko-en-literary", cache_dir="D:/ai_data")
except Exception as e:
    print(f"데이터셋 로드 실패: {e}")
    print("터미널에서 'huggingface-cli login'을 실행하여 로그인이 필요한지 확인하세요.")
    exit()

print("데이터 처리 중...")

# [변경] 데이터셋에서 텍스트 추출
# 이 데이터셋은 보통 'train' 분할을 가지고 있으며, 컬럼은 'ko'(한국어), 'en'(영어) 등으로 구성됩니다.
# 한국어 모델을 만들기 위해 'ko' 컬럼만 뽑아서 하나의 긴 텍스트로 합칩니다.
# 데이터가 너무 많으면 메모리가 터질 수 있으므로, 예시로 앞부분 30,000개 샘플만 사용하겠습니다.
# 전체를 다 쓰고 싶다면 [:30000]을 제거하고 ds['train'] 전체를 순회하세요.

sample_limit = 30000 # 학습에 사용할 문장 개수 제한 (메모리 절약용)
raw_text_list = []

# 데이터 구조 확인 및 추출
if 'train' in ds:
    data_source = ds['train']
else:
    data_source = ds # 분할이 없는 경우

count = 0
for item in data_source:
    if 'ko' in item: # 'ko' 컬럼이 한국어라고 가정
        raw_text_list.append(item['ko'])
        count += 1
    if count >= sample_limit:
        break

# 문장들을 줄바꿈(\n)으로 이어 붙여 하나의 긴 문자열로 만듭니다.
#text = "\n".join(raw_text_list)
text = "\n<|endoftext|>\n".join(raw_text_list)

print(f"데이터 길이: {len(text)} 글자")
print(f"데이터 앞부분 예시:\n{text[:100]}...")

# [중요 변경] 토크나이저를 'gpt_simple.py' 설정에 맞춰 'cl100k_base'로 변경
# gpt_simple.py의 vocab_size가 100277이므로 gpt2(50257)를 쓰면 안 됩니다.
enc = tiktoken.get_encoding("cl100k_base")

# 토큰화 (텍스트 -> 숫자 변환)
#data = torch.tensor(enc.encode(text), dtype=torch.long)
data = torch.tensor(enc.encode(text, allowed_special={'<|endoftext|>'}), dtype=torch.long)
n = int(0.9 * len(data)) 
train_data = data[:n]
val_data = data[n:]

print(f"총 토큰 수: {len(data)}")

# --- 2. 배치 뽑기 함수 ---
def get_batch(split):
    data_source = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_source) - GPTConfig.block_size, (GPTConfig.batch_size,))
    x = torch.stack([data_source[i:i+GPTConfig.block_size] for i in ix])
    y = torch.stack([data_source[i+1:i+GPTConfig.block_size+1] for i in ix])
    return x.to(device), y.to(device)

# --- 3. 손실(Loss) 계산 함수 ---
@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(200)
        for k in range(200):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# --- 4. 모델 및 학습 시작 ---
model = GPT()
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print(f"--- 학습 시작 (총 {max_iters} 단계) ---")
start_time = time.time()

for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(model)
        print(f"단계 {iter}: 훈련 오차 {losses['train']:.4f}, 검증 오차 {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

end_time = time.time()
print(f"--- 학습 완료! (소요 시간: {end_time - start_time:.2f}초) ---")

# --- 5. 모델 저장 ---
save_path = 'model_weights.pth'
torch.save(model.state_dict(), save_path)
print(f"모델이 '{save_path}'에 저장되었습니다.")