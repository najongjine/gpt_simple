import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import os
import requests
import time

# 님께서 만든 모델 파일(gpt_simple.py)에서 클래스들을 가져옵니다.
# 주의: gpt_simple.py 파일이 같은 폴더에 있어야 합니다.
from gpt_simple import GPT, GPTConfig

# --- 설정 ---
# 학습 횟수 (원리를 배우는 용도니 짧게 잡았습니다. 제대로 하려면 5000번 이상 추천)
max_iters =6000  
eval_interval = 100  # 100번마다 얼마나 잘하고 있나 중간 점검
learning_rate = 3e-4 # 학습 속도 (너무 크면 체하고, 너무 작으면 더딤)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"사용 장치: {device}")

# --- 1. 데이터 준비 (한국어/영어) ---
# 데이터를 담을 파일 이름
file_path = 'input.txt'

# 파일이 없으면 다운로드 (기본: 셰익스피어 영어 데이터)
if not os.path.exists(file_path):
    print("데이터가 없어서 다운로드합니다... (TinyShakespeare)")
    # 한국어로 학습하고 싶다면?
    # 1. 인터넷에서 구한 한국어 텍스트 파일(소설, 뉴스 등)을 'input.txt'로 저장하세요.
    # 2. 혹은 아래 url을 한국어 텍스트 파일 주소로 바꾸세요.
    url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(requests.get(url).text)

# 데이터 읽기
with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

print(f"데이터 길이: {len(text)} 글자")
print(f"데이터 앞부분 예시:\n{text[:100]}...")

# 토크나이저 준비 (GPT-2 기준)
# 팁: 한국어는 tiktoken이 토큰을 좀 잘게 쪼개지만, 학습 원리 배우는 덴 문제없습니다.
enc = tiktoken.get_encoding("cl100k_base")
data = torch.tensor(enc.encode(text), dtype=torch.long)
n = int(0.9 * len(data)) # 90%는 학습용, 10%는 검증용(시험문제)
train_data = data[:n]
val_data = data[n:]

# --- 2. 배치 뽑기 함수 (문제집 만들기) ---
def get_batch(split):
    # 학습용인지 검증용인지 선택
    data_source = train_data if split == 'train' else val_data
    
    # 랜덤한 위치(ix)를 여러 개 뽑습니다.
    # 예: 책의 50페이지, 120페이지, 3페이지... 이렇게 무작위로 펼침
    ix = torch.randint(len(data_source) - GPTConfig.block_size, (GPTConfig.batch_size,))
    
    # 해당 위치에서 block_size만큼 글자를 가져옵니다. (문제: x)
    x = torch.stack([data_source[i:i+GPTConfig.block_size] for i in ix])
    
    # 바로 다음 글자를 가져옵니다. (정답: y)
    y = torch.stack([data_source[i+1:i+GPTConfig.block_size+1] for i in ix])
    
    return x.to(device), y.to(device)

# --- 3. 손실(Loss) 계산 함수 ---
# 훈련에는 영향 안 주고, 그냥 점수만 매겨보는 용도
@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval() # 평가 모드 (드롭아웃 끔)
    for split in ['train', 'val']:
        losses = torch.zeros(50) # 200번 정도 랜덤하게 뽑아서 평균 내봄
        for k in range(50):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # 다시 학습 모드
    return out

# --- 4. 모델 및 선생님(Optimizer) 준비 ---
model = GPT() # 님 코드로 만든 그 로봇
model = model.to(device)

# 선생님 (AdamW 알고리즘이 국룰입니다)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print(f"--- 학습 시작 (총 {max_iters} 단계) ---")
start_time = time.time()

for iter in range(max_iters):

    # 일정 주기마다 중간 점검 (성적표 출력)
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(model)
        print(f"단계 {iter}: 훈련 오차 {losses['train']:.4f}, 검증 오차 {losses['val']:.4f}")

    # 1. 문제 풀기 (배치 가져오기)
    xb, yb = get_batch('train')

    # 2. 시험 보기 (Forward)
    logits, loss = model(xb, yb)

    # 3. 뇌 초기화 (이전 배치의 기울기 삭제)
    optimizer.zero_grad(set_to_none=True)

    # 4. 피드백 받기 (Backward - 역전파)
    # 틀린 만큼 어느 방향으로 수정해야 할지 계산
    loss.backward()

    # 5. 지능 수정하기 (Update)
    optimizer.step()

end_time = time.time()
print(f"--- 학습 완료! (소요 시간: {end_time - start_time:.2f}초) ---")

# --- 5. 모델 저장 ---
# 학습된 뇌(파라미터들)를 파일로 저장합니다.
save_path = 'model_weights.pth'
torch.save(model.state_dict(), save_path)
print(f"모델이 '{save_path}'에 저장되었습니다.")