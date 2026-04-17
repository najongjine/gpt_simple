import torch
import tiktoken
import os
import time
import json  # [추가] json 파일 읽기용
from datasets import load_dataset
from gpt_simple import GPT, GPTConfig

# --- 설정 ---
learning_rate = 5e-5
max_iters = 1000  # 데이터가 늘어났으니 조금 더 학습
eval_interval = 100
device = 'cuda' if torch.cuda.is_available() else 'cpu'

load_path = 'base_model_weights.pth'       # Base Model (기반)
save_path = 'custom_instruct_model.pth' # 새로 저장될 모델

# --- 1. 데이터 준비 (KoAlpaca + 내 데이터) ---
print("1. KoAlpaca 데이터 로드 중...")
try:
    ds_alpaca = load_dataset("beomi/KoAlpaca-v1.1a", split="train")
except Exception as e:
    print(f"KoAlpaca 로드 실패: {e}")
    exit()

print("2. 나만의 데이터(my_inst_data.json) 로드 중...")
custom_data = []
if os.path.exists("my_inst_data.json"):
    with open("my_inst_data.json", "r", encoding="utf-8") as f:
        custom_data = json.load(f)
    print(f"-> 내 데이터 {len(custom_data)}개를 찾았습니다!")
else:
    print("-> my_inst_data.json 파일이 없습니다. KoAlpaca만 사용합니다.")

# [중요] 데이터 합치기 & 포맷팅
formatted_data = []

def format_prompt(inst, out):
    return f"### 질문:\n{inst}\n\n### 답변:\n{out}<|endoftext|>\n"

# A. KoAlpaca 데이터 변환 (너무 많으면 5000개만 사용)
print("데이터 합치는 중...")
for item in ds_alpaca.select(range(5000)): 
    formatted_data.append(format_prompt(item['instruction'], item['output']))

# B. 내 데이터 변환 (중요하니까 10번 반복해서 강조 학습!)
# 딥러닝에서는 데이터 양이 적으면 무시될 수 있어서, 중요한 데이터는 복사해서 양을 늘리기도 합니다.
if custom_data:
    for _ in range(10):  # 내 데이터를 10배로 뻥튀기
        for item in custom_data:
            formatted_data.append(format_prompt(item['instruction'], item['output']))

full_text = "".join(formatted_data)
print(f"총 학습 데이터 길이: {len(full_text)} 글자")

# 토크나이징
enc = tiktoken.get_encoding("cl100k_base")
train_ids = enc.encode(full_text, allowed_special={'<|endoftext|>'})
train_data = torch.tensor(train_ids, dtype=torch.long)

# --- 2. 모델 로드 (Base Model) ---
model = GPT()
model.to(device)

if os.path.exists(load_path):
    print(f"Base Model 로드: {load_path}")
    model.load_state_dict(torch.load(load_path, map_location=device))
else:
    print("Base Model이 없습니다. 먼저 llm_simple_train_bigtxt.py를 실행하세요.")
    exit()

# --- 3. 학습 루프 ---
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

def get_batch():
    ix = torch.randint(len(train_data) - GPTConfig.block_size, (GPTConfig.batch_size,))
    x = torch.stack([train_data[i:i+GPTConfig.block_size] for i in ix])
    y = torch.stack([train_data[i+1:i+GPTConfig.block_size+1] for i in ix])
    return x.to(device), y.to(device)

model.train()
print("학습 시작! (내 데이터 내용을 잘 배우는지 지켜보세요)")
start_time = time.time()

for iter in range(max_iters):
    xb, yb = get_batch()
    logits, loss = model(xb, yb)
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if iter % eval_interval == 0:
        print(f"단계 {iter}/{max_iters} | Loss: {loss.item():.4f}")

print("학습 완료! 모델 저장 중...")
torch.save(model.state_dict(), save_path)
print(f"✅ 저장 완료: {save_path}")