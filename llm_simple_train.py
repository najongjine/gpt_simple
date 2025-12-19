import torch
import numpy as np
import os
import time
from gpt_simple import GPT, GPTConfig

# --- 설정 ---
max_iters = 20000     # 3GB 데이터면 5000번 택도 없음. 일단 2만 번 해보고 결과 보셈.
eval_interval = 500   # 자주 체크하면 시간 낭비
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 1. 데이터 준비 (numpy memory map 사용) ---
# prepare_data.py 로 만든 bin 파일을 씁니다.
train_data = np.memmap('train.bin', dtype=np.uint16, mode='r')
val_data = np.memmap('val.bin', dtype=np.uint16, mode='r')

# --- 2. 배치 뽑기 함수 (memmap용 수정) ---
def get_batch(split):
    data_source = train_data if split == 'train' else val_data
    # 3GB 넘는 데이터에서 랜덤 위치 뽑기
    ix = torch.randint(len(data_source) - GPTConfig.block_size, (GPTConfig.batch_size,))
    
    # int64(long)로 변환해서 가져옴
    x = torch.stack([torch.from_numpy((data_source[i:i+GPTConfig.block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data_source[i+1:i+GPTConfig.block_size+1]).astype(np.int64)) for i in ix])
    
    # VRAM으로 올릴 때 non_blocking=True 쓰면 쬐끔 더 빠름
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

# --- 3. 손실 계산 함수 (동일) ---
@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(100)
        for k in range(100):
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
scaler = torch.cuda.amp.GradScaler() # 16비트 연산 가속기 (속도 2배, 메모리 절약)

print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
print(f"학습 시작 (총 {max_iters} 단계)")

start_time = time.time()

for iter in range(max_iters):
    # 평가
    if iter % eval_interval == 0:
        losses = estimate_loss(model)
        print(f"[{iter}/{max_iters}] train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}")

    xb, yb = get_batch('train')

    # Mixed Precision Training (속도 향상 치트키)
    # float32 대신 float16을 섞어 써서 계산 속도를 높임
    with torch.cuda.amp.autocast():
        logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward() # loss 스케일링
    scaler.step(optimizer)
    scaler.update()

end_time = time.time()
print(f"학습 완료! 소요 시간: {(end_time - start_time)/60:.2f}분")

torch.save(model.state_dict(), 'model_weights.pth')