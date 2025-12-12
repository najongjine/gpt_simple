import torch
import tiktoken
import os

# 우리가 정의했던 모델 구조(설계도)를 가져옵니다.
from gpt_simple import GPT, GPTConfig

# --- 1. 설정 및 준비 ---
# 모델 파일 경로
model_path = 'model_weights.pth'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"사용 장치: {device}")

# 모델 파일이 있는지 확인
if not os.path.exists(model_path):
    print("오류: 학습된 모델 파일(model_weights.pth)이 없습니다!")
    print("먼저 llm_simple_train.py를 실행해서 모델을 학습시켜주세요.")
    exit()

# --- 2. 뇌(모델) 불러오기 ---
print("모델을 로딩 중입니다... (뇌세포 깨우는 중)")

# A. 빈 깡통(구조) 만들기
# 학습할 때랑 똑같은 구조의 빈 모델을 먼저 만듭니다.
model = GPT()
model.to(device)

# B. 학습된 기억(가중치) 주입하기
# map_location: GPU에서 학습한 걸 CPU에서 돌릴 때(혹은 반대) 에러 안 나게 해줌
checkpoint = torch.load(model_path, map_location=device, weights_only=True)
model.load_state_dict(checkpoint)

# C. 평가 모드로 전환 (중요!)
# 드롭아웃(Dropout) 같은 학습 전용 기능을 끕니다. 이걸 안 하면 대답이 이상하게 나옵니다.
model.eval()

print("로딩 완료! 대화를 시작합니다.")

# --- 3. 대화 루프 ---
# 토크나이저 준비 (GPT-2 기준)
enc = tiktoken.get_encoding("gpt2")

while True:
    # 사용자 입력 받기
    user_input = input("\n문장을 입력하세요 (종료하려면 q 입력): ")
    
    if user_input.lower() == 'q':
        print("종료합니다.")
        break
    
    if not user_input:
        continue

    # 1. 입력을 숫자로 변환 (Encoding)
    # 텍스트 -> 토큰 ID 리스트 -> 텐서(Tensor)
    input_ids = enc.encode(user_input)
    
    # unsqueeze(0): (T) 모양을 (1, T) 모양으로 바꿈. 
    # 모델은 항상 '배치(Batch)' 단위로 입력을 받기 때문에, [1개짜리 묶음]이라고 속여야 함.
    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

    # 2. 다음 단어 생성 (Generation)
    # max_new_tokens: 몇 글자나 더 떠들게 할지 설정 (여기선 100토큰)
    print("생성 중...", end="", flush=True)
    
    # torch.no_grad(): "생성할 때는 공부(기울기 계산) 하지 마" (메모리 아끼고 속도 빨라짐)
    with torch.no_grad():
        generated_ids = model.generate(input_tensor, max_new_tokens=100)

    # 3. 숫자를 다시 텍스트로 변환 (Decoding)
    # generated_ids[0]: 배치의 첫 번째 결과물 (우린 문장 하나만 넣었으니 0번)
    # tolist(): 텐서를 일반 파이썬 리스트로 변환
    decoded_text = enc.decode(generated_ids[0].tolist())

    print("\n" + "="*30)
    print(f"결과:\n{decoded_text}")
    print("="*30)