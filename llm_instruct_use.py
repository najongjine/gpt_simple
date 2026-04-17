import torch
import tiktoken
import os
from gpt_simple import GPT, GPTConfig

# --- 1. 설정 ---
# [변경] 방금 학습시킨 Instruct 모델 파일명으로 변경
model_path = 'custom_instruct_model.pth' 
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"사용 장치: {device}")

if not os.path.exists(model_path):
    print(f"오류: {model_path} 파일이 없습니다.")
    exit()

# --- 2. 모델 로드 ---
print("Instruct 모델 로딩 중...")
model = GPT()
model.to(device)
# weights_only=True는 보안 경고를 피하기 위함 (없어도 됨)
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()
print("준비 완료! (종료: q)")

# --- 3. 대화 루프 ---
enc = tiktoken.get_encoding("cl100k_base")

while True:
    user_input = input("\n질문(Q): ")
    if user_input.lower() == 'q':
        break
    if not user_input:
        continue

    # [핵심 변경] 사용자가 입력한 말을 학습 데이터와 똑같은 '틀'에 끼워넣기
    prompt = f"### 질문:\n{user_input}\n\n### 답변:\n"

    # 인코딩 (특수 토큰 허용)
    input_ids = enc.encode(prompt, allowed_special={'<|endoftext|>'})
    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

    print("생성 중...", end="", flush=True)

    with torch.no_grad():
        # 답변 생성 (max_new_tokens를 좀 넉넉하게)
        generated_ids = model.generate(input_tensor, max_new_tokens=200)

    # 디코딩
    full_response = enc.decode(generated_ids[0].tolist())

    # [후처리] 질문 부분은 자르고, 답변만 깔끔하게 보여주기
    # "### 답변:" 뒷부분만 가져오고, <|endoftext|>가 나오면 거기서 끊음
    answer = full_response.split("### 답변:\n")[-1].split("<|endoftext|>")[0]

    print("\n" + "="*30)
    print(f"AI 답변: {answer.strip()}")
    print("="*30)