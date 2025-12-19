import os
import tiktoken
import numpy as np

input_file_path = 'input.txt' # 형이 구한 3GB 소설 파일 경로
train_file_path = 'train.bin'
val_file_path = 'val.bin'

def process_data():
    # 1. 파일이 있는지 확인
    if not os.path.exists(input_file_path):
        print("파일이 없습니다. 경로를 확인하세요.")
        return

    # 2. GPT-2 토크나이저 로드
    enc = tiktoken.get_encoding("gpt2")
    
    # 3. 파일을 청크(조각) 단위로 읽어서 처리 (램 폭발 방지)
    # 3GB를 한 번에 못 읽으니까 100MB씩 끊어서 읽음
    chunk_size = 100 * 1024 * 1024 
    
    # 전체 토큰 개수를 모르니 일단 리스트에 모으지 않고 바로바로 저장하는 방식은 복잡하니
    # 여기서는 "uint16" (2바이트 정수)으로 저장해서 용량을 줄임
    # GPT2 토큰은 50257개라 uint16(65535까지 표현)으로 충분함
    
    token_list = []
    
    print("토큰화 시작... (시간 좀 걸림)")
    with open(input_file_path, 'r', encoding='utf-8') as f:
        while True:
            text_chunk = f.read(chunk_size)
            if not text_chunk:
                break
            # 토큰화해서 리스트에 추가
            token_list.extend(enc.encode(text_chunk))
            print(f"현재 수집된 토큰 수: {len(token_list):,}")
    
    total_tokens = len(token_list)
    print(f"총 토큰 수: {total_tokens:,}")

    # 4. 데이터 쪼개기 (90% 학습, 10% 검증)
    split_idx = int(total_tokens * 0.9)
    train_tokens = np.array(token_list[:split_idx], dtype=np.uint16)
    val_tokens = np.array(token_list[split_idx:], dtype=np.uint16)

    # 5. 파일로 저장 (bin 파일)
    print("파일로 저장 중...")
    train_tokens.tofile(train_file_path)
    val_tokens.tofile(val_file_path)
    print("완료! train.bin, val.bin 생성됨.")

if __name__ == '__main__':
    process_data()