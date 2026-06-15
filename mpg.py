import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. 저장해둔 모델, 스케일러, 딕셔너리 불러오기
# (파일들이 app.py와 같은 폴더에 있어야 합니다)
@st.cache_resource # 모델을 매번 새로 로드하지 않고 캐싱하여 속도를 높입니다
def load_models():
    model = joblib.load("kmeans_model.pkl")
    scaler = joblib.load("scaler.pkl")
    grade_dict = joblib.load("grade_dict.pkl")
    return model, scaler, grade_dict

try:
    model, scaler, grade_dict = load_models()
except FileNotFoundError:
    st.error("⚠️ 모델 파일(pkl)을 찾을 수 없습니다. 코랩에서 다운로드 받아 현재 폴더에 넣어주세요!")
    st.stop()

# 2. 웹페이지 제목 및 레이아웃 설정
st.set_page_config(page_title="자동차 친환경 등급 예측", layout="centered")
st.title("🚗 AI 기반 자동차 배출가스 성향 예측 서비스")
st.markdown("차량의 스펙을 입력하면 AI가 유사한 차량 군집을 분석하여 친환경 성향 등급을 알려줍니다.")
st.write("---")

# 3. 사용자 입력 양식 (UI) 만들기
st.subheader("📋 차량 스펙 입력")

col1, col2 = st.columns(2)

with col1:
    input_mpg = st.number_input("복합 연비 (mpg 단위)", min_value=1.0, max_value=100.0, value=25.0, step=0.1)
    input_disp = st.number_input("배기량 (리터 단위)", min_value=0.5, max_value=10.0, value=2.0, step=0.1)

with col2:
    input_cyl = st.selectbox("실린더 개수", options=[3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0], index=1)
    input_drive_str = st.selectbox("구동 방식", options=["fwd (전륜)", "rwd (후륜)", "awd (상시사륜)", "4wd (사륜)"])

# 구동 방식 입력값 처리 (원래 데이터프레임 매핑 기준 적용)
drive_clean = input_drive_str.split(" ")[0] # "fwd (전륜)" -> "fwd"
drive_mapping = {'fwd': 0.0, 'rwd': 1.0, 'awd': 2.0, '4wd': 3.0}
input_drive_num = drive_mapping.get(drive_clean, 0.0)

st.write("---")

# 4. '예측하기' 버튼 클릭 시 로직 실행
if st.button("🔥 AI 등급 분석 시작", use_container_width=True):
    
    # 학습에 사용했던 속성 구조와 똑같이 2차원 배열(데이터프레임) 데이터 생성
    # [복합연비, 배기량, 구동방식] -> 스케일러 학습 당시의 변수명이나 순서와 완벽히 일치해야 합니다.
    user_data = pd.DataFrame([{
        '복합연비': input_mpg,
        '배기량': input_disp,
        '구동방식': input_drive_num
    }])
    
    # 스케일러로 유저 입력값 표준화 변환
    user_scaled = scaler.transform(user_data)
    
    # K-Means 모델로 어떤 군집(cluster)에 속하는지 예측
    pred_cluster = model.predict(user_scaled)[0]
    
    # 결과 결과 매핑
    result_text = grade_dict.get(pred_cluster, "알 수 없는 등급")
    
    # 5. 화면에 분석 결과 예쁘게 띄워주기
    st.subheader("📊 AI 분석 결과")
    
    # 군집 번호별로 디자인에 포인트를 주기 위한 로직
    if "높음" in result_text or "낮음" in result_text:
        # 본인의 grade_dict 텍스트 성향에 맞춰서 컬러 박스를 바꿀 수 있습니다.
        if 1 == pred_cluster: # 예시: 1번이 친환경성 높음(좋은 거)일 때
            st.success(f"🎉 분석 완료: **{result_text}**")
        elif 0 == pred_cluster: # 예시: 0번이 친환경성 낮음(나쁜 거)일 때
            st.error(f"🚨 분석 완료: **{result_text}**")
        else:
            st.info(f"ℹ️ 분석 완료: **{result_text}**")
    else:
        st.info(f"ℹ️ 분석 완료: **{result_text}**")
        
    # 부가 설명 출력
    st.caption(f"본 결과는 입력하신 스펙과 가장 유사한 {drive_clean.upper()} 구동방식의 {int(input_cyl)}기통 차량들의 통계 군집(Cluster {pred_cluster})을 기반으로 계산되었습니다.")
