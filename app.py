# -*- coding: utf-8 -*-
# Streamlit 대시보드 - 최종 완성 버전 (모델 이식성 최종 수정)

import os, warnings, sys, types, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib
import openai
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import os
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

# .env 파일에서 환경 변수를 불러옴
load_dotenv()

# 1. 환경 분기: 현재 앱이 어디서 실행되는지 확인
if st.runtime.exists():
    # 2. Streamlit Cloud 환경: st.secrets에서 키 로드
    api_key = st.secrets.get(API_KEY_NAME)
else:
    # 3. 로컬 환경: .env 파일에서 키 로드
    try:
        load_dotenv() # .env 파일 내용을 환경 변수에 로드
        api_key = os.environ.get(API_KEY_NAME)
    except Exception:
        # load_dotenv가 실패하더라도 기본적으로는 os.environ에 접근을 시도합니다.
        # 이 예제에서는 명시적인 에러 대신, 키가 없는 경우 최종 에러 메시지를 사용합니다.
        api_key = os.environ.get(API_KEY_NAME)


# 4. API 키 유효성 검사 및 설정
if api_key:
    # 키가 존재하면 OpenAI 라이브러리에 설정합니다.
    openai.api_key = api_key
    st.success("OpenAI API 키가 성공적으로 설정되었습니다.")
else:
    # 키를 찾지 못했을 때 최종 에러 메시지 출력 및 중단
    error_msg = f"OpenAI API 키를 찾을 수 없습니다. "
    if st.runtime.exists():
        error_msg += f"Streamlit Cloud **Secrets**에 '{API_KEY_NAME}'을 등록했는지 확인하세요."
    else:
        error_msg += f"**로컬 .env 파일**에 '{API_KEY_NAME}'을 올바르게 설정했는지 확인하세요."
        
    st.error(error_msg)
    # 키가 없으면 더 이상 앱 로직이 진행되지 않도록 합니다.
    st.stop()

# =========================
# 공정별 변수 정의
# =========================
PROCESS_FEATURES = {
    "PCB 압착 공정": ['X_01','X_03','X_05','X_06','X_07','X_08','X_09'],
    "패드 공정": ['X_24','X_25','X_26','X_27','X_28','X_29'],
    "스크류 조립": ['X_33','X_34','X_35','X_36','X_37'],
    "안테나 조립": ['X_13','X_14','X_15','X_16','X_17','X_18'],
    "레이돔 공정": ['X_50','X_51','X_52','X_53','X_54','X_55','X_56'],
    "SMT 공정": ['X_19','X_20','X_21','X_22']
}

# =========================
# 0) 고정 경로/상수 (상대 경로로 수정)
# =========================
TRAIN_DEFAULT_PATH = "merged_train.csv"
SPEC_FILE_NAME = "y_feature_spec_info.csv"
COST_FILE_NAME = "Y_Cost.csv"
MODEL_DIR = "pre_trained_models"
X_FEATURE_INFO_PATH = "x_feature_info.csv"
Y_FEATURE_INFO_PATH = "y_feature_info.csv"
ARTIFACTS_PATH = "model_artifacts.joblib"


PAGES = [
    "메인 대시보드",
    "포지셔닝 전략",
    "공정 최적화",
    "자동 보고서 생성"
]

# =========================
# 페이지 설정 및 스타일
# =========================
st.set_page_config(
    page_title="FMCW 품질관리 대시보드",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    .stApp { background-color: #ffffff !important; }
    .main .block-container { padding: 2rem 3rem; max-width: 1400px; }
    .main-header { font-size: 3.2rem; font-weight: 800; color: #FA8072 !important; text-align: center; margin: 1rem 0 0.5rem; text-shadow: 2px 2px 4px rgba(250,128,114,0.2); }
    .sub-header { color: #666 !important; text-align: center; font-size: 1.3rem; margin-bottom: 2rem; font-weight: 300; }
    .section-title { font-size: 1.7rem; font-weight: 700; color: #FA8072 !important; margin: 2rem 0 1rem; padding-bottom: 0.5rem; border-bottom: 3px solid #FA8072 !important; }
    .section-header { font-size: 1.6rem; font-weight: 700; color: #FA8072 !important; margin: 1.5rem 0 1rem; }
    .kpi-card { background: #ffffff !important; border: 2px solid #e0e0e0 !important; border-radius: 12px; padding: 1.5rem; box-shadow: 0 2px 8px rgba(0,0,0,0.08) !important; text-align: center; margin-bottom: 1rem; }
    .kpi-label { font-size: 1.0rem; color: #666 !important; font-weight: 600; margin-bottom: 0.5rem; }
    .kpi-value { font-size: 2.2rem; font-weight: 800; color: #FA8072 !important; }
    .kpi-value.success { color: #28a745 !important; }
    .kpi-value.warning { color: #ffc107 !important; }
    .kpi-value.danger { color: #dc3545 !important; }
    .kpi-box { background: #ffffff !important; border: 2px solid #e0e0e0 !important; border-radius: 12px; padding: 1.2rem; text-align: center; margin-bottom: 1rem; box-shadow: 0 2px 8px rgba(0,0,0,0.08) !important; }
    .kpi-title { font-size: 1.0rem; color: #666 !important; font-weight: 600; margin-bottom: 0.5rem; }
    .kpi-box .kpi-value { font-size: 2.0rem; font-weight: 800; color: #FA8072 !important; }
    .chart-container { background: #ffffff !important; border: 2px solid #e0e0e0 !important; border-radius: 12px; padding: 1.5rem; box-shadow: 0 2px 8px rgba(0,0,0,0.08) !important; margin-bottom: 1.5rem; }
    .stButton > button { background: linear-gradient(135deg, #FA8072 0%, #FF6B6B 100%) !important; color: white !important; border: none !important; border-radius: 12px !important; padding: 0.75rem 2rem !important; font-weight: 600 !important; font-size: 1rem !important; box-shadow: 0 2px 8px rgba(250,128,114,0.3) !important; }
    .stButton > button:hover { box-shadow: 0 4px 12px rgba(250,128,114,0.5) !important; }
    [data-testid="stSidebar"] { display: none !important; }
    .stSelectbox > div > div { background: #ffffff !important; border: 2px solid #e0e0e0 !important; border-radius: 12px !important; }
    .stRadio > div { background: #f8f9fa !important; padding: 0.5rem; border-radius: 12px; }
    </style>
""", unsafe_allow_html=True)

# =========================
# 유틸리티 함수
# =========================
@st.cache_resource
def load_and_reconstruct_pipeline():
    """
    저장된 모델 부품들(artifacts)을 불러와 파이프라인 객체들을 재조립합니다.
    """
    if not os.path.exists(ARTIFACTS_PATH):
        st.error(f"모델 부품 파일({ARTIFACTS_PATH})을 찾을 수 없습니다. "
                 f"먼저 save_model_parts.py를 실행하여 파일을 생성하세요.")
        return None

    saved_artifacts = joblib.load(ARTIFACTS_PATH)
    
    # 1. Imputer 재조립 (scikit-learn 버전 호환성 확보)
    imputer = SimpleImputer(strategy='median')
    num_features = len(saved_artifacts['imputer_stats'])
    dummy_data = np.zeros((1, num_features))
    imputer.fit(dummy_data)
    imputer.statistics_ = saved_artifacts['imputer_stats']

    # === 변경된 부분: PCA 재조립 로직 수정 ===
    # 2. PCA 재조립 (scikit-learn 버전 호환성 확보)
    reconstructed_pca_info = []
    for pca_parts in saved_artifacts['pca_info']:
        pca = PCA(n_components=pca_parts['n_components'])
        
        # PCA 객체 초기화를 위해 가상 데이터로 fit 수행
        num_input_features = len(pca_parts['mean'])
        # n_samples는 n_components보다 크거나 같아야 함
        dummy_pca_data = np.zeros((max(pca_parts['n_components'], 1), num_input_features))
        pca.fit(dummy_pca_data)
        
        # 실제 학습된 파라미터로 덮어쓰기
        pca.mean_ = pca_parts['mean']
        pca.components_ = pca_parts['components']
        
        reconstructed_pca_info.append({
            'group_cols': pca_parts['group_cols'],
            'component_cols': pca_parts['component_cols'],
            'pca': pca
        })

    final_artifacts = {
        'imputer': imputer,
        'pca_info': reconstructed_pca_info,
        'models': saved_artifacts['models'],
        'pre_drop_cols': saved_artifacts.get('pre_drop_cols', []),
        'imputer_feature_names': saved_artifacts.get('imputer_feature_names', []),
        'final_feature_names': saved_artifacts['final_feature_names'],
        'target_columns': saved_artifacts['target_columns']
    }

    return {"type": "refactored", "artifacts": final_artifacts}


@st.cache_data
def load_feature_info(x_path=X_FEATURE_INFO_PATH, y_path=Y_FEATURE_INFO_PATH):
    try:
        x_info = pd.read_csv(x_path)
    except FileNotFoundError:
        st.warning(f"X Feature 정보 파일을 찾을 수 없습니다: {x_path}")
        x_info = pd.DataFrame(columns=['Feature', '설명'])
    try:
        y_info = pd.read_csv(y_path)
    except FileNotFoundError:
        st.warning(f"Y Feature 정보 파일을 찾을 수 없습니다: {y_path}")
        y_info = pd.DataFrame(columns=['Feature', '설명'])
    return x_info, y_info

@st.cache_data
def load_spec_and_cost(spec_path=SPEC_FILE_NAME, cost_path=COST_FILE_NAME):
    if not os.path.exists(spec_path):
        st.error(f"스펙 파일을 찾을 수 없습니다: {spec_path}")
        return None, None, None
    if not os.path.exists(cost_path):
        st.error(f"비용 파일을 찾을 수 없습니다: {cost_path}")
        return None, None, None

    def _read_any(p):
        return pd.read_excel(p) if p.lower().endswith((".xlsx", ".xls")) else pd.read_csv(p, encoding="utf-8")

    spec_df = _read_any(spec_path)
    cost_df = _read_any(cost_path)

    need_spec = {"Feature", "최소", "최대"}
    need_cost = {"Feature", "Cost"}
    if not need_spec.issubset(spec_df.columns):
        st.error("스펙 파일에는 'Feature','최소','최대' 컬럼이 필요합니다.")
        return None, None, None
    if not need_cost.issubset(cost_df.columns):
        st.error("비용 파일에는 'Feature','Cost' 컬럼이 필요합니다.")
        return None, None, None
    if "검사 시간 (초)" not in cost_df.columns:
        cost_df["검사 시간 (초)"] = 0

    y_spec_dict = spec_df.set_index("Feature").to_dict("index")
    return spec_df, cost_df, y_spec_dict

@st.cache_data
def load_data(path=TRAIN_DEFAULT_PATH):
    if not os.path.exists(path):
        st.error(f"데이터 파일을 찾을 수 없습니다: {path}")
        st.stop()
    return pd.read_csv(path)

@st.cache_resource
def train_or_load_simple_models(df, y_cols, y_spec_dict):
    os.makedirs(MODEL_DIR, exist_ok=True)
    X_cols = [c for c in df.columns if c.startswith("X_")]
    if not X_cols:
        return {}
    X = df[X_cols].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.mean())
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    models = {}
    for y in y_cols:
        if y not in y_spec_dict:
            continue
        fpath = os.path.join(MODEL_DIR, f"{y}_lr_model.joblib")
        try:
            if os.path.exists(fpath):
                models[y] = joblib.load(fpath)
                continue
        except Exception:
            pass
        y_s = pd.to_numeric(df[y], errors="coerce")
        spec = y_spec_dict[y]
        y_fail = ((y_s < spec["최소"]) | (y_s > spec["최대"])).astype(int)
        if y_fail.sum() < 10 or y_fail.sum() == len(y_fail):
            continue
        lr = LogisticRegression(solver="liblinear", class_weight="balanced", random_state=42).fit(Xs, y_fail)
        models[y] = (lr, scaler, X_cols)
        joblib.dump(models[y], fpath)
    return models

def transform_X_for_models_refactored(X_df, art):
    drop_cols = art.get('pre_drop_cols', [])
    X_df_dropped = X_df.drop(columns=[c for c in drop_cols if c in X_df.columns], errors='ignore').copy()

    imputer_cols = art.get('imputer_feature_names', [])
    if not imputer_cols:
        st.error("모델 아티팩트 파일에 'imputer_feature_names' 정보가 없습니다.")
        return None
    
    for col in imputer_cols:
        if col not in X_df_dropped.columns:
            X_df_dropped[col] = 0.0
            
    X_df_ordered = X_df_dropped[imputer_cols]
    
    imputer = art['imputer']
    X_imp = pd.DataFrame(imputer.transform(X_df_ordered), columns=X_df_ordered.columns, index=X_df_ordered.index)
    
    X_trans = X_imp.copy()
    for g in art['pca_info']:
        cols = [c for c in g['group_cols'] if c in X_trans.columns]
        if not cols: continue
        Z = g['pca'].transform(X_trans[cols])
        comp_cols = g['component_cols']
        X_trans = pd.concat([X_trans.drop(columns=cols),
                             pd.DataFrame(Z, columns=comp_cols, index=X_trans.index)], axis=1)
    
    final_names = art['final_feature_names']
    for c in final_names:
        if c not in X_trans.columns:
            X_trans[c] = 0.0
            
    return X_trans[final_names]

def predict_multi_any(pipe_obj, X_src):
    ptype = pipe_obj['type']; art = pipe_obj['artifacts']
    if ptype == 'refactored':
        X_model = transform_X_for_models_refactored(X_src, art)
        if X_model is None: return pd.DataFrame()
        pred = {t: art['models'][t].predict(X_model) for t in art['target_columns']}
        return pd.DataFrame(pred, index=X_model.index)
    return pd.DataFrame()

def _first_valid_target(y_candidates, y_spec_dict):
    for y in y_candidates:
        if y in y_spec_dict:
            return y
    return None

def _ensure_numeric_series(s):
    return pd.to_numeric(s, errors="coerce") if not pd.api.types.is_numeric_dtype(s) else s

def get_feature_name(feature_code, info_dict, short=False):
    full_name = feature_code
    return full_name[:40] + '...' if short and len(full_name) > 40 else full_name

# =========================
# 데이터 로드
# =========================
df_raw = load_data()
spec_df, cost_df, y_spec_dict = load_spec_and_cost()
x_feature_info, y_feature_info = load_feature_info()
if spec_df is None:
    st.stop()

df = df_raw.copy()
if "X_33" in df.columns:
    df = df[df["X_33"] <= 6].copy()

y_cols = [c for c in df.columns if c.startswith("Y_")]
x_cols = [c for c in df.columns if c.startswith("X_")]

if 'ai_models' not in st.session_state:
    st.session_state['ai_models'] = {}
if 'ai_kpi' not in st.session_state:
    st.session_state['ai_kpi'] = None
if 'stat_kpi' not in st.session_state:
    st.session_state['stat_kpi'] = {}
if 'current_stat_kpi_criterion' not in st.session_state:
    st.session_state['current_stat_kpi_criterion'] = '비용 효율성 최우선 (비용당 불합격률 기준)'

# =========================
# 네비게이션
# =========================
st.markdown('<div class="main-header">FMCW 품질관리 대시보드</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("메인", use_container_width=True):
        st.session_state.page = PAGES[0]
with col2:
    if st.button("포지셔닝", use_container_width=True):
        st.session_state.page = PAGES[1]
with col3:
    if st.button("공정 최적화", use_container_width=True):
        st.session_state.page = PAGES[2]

with col4:
    if st.button("자동 보고서 생성", use_container_width=True):
        st.session_state.page = PAGES[3]



page = st.session_state.get("page", PAGES[0])
st.markdown("---")

# =========================
# 메인 대시보드
# =========================
if page == "메인 대시보드":
    st.markdown('<div class="sub-header">실시간 품질 모니터링 및 핵심 인사이트</div>', unsafe_allow_html=True)
    
    y_default = _first_valid_target(y_cols, y_spec_dict)
    if y_default is None:
        st.warning("스펙 파일과 일치하는 Y 지표가 없습니다.")
        st.stop()

    col1, col2 = st.columns([1, 2])
    with col1:
        y_target = st.selectbox("분석 지표 선택",
                                [y for y in y_cols if y in y_spec_dict],
                                index=[y for y in y_cols if y in y_spec_dict].index(y_default))
    with col2:
        slice_choice = st.radio("데이터 필터", ["전체", "합격", "불합격"], horizontal=True)

    lo, hi = y_spec_dict[y_target]["최소"], y_spec_dict[y_target]["최대"]
    y_series = _ensure_numeric_series(df[y_target])
    pass_mask = (y_series >= lo) & (y_series <= hi)
    fail_mask = ~pass_mask
    n_pass, n_fail = int(pass_mask.sum()), int(fail_mask.sum())
    view_df = df.loc[pass_mask] if slice_choice == "합격" else (df.loc[fail_mask] if slice_choice == "불합격" else df)

    st.markdown('<div class="section-title">핵심 지표</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">총 샘플 수</div><div class="kpi-value">{len(df):,}</div></div>', unsafe_allow_html=True)
    with col2:
        pass_rate = (n_pass/len(df)*100) if len(df) else 0
        color_class = "success" if pass_rate >= 95 else ("warning" if pass_rate >= 90 else "danger")
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">합격률</div><div class="kpi-value {color_class}">{pass_rate:.1f}%</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">합격 건수</div><div class="kpi-value success">{n_pass:,}</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">불합격 건수</div><div class="kpi-value danger">{n_fail:,}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">데이터 분석</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container"><p><b>합격/불합격 비율</b></p>', unsafe_allow_html=True)
        pie_df = pd.DataFrame({"상태": ["합격", "불합격"], "건수": [n_pass, n_fail]})
        fig = px.pie(pie_df, names="상태", values="건수", hole=0.4, color="상태",
                     color_discrete_map={"합격": "#28a745", "불합격": "#dc3545"})
        fig.update_traces(textposition="inside", textinfo="percent+label", textfont_size=14)
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="chart-container"><p><b>분포 및 스펙 범위</b></p>', unsafe_allow_html=True)
        fig_hist = px.histogram(view_df, x=y_target, nbins=50, color_discrete_sequence=['#FA8072'])
        fig_hist.add_vline(x=lo, line_dash="dash", line_color="red")
        fig_hist.add_vline(x=hi, line_dash="dash", line_color="red")
        fig_hist.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig_hist, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">주요 영향 변수 분석</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<b>변수별 불량 영향도 (Top 15)</b>', unsafe_allow_html=True)
        
        if not st.session_state["ai_models"]:
            st.session_state["ai_models"] = train_or_load_simple_models(df, y_cols, y_spec_dict)

        imp_df = pd.DataFrame() 

        if y_target in st.session_state["ai_models"]:
            model, scaler, x_use_cols = st.session_state["ai_models"][y_target]
            imp = np.abs(model.coef_.ravel())
            imp_df = pd.DataFrame({"Feature": x_use_cols, "중요도": imp}).sort_values("중요도", ascending=False).head(15)
        else:
            x_use = [c for c in x_cols if pd.api.types.is_numeric_dtype(df[c])]
            y_bin = (~pass_mask).astype(int)
            corrs = {}
            for c in x_use:
                xc = df[c]
                if xc.notna().sum() > 1:
                    try:
                        corrs[c] = np.abs(np.corrcoef(xc.fillna(xc.mean()), y_bin)[0, 1])
                    except:
                        corrs[c] = 0.0
                else:
                    corrs[c] = 0.0
            imp_df = pd.DataFrame({"Feature": list(corrs.keys()), "중요도": list(corrs.values())}).sort_values("중요도", ascending=False).head(15)
        
        fig_imp = px.bar(imp_df, x="Feature", y="중요도", color="중요도",
                         color_continuous_scale=[[0, '#FFE4E1'], [1, '#FA8072']])
        fig_imp.update_layout(height=450, xaxis_title=None, yaxis_title="중요도 점수", showlegend=False)
        st.plotly_chart(fig_imp, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<b>주요 변수별 합격/불합격 분포 (Top 3)</b>', unsafe_allow_html=True)
        
        if not imp_df.empty:
            top_3_features = imp_df["Feature"].head(3).tolist()
            
            temp_df = df[top_3_features].copy()
            temp_df['상태'] = np.where(pass_mask, '합격', '불합격')
            
            for feature in top_3_features:
                st.markdown(f"**{feature}**")
                fig_box = px.box(temp_df, x='상태', y=feature, color='상태',
                                 color_discrete_map={"합격": "#28a745", "불합격": "#dc3545"})
                fig_box.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.info("영향도 분석 데이터가 없습니다.")
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<p class="section-header">3. 공정별 관리도 (Control Chart)</p>', unsafe_allow_html=True)
    import time as _time
    selected_process = st.selectbox("분석할 공정을 선택하세요.", options=list(PROCESS_FEATURES.keys()))
    if selected_process:
        features_in_process = [f for f in PROCESS_FEATURES[selected_process] if f in df.columns]
        if features_in_process:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            for feature in features_in_process:
                s = pd.to_numeric(df[feature], errors='coerce')
                mean_val = s.mean()
                std_val  = s.std(ddof=1)
                if pd.isna(std_val) or std_val == 0:
                    UCL = mean_val
                    LCL = mean_val
                else:
                    UCL = mean_val + 1.96 * std_val
                    LCL = mean_val - 1.96 * std_val

                n_show = min(100, len(s))
                y_show = s.head(n_show).reset_index(drop=True)

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(n_show)), y=y_show,
                    mode='lines+markers', name='측정값',
                    line=dict(color='#1d3557', width=2),
                    marker=dict(size=6, color='#1d3557')
                ))
                fig.add_trace(go.Scatter(
                    x=list(range(n_show)), y=[mean_val] * n_show,
                    mode='lines', name='평균 (CL)', line=dict(color='#28a745', width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=list(range(n_show)), y=[UCL] * n_show,
                    mode='lines', name='UCL', line=dict(color='#dc3545', width=2, dash='dash')
                ))
                fig.add_trace(go.Scatter(
                    x=list(range(n_show)), y=[LCL] * n_show,
                    mode='lines', name='LCL', line=dict(color='#dc3545', width=2, dash='dash')
                ))
                if len(s):
                    fig.add_hrect(y0=UCL, y1=float(s.max()) * 1.05, fillcolor="red", opacity=0.1, line_width=0)
                    fig.add_hrect(y0=float(s.min()) * 0.95, y1=LCL, fillcolor="red", opacity=0.1, line_width=0)
                fig.update_layout(
                    title=f"<b>{feature} 관리도</b><br><sub>평균: {mean_val:.4f} | UCL: {UCL:.4f} | LCL: {LCL:.4f}</sub>",
                    xaxis_title="샘플 번호", yaxis_title="측정값",
                    template="plotly_white", height=400, hovermode='x unified',
                    legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom")
                )
                st.plotly_chart(fig, use_container_width=True)

                if len(s) and (UCL != LCL):
                    outliers_count = ((s < LCL) | (s > UCL)).sum()
                    outliers_ratio = (outliers_count / len(s) * 100) if len(s) else 0
                    if outliers_count > 0:
                        st.warning(f"{feature}: {outliers_count}건 ({outliers_ratio:.2f}%)이 관리한계를 벗어났습니다.")
                    else:
                        st.success(f"{feature}: 모든 데이터가 관리한계 내에 있습니다.")
                else:
                    st.info(f"{feature}: 분산이 거의 0이어서 CL/UCL/LCL 구분이 의미 없습니다.")
                st.markdown("---")
            st.markdown('</div>', unsafe_allow_html=True)
            st.info("관리도 해석: 빨간 점선(UCL/LCL)을 벗어나는 점은 공정이 통계적으로 불안정한 상태를 의미합니다.")
        else:
            st.warning(f"'{selected_process}' 공정에 해당하는 데이터 변수가 없습니다.")

    st.markdown('<p class="section-header">3-추가: 공정별 관리도 - 실시간 스트리밍</p>', unsafe_allow_html=True)

    use_stream = st.checkbox(
        "실시간 스트리밍 모드 (라인이 ID 순서대로 누적, 축 자동)",
        value=False,
        help="체크하면 선택 공정의 모든 변수 라인이 ID 순서대로 동시에 늘어납니다."
    )

    if use_stream:
        candidate_id_cols = [c for c in ['id','test_id','sample_id','filename','uid','battery_id','index'] if c in df.columns]
        df_stream = df
        if not candidate_id_cols:
            df_stream = df.copy()
            df_stream['index'] = np.arange(len(df_stream))
            candidate_id_cols = ['index']
        id_col = st.selectbox("정렬(스트리밍) 기준 ID/순서 컬럼", options=candidate_id_cols, index=0, key="stream_id_col")

        default_proc_idx = list(PROCESS_FEATURES.keys()).index(selected_process) if selected_process in PROCESS_FEATURES else 0
        sel_proc_stream = st.selectbox(
            "스트리밍할 공정", options=list(PROCESS_FEATURES.keys()),
            index=default_proc_idx, key="stream_proc"
        )
        cols_per_row = st.slider("열 수(한 행 그래프 수)", 1, 4, 2, 1, key="stream_cols")

        features_in_proc = [f for f in PROCESS_FEATURES.get(sel_proc_stream, []) if f in df_stream.columns]
        if not features_in_proc:
            st.warning(f"'{sel_proc_stream}' 공정에 해당하는 데이터 변수가 없습니다.")
        else:
            cc1, cc2, cc3, cc4 = st.columns([1,1,2,2])
            with cc1:
                if st.button("이전", key="stream_prev_all"):
                    st.session_state.setdefault("stream_pos", 1)
                    st.session_state["stream_pos"] = max(1, st.session_state["stream_pos"] - 1)
            with cc2:
                if st.button("다음", key="stream_next_all"):
                    st.session_state.setdefault("stream_pos", 1)
                    st.session_state["stream_pos"] = min(len(df_stream), st.session_state["stream_pos"] + 1)
            with cc3:
                st.session_state.setdefault("stream_play", False)
                play_label = "일시정지" if st.session_state["stream_play"] else "재생"
                if st.button(play_label, key="stream_play_toggle"):
                    st.session_state["stream_play"] = not st.session_state["stream_play"]
                    if st.session_state["stream_play"]:
                        st.session_state["stream_last_tick"] = _time.time()
                        st.session_state["stream_just_toggled"] = True
                    else:
                        st.session_state["stream_just_toggled"] = False
            with cc4:
                step_ms = st.slider("재생 간격(ms)", 10, 1000, 60, 10, key="stream_step_ms")

            df_sorted = df_stream.sort_values(by=id_col, kind="stable").reset_index(drop=True)
            total_len = len(df_sorted)

            st.session_state.setdefault("stream_pos", 1)
            st.session_state.setdefault("stream_last_tick", 0.0)

            sig = f"{id_col}__{sel_proc_stream}"
            if st.session_state.get("stream_sig") != sig:
                st.session_state["stream_sig"] = sig
                st.session_state["stream_pos"] = 1
                st.session_state["stream_last_tick"] = _time.time()
                st.session_state["stream_just_toggled"] = False

            pos = st.session_state["stream_pos"] 

            stats = {}
            for feat in features_in_proc:
                s = pd.to_numeric(df_sorted[feat], errors='coerce')
                m  = float(s.mean())
                sd = float(s.std(ddof=1)) if len(s) > 1 else 0.0
                if sd == 0 or pd.isna(sd):
                    sd = 1e-9
                stats[feat] = (m, m + 1.96*sd, m - 1.96*sd)

            n = len(features_in_proc)
            rows = (n + cols_per_row - 1) // cols_per_row
            feats_iter = iter(features_in_proc)

            for _ in range(rows):
                row_cols = st.columns(cols_per_row)
                for c in row_cols:
                    try:
                        feat = next(feats_iter)
                    except StopIteration:
                        break

                    y_full = pd.to_numeric(df_sorted[feat], errors='coerce').reset_index(drop=True)
                    x_full = np.arange(total_len)

                    y_draw = y_full.copy()
                    if pos < total_len:
                        y_draw.iloc[pos:] = np.nan

                    mean_val, UCL, LCL = stats[feat]
                    cur_id_val = df_sorted.iloc[pos-1][id_col] if pos-1 < total_len else df_sorted.iloc[-1][id_col]

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=x_full, y=y_draw,
                        mode='lines', name='측정값(누적)',
                        line=dict(width=2)
                    ))
                    if pos >= 1:
                        fig.add_trace(go.Scatter(
                            x=[pos-1], y=[y_full.iloc[pos-1]],
                            mode='markers', name='현재 포인트',
                            marker=dict(size=12, symbol='circle-open')
                        ))
                    fig.add_trace(go.Scatter(x=x_full, y=[mean_val]*total_len, mode='lines', name='평균 (CL)', line=dict(width=2)))
                    fig.add_trace(go.Scatter(x=x_full, y=[UCL]*total_len,  mode='lines', name='UCL', line=dict(width=2, dash='dash')))
                    fig.add_trace(go.Scatter(x=x_full, y=[LCL]*total_len,  mode='lines', name='LCL', line=dict(width=2, dash='dash')))
                    if pos >= 1:
                        fig.add_vline(x=pos-1, line_width=1, line_dash="dot", line_color="gray")

                    fig.update_layout(
                        title=(f"<b>{feat}</b><br><sub>ID={cur_id_val} | pos {pos}/{total_len} | "
                               f"CL={mean_val:.4f} · UCL={UCL:.4f} · LCL={LCL:.4f}</sub>"),
                        xaxis=dict(title="순서(0 ~ N-1)", autorange=True),
                        yaxis=dict(title="측정값",       autorange=True),
                        template="plotly_white", height=340, hovermode='x unified',
                        legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom")
                    )
                    c.plotly_chart(fig, use_container_width=True)

            total_outliers = 0
            for feat in features_in_proc:
                s = pd.to_numeric(df_sorted[feat], errors='coerce')
                _, UCL, LCL = stats[feat]
                total_outliers += int(((s > UCL) | (s < LCL)).sum())
            ratio = (total_outliers / (total_len * len(features_in_proc)) * 100) if total_len > 0 else 0.0

            k1, k2, k3 = st.columns(3)
            k1.metric("누적 이상치 총합(전체 변수)", f"{total_outliers:,}")
            k2.metric("평균 이상치 비율(%)", f"{ratio:.2f}%")
            if pos-1 < total_len:
                cur_id_val = df_sorted.iloc[pos-1][id_col]
            else:
                cur_id_val = df_sorted.iloc[-1][id_col]
            k3.metric("현재 ID", f"{cur_id_val}")

            now = _time.time()
            last = st.session_state["stream_last_tick"]

            if st.session_state["stream_play"]:
                if last == 0 or st.session_state.get("stream_just_toggled", False):
                    st.session_state["stream_last_tick"] = now
                    st.session_state["stream_just_toggled"] = False
                    try:
                        st.rerun()
                    finally:
                        st.stop()

                elapsed_ms = (now - last) * 1000.0
                if elapsed_ms >= step_ms:
                    st.session_state["stream_pos"] = min(total_len, st.session_state["stream_pos"] + 1)
                    st.session_state["stream_last_tick"] = now
                    try:
                        st.rerun()
                    finally:
                        st.stop()

# =========================
# 포지셔닝 전략
# =========================
elif page == "포지셔닝 전략":
    st.markdown('<div class="sub-header">AI 예측 기반 품질 포지셔닝 분석</div>', unsafe_allow_html=True)

    art_pack = load_and_reconstruct_pipeline() # 함수 호출 변경
    
    if art_pack is None:
        st.stop()

    y_choices = art_pack['artifacts']['target_columns']
    
    st.markdown('<p class="section-header">예측 모드 선택</p>', unsafe_allow_html=True)
    prediction_mode = st.radio(
        "예측 방식을 선택하세요",
        ["전체 데이터 예측", "공정 수치 직접 입력 (단일 제품)"],
        horizontal=True
    )
    
    if prediction_mode == "공정 수치 직접 입력 (단일 제품)":
        st.markdown('<p class="section-header">공정 변수 입력</p>', unsafe_allow_html=True)
        st.info("공정 변수값을 입력하면 해당 조건에서 모든 Y 지표(14개)를 한 번에 예측합니다.")
        
        excluded_vars = ['X_antenna_std', 'X_press_std', 'X_screw_depth_mean', 'X_screw_std', 'X_total_press']
        
        input_data = {}
        
        with st.expander("공정 변수 입력 (클릭하여 펼치기)", expanded=True):
            for process_name, features in PROCESS_FEATURES.items():
                available_features = [f for f in features if f in x_cols and f not in excluded_vars]
                
                if available_features:
                    st.markdown(f"**{process_name}**")
                    cols = st.columns(3)
                    for idx, feature in enumerate(available_features):
                        with cols[idx % 3]:
                            default_val = float(df[feature].mean())
                            min_val = float(df[feature].min())
                            max_val = float(df[feature].max())
                            input_data[feature] = st.number_input(
                                f"{feature}",
                                value=default_val,
                                min_value=min_val,
                                max_value=max_val,
                                step=(max_val - min_val) / 100,
                                format="%.4f",
                                key=f"input_{feature}"
                            )
                    st.markdown("---")
            
            remaining_features = [x for x in x_cols
                                  if x not in sum(PROCESS_FEATURES.values(), [])
                                  and x not in excluded_vars]
            if remaining_features:
                st.markdown("**기타 공정 변수**")
                cols = st.columns(3)
                for idx, feature in enumerate(remaining_features):
                    with cols[idx % 3]:
                        default_val = float(df[feature].mean())
                        min_val = float(df[feature].min())
                        max_val = float(df[feature].max())
                        input_data[feature] = st.number_input(
                            f"{feature}",
                            value=default_val,
                            min_value=min_val,
                            max_value=max_val,
                            step=(max_val - min_val) / 100,
                            format="%.4f",
                            key=f"input_{feature}"
                        )
        
        with st.expander("자동 계산 변수 (입력 불필요)", expanded=False):
            st.markdown("""
            다음 변수들은 다른 변수들로부터 자동으로 계산되므로 입력이 필요하지 않습니다:
            - **X_antenna_std**: 안테나 관련 표준편차
            - **X_press_std**: 압착 관련 표준편차
            - **X_screw_depth_mean**: 스크류 깊이 평균
            - **X_screw_std**: 스크류 관련 표준편차
            - **X_total_press**: 전체 압착 합계
            
            이 변수들은 예측 시 자동으로 평균값으로 설정됩니다.
            """)
        
        run_pred = st.button("전체 Y 지표 예측 실행", type="primary")
        
        if run_pred:
            with st.spinner("모든 Y 지표 예측 수행 중..."):
                input_df = pd.DataFrame([input_data])
                for col in x_cols:
                    if col not in input_df.columns:
                        input_df[col] = df[col].mean()
                input_df = input_df[x_cols]
                
                try:
                    pred_df = predict_multi_any(art_pack, input_df)
                    st.session_state["pos_pred_df_manual"] = pred_df
                    st.session_state["pos_input_data"] = input_data
                    st.success("모든 Y 지표 예측 완료!")
                except Exception as e:
                    st.error(f"예측 실패: {e}")
                    st.stop()
        
        if "pos_pred_df_manual" in st.session_state and st.session_state.get("pos_pred_df_manual") is not None:
            pred_df = st.session_state["pos_pred_df_manual"]
            
            st.markdown('<div class="section-title">전체 Y 지표 예측 결과 (14개)</div>', unsafe_allow_html=True)
            
            results = []
            for y_col in y_choices:
                if y_col in pred_df.columns and y_col in y_spec_dict:
                    spec = y_spec_dict[y_col]
                    lo, hi = spec["최소"], spec["최대"]
                    pred_value = float(pred_df[y_col].iloc[0])
                    
                    in_spec = lo <= pred_value <= hi
                    spec_status = "합격" if in_spec else "불합격"
                    
                    position = "스펙 밖"
                    if in_spec:
                        spec_data = df[y_col][(df[y_col] >= lo) & (df[y_col] <= hi)]
                        if not spec_data.empty:
                            q1 = spec_data.quantile(0.25)
                            q3 = spec_data.quantile(0.75)
                            if pred_value <= q1: position = "하위 25%"
                            elif pred_value >= q3: position = "상위 25%"
                            else: position = "중위 50%"
                        else:
                            position = "스펙 내"
                    
                    results.append({
                        "Y 지표": y_col,
                        "예측값": f"{pred_value:.4f}",
                        "스펙 범위": f"{lo:.4f} ~ {hi:.4f}",
                        "스펙 판정": spec_status,
                        "포지셔닝": position
                    })
            
            results_df = pd.DataFrame(results)
            
            pass_count = results_df["스펙 판정"].str.contains("합격").sum()
            fail_count = len(results_df) - pass_count
            pass_rate = (pass_count / len(results_df) * 100) if len(results_df) > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f'<div class="kpi-card"><div class="kpi-label">전체 Y 지표 수</div><div class="kpi-value">{len(results_df)}개</div></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="kpi-card"><div class="kpi-label">합격 지표 수</div><div class="kpi-value success">{pass_count}개</div></div>', unsafe_allow_html=True)
            with col3:
                st.markdown(f'<div class="kpi-card"><div class="kpi-label">전체 합격률</div><div class="kpi-value {"success" if pass_rate >= 90 else "warning" if pass_rate >= 70 else "danger"}">{pass_rate:.1f}%</div></div>', unsafe_allow_html=True)
            
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown("**상세 예측 결과**")
            
            def highlight_status(row):
                if "합격" in row["스펙 판정"]:
                    return ['background-color: #d4edda']*len(row)
                else:
                    return ['background-color: #f8d7da']*len(row)
            
            st.dataframe(
                results_df.style.apply(highlight_status, axis=1),
                use_container_width=True,
                height=500
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="section-title">예측 결과 시각화</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.markdown("**스펙 판정 분포**")
                status_counts = results_df["스펙 판정"].value_counts()
                fig_pie = px.pie(
                    values=status_counts.values,
                    names=status_counts.index,
                    color=status_counts.index,
                    color_discrete_map={"합격": "#28a745", "불합격": "#dc3545"}
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                fig_pie.update_layout(showlegend=True, height=400)
                st.plotly_chart(fig_pie, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.markdown("**포지셔닝 분포 (합격 지표만)**")
                pass_results = results_df[results_df["스펙 판정"].str.contains("합격")]
                if not pass_results.empty:
                    position_counts = pass_results["포지셔닝"].value_counts()
                    colors_map = {"하위 25%": "#ffc107", "중위 50%": "#28a745", "상위 25%": "#007bff"}
                    fig_pos = px.pie(
                        values=position_counts.values,
                        names=position_counts.index,
                        color=position_counts.index,
                        color_discrete_map=colors_map
                    )
                    fig_pos.update_traces(textposition='inside', textinfo='percent+label')
                    fig_pos.update_layout(showlegend=True, height=400)
                    st.plotly_chart(fig_pos, use_container_width=True)
                else:
                    st.warning("합격한 지표가 없습니다.")
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="section-title">개별 지표 예측값 상세 분석</div>', unsafe_allow_html=True)
            st.info("각 품질 지표의 예측값이 원본 데이터 분포 및 스펙 범위 내에서 어디에 위치하는지 확인합니다.")

            for index, row in results_df.iterrows():
                y_col = row["Y 지표"]
                pred_val = float(row["예측값"])
                spec_status = row["스펙 판정"]
                spec = y_spec_dict[y_col]
                lo_spec, hi_spec = spec["최소"], spec["최대"]
                
                pred_line_color = "#28a745" if "합격" in spec_status else "#dc3545"

                st.markdown(f'<div class="chart-container">', unsafe_allow_html=True)
                st.markdown(f"#### {y_col} 예측값 분석 (결과: {spec_status})")
                
                fig_hist = px.histogram(df, x=y_col, nbins=100,
                                        title=f"<b>{y_col}</b>의 원본 데이터 분포 및 예측 위치",
                                        color_discrete_sequence=['#AAAAAA'])
                
                fig_hist.add_vline(x=lo_spec, line_dash="dash", line_color="red", 
                                   annotation_text="스펙 하한", annotation_position="top left")
                fig_hist.add_vline(x=hi_spec, line_dash="dash", line_color="red", 
                                   annotation_text="스펙 상한", annotation_position="top right")

                fig_hist.add_vline(x=pred_val, line_dash="solid", line_color=pred_line_color, line_width=3,
                                   annotation_text=f"예측값: {pred_val:.4f}", annotation_position="top")

                fig_hist.update_layout(height=400)
                st.plotly_chart(fig_hist, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="section-title">입력된 공정 데이터</div>', unsafe_allow_html=True)
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            input_summary = pd.DataFrame(list(st.session_state["pos_input_data"].items()),
                                         columns=["변수", "입력값"])
            st.dataframe(input_summary, use_container_width=True, height=300)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.download_button(
                "전체 예측 결과 CSV 다운로드",
                results_df.to_csv(index=False).encode("utf-8-sig"),
                "all_y_predictions.csv",
                "text/csv"
            )
        else:
            st.info("상단의 공정 변수를 입력하고 '전체 Y 지표 예측 실행' 버튼을 클릭하세요.")
    
    else:  # 전체 데이터 예측 모드
        y_default = _first_valid_target(y_choices, y_spec_dict)
        if y_default is None:
            st.warning("스펙과 일치하는 대상 Y가 없습니다.")
            st.stop()
        
        y_target = st.selectbox("분석 대상 지표", y_choices, index=y_choices.index(y_default))
        
        run_pred = st.button("예측 실행", type="primary")
        
        if run_pred:
            with st.spinner("AI 예측 수행 중..."):
                X_src = df[[c for c in df.columns if c.startswith("X_")]].apply(pd.to_numeric, errors="coerce").fillna(0)
                try:
                    pred_df = predict_multi_any(art_pack, X_src)
                    st.session_state["pos_pred_df"] = pred_df
                    st.success("예측 완료!")
                except Exception as e:
                    st.error(f"예측 실패: {e}")
                    st.stop()

        pred_df = st.session_state.get("pos_pred_df")
        if pred_df is None:
            st.info("상단 버튼을 눌러 예측을 수행하세요.")
            st.stop()
            
        if y_target not in pred_df.columns:
            st.error("예측 데이터가 없습니다.")
            st.stop()

        lo, hi = y_spec_dict[y_target]["최소"], y_spec_dict[y_target]["최대"]
        pred_s = _ensure_numeric_series(pred_df[y_target])
        in_spec = pred_s.between(lo, hi, inclusive="both")
        s_in = pred_s[in_spec]
        
        if s_in.empty:
            cats = pd.Series("스펙 밖", index=pred_s.index)
            q1 = q3 = np.nan
        else:
            q1, q3 = s_in.quantile(0.25), s_in.quantile(0.75)
            cats = pd.Series("중위 50%", index=pred_s.index)
            cats.loc[in_spec & (pred_s <= q1)] = "하위 25%"
            cats.loc[in_spec & (pred_s >= q3)] = "상위 25%"
            cats.loc[~in_spec] = "스펙 밖"

        view = pd.DataFrame({"예측값": pred_s, "포지셔닝": cats})
        total = len(view)
        in_spec_cnt = int((view["포지셔닝"] != "스펙 밖").sum())

        st.markdown('<div class="section-title">포지셔닝 요약</div>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'<div class="kpi-card"><div class="kpi-label">전체 샘플</div><div class="kpi-value">{total:,}</div></div>', unsafe_allow_html=True)
        with col2:
            spec_rate = (in_spec_cnt/total*100 if total else 0)
            st.markdown(f'<div class="kpi-card"><div class="kpi-label">스펙 내 비율</div><div class="kpi-value success">{spec_rate:.1f}%</div></div>', unsafe_allow_html=True)
        with col3:
            q1_text = f"{q1:.2f}" if not np.isnan(q1) else 'N/A'
            st.markdown(f'<div class="kpi-card"><div class="kpi-label">Q1 (25%)</div><div class="kpi-value">{q1_text}</div></div>', unsafe_allow_html=True)
        with col4:
            q3_text = f"{q3:.2f}" if not np.isnan(q3) else 'N/A'
            st.markdown(f'<div class="kpi-card"><div class="kpi-label">Q3 (75%)</div><div class="kpi-value">{q3_text}</div></div>', unsafe_allow_html=True)

        st.markdown('<div class="section-title">포지셔닝 분석 시각화</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="chart-container"><p><b>포지셔닝 구간 분포</b></p>', unsafe_allow_html=True)
            pie_df = view["포지셔닝"].value_counts().reset_index()
            pie_df.columns = ["구간", "건수"]
            colors = {"하위 25%": "#ffc107", "중위 50%": "#28a745", "상위 25%": "#007bff", "스펙 밖": "#dc3545"}
            fig = px.pie(pie_df, names="구간", values="건수", hole=0.4, color="구간", color_discrete_map=colors)
            fig.update_traces(textposition="inside", textinfo="percent+label")
            fig.update_layout(showlegend=True, height=400)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="chart-container"><p><b>예측값 분포</b></p>', unsafe_allow_html=True)
            fig = px.histogram(view, x="예측값", nbins=50, color_discrete_sequence=['#FA8072'])
            fig.add_vline(x=lo, line_dash="dash", line_color="red")
            fig.add_vline(x=hi, line_dash="dash", line_color="red")
            if not np.isnan(q1):
                fig.add_vline(x=q1, line_dash="dot", line_color="orange")
            if not np.isnan(q3):
                fig.add_vline(x=q3, line_dash="dot", line_color="blue")
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-title">상세 결과</div>', unsafe_allow_html=True)
        st.dataframe(view.head(100), use_container_width=True)
        st.download_button("CSV 다운로드", view.to_csv(index=False).encode("utf-8-sig"),
                           f"positioning_{y_target}.csv", "text/csv")
# =========================
# 통계 기반 순서 최적화
# =========================
elif page == "공정 최적화":
    mode = st.radio(
        "분석 모드 선택",
        ("통계 기반 최적화", "AI 기반 최적화"),
        horizontal=True
    )
    if mode == "통계 기반 최적화":
        st.markdown('<p class="main-header">통계 기반 검사 순서 최적화</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">불합격률과 비용/시간 데이터로 최적 검사 순서를 결정합니다.</p>', unsafe_allow_html=True)
        
        total_samples = len(df)
        if total_samples == 0 or cost_df.empty:
            st.error("데이터 또는 비용 정보가 없습니다.")
            st.stop()
        
        cost_map = cost_df.set_index('Feature')['Cost'].to_dict()
        time_map = cost_df.set_index('Feature')['검사 시간 (초)'].to_dict()
        
        st.markdown('<p class="section-header">1단계: 기준 선택</p>', unsafe_allow_html=True)
        optimization_criterion = st.selectbox(
            "최적화 기준",
            ['비용 효율성 최우선 (비용당 불합격률 기준)', 
            '품질 위험 최우선 (불합격률 기준)', 
            '검사 시간 최우선 (시간당 불합격률 기준)'],
            index=0
        )
        
        if st.button("시뮬레이션 실행", type='primary'):
            sort_by_key = '비용당 불합격률' if '비용' in optimization_criterion else ('시간당 불합격률' if '시간' in optimization_criterion else '불합격률')
            
            with st.spinner("시뮬레이션 중..."):
                y_in_spec = [y for y in y_cols if y in y_spec_dict]
                recs = []
                for y in y_in_spec:
                    spec = y_spec_dict[y]
                    y_s = _ensure_numeric_series(df[y])
                    fail = ((y_s < spec['최소']) | (y_s > spec['최대'])).sum()
                    rate = fail / total_samples if total_samples else 0
                    c, t = cost_map.get(y, 0), time_map.get(y, 0)
                    c_per = (rate / c) if c > 0 else (np.inf if rate > 0 else 0)
                    t_per = (rate / t) if t > 0 else (np.inf if rate > 0 else 0)
                    recs.append({'Feature': y, '불합격 횟수': fail, '검사 비용 (단가)': c, '검사 시간 (초)': t,
                                '비용당 불합격률': c_per, '불합격률': rate, '시간당 불합격률': t_per})
                
                analysis_df = pd.DataFrame(recs).sort_values(sort_by_key, ascending=False).reset_index(drop=True)
                
                sim_df = df.sample(min(5000, len(df)), random_state=42)
                base_order = sorted(y_in_spec)
                base_cost = base_time = 0
                for _, row in sim_df.iterrows():
                    for y in base_order:
                        base_cost += cost_map.get(y, 0)
                        base_time += time_map.get(y, 0)
                        spec = y_spec_dict[y]
                        val = row.get(y, np.nan)
                        if pd.notna(val) and ((val < spec['최소']) or (val > spec['최대'])):
                            break
                
                total_baseline_cost = (base_cost / len(sim_df)) * total_samples
                total_baseline_time = (base_time / len(sim_df)) * total_samples
                
                cumulative_failures = 0
                current_optimized_cost = 0
                current_optimized_time = 0
                analysis_df['누적 불합격 횟수'] = 0
                analysis_df['생략 가능 횟수'] = 0
                analysis_df['절감 비용'] = 0
                analysis_df['절감 시간 (초)'] = 0
                
                for i in range(len(analysis_df)):
                    current_y = analysis_df.loc[i, 'Feature']
                    cost = cost_map.get(current_y, 0)
                    time_sec = time_map.get(current_y, 0)
                    fail_count = analysis_df.loc[i, '불합격 횟수']
                    checks_before = max(0, total_samples - cumulative_failures)
                    current_optimized_cost += checks_before * cost
                    current_optimized_time += checks_before * time_sec
                    analysis_df.loc[i, '생략 가능 횟수'] = cumulative_failures
                    analysis_df.loc[i, '절감 비용'] = cumulative_failures * cost
                    analysis_df.loc[i, '절감 시간 (초)'] = cumulative_failures * time_sec
                    cumulative_failures += fail_count
                    analysis_df.loc[i, '누적 불합격 횟수'] = cumulative_failures
                
                total_savings = total_baseline_cost - current_optimized_cost
                total_time_savings = total_baseline_time - current_optimized_time
                total_failures_sum = analysis_df['불합격 횟수'].sum()
                if total_failures_sum > 0:
                    analysis_df['누적 불합격 발견율 (%)'] = (analysis_df['누적 불합격 횟수'] / total_failures_sum) * 100
                else:
                    analysis_df['누적 불합격 발견율 (%)'] = 0
                
                kpi_result = (current_optimized_cost, total_savings, current_optimized_time, 
                            total_time_savings, analysis_df, total_baseline_cost, total_baseline_time)
                st.session_state['stat_kpi'][optimization_criterion] = kpi_result
                st.session_state['current_stat_kpi_criterion'] = optimization_criterion
                st.success("완료!")
        
        current_kpi_result = st.session_state['stat_kpi'].get(st.session_state['current_stat_kpi_criterion'])
        
        if not current_kpi_result:
            st.info("시뮬레이션 버튼을 클릭하세요.")
            st.stop()
        
        total_opt_cost, total_savings, total_opt_time, total_time_savings, analysis_df, total_baseline_cost, total_baseline_time = current_kpi_result
        savings_rate = (total_savings / total_baseline_cost * 100) if total_baseline_cost > 0 else 0
        time_savings_min = total_time_savings / 60
        time_savings_rate = (total_time_savings / total_baseline_time * 100) if total_baseline_time > 0 else 0
        
        st.markdown('<p class="section-header">2단계: 절감 성과</p>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'<div class="kpi-box bg-green"><div class="kpi-title">비용 절감액</div><div class="kpi-value">\\{total_savings:,.0f}</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="kpi-box bg-blue"><div class="kpi-title">비용 절감률</div><div class="kpi-value">{savings_rate:.1f}%</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="kpi-box bg-green"><div class="kpi-title">시간 절감</div><div class="kpi-value">{time_savings_min:,.1f}분</div></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="kpi-box bg-blue"><div class="kpi-title">시간 절감률</div><div class="kpi-value">{time_savings_rate:.1f}%</div></div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        optimal_order_df = analysis_df.copy()
        optimal_order_df.index = optimal_order_df.index + 1
        optimal_order_df = optimal_order_df.rename_axis('순서')
        optimal_order_df['표시 이름'] = optimal_order_df.index.astype(str) + ". " + optimal_order_df['Feature']
        
        st.markdown('<p class="section-header">1. 절감 비용 및 시간 기여도</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="chart-container"><p><b>절감 비용 기여도</b></p>', unsafe_allow_html=True)
            plot_df_cost = optimal_order_df[optimal_order_df['절감 비용'] > 0].copy().reset_index()
            fig_cost = px.bar(plot_df_cost, x='절감 비용', y='표시 이름', orientation='h', color='순서',
                        color_continuous_scale=[[0, '#FFE4E1'], [1, '#FA8072']])
            fig_cost.update_layout(yaxis={'categoryorder':'total ascending'}, height=500, yaxis_title=None)
            st.plotly_chart(fig_cost, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="chart-container"><p><b>절감 시간 기여도 (초)</b></p>', unsafe_allow_html=True)
            plot_df_time = optimal_order_df[optimal_order_df['절감 시간 (초)'] > 0].copy().reset_index()
            fig_time = px.bar(plot_df_time, x='절감 시간 (초)', y='표시 이름', orientation='h', color='순서',
                        color_continuous_scale=[[0, '#E3F2FD'], [1, '#2196F3']])
            fig_time.update_layout(yaxis={'categoryorder':'total ascending'}, height=500, yaxis_title=None)
            st.plotly_chart(fig_time, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<p class="section-header">2. 누적 불합격 발견율</p>', unsafe_allow_html=True)
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        if optimal_order_df['불합격 횟수'].sum() > 0:
            fig = px.line(optimal_order_df.reset_index(), x='순서', y='누적 불합격 발견율 (%)', 
                        markers=True, color_discrete_sequence=['#FA8072'])
            fig.update_layout(yaxis=dict(range=[0, 105]), height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("불합격 건수가 없습니다.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<p class="section-header">3. 누적 비용/시간 변화</p>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        cum_df = optimal_order_df.copy()
        cum_df["이전_누적불합격"] = cum_df["불합격 횟수"].cumsum().shift(fill_value=0)
        cum_df["점검건수"] = (total_samples - cum_df["이전_누적불합격"]).clip(lower=0)
        cum_df["증가비용"] = cum_df.apply(lambda r: cost_map.get(r["Feature"], 0) * r["점검건수"], axis=1)
        cum_df["증가시간"] = cum_df.apply(lambda r: time_map.get(r["Feature"], 0) * r["점검건수"], axis=1)
        cum_df["누적비용"] = cum_df["증가비용"].cumsum()
        cum_df["누적시간"] = cum_df["증가시간"].cumsum()
        cum_df["Step"] = np.arange(1, len(cum_df) + 1)
        
        with col1:
            st.markdown('<div class="chart-container"><p><b>누적 비용</b></p>', unsafe_allow_html=True)
            fig = px.line(cum_df, x="Step", y="누적비용", markers=True, color_discrete_sequence=['#FA8072'])
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="chart-container"><p><b>누적 시간</b></p>', unsafe_allow_html=True)
            fig = px.line(cum_df, x="Step", y="누적시간", markers=True, color_discrete_sequence=['#FF6B6B'])
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<p class="section-header">4단계: 최적 검사 순서</p>', unsafe_allow_html=True)
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        display_sort_col = '비용당 불합격률' if '비용' in st.session_state['current_stat_kpi_criterion'] else ('시간당 불합격률' if '시간' in st.session_state['current_stat_kpi_criterion'] else '불합격률')
        detail_cols = ['Feature', '불합격 횟수', '검사 비용 (단가)', '검사 시간 (초)', display_sort_col, '생략 가능 횟수', '절감 비용', '절감 시간 (초)']
        st.dataframe(optimal_order_df[detail_cols], use_container_width=True, height=500)
        st.markdown('</div>', unsafe_allow_html=True)

    elif mode == "AI 기반 최적화":
        st.markdown('<p class="main-header">AI 예측 기반 검사 순서 최적화</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">AI가 개별 제품의 불량 확률을 실시간으로 예측하여 최적 검사 순서를 결정합니다.</p>', unsafe_allow_html=True)
        
        ai_models = st.session_state.get('ai_models', {})
        
        st.markdown('<p class="section-header">1단계: AI 예측 모델 학습</p>', unsafe_allow_html=True)
        if not ai_models:
            if st.button("AI 모델 학습 시작", type="primary"):
                with st.spinner("학습 중..."):
                    ai_models = train_or_load_simple_models(df, y_cols, y_spec_dict)
                    st.session_state['ai_models'] = ai_models
                st.success(f"학습 완료! ({len(ai_models)}개 모델)")
                st.rerun()
            else:
                st.info("버튼을 클릭하여 AI 모델을 학습하세요.")
                st.stop()
        else:
            st.success(f"AI 모델 준비 완료 ({len(ai_models)}개)")
        
        st.markdown('<p class="section-header">2단계: 시뮬레이션</p>', unsafe_allow_html=True)
        total_samples = len(df)
        st.info(f"샘플 {min(5000, total_samples):,}개 기반 시뮬레이션 (전체 {total_samples:,}개로 환산)")
        
        if st.button("시뮬레이션 실행", type="primary"):
            cost_map = cost_df.set_index('Feature')['Cost'].to_dict()
            time_map = cost_df.set_index('Feature')['검사 시간 (초)'].to_dict()
            y_in_spec = [y for y in y_cols if y in y_spec_dict]
            
            with st.spinner("시뮬레이션 중..."):
                sim_df = df.sample(min(5000, len(df)), random_state=42)
                
                # 기준선
                base_order = sorted(y_in_spec)
                base_cost = base_time = 0
                for _, row in sim_df.iterrows():
                    for y in base_order:
                        base_cost += cost_map.get(y, 0)
                        base_time += time_map.get(y, 0)
                        spec = y_spec_dict[y]
                        val = row.get(y, np.nan)
                        if pd.notna(val) and ((val < spec['최소']) or (val > spec['최대'])):
                            break
                
                total_baseline_cost = (base_cost / len(sim_df)) * total_samples
                total_baseline_time = (base_time / len(sim_df)) * total_samples
                
                # AI 순서
                cost_sum = time_sum = 0
                for _, row in sim_df.iterrows():
                    preds = {}
                    for y in y_in_spec:
                        if y not in ai_models:
                            continue
                        model, scaler, Xc = ai_models[y]
                        use_cols = [c for c in Xc if c in row.index]
                        if not use_cols: continue
                        Xs = row[use_cols].fillna(0).values.reshape(1, -1)
                        try:
                            pr = model.predict_proba(scaler.transform(Xs))[0, 1]
                        except:
                            pr = 0.0
                        preds[y] = pr
                    ai_order = sorted(preds, key=preds.get, reverse=True)
                    for y in ai_order:
                        cost_sum += cost_map.get(y, 0)
                        time_sum += time_map.get(y, 0)
                        spec = y_spec_dict[y]
                        val = row.get(y, np.nan)
                        if pd.notna(val) and ((val < spec['최소']) or (val > spec['최대'])):
                            break

                cost_scaled = (cost_sum / len(sim_df)) * total_samples
                time_scaled = (time_sum / len(sim_df)) * total_samples
                tot_cost_save = total_baseline_cost - cost_scaled
                tot_time_save = total_baseline_time - time_scaled

                st.session_state['ai_kpi'] = (cost_scaled, tot_cost_save, time_scaled, tot_time_save, 
                                            pd.DataFrame(), total_baseline_cost, total_baseline_time)
                st.success("시뮬레이션 완료!")

        ai_kpi = st.session_state.get('ai_kpi')
        if ai_kpi:
            total_opt_cost, total_cost_savings, total_opt_time, total_time_savings, _, total_baseline_cost, total_baseline_time = ai_kpi
            
            ai_savings_rate = (total_cost_savings / total_baseline_cost * 100) if total_baseline_cost > 0 else 0
            ai_time_savings_min = total_time_savings / 60
            ai_time_savings_rate = (total_time_savings / total_baseline_time * 100) if total_baseline_time > 0 else 0
            
            st.markdown("---")
            st.markdown('<p class="section-header">AI 최적화 성과</p>', unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f'<div class="kpi-box bg-cost"><div class="kpi-title">비용 절감액</div><div class="kpi-value">\\{total_cost_savings:,.0f}</div></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="kpi-box bg-cost"><div class="kpi-title">비용 절감률</div><div class="kpi-value">{ai_savings_rate:.1f}%</div></div>', unsafe_allow_html=True)
            with col3:
                st.markdown(f'<div class="kpi-box bg-time"><div class="kpi-title">시간 절감</div><div class="kpi-value">{ai_time_savings_min:,.1f}분</div></div>', unsafe_allow_html=True)
            with col4:
                st.markdown(f'<div class="kpi-box bg-time"><div class="kpi-title">시간 절감률</div><div class="kpi-value">{ai_time_savings_rate:.1f}%</div></div>', unsafe_allow_html=True)
            
            st.markdown('<p class="section-header">비교</p>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="chart-container"><p><b>비용 비교</b></p>', unsafe_allow_html=True)
                compare_df = pd.DataFrame({"방식": ["기준선", "AI"], "비용": [total_baseline_cost, total_opt_cost]})
                fig = px.bar(compare_df, x="방식", y="비용", color="방식",
                            color_discrete_map={"기준선": "#dc3545", "AI": "#28a745"})
                fig.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="chart-container"><p><b>시간 비교</b></p>', unsafe_allow_html=True)
                time_df = pd.DataFrame({"방식": ["기준선", "AI"], "시간(분)": [total_baseline_time/60, total_opt_time/60]})
                fig = px.bar(time_df, x="방식", y="시간(분)", color="방식",
                            color_discrete_map={"기준선": "#dc3545", "AI": "#28a745"})
                fig.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("시뮬레이션 버튼을 클릭하세요.")

# =========================
# 자동 보고서 생성
# =========================
elif page == "자동 보고서 생성":
    st.markdown('<p class="main-header">자동 보고서 생성</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">분석 결과를 바탕으로 AI가 자동으로 보고서를 작성합니다.</p>', unsafe_allow_html=True)

    st.info("보고서를 생성하려면 먼저 '통계 기반 순서 최적화' 또는 'AI 기반 순서 최적화' 페이지에서 시뮬레이션을 한 번 이상 실행해야 합니다.")

    report_col1, report_col2, report_col3 = st.columns(3)

    process_knowledge = """
    - SMT (납땜, X_50~X_56): 납의 양이 너무 많으면 기생 정전용량/인덕턴스가 증가하여 임피던스 불일치를 유발하고, 너무 적으면 접촉 불량으로 SNR/Gain이 저하됩니다.
    - PCB 프레스 (누름량, X_01~X_06): 누르는 힘이 과도하면 PCB나 패드가 변형되어 Gain/SNR 불균형을 초래하고, 부족하면 접지 불량으로 노이즈가 유입됩니다.
    - 방열재(TIM) 적용 (X_03, X_07 등): 방열재가 과도하면 유전율 특성 변화로 주파수가 틀어질 수 있고, 부족하면 소자 온도 상승으로 노이즈가 증가하고 SNR이 저하됩니다.
    - 스크류 체결 (X_19~X_37): 체결 토크가 너무 강하면 하우징 변형을, 너무 약하면 부품 간 유격 및 성능 저하를 야기할 수 있습니다.
    """
    with report_col1:
        if st.button("포지셔닝 보고서 작성", use_container_width=True):
            positioning_df_manual = st.session_state.get("pos_pred_df_manual")
            positioning_df_default = st.session_state.get("pos_pred_df")

            # 우선 manual이 있으면 사용, 없으면 default
            if positioning_df_manual is not None and not positioning_df_manual.empty:
                positioning_df = positioning_df_manual
            elif positioning_df_default is not None and not positioning_df_default.empty:
                positioning_df = positioning_df_default
            else:
                positioning_df = None

            if positioning_df is None:
                st.warning("먼저 '포지셔닝 전략' 페이지에서 시뮬레이션을 실행해주세요.")


            else:
                with st.spinner("AI가 포지셔닝 자동 보고서를 작성 중입니다..."):
                    try:
                        # 포지셔닝 요약
                        y_cols = [col for col in positioning_df.columns if col.startswith("Y_")]
                        total_samples = len(positioning_df)
                        in_spec_count = sum(
                            positioning_df[col].between(y_spec_dict[col]["최소"], y_spec_dict[col]["최대"])
                            for col in y_cols
                        )
                        fail_count = len(y_cols) * total_samples - in_spec_count

                        # summary, feature info (기존에 있으면 사용)
                        summary_md = st.session_state.get("positioning_summary", pd.DataFrame()).to_markdown(index=False) \
                            if "positioning_summary" in st.session_state else "요약 데이터 없음"
                        feature_info_md = y_feature_info[['Feature','설명']].to_markdown(index=False) if 'Feature' in y_feature_info.columns else "-"

                        # 프롬프트
                        prompt = f"""
                        당신은 제조 공정 분석 및 품질 최적화 전문 데이터 분석가입니다.
                        다음은 AI 기반 포지셔닝 분석 결과 데이터입니다.

                        [분석 개요]
                        - 전체 샘플 수: {total_samples:,}개
                        - Y 지표 수: {len(y_cols)}
                        - 스펙 내 지표 수: {in_spec_count}
                        - 불량 지표 수: {fail_count}
                        - 주요 성능 지표 요약: {summary_md}
                        - Y 지표 설명: {feature_info_md}

                        ---
                        위의 내용을 바탕으로, 포지셔닝 결과를 분석한 자동 보고서를 작성해 주세요. 
                        작성 시 다음 사항을 고려:

                        1. **프로젝트 개요**: 포지셔닝의 목적(샘플의 특성에 따른 성능 위치/균형 분석)을 설명.
                        2. **분석 방법**: 포지셔닝이 어떤 기준으로 샘플을 분류하고, 불량 위치를 어떻게 시각화했는지 설명.
                        3. **성과 요약**: 정상/불량 샘플 비율, 주요 Y 성능 지표의 특징, 불균형 원인 요약.
                        4. **의의 및 활용 방안**: 포지셔닝 분석을 통해 얻을 수 있는 통찰(예: 공정 변수별 불량 클러스터 탐지, 예측 모델 개선 방향)을 설명, 품질 개선 및 사전 예측 시스템 구축에 어떻게 활용할 수 있는지 제안.

                        - 글자수는 최소 5000자 이상, 보고서 대상은 같은 직무를 하는 품질검사관 대상.
                        - 사실만 작성, 거짓 내용 절대 포함 금지.
                        - 인터넷 검색과 참조 허용, 그러나 참조시 이름, 링크 남기기.
                        """

                        client = openai.OpenAI(api_key=openai.api_key)
                        response = client.chat.completions.create(
                            model="gpt-4o",
                            messages=[{"role": "user", "content": prompt}]
                        )
                        st.session_state['generated_report'] = response.choices[0].message.content
                        st.success("포지셔닝 보고서 생성 완료!")
                    except Exception as e:
                        st.error(f"포지셔닝 보고서 생성 중 오류 발생: {e}")
                        st.stop()
    
    with report_col2:
        if st.button("통계 최적화 결과 보고서 생성", type="primary", use_container_width=True):
            criterion = st.session_state.get('current_stat_kpi_criterion')
            if not criterion or criterion not in st.session_state.get('stat_kpi', {}):
                st.warning("먼저 '통계 기반 순서 최적화' 페이지에서 시뮬레이션을 실행해주세요.")

            
            with st.spinner("AI가 통계 기반 보고서를 작성 중입니다..."):
                try:
                    kpi_data = st.session_state['stat_kpi'][criterion]
                    total_opt_cost, total_savings, total_opt_time, total_time_savings, analysis_df, total_baseline_cost, total_baseline_time = kpi_data
                    savings_rate = (total_savings / total_baseline_cost * 100) if total_baseline_cost > 0 else 0
                    time_savings_min = total_time_savings / 60
                    time_savings_rate = (total_time_savings / total_baseline_time * 100) if total_baseline_time > 0 else 0
                    
                    merged_df = pd.merge(analysis_df, y_feature_info, on="Feature", how="left")
                    merged_df['설명'] = merged_df['설명'].fillna('-')
                    
                    cost_info_md = merged_df[['Feature', '설명', '불합격률', '검사 비용 (단가)', '검사 시간 (초)']].to_markdown(index=False)
                    sort_col = '비용당 불합격률' if '비용' in criterion else ('시간당 불합격률' if '시간' in criterion else '불합격률')
                    order_df = merged_df[['Feature', '설명', sort_col]].copy()
                    order_df.index = order_df.index + 1
                    inspection_order_md = order_df.to_markdown()


                    prompt = f"""
                    당신은 제조 공정 분석 및 품질 최적화 전문 데이터 분석가입니다.
                    다음은 통계 기반 검사 순서 최적화 대시보드의 결과입니다.
                    [공정 배경 지식]
                    {process_knowledge}
                    [분석 개요]
                    - 공정 목적: 불합격률, 검사 비용, 검사 시간을 종합하여 검사 순서를 최적화
                    - 선택된 최적화 기준: {criterion}
                    - 총 샘플 수: {len(df):,}개
                    [성과 요약]
                    - 기준선 대비 최적화 후 절감 비용: ₩{total_savings:,.0f}
                    - 비용 절감률: {savings_rate:.1f}%
                    - 검사 시간 절감: {time_savings_min:,.1f}분
                    - 시간 절감률: {time_savings_rate:.1f}%
                    [출력 데이터]
                    - 총 기준 검사 비용: ₩{total_baseline_cost:,.0f}
                    - 최적화 후 검사 비용: ₩{total_opt_cost:,.0f}
                    - 총 기준 검사 시간(분): {total_baseline_time/60:,.1f}
                    - 최적화 후 검사 시간(분): {total_opt_time/60:,.1f}
                    [추가 정보]
                    - 각 Y 성능 지표별 상세 정보:
                    {cost_info_md}
                    - 통계 기반 최적 검사 순서 결과:
                    {inspection_order_md}
                    ---
                    위의 내용을 바탕으로, 보고서 형식으로 작성해 주세요. 작성 시 다음 사항을 고려:
                    1. **프로젝트 개요**: 검사 순서 최적화의 목적과 원리를 설명, 보고서의 목적과 조건(예: 비용/시간 감소를 통한 회사 이익 기여)을 반영
                    2. **성능 검사 지표**: y_1~y_14까지의 이름과 설명
                    3. **분석 방법**: 사용자가 설정한 기준에 따라 검사 순서를 어떻게 최적화했는지 설명, 이해하기 쉽게 작성
                    4. **성과 요약**: 최적화 전과 후의 검사 비용, 시간, 절감률을 표로 요약.
                    5. **의의 및 활용 방안**: 공정 배경 지식을 참고하여, 왜 특정 검사를 먼저 하는 것이 효율적인지에 대한 통찰을 포함하여 제조 품질 관리에서 최적화의 실효성과 기대효과 설명. 단기/장기 적용 가능성을 포함하여 구체적으로 제언.
                    - 글자수는 최소 5000자 이상, 보고서 대상은 같은 직무를 하는 품질검사관 대상.
                    - 사실만 작성, 거짓 내용 절대 포함 금지.
                    - 인터넷 검색과 참조 허용, 그러나 참조시 이름, 링크 남기기.
                    """
                    client = openai.OpenAI(api_key=openai.api_key)
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "user", "content": prompt}]
                    )
                    st.session_state['generated_report'] = response.choices[0].message.content
                    st.success("보고서 생성이 완료되었습니다!")
                except Exception as e:
                    st.error(f"보고서 생성 중 오류 발생: {e}")
                    st.stop()
    
    with report_col3:
        if st.button("AI 최적화 결과 보고서 생성", type="primary", use_container_width=True):
            if 'ai_kpi' not in st.session_state:
                st.warning("먼저 'AI 기반 순서 최적화' 페이지에서 시뮬레이션을 실행해주세요.")
                st.stop()
            
            with st.spinner("AI가 AI 기반 보고서를 작성 중입니다..."):
                try:
                    ai_models = st.session_state.get('ai_models', {})
                    total_opt_cost, total_cost_savings, total_opt_time, total_time_savings, _, total_baseline_cost, total_baseline_time = st.session_state['ai_kpi']
                    ai_savings_rate = (total_cost_savings / total_baseline_cost * 100) if total_baseline_cost > 0 else 0
                    ai_time_savings_min = total_time_savings / 60
                    ai_time_savings_rate = (total_time_savings / total_baseline_time * 100) if total_baseline_time > 0 else 0
                    
                    cost_info_df = pd.merge(cost_df, y_feature_info, on="Feature", how="left")
                    cost_info_df['설명'] = cost_info_df['설명'].fillna('-')
                    cost_info_df = cost_info_df[['Feature', '설명', 'Cost', '검사 시간 (초)']].rename(columns={'Cost': '검사 비용 (단가)'})
                    relevant_features = list(ai_models.keys())
                    cost_info_df = cost_info_df[cost_info_df['Feature'].isin(relevant_features)]
                    cost_info_md = cost_info_df.to_markdown(index=False)
                    inspection_order_md = "AI 최적화는 각 샘플의 공정 데이터(X값)에 따라 동적으로 불량 확률을 예측하여 실시간으로 최적의 검사 순서를 결정합니다. 따라서 고정된 단일 순서는 존재하지 않습니다."

                    prompt = f"""
                    당신은 제조 공정 분석 및 품질 최적화 전문 데이터 분석가입니다.
                    다음은 AI 기반 검사 순서 최적화 대시보드의 실행 결과 데이터입니다.
                    [공정 배경 지식]
                    {process_knowledge}
                    [분석 개요]
                    - 공정 목적: AI 예측 모델을 이용해 각 Y 성능 지표별 불량 확률을 예측하고, 이를 바탕으로 검사 순서를 최적화하여 비용과 시간을 절감함
                    - 사용된 AI 모델 수: {len(ai_models)}개
                    - 기준 비교 방식: 고정된 기본 순서 vs. AI 동적 최적 순서
                    - 시뮬레이션 샘플 수: {len(df):,}개
                    [성과 요약]
                    - 전체 검사 기준 대비 AI 최적화 후 비용 절감액: ₩{total_cost_savings:,.0f}
                    - 비용 절감률: {ai_savings_rate:.1f}%
                    - 전체 시간 절감: {ai_time_savings_min:,.1f}분
                    - 시간 절감률: {ai_time_savings_rate:.1f}%
                    [출력 데이터]
                    - 총 기준 검사 비용: ₩{total_baseline_cost:,.0f}
                    - 최적화 후 검사 비용: ₩{total_opt_cost:,.0f}
                    - 총 기준 검사 시간(분): {total_baseline_time/60:,.1f}
                    - 최적화 후 검사 시간(분): {total_opt_time/60:,.1f}
                    [추가 정보]
                    - 각 성능 지표별 검사 단가 및 검사 시간 정보:
                    {cost_info_md}
                    - AI 예측 기반 순서 결정 방식: {inspection_order_md}
                    ---
                    위의 내용을 바탕으로, 보고서 형식으로 작성해 주세요. 작성 시 다음 사항을 고려:
                    1. **프로젝트 개요**: 검사 순서 최적화의 목적과 원리를 설명, 보고서의 목적과 조건(예: 비용/시간 감소를 통한 회사 이익 기여)을 반영
                    2. **성능 검사 지표**: y_1~y_14까지의 이름과 설명
                    3. **분석 방법**: 사용자가 설정한 기준에 따라 검사 순서를 어떻게 최적화했는지 설명, 이해하기 쉽게 작성
                    4. **성과 요약**: 최적화 전과 후의 검사 비용, 시간, 절감률을 표로 요약.
                    5. **의의 및 활용 방안**: 공정 배경 지식을 참고하여, 왜 특정 검사를 먼저 하는 것이 효율적인지에 대한 통찰을 포함하여 제조 품질 관리에서 최적화의 실효성과 기대효과 설명. 단기/장기 적용 가능성을 포함하여 구체적으로 제언.
                    - 글자수는 최소 5000자 이상, 보고서 대상은 같은 직무를 하는 품질검사관 대상.
                    - 사실만 작성, 거짓 내용 절대 포함 금지.
                    - 인터넷 검색과 참조 허용, 그러나 참조시 이름, 링크 남기기.
                    """

                    client = openai.OpenAI(api_key=openai.api_key)
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "user", "content": prompt}]
                    )
                    st.session_state['generated_report'] = response.choices[0].message.content
                    st.success("보고서 생성이 완료되었습니다!")
                except Exception as e:
                    st.error(f"보고서 생성 중 오류 발생: {e}")
                    st.stop()

    if 'generated_report' in st.session_state:
        st.markdown('<div class="section-title">생성된 보고서</div>', unsafe_allow_html=True)
        report_text = st.session_state['generated_report']
        
        st.markdown(report_text)
        
        st.download_button(
            label="보고서 다운로드 (Markdown)",
            data=report_text.encode('utf-8-sig'),
            file_name="optimization_report.md",
            mime="text/markdown"
        )


# 푸터
st.markdown("---")
st.markdown('<p style="text-align:center;color:#999;font-size:0.9rem;">FMCW 품질관리 대시보드 | Powered by AI</p>', unsafe_allow_html=True)


