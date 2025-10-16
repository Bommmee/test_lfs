import pandas as pd
import joblib
import cloudpickle
from pathlib import Path

# --- 사용자 설정 ---
# 기존 prediction_pipeline.pkl 파일 경로를 확인해주세요.
PRED_PIPE_PATH = str(
    Path(r"C:\Users\Taeyoung\4차 프로젝트 정리\prediction_pipeline.pkl")
)
# 새로 저장할 부품 파일 이름
ARTIFACTS_PATH = "model_artifacts.joblib"
# pkl 생성 시 사용했던 원본 데이터 경로 (컬럼 순서 재구성을 위해 필요)
ORIGINAL_TRAIN_PATH = str(Path(r"C:\Users\Taeyoung\4차 프로젝트 정리\merged_train.csv"))
# --------------------

print(f"기존 파이프라인 파일 로드 중: {PRED_PIPE_PATH}")
try:
    with open(PRED_PIPE_PATH, "rb") as f:
        artifacts = cloudpickle.load(f)
except FileNotFoundError:
    print(f"오류: {PRED_PIPE_PATH} 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
    exit()
except Exception as e:
    print(f"파일 로드 중 오류 발생: {e}")
    exit()

if not isinstance(artifacts, dict):
    print("오류: .pkl 파일이 예상한 딕셔너리 형태가 아닙니다.")
    exit()

print("파이프라인 로드 성공!")

# === Imputer가 학습한 실제 컬럼 순서 저장 (핵심 수정) ===
imputer_feature_names = []
try:
    # 최신 scikit-learn에서는 이 속성으로 정확한 순서를 가져올 수 있음
    imputer_feature_names = artifacts["imputer"].feature_names_in_.tolist()
    print("Imputer의 'feature_names_in_' 속성에서 컬럼 순서 정보를 추출했습니다.")
except AttributeError:
    print(
        "경고: 'feature_names_in_'을 찾을 수 없어, 원본 데이터 기준으로 컬럼 순서를 재구성합니다."
    )
    # 구버전 호환용: pkl 생성 스크립트와 동일한 로직으로 컬럼 순서를 재구성
    try:
        original_train_data = pd.read_csv(ORIGINAL_TRAIN_PATH)
        x_full_cols = original_train_data.filter(regex="^X_").columns
        drop_cols = artifacts.get("pre_drop_cols", [])
        imputer_feature_names = [col for col in x_full_cols if col not in drop_cols]
        print(
            f"원본 데이터에서 {len(imputer_feature_names)}개의 컬럼 순서를 성공적으로 재구성했습니다."
        )
    except Exception as e:
        print(
            f"오류: 원본 학습 데이터를 로드하여 컬럼을 재구성하는 데 실패했습니다: {e}"
        )
        # 이 경우에도 오류가 나면 수동으로 컬럼 순서를 맞춰야 할 수 있습니다.
        imputer_feature_names = []

if not imputer_feature_names:
    print(
        "치명적 오류: Imputer의 컬럼 순서를 결정할 수 없습니다. 스크립트를 중단합니다."
    )
    exit()


# 저장할 부품들을 담을 딕셔너리
export_artifacts = {
    "pre_drop_cols": artifacts.get("pre_drop_cols", []),
    "imputer_feature_names": imputer_feature_names,  # Imputer의 컬럼 순서 정보 추가
    "target_columns": artifacts["target_columns"],
    "final_feature_names": artifacts["final_feature_names"],
    "imputer_stats": artifacts["imputer"].statistics_,
    "pca_info": [],
    "models": artifacts["models"],
}

print("PCA 정보 추출 중...")
for g in artifacts["pca_info"]:
    export_artifacts["pca_info"].append(
        {
            "group_cols": g["group_cols"],
            "component_cols": g["component_cols"],
            "n_components": g["pca"].n_components_,
            "mean": g["pca"].mean_,
            "components": g["pca"].components_,
        }
    )

print("개별 모델은 StackingRegressor 객체 그대로 저장합니다.")

print(f"추출 완료! 부품들을 '{ARTIFACTS_PATH}' 파일로 저장합니다...")
joblib.dump(export_artifacts, ARTIFACTS_PATH)

print("\n" + "=" * 50)
print("저장 완료!")
print("이제 'dashboard.py'와 함께 다음 파일들을 다른 컴퓨터로 옮기세요:")
print(f"  - {ARTIFACTS_PATH}")
print("  - 각종 csv 데이터 파일들")
print("=" * 50)
