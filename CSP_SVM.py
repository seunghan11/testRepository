import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mne.decoding import CSP
from sklearn.metrics import classification_report

# 1. 데이터 생성 (가상의 SSVEP 데이터)
# 데이터 구조: (samples, channels, timepoints)
np.random.seed(42)
n_samples = 100  # 총 샘플 수
n_channels = 8   # 채널 수
n_timepoints = 500  # 각 샘플의 시간 포인트 수
n_classes = 6     # 분류 클래스 수

# 가상의 데이터와 레이블 생성
X = np.random.randn(n_samples, n_channels, n_timepoints)  # 랜덤 SSVEP 신호
y = np.random.randint(0, n_classes, size=n_samples)  # 0 ~ 5 레이블

# 2. 데이터 분할 (학습 및 테스트 세트)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. CSP 및 SVM 파이프라인 설정
csp = CSP(n_components=4, reg=None, log=True, cov_est='concat')  # CSP 설정
svm = SVC(kernel='linear', C=1, probability=True, random_state=42)  # SVM 설정

# 파이프라인 구성
pipeline = Pipeline([
    ('csp', csp),
    ('svm', svm)
])

# 4. CSP 피팅 및 분류
pipeline.fit(X_train, y_train)

# 5. 테스트 데이터로 예측
y_pred = pipeline.predict(X_test)

# 6. 성능 평가
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 7. CSP 패턴 시각화 (선택적)
import numpy as np
import matplotlib.pyplot as plt

# CSP 패턴 수 확인
n_patterns = len(csp.patterns_)

# 서브플롯 행렬 계산 (최대 4개까지 표시)
n_rows = int(np.ceil(np.sqrt(n_patterns)))
n_cols = int(np.ceil(n_patterns / n_rows))

plt.figure(figsize=(10, 6))
for i, pattern in enumerate(csp.patterns_):
    plt.subplot(n_rows, n_cols, i + 1)  # 서브플롯 행렬 동적 생성
    plt.plot(pattern)
    plt.title(f'CSP Pattern {i + 1}')
plt.tight_layout()
plt.show()