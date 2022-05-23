###### 201795085 조병률 ###################################################
##########################################################################
import numpy as np
import pandas as pd
#
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
# 성능 확인
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
# roc curve
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

##### 데이터 ##############################################################
##########################################################################
### csv파일을 데이터프레임 객체로 생성
data = pd.read_csv(r'/Users/ryul/Desktop/workspace/big-data/pythonProject/diabetes_logistic_regression/diabetes.csv', index_col=None, encoding='CP949', engine='python')
# print(data.info())
# print(data)

### 'Outcome' 피처를 회귀식의 종속 변수 Y로 설정
Y = data['Outcome']
### 종속 변수를 제외한 모든 피처를 독립 변수 X로 설정 
X = data.drop(['Outcome'], axis=1) # inplace=False 기본값



##### 정규 분포 형태로 표준화 #################################################
##########################################################################
### 정규 분포 스케일러를 위한 객체 생성
scaler = StandardScaler()
diabetes_scaled = scaler.fit_transform(X)
# print(diabetes_scaled)

### 스케일링한 피처들을 재할당 
X = diabetes_scaled


##### 예측 ################################################################
##########################################################################
### X, Y 데이터를 학습 데이터와 평가 데이터로 7:3비율로 분할 (test_size=0.3) 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

### 로지스틱 회귀 분석 : 객체 생성
lr_diabetes = LogisticRegression()
### 로지스틱 회귀 분석 : 모델 훈련 
lr_diabetes.fit(X_train, Y_train)

### 테스트 데이터에서 5개 예측
print(lr_diabetes.predict(X_test[:5]))

### 예측에 대한 확률 predict_proba() : y= 1/1+e^-z
print(lr_diabetes.predict_proba(X_test[:5]))
##########################################################################


###### 선형 회귀 분석 : 평가 데이터에 대한 예측 수행 -> 예측 결과(Y_predict) ########
##########################################################################
Y_predict = lr_diabetes.predict(X_test)
# print(Y_predict)

print('Y 절편 값 : ', np.round(lr_diabetes.intercept_, 2))
print('회귀 계수 값 : ', np.round(lr_diabetes.coef_, 2))


##### 모델 성능 확인 ########################################################
##########################################################################                    
###  혼동(오차) 행렬 : TN(141), FP(16), FN(35), TP(39)
print (confusion_matrix(Y_test, Y_predict))
### 정확도 : (TN+FN)/(TN+FP+FN+TP)
### 정밀도 : TP/(FP+TP)
### 재현율 : TP/(FN+TP)
### F1   : 2*(정밀도 * 재현율)/(정밀도 + 재현율)

accuracy = accuracy_score(Y_test, Y_predict) # (141+39)/(141+16+35+39)=0.779
precision = precision_score(Y_test, Y_predict) # 39/(16+39)=0.709
recall = recall_score(Y_test, Y_predict) # 39/(35+39)=0.527
f1 = f1_score(Y_test, Y_predict) # 2*(0.709*0.527)/(0.709+0.527)=0.605
roc_auc = roc_auc_score(Y_test, Y_predict) 

print('정확도: {0:.3f}, 정밀도: {1:.3f}, 재현율: {2:.3f}. F1: {3:.3f}'.format(accuracy, precision, recall, f1))
print('ROC_AUC : {0:.3f}'.format(roc_auc))


##### ROC CURVE ##########################################################
##########################################################################
###  roc curve 패키지로 fpr, tpr 뽑기
model_fpr, model_tpr, threshold1 = roc_curve(Y_test, Y_predict)
### 랜덤하게 했을때는 fpr, tpr 모두 0.5, 0.5로 나온다
random_fpr, random_tpr, threshold2 = roc_curve(Y_test, [0 for i in range(len(X_test))])

plt.figure(figsize = (10,10))
plt.plot(model_fpr, model_tpr, marker = '.', label = "Logistic")
plt.plot(random_fpr, random_tpr, linestyle = '--', label = "Random")

plt.xlabel("False Positive Rate", size = 20)
plt.ylabel("True Positive Rate", size = 20)

plt.legend(fontsize = 20)
plt.title("ROC curve", size = 20)
plt.show()