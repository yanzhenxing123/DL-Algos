"""
@Author: yanzx
@Date: 2025/3/2 21:54
@Description:
"""

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 加载数据
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)

# 训练 AdaBoost 模型
model = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)
model.fit(X_train, y_train)

print(X_train)
print(y_train)

# 预测
y_pred = model.predict(X_test)
print(y_pred)
print("AdaBoost 准确率:", accuracy_score(y_test, y_pred))
