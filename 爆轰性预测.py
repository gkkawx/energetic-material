import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from  sklearn.model_selection import train_test_split
from  sklearn.preprocessing import StandardScaler
from  sklearn.ensemble import RandomForestRegressor
from  sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ===================== 1. 模拟含能材料数据集 =====================
# 特征：分子氧平衡OB、氮含量N%、晶体密度ρ、生成焓ΔH、分子分子量Mw
# 标签：爆速D(km/s)、CJ爆压Pcj(GPa)
np.random.seed(42)  # 固定随机种子保证复现
sample_num = 200

# 构造模拟材料特征
oxygen_balance = np.random.uniform(-120, 20, sample_num)
nitrogen_ratio = np.random.uniform(10, 45, sample_num)
crystal_density = np.random.uniform(1.4, 2.0, sample_num)
formation_enthalpy = np.random.uniform(-400, 800, sample_num)
molecular_weight = np.random.uniform(100, 400, sample_num)

# 构造近似爆轰参数（物理经验拟合公式模拟真实标签）
detonation_velocity = (crystal_density * 4.2) + (nitrogen_ratio * 0.06) + (oxygen_balance * 0.008) + np.random.normal(0, 0.15, sample_num)
cj_pressure = (crystal_density ** 2 * 12) + (nitrogen_ratio * 0.12) + np.random.normal(0, 0.8, sample_num)

# 组装DataFrame
data = pd.DataFrame({"OB": oxygen_balance,"N_ratio": nitrogen_ratio,"rho": crystal_density,"delta_H": formation_enthalpy,"Mw": molecular_weight,"D_km_s": detonation_velocity,"Pcj_GPa": cj_pressure})
print("数据集前5行：")
print(data.head())

# ===================== 2. 划分特征与预测目标 =====================
X = data[["OB", "N_ratio", "rho", "delta_H", "Mw"]]
y = data[["D_km_s", "Pcj_GPa"]]

# 训练集80%，测试集20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===================== 3. 构建多输出AI回归模型 =====================
# 随机森林适合含能材料非线性小样本数据
model = RandomForestRegressor(n_estimators=120, max_depth=8, random_state=42)
model.fit(X_train_scaled, y_train)

# ===================== 4. 模型预测与性能评估 =====================
y_pred = model.predict(X_test_scaled)

# 分别评估爆速、爆压预测精度
def eval_metrics(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f"\n==== {name} 模型评价指标 ====")
    print(f"MAE(平均绝对误差): {mae:.4f}")
    print(f"RMSE(均方根误差): {rmse:.4f}")
    print(f"R²决定系数: {r2:.4f}")

eval_metrics(y_test["D_km_s"], y_pred[:, 0], "爆速 D(km/s)")
eval_metrics(y_test["Pcj_GPa"], y_pred[:, 1], "CJ爆压 Pcj(GPa)")

# 特征重要性（判断哪种分子参数对爆轰影响最大）
feature_importance = pd.Series(model.feature_importances_, index=X.columns)
print("\n各分子描述符特征重要性：")
print(feature_importance.sort_values(ascending=False))

# ===================== 5. 可视化预测结果对比 =====================
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
# 爆速对比
ax1.scatter(y_test["D_km_s"], y_pred[:, 0], c="#d62728", alpha=0.7)
ax1.plot([y_test["D_km_s"].min(), y_test["D_km_s"].max()],[y_test["D_km_s"].min(), y_test["D_km_s"].max()], "k--")
ax1.set_xlabel("真实爆速 D (km/s)")
ax1.set_ylabel("AI预测爆速 D (km/s)")
ax1.set_title("爆速预测值 vs 真实值")
ax1.grid(alpha=0.3)

# 爆压对比
ax2.scatter(y_test["Pcj_GPa"], y_pred[:, 1], c="#1f77b4", alpha=0.7)
ax2.plot([y_test["Pcj_GPa"].min(), y_test["Pcj_GPa"].max()],[y_test["Pcj_GPa"].min(), y_test["Pcj_GPa"].max()], "k--")
ax2.set_xlabel("真实CJ爆压 Pcj (GPa)")
ax2.set_ylabel("AI预测CJ爆压 Pcj (GPa)")
ax2.set_title("CJ爆压预测值 vs 真实值")
ax2.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ===================== 6. 单种新含能材料爆轰性能预测 =====================
# 输入新材料分子特征：OB、N含量、密度、生成焓、分子量
new_material = np.array([[-35, 32.5, 1.82, 420, 226]])
new_material_scaled = scaler.transform(new_material)
pred_result = model.predict(new_material_scaled)

print("\n==== 新含能材料爆轰性能预测结果 ====")
print(f"预测爆速 D = {pred_result[0][0]:.3f} km/s")
print(f"预测CJ爆压 Pcj = {pred_result[0][1]:.3f} GPa")
