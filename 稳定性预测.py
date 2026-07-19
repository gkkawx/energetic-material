import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ===================== 1. 模拟含能材料数据集 =====================
# 输入特征：氧平衡OB、氮含量N%、晶体密度ρ、生成焓ΔH、分子总键能、硝基数目
np.random.seed(2026)
n_samples = 220

OB = np.random.uniform(-130, 15, n_samples)
N_ratio = np.random.uniform(8, 48, n_samples)
rho = np.random.uniform(1.35, 2.05, n_samples)
dH = np.random.uniform(-450, 900, n_samples)
bond_energy = np.random.uniform(300, 1200, n_samples)
nitro_num = np.random.randint(1, 8, n_samples)

# 稳定性标签：热分解温度Td(℃)、撞击感度H50(cm)
Td = 220 - abs(OB)*0.4 - nitro_num*12 + bond_energy*0.12 + np.random.normal(0, 7, n_samples)
H50 = 65 - nitro_num*7.5 - abs(OB)*0.15 + rho*12 + np.random.normal(0, 3, n_samples)

# 构建数据表
df = pd.DataFrame({
    "OB": OB,
    "N_ratio": N_ratio,
    "rho": rho,
    "delta_H": dH,
    "total_bond_E": bond_energy,
    "nitro_count": nitro_num,
    "Td_decomp": Td,    # 热分解温度（越高热稳定性越好）
    "H50": H50          # 撞击感度（越高越钝感、稳定）
})
print("数据集前6行：")
print(df.head())

# ===================== 2. 数据集划分与标准化 =====================
X = df[["OB", "N_ratio", "rho", "delta_H", "total_bond_E", "nitro_count"]]
y = df[["Td_decomp", "H50"]]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2026
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# ===================== 3. 训练多输出AI模型 =====================
model = RandomForestRegressor(n_estimators=150, max_depth=9, random_state=2026)
model.fit(X_train_sc, y_train)
y_pred = model.predict(X_test_sc)

# ===================== 4. 模型评估函数 =====================
def evaluate(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n【{name}预测精度】")
    print(f"MAE = {mae:.2f} | RMSE = {rmse:.2f} | R² = {r2:.3f}")

evaluate(y_test["Td_decomp"], y_pred[:,0], "热分解温度Td")
evaluate(y_test["H50"], y_pred[:,1], "撞击感度H50")

# 输出特征重要性（判断哪些结构影响稳定性）
feat_import = pd.Series(model.feature_importances_, index=X.columns)
print("\n各分子特征对稳定性的影响权重：")
print(feat_import.sort_values(ascending=False))

# ===================== 5. 绘图：真实值vs预测值 =====================
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

# 热分解温度
ax1.scatter(y_test["Td_decomp"], y_pred[:,0], c="#c82423", alpha=0.7)
ax1.plot([min(y_test["Td_decomp"]), max(y_test["Td_decomp"])],
         [min(y_test["Td_decomp"]), max(y_test["Td_decomp"])], "k--")
ax1.set_xlabel("实测热分解温度 Td (℃)")
ax1.set_ylabel("AI预测热分解温度 Td (℃)")
ax1.set_title("热稳定性预测对比")
ax1.grid(alpha=0.3)

# 撞击感度
ax2.scatter(y_test["H50"], y_pred[:,1], c="#2878b5", alpha=0.7)
ax2.plot([min(y_test["H50"]), max(y_test["H50"])],
         [min(y_test["H50"]), max(y_test["H50"])], "k--")
ax2.set_xlabel("实测H50撞击感度 (cm)")
ax2.set_ylabel("AI预测H50撞击感度 (cm)")
ax2.set_title("撞击稳定性预测对比")
ax2.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ===================== 6. 预测全新含能材料稳定性 =====================
# 特征顺序：[OB, N_ratio, rho, delta_H, total_bond_E, nitro_count]
new_mol = np.array([[-28, 34.2, 1.84, 460, 820, 4]])
new_sc = scaler.transform(new_mol)
res = model.predict(new_sc)

print("\n===== 新材料稳定性预测结果 =====")
print(f"预测热分解温度 Td = {res[0][0]:.2f} ℃")
print(f"预测撞击感度 H50 = {res[0][1]:.2f} cm")
print("说明：Td越高热稳定性越好；H50越高撞击越钝感、安全稳定")
