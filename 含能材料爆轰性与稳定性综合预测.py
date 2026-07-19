# ===================== 1. 导入所需全部第三方库 =====================
import numpy as np
# numpy：数值计算，用于随机生成特征、数学运算、数组处理
import pandas as pd
# pandas：表格处理，构造数据集DataFrame、筛选最优样本、数据查看
import matplotlib.pyplot as plt
# matplotlib：绘图库，绘制真实值vs预测值对比散点图

# 数据集划分工具
from sklearn.model_selection import train_test_split
# 特征标准化工具，消除特征量纲差异
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# 随机森林回归模型，用于多输出预测爆轰、稳定性能
from sklearn.ensemble import RandomForestRegressor
# 模型评价指标：平均绝对误差、均方误差、决定系数R²
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ===================== 2. 生成含能材料仿真数据集 =====================
# 固定随机种子，每次运行生成完全相同的数据，保证实验可复现
np.random.seed(2026)
# 设置样本总量220组含能分子
n_samples = 220

# 随机生成6项分子结构特征，均为均匀分布
# OB氧平衡：取值区间-130 ~ 15
OB = np.random.uniform(-130, 15, n_samples)
# N_ratio氮含量百分比：8 ~ 48
N_ratio = np.random.uniform(8, 48, n_samples)
# rho晶体密度：1.35 ~ 2.05 g/cm³
rho = np.random.uniform(1.35, 2.05, n_samples)
# delta_H生成焓：-450 ~ 900
delta_H = np.random.uniform(-450, 900, n_samples)
# total_bond_E分子总键能：300 ~ 1200
total_bond_E = np.random.uniform(300, 1200, n_samples)
# nitro_count硝基数目：随机整数1~7个
nitro_count = np.random.randint(1, 8, n_samples)

# 构造爆轰性能标签（模拟真实物理规律+高斯噪声模拟实验误差）
# D爆速：密度、氮含量、氧平衡共同决定，叠加正态随机噪声
D = (rho * 4.2) + (N_ratio * 0.06) + (OB * 0.008) + np.random.normal(0, 0.15, n_samples)
# Pcj CJ爆压：密度平方主导，叠加噪声
Pcj = (rho ** 2 * 12) + (N_ratio * 0.12) + np.random.normal(0, 0.8, n_samples)

# 构造稳定安全性能标签
# Td热分解温度：硝基数越多、氧平衡绝对值越大，热稳定性越差；总键能越高越稳定
Td = 220 - abs(OB)*0.4 - nitro_count*12 + total_bond_E*0.12 + np.random.normal(0, 7, n_samples)
# H50撞击感度：硝基数多、氧平衡差则敏感；密度越高越钝感；数值越大越安全
H50 = 65 - nitro_count*7.5 - abs(OB)*0.15 + rho*12 + np.random.normal(0, 3, n_samples)

# 将全部特征、性能标签组装成二维表格DataFrame
df = pd.DataFrame({
    "OB": OB,
    "N_ratio": N_ratio,
    "rho": rho,
    "delta_H": delta_H,
    "total_bond_E": total_bond_E,
    "nitro_count": nitro_count,
    "D_km_s": D,
    "Pcj_GPa": Pcj,
    "Td_decomp": Td,
    "H50": H50
})
# 在控制台打印数据集前6行，直观查看生成数据
print("数据集前6行原始数据：")
print(df.head(), "\n")

# ===================== 3. 优化版综合得分计算（分层权重+短板惩罚） =====================
# 初始化最大最小归一化器，把爆速、爆压、Td、H50缩放到0~1区间，消除量纲差异
mm_scaler = MinMaxScaler()
# 需要归一化的4个性能指标列名
norm_cols = ["D_km_s", "Pcj_GPa", "Td_decomp", "H50"]
# 归一化并新增4列保存归一化结果
df[["D_norm", "Pcj_norm", "Td_norm", "H50_norm"]] = mm_scaler.fit_transform(df[norm_cols])

# 第一层：指标内部行业加权（工程经验权重）
# 爆轰分项得分：爆速权重0.6（更重要），爆压权重0.4
df["score_det"] = 0.6 * df["D_norm"] + 0.4 * df["Pcj_norm"]
# 稳定分项得分：撞击感度H50权重0.6（生产安全核心），热分解温度0.4
df["score_stab"] = 0.6 * df["H50_norm"] + 0.4 * df["Td_norm"]

# 第二层：全局偏好权重，平衡爆轰威力与安全性
w_d = 0.5    # 爆轰性能权重
w_s = 0.5    # 稳定安全权重
# 基础均衡加权得分
df["score_balance"] = w_d * df["score_det"] + w_s * df["score_stab"]

# 第三层：短板惩罚逻辑，淘汰单项性能极差的平庸样本
T_det = 0.35    # 爆轰分项最低合格阈值，低于该值扣分
T_stab = 0.35   # 稳定分项最低合格阈值
lam = 1.2       # 惩罚强度系数，数值越大短板扣分越严重

# 自定义惩罚函数，输入单一样本的爆轰分、稳定分，返回扣分值
def calc_penalty(d, s):
    p = 0
    # 爆轰分不达标，施加负惩罚
    if d < T_det:
        p -= lam * (T_det - d)
    # 稳定分不达标，施加负惩罚
    if s < T_stab:
        p -= lam * (T_stab - s)
    return p

# 按行遍历表格，对每个样本计算惩罚值，新增penalty列
df["penalty"] = df.apply(lambda row: calc_penalty(row["score_det"], row["score_stab"]), axis=1)
# 最终综合得分 = 基础均衡分 + 短板惩罚分
df["Final综合得分"] = df["score_balance"] + df["penalty"]

# 筛选三类最优样本
# idxmax()找到得分最大值所在行索引；loc提取整行数据；copy()复制避免原表联动修改
top_detonation_raw = df.loc[df["score_det"].idxmax()].copy()
top_stable_raw = df.loc[df["score_stab"].idxmax()].copy()
top_balance_raw = df.loc[df["Final综合得分"].idxmax()].copy()

# 控制台打印三组最优完整原始参数
print("="*60)
print("【原始数据-爆轰性能最高样本完整参数】")
print("="*60)
print(top_detonation_raw)

print("\n" + "="*60)
print("【原始数据-稳定安全性最高样本完整参数】")
print("="*60)
print(top_stable_raw)

print("\n" + "="*60)
print(f"【原始数据-均衡最优样本 爆轰权重{w_d} 稳定权重{w_s} 带短板惩罚】")
print("="*60)
print(top_balance_raw)
print("\n")

# ===================== 4. 数据集划分与标准化 =====================
# X：模型输入特征（6个分子描述符）
X = df[["OB", "N_ratio", "rho", "delta_H", "total_bond_E", "nitro_count"]]
# y_energy：能量输出标签（爆速、爆压）
y_energy = df[["D_km_s", "Pcj_GPa"]]
# y_safe：安全稳定输出标签（热分解温度、撞击感度）
y_safe = df[["Td_decomp", "H50"]]

# train_test_split：划分训练集80%、测试集20%，随机种子固定保证划分不变
X_train, X_test, y_e_train, y_e_test, y_s_train, y_s_test = train_test_split(
    X, y_energy, y_safe, test_size=0.2, random_state=2026
)

# 初始化Z-score标准化器，消除特征量纲，加速树模型收敛
scaler = StandardScaler()
# fit_transform：训练集计算均值方差并标准化
X_train_sc = scaler.fit_transform(X_train)
# transform：测试集复用训练集均值方差，不泄露测试集信息
X_test_sc = scaler.transform(X_test)

# ===================== 5. 训练两套随机森林模型 =====================
# 能量预测随机森林：150棵决策树，单树最大深度9，固定随机种子
model_energy = RandomForestRegressor(n_estimators=150, max_depth=9, random_state=2026)
# fit：使用标准化训练集拟合模型，学习分子特征与爆轰性能映射关系
model_energy.fit(X_train_sc, y_e_train)
# predict：测试集预测爆速、爆压
y_e_pred = model_energy.predict(X_test_sc)

# 稳定性能预测随机森林，超参数与能量模型保持一致
model_safe = RandomForestRegressor(n_estimators=150, max_depth=9, random_state=2026)
model_safe.fit(X_train_sc, y_s_train)
y_s_pred = model_safe.predict(X_test_sc)

# ===================== 6. 模型评估函数 =====================
# 封装评价指标计算逻辑，避免重复代码
def eval_model(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)    # 平均绝对误差
    mse = mean_squared_error(y_true, y_pred)     # 均方误差
    rmse = np.sqrt(mse)                         # 均方根误差，和指标量纲一致
    r2 = r2_score(y_true, y_pred)               # 决定系数，越接近1拟合效果越好
    print(f"\n==== {name} 预测指标 ====")
    print(f"MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.3f}")

# 分别评估4个输出指标的预测精度
eval_model(y_e_test["D_km_s"], y_e_pred[:,0], "爆速 D(km/s)")
eval_model(y_e_test["Pcj_GPa"], y_e_pred[:,1], "CJ爆压 Pcj(GPa)")
eval_model(y_s_test["Td_decomp"], y_s_pred[:,0], "热分解温度 Td(℃)")
eval_model(y_s_test["H50"], y_s_pred[:,1], "撞击感度 H50(cm)")

# 提取能量模型特征重要性，转为Series并按降序输出
print("\n==== 能量模型特征重要性 ====")
feat_e = pd.Series(model_energy.feature_importances_, index=X.columns)
print(feat_e.sort_values(ascending=False))

# 提取稳定模型特征重要性
print("\n==== 稳定感度模型特征重要性 ====")
feat_s = pd.Series(model_safe.feature_importances_, index=X.columns)
print(feat_s.sort_values(ascending=False))

# ===================== 7. 绘制预测对比图 =====================
# 设置绘图中文显示，解决中文方框乱码
plt.rcParams["font.sans-serif"] = ["SimHei"]
# 解决负号显示异常
plt.rcParams["axes.unicode_minus"] = False
# 创建2行2列画布，画布尺寸14×10英寸
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 第一张子图：爆速真实值vs预测值
ax = axes[0,0]
# 散点绘制，红色，透明度0.7
ax.scatter(y_e_test["D_km_s"], y_e_pred[:,0], c="#d62728", alpha=0.7)
# 绘制y=x虚线，代表完美预测基准线
ax.plot([y_e_test["D_km_s"].min(), y_e_test["D_km_s"].max()],
        [y_e_test["D_km_s"].min(), y_e_test["D_km_s"].max()], "k--")
ax.set_title("爆速预测值vs真实值")
ax.set_xlabel("真实爆速")
ax.set_ylabel("预测爆速")
ax.grid(alpha=0.3) # 浅色网格线

# 第二张子图：CJ爆压对比
ax = axes[0,1]
ax.scatter(y_e_test["Pcj_GPa"], y_e_pred[:,1], c="#1f77b4", alpha=0.7)
ax.plot([y_e_test["Pcj_GPa"].min(), y_e_test["Pcj_GPa"].max()],
        [y_e_test["Pcj_GPa"].min(), y_e_test["Pcj_GPa"].max()], "k--")
ax.set_title("CJ爆压预测值vs真实值")
ax.set_xlabel("真实爆压")
ax.set_ylabel("预测爆压")
ax.grid(alpha=0.3)

# 第三张子图：热分解温度对比
ax = axes[1,0]
ax.scatter(y_s_test["Td_decomp"], y_s_pred[:,0], c="#2ca02c", alpha=0.7)
ax.plot([y_s_test["Td_decomp"].min(), y_s_test["Td_decomp"].max()],
        [y_s_test["Td_decomp"].min(), y_s_test["Td_decomp"].max()], "k--")
ax.set_title("热分解温度预测对比")
ax.set_xlabel("实测Td")
ax.set_ylabel("预测Td")
ax.grid(alpha=0.3)

# 第四张子图：撞击感度H50对比
ax = axes[1,1]
ax.scatter(y_s_test["H50"], y_s_pred[:,1], c="#9467bd", alpha=0.7)
ax.plot([y_s_test["H50"].min(), y_s_test["H50"].max()],
        [y_s_test["H50"].min(), y_s_test["H50"].max()], "k--")
ax.set_title("撞击感度H50预测对比")
ax.set_xlabel("实测H50")
ax.set_ylabel("预测H50")
ax.grid(alpha=0.3)

# 自动调整子图间距，防止标题、坐标轴重叠
plt.tight_layout()
# 弹出图像窗口展示绘图结果
plt.show()

# ===================== 8. 新含能材料综合预测 =====================
# 输入新分子6项特征：OB, N_ratio, rho, delta_H, total_bond_E, nitro_count
new_sample = np.array([[-28, 34.2, 1.84, 460, 820, 4]])
# 使用训练集标准化规则对新样本缩放
new_sc = scaler.transform(new_sample)
# 分别调用两套模型预测爆轰、稳定性能
pred_e = model_energy.predict(new_sc)
pred_s = model_safe.predict(new_sc)

# 控制台输出新材料全部预测指标
print("\n========== 新材料综合预测结果 ==========")
print(f"预测爆速 D = {pred_e[0][0]:.3f} km/s")
print(f"预测CJ爆压 Pcj = {pred_e[0][1]:.3f} GPa")
print(f"预测热分解温度 Td = {pred_s[0][0]:.2f} ℃")
print(f"预测撞击感度 H50 = {pred_s[0][1]:.2f} cm")
print("说明：Td越高热稳定越好；H50数值越大撞击越钝感、安全性越高")
