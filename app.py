import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

st.set_page_config(page_title="多元线性回归-期末成绩预测系统", layout="wide")

# ========== 会话状态：保存模型、训练状态 ==========
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False
if "model" not in st.session_state:
    st.session_state.model = None
if "X_train_cnt" not in st.session_state:
    st.session_state.X_train_cnt = 0
if "X_test_cnt" not in st.session_state:
    st.session_state.X_test_cnt = 0
if "coefs" not in st.session_state:
    st.session_state.coefs = None
if "intercept" not in st.session_state:
    st.session_state.intercept = None
if "metrics" not in st.session_state:
    st.session_state.metrics = dict()
if "feature_list" not in st.session_state:
    st.session_state.feature_list = []
if "y_test_real" not in st.session_state:
    st.session_state.y_test_real = None
if "y_test_pred" not in st.session_state:
    st.session_state.y_test_pred = None

# ========== 1.系统介绍 ==========
st.title("基于多元线性回归的期末成绩预测系统")
st.markdown("""
**系统介绍**
- 预测任务：根据学生多项学习特征预测期末成绩（连续数值）
- 输入特征：每日学习时长、出勤率、每周复习次数
- 预测目标：期末成绩
- 使用模型：多元线性回归
""")

# ========== 2.数据载入 ==========
st.subheader("1. 数据载入与查看")
try:
    df = pd.read_csv("score_data.csv")
except FileNotFoundError:
    np.random.seed(2026)
    df = pd.DataFrame({
        "study_hours": np.random.uniform(2, 10, 30),
        "attendance": np.random.randint(68, 99, 30),
        "review_times": np.random.randint(1, 5, 30),
    })
    df["final_score"] = 5*df["study_hours"] + 0.3*df["attendance"] + 3*df["review_times"] + np.random.normal(0,3,30)

st.dataframe(df, use_container_width=True)
st.write(f"数据集总样本数：{df.shape[0]}，总字段数：{df.shape[1]}")
st.write("数据统计信息：")
st.dataframe(df.describe())

# ========== 3.训练设置 ==========
st.subheader("2. 训练设置")
all_features = ["study_hours", "attendance", "review_times"]
target_col = "final_score"

select_features = st.multiselect(
    "选择输入特征（至少选择1项）",
    options=all_features,
    default=all_features
)
test_size_slider = st.slider("测试集占比", min_value=0.1, max_value=0.4, value=0.2, step=0.05)

# ========== 训练按钮 ==========
train_btn = st.button("训练模型", type="primary")

if train_btn:
    if len(select_features) < 1:
        st.error("请至少选择一个输入特征！")
    else:
        with st.spinner("训练中……"):
            X = df[select_features]
            y = df[target_col]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size_slider, random_state=42
            )
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred_test = model.predict(X_test)

            st.session_state.model_trained = True
            st.session_state.model = model
            st.session_state.X_train_cnt = X_train.shape[0]
            st.session_state.X_test_cnt = X_test.shape[0]
            st.session_state.coefs = model.coef_
            st.session_state.intercept = model.intercept_
            st.session_state.feature_list = select_features
            st.session_state.y_test_real = y_test.values
            st.session_state.y_test_pred = y_pred_test

            mae = mean_absolute_error(y_test, y_pred_test)
            mse = mean_squared_error(y_test, y_pred_test)
            rmse = np.sqrt(mse)
            st.session_state.metrics = {"MAE": mae, "MSE": mse, "RMSE": rmse}
        st.success("✅ 训练成功！")

# ========== 训练结果展示 ==========
if st.session_state.model_trained:
    st.subheader("3. 模型训练结果")
    st.write(f"训练集样本数量：{st.session_state.X_train_cnt}")
    st.write(f"测试集样本数量：{st.session_state.X_test_cnt}")
    st.write("使用输入特征：", st.session_state.feature_list)
    st.write("预测目标：", target_col)
    st.write(f"模型截距 intercept = {st.session_state.intercept:.3f}")

    coef_df = pd.DataFrame({
        "特征": st.session_state.feature_list,
        "回归系数": [round(c, 4) for c in st.session_state.coefs]
    })
    st.dataframe(coef_df)
    st.markdown("**系数解释：** 在其他特征不变条件下，该特征每增加1单位，期末成绩平均变化量等于对应系数。")

    # ========== 模型评价指标 ==========
    st.subheader("4. 模型评价指标")
    met = st.session_state.metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE 平均绝对误差", round(met["MAE"], 3))
    col2.metric("MSE 均方误差", round(met["MSE"], 3))
    col3.metric("RMSE 均方根误差", round(met["RMSE"], 3))

    # ========== 两张可视化图表 ==========
    st.subheader("5. 数据与结果可视化")
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    ax1.scatter(st.session_state.y_test_real, st.session_state.y_test_pred, color="steelblue", alpha=0.8)
    lo = min(st.session_state.y_test_real.min(), st.session_state.y_test_pred.min())
    hi = max(st.session_state.y_test_real.max(), st.session_state.y_test_pred.max())
    ax1.plot([lo, hi], [lo, hi], "r--", label="理想拟合线")
    ax1.set_xlabel("真实期末成绩")
    ax1.set_ylabel("预测期末成绩")
    ax1.set_title("图1：真实值与预测值对比散点图")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.hist(df["final_score"], bins=8, color="skyblue", edgecolor="black")
    ax2.set_xlabel("期末成绩")
    ax2.set_ylabel("样本数量")
    ax2.set_title("图2：期末成绩样本分布直方图")
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)

    # ========== 在线预测 ==========
    st.subheader("6. 在线预测模块")
    sh = st.number_input("每日学习时长（小时）", min_value=0.0, max_value=12.0, value=5.0, step=0.5)
    att = st.number_input("出勤率（0~100）", min_value=0.0, max_value=100.0, value=85.0)
    rev = st.number_input("每周复习次数", min_value=0, max_value=10, value=2)

    pred_btn = st.button("执行预测")
    if pred_btn:
        input_arr = np.array([[sh, att, rev]])
        pred_res = st.session_state.model.predict(input_arr)
        st.info(f"📌 预测期末成绩：**{pred_res[0]:.2f} 分**")
else:
    st.warning("⚠️ 模型尚未训练，请先点击【训练模型】按钮完成训练，之后才可以进行预测！")
