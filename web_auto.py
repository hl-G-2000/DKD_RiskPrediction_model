import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import shap


def VSpace(px):
    """一个简单的函数，用于在 Streamlit 中创建指定像素的垂直空间"""
    st.markdown(f'<div style="margin-top: {px}px;"></div>', unsafe_allow_html=True)


# Load the trained AutoGluon model
# 模型路径：./Result_auto_DKD_s73_try6/
predictor = TabularPredictor.load('./DKD_model')  
best_model = "LightGBM_BAG_L1/T3"  # 最佳模型名称

# Define the feature options
Gender_options = {
    '0': 'Female',  
    '1': 'Male'  
}
Inj_Freq_options = {
    '0': 'No Insulin',  
    '1': 'With Insulin'  
}

# Streamlit UI
st.title("Diabetic Kidney Disease (DKD) Risk Predictor")  

# 如果有图片，可以取消注释
# image = Image.open("Snipaste_2025-07-01_13-45-35.png")
# st.image(image)


# Sidebar for input options
st.sidebar.header("Input Patient Data")  # 侧边栏输入样本数据

Age = st.sidebar.number_input("Age (年龄):", min_value=18, max_value=100, value=60)
Gender = st.sidebar.selectbox("Gender (性别):", options=list(Gender_options.keys()), format_func=lambda x: Gender_options[x])
DM_Duration = st.sidebar.number_input("DM Duration (糖尿病病程, 年):", min_value=0.0, max_value=50.0, value=5.0, step=0.5)
Inj_Freq = st.sidebar.selectbox("Insulin Use (胰岛素使用):", options=list(Inj_Freq_options.keys()), format_func=lambda x: Inj_Freq_options[x])

st.sidebar.subheader("Laboratory Tests (实验室检查)")
Glu = st.sidebar.number_input("Glucose (Glu, mmol/L):", min_value=0.0, max_value=30.0, value=7.0, step=0.1)
HbA1c = st.sidebar.number_input("HbA1c (%):", min_value=4.0, max_value=15.0, value=7.0, step=0.1)
Cr = st.sidebar.number_input("Creatinine (Cr, μmol/L):", min_value=0.0, max_value=500.0, value=80.0, step=1.0)
SBP = st.sidebar.number_input("Systolic BP (SBP, mmHg):", min_value=80, max_value=220, value=130)
DBP = st.sidebar.number_input("Diastolic BP (DBP, mmHg):", min_value=40, max_value=140, value=80)
TC = st.sidebar.number_input("Total Cholesterol (TC, mmol/L):", min_value=0.0, max_value=15.0, value=5.0, step=0.1)
LDL_C = st.sidebar.number_input("LDL-C (mmol/L):", min_value=0.0, max_value=10.0, value=3.0, step=0.1)
HDL_C = st.sidebar.number_input("HDL-C (mmol/L):", min_value=0.0, max_value=5.0, value=1.2, step=0.1)
TG = st.sidebar.number_input("Triglycerides (TG, mmol/L):", min_value=0.0, max_value=20.0, value=1.5, step=0.1)
ALT = st.sidebar.number_input("ALT (U/L):", min_value=0.0, max_value=500.0, value=25.0, step=1.0)
AST = st.sidebar.number_input("AST (U/L):", min_value=0.0, max_value=500.0, value=25.0, step=1.0)
PLT = st.sidebar.number_input("Platelet (PLT, 10^9/L):", min_value=0.0, max_value=800.0, value=200.0, step=1.0)
HGB = st.sidebar.number_input("Hemoglobin (HGB, g/L):", min_value=0.0, max_value=200.0, value=140.0, step=1.0)
WBC = st.sidebar.number_input("WBC (10^9/L):", min_value=0.0, max_value=50.0, value=7.0, step=0.1)
BMI = st.sidebar.number_input("BMI (kg/m²):", min_value=10.0, max_value=60.0, value=25.0, step=0.1)

# 添加一个 50 像素的垂直空白
VSpace(50)

st.subheader("Process the input and make a prediction")
# Process the input and make a prediction
# 注意：特征顺序需要与训练时一致
feature_values = [Age, Gender, DM_Duration, Inj_Freq, Glu, HbA1c, Cr, SBP, DBP, 
                  TC, LDL_C, HDL_C, TG, ALT, AST, PLT, HGB, WBC, BMI]
feature_names = ["Age", "Gender", "DM_Duration", "Inj_Freq", "Glu", "HbA1c", "Cr", 
                 "SBP", "DBP", "TC", "LDL_C", "HDL_C", "TG", "ALT", "AST", 
                 "PLT", "HGB", "WBC", "BMI"]
features = pd.DataFrame([feature_values], columns=feature_names) 

if st.button("Make Prediction"):  # 如果点击了预测按钮
    # Predict the class and probabilities using AutoGluon
    predicted_proba_df = predictor.predict_proba(features, model=best_model)
    predicted_proba = predicted_proba_df.values[0]  # [prob_class_0, prob_class_1]
    predicted_class = predictor.predict(features, model=best_model).values[0]  # 预测结果

    # Display the prediction results
    st.write(f"**Predicted Class (0 = Non-DKD, 1 = DKD):** {predicted_class}")  # 显示预测类别
    st.write(f"**Prediction Probabilities:** {predicted_proba}")  # 显示各类别的预测概率

    # Generate advice based on the prediction result
    probability = predicted_proba[predicted_class] * 100  # 根据预测类别获取对应的概率，并转化为百分比


    # Visualize the prediction probabilities
    sample_prob = {
        'No DKD': predicted_proba[0],  # DKD不发生的概率
        'DKD': predicted_proba[1]  # DKD发生的概率
    }
    
    VSpace(20)
    # Set figure size
    plt.figure(figsize=(4, 1))  # 设置图形大小
    plt.rc('ytick', labelsize=8) # 设置所有Y轴刻度的字体大小
    plt.rc('xtick', labelsize=8) # 设置所有X轴刻度的字体大小
    # Create bar chart
    bars = plt.barh(['No DKD', 'DKD'], 
                    [sample_prob['No DKD'], sample_prob['DKD']], 
                    height=0.6, edgecolor="black", color=['#81abd3','#fcd6d3'])  # 绘制水平条形图

    # Add title and labels, set font bold and increase font size
    plt.title("Prediction Probability for DKD", fontsize=12, fontweight='bold')  # 添加图表标题，并设置字体大小和加粗
    plt.xlabel("Probability", fontsize=7 )  # 添加X轴标签，并设置字体大小和加粗

    # Add probability text labels, adjust position to avoid overlap, set font bold
    for i, v in enumerate([sample_prob['No DKD'], sample_prob['DKD']]):  # 为每个条形图添加概率文本标签
        plt.text(v + 0.01, i, f"{v:.2f}", va='center', fontsize=6, color='black' )  # 设置标签位置、字体加粗

    # Hide other axes (top, right, bottom)
    plt.gca().spines['top'].set_visible(False)  # 隐藏顶部边框
    plt.gca().spines['right'].set_visible(False)  # 隐藏右边框

    # Show the plot
    st.pyplot(plt, use_container_width=True)  # 显示图表
    

    if predicted_class == 1:  # 如果预测为DKD发生，给出相关建议
        advice = (
            f"**Recommendation:** According to our model, the probability of Diabetic Kidney Disease (DKD) is {probability:.1f}%, which is considered **High risk**. "
            f"We recommend you discuss these findings with your doctor or nephrologist as soon as possible to determine the next steps for kidney[Mam- protection and treatment."
        )  
    else:  # 如果预测为DKD低风险
        advice = (
            f"**Recommendation:** According to our model, the patient is at **low risk** for DKD. "
            f"The probability of **not developing DKD** is **{probability:.1f}%**. "
            "However, it is still important to continue regular monitoring of kidney function and blood glucose control. "
            "Please maintain good diabetes management and have regular check-ups."
        )  

    st.write(advice)  # 显示建议
    
    VSpace(50)

    st.subheader("Feature importance")
    # 获取 AutoGluon 模型的底层估计器
    try:
        # 对于 LightGBM 模型
        model_obj = predictor._trainer.load_model(best_model)
        if hasattr(model_obj, 'model'):
            model_estimator = model_obj.model
        else:
            model_estimator = model_obj
        
        explainer = shap.TreeExplainer(model_estimator)
        shap_values = explainer.shap_values(features.values)
        # 对于二分类，取正类的 SHAP 值
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    except Exception as e:
        st.error(f"SHAP explanation failed: {str(e)}")
        shap_values = None

    if shap_values is not None:
        fig, ax = plt.subplots(figsize=(5, 2.5))
        # 处理 expected_value
        if isinstance(explainer.expected_value, list):
            base_val = explainer.expected_value[1]
        else:
            base_val = explainer.expected_value
            
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0] if len(shap_values.shape) > 1 else shap_values,
                base_values=base_val,
                data=features.values[0],
                feature_names=features.columns.tolist()
            ) 
        )
        plt.savefig("shap_waterfall_plot.png", bbox_inches='tight', dpi=300)
        plt.close(fig)
        st.image("shap_waterfall_plot.png", use_container_width=True)


