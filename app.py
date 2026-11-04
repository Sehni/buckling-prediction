import streamlit as st
import pandas as pd
import numpy as np
import io
import joblib
from datap import Data

# ==========================================
# 0. 准备模型和特征 (使用缓存避免重复加载)
# ==========================================
st.set_page_config(page_title="屈曲载荷预测系统", page_icon="🛡️", layout="wide")

@st.cache_resource
def load_assets():
    model = joblib.load('best_model.pkl')
    data = Data()
    return model, data

model, data = load_assets()

feature_names = model.feature_names_in_
raw_features = data.features.copy()  # 使用 copy 防止修改原列表
if 'buckling' in raw_features:
    raw_features.remove('buckling')

trans = {
    'top': '顶点内凹（单位：mm）', 
    'crown': '球冠区最大内凹（单位：mm）', 
    'material': '材料屈服强度（单位：MPa）', 
    'd': '直径（单位：mm）',
    't': '壁厚（单位：mm）', 
    'd/2h': '径高比（半径/高度）', 
    'fai': '过渡区外凸范围（0°~X°）', 
    'knuckle': '过渡区整体缺陷最大外凸（单位：mm）', 
    'xita': '过渡区局部缺陷环向范围（±X°）', 
    'smax': '局部缺陷最大外凸（单位：mm）'
}
default = data.default

def predict(input_df):
    dt = Data()
    input_df['buckling'] = 0  # 补一列骗过 translate
    dt.translate(input_df)
    
    # 拿到 translate 处理后的 DataFrame
    df_processed = dt.good_dfs[0]
    
    # 强制剔除多余的列，并严格按照模型训练时的顺序重排
    df_processed = df_processed[model.feature_names_in_].copy()
    
    return model.predict(df_processed)

# ==========================================
# 1. 侧边栏导航 (替代原来的 switch_page 按钮)
# ==========================================
st.sidebar.title("🛡️ 导航菜单")
page = st.sidebar.radio("请选择功能", ["欢迎首页", "单次参数预测", "批量 Excel 预测"])

# ==========================================
# 2. 页面一：欢迎页
# ==========================================
if page == "欢迎首页":
    st.markdown("""
        <div style="text-align: center; margin-top: 100px;">
            <h1 style="color: #2c3e50; font-size: 40px; font-weight: bold; margin-bottom: 15px;">
                内压薄壁封头屈曲载荷预测系统
            </h1>
            <p style="color: #7f8c8d; font-size: 20px; margin-bottom: 60px;">
                基于随机森林模型与贝叶斯优化
            </p>
        </div>
    """, unsafe_allow_html=True)
    st.info("👈 请在左侧侧边栏选择对应的功能开始使用。")

# ==========================================
# 3. 页面二：单次预测页
# ==========================================
elif page == "单次参数预测":
    st.title("单次参数预测")
    
    # 将页面分为左右两栏，左侧放输入，右侧放结果
    col_left, col_right = st.columns([1.2, 1])
    
    with col_left:
        st.markdown("### 输入特征参数")
        with st.form("single_predict_form"):
            input_values = {}
            # 将 10 个输入框分成两列显示，避免页面太长
            col_a, col_b = st.columns(2)
            
            for i, name in enumerate(raw_features):
                with col_a if i % 2 == 0 else col_b:
                    input_values[name] = st.number_input(
                        label=f"{trans[name]}:",
                        value=float(default[name]),
                        step=0.1,
                        format="%.2f"
                    )
            
            # 提交按钮
            submit_btn = st.form_submit_button("🚀 开始预测", use_container_width=True)
            
    with col_right:
        st.markdown("### 预测结果")
        # 占位符，用于稍后填充结果
        result_placeholder = st.empty()
        
        if submit_btn:
            with st.spinner("正在计算中..."):
                try:
                    input_df = pd.DataFrame([input_values], columns=raw_features)
                    result = predict(input_df)[0]
                    
                    # 渲染原版的渐变卡片
                    html_card = f"""
                    <div style="background: linear-gradient(135deg, #e0f7fa 0%, #b2ebf2 100%); border-left: 8px solid #00bcd4; padding: 30px; border-radius: 8px; margin-top: 20px;">
                        <h3 style="margin: 0; color: #006064;">✅ 预测成功</h3>
                        <p style="font-size: 48px; font-weight: bold; color: #00838f; margin: 15px 0;">{result:.4f} <span style="font-size: 20px;">MPa</span></p>
                    </div>
                    """
                    result_placeholder.markdown(html_card, unsafe_allow_html=True)
                except Exception as e:
                    result_placeholder.error(f"错误：{e}")
        else:
            result_placeholder.markdown("""
            <div style="text-align:center; color:#999; margin-top:50px; border: 1px dashed #ccc; padding: 50px; border-radius: 8px;">
                <h3>等待输入...</h3>
            </div>
            """, unsafe_allow_html=True)

# ==========================================
# 4. 页面三：批量预测页
# ==========================================
elif page == "批量 Excel 预测":
    st.title("批量 Excel 数据预测")
    
    # --- 生成格式要求 HTML ---
    th_style = "border: 1px solid #ddd; padding: 10px; background-color: #f2f2f2; font-weight: bold;"
    td_style = "border: 1px solid #ddd; padding: 10px; text-align: center;"
    header_html = "".join([f"<th style='{th_style}'>{trans[col]}</th>" for col in raw_features])
    default_html = "".join([f"<td style='{td_style}'>{default[col]}</td>" for col in raw_features])
    ellipsis_html = "".join([f"<td style='{td_style}'>...</td>" for _ in raw_features])

    example_table_html = f"""
    <div style="background-color: #e8f4f8; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
        <h4 style="margin-top:0; color: #005b82;">上传文件格式要求：</h4>
        <p>请上传 <b>.xlsx</b> 或 <b>.xls</b> 文件。第一行是特征列名，且顺序需要与下方示例一致。</p>
        <table style="width: 100%; border-collapse: collapse; margin-top: 15px; background: white;">
            <tr>{header_html}</tr>
            <tr>{default_html}</tr>
            <tr>{ellipsis_html}</tr>
        </table>
    </div>
    """
    st.markdown(example_table_html, unsafe_allow_html=True)

    # --- 文件上传与处理 ---
    uploaded_file = st.file_uploader("📁 上传数据文件", type=["xlsx", "xls"])
    
    if uploaded_file is not None:
        if st.button("开始批量预测", type="primary"):
            with st.spinner("正在计算中，请稍候..."):
                try:
                    # 1. 读取文件
                    df_upload = pd.read_excel(uploaded_file)
                    
                    # 2. 校验列数
                    new_cols = df_upload.columns
                    missing_cols_count = len(raw_features) - len(new_cols)
                    if missing_cols_count != 0:
                        st.error(f"❌ 文件缺少 {missing_cols_count} 列。请确保特征数与示例一致！")
                        st.stop()
                        
                    # 3. 换表头并预测
                    df_upload.columns = raw_features
                    predictions = predict(df_upload)
                    
                    # 4. 还原表头并添加结果
                    df_upload.columns = new_cols
                    df_upload['预测屈曲载荷(MPa)'] = predictions
                    
                    # 5. 生成 Excel 下载流
                    output_stream = io.BytesIO()
                    with pd.ExcelWriter(output_stream, engine='openpyxl') as writer:
                        df_upload.to_excel(writer, index=False)
                    
                    # 6. 成功提示与下载按钮
                    st.success(f"✅ 预测完成！共成功处理了 {len(df_upload)} 条数据。")
                    st.download_button(
                        label="📥 点击下载预测结果 (Excel)",
                        data=output_stream.getvalue(),
                        file_name="预测结果.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        type="primary"
                    )
                    
                except Exception as e:
                    st.error(f"❌ 处理失败，错误信息：{e}")