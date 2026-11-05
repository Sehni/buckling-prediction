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
    model = joblib.load('best_model1.pkl')
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
page = st.sidebar.radio("请选择功能", ["欢迎首页", "单次参数预测", "批量 Excel 预测", "敏感性分析"])

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

# ==========================================
# 5. 页面四：敏感性分析
# ==========================================
elif page == "敏感性分析":
    st.title("📊 缺陷参数敏感性分析")
    st.markdown("""
    本章节通过调用随机森林模型进行“虚拟实验”，分析当单一缺陷参数变化时，对封头屈曲载荷的影响规律。
    **无需重新进行耗时的有限元仿真，即可快速评估缺陷敏感度。**
    """)
    
    # 分为两个 Tab：单变量敏感性、全局特征重要性
    tab1, tab2 = st.tabs(["📈 单变量敏感性曲线 (定量)", "🏆 全局特征重要性 (定性)"])
    
    # --- Tab 1: 单变量敏感性曲线 ---
    with tab1:
        col_ctrl, col_chart = st.columns([1, 2.5])
        
        with col_ctrl:
            st.markdown("### ⚙️ 实验设置")
            st.info("保持其他参数不变，单独调节一个参数，观察载荷变化。")
            
            # 1. 选择要分析的参数
            target_feature = st.selectbox(
                "选择要分析的参数", 
                options=raw_features,
                format_func=lambda x: trans[x], # 显示中文名
                index=raw_features.index('top') # 默认选中鼓包高度
            )
            
            # 2. 设置扫描范围
            st.write("设置扫描范围：")
            # 默认根据原特征适当放宽范围
            default_min = 0.0
            default_max = float(default[target_feature]) * 3 + 10.0
            min_val = st.number_input("最小值", value=default_min, step=0.1)
            max_val = st.number_input("最大值", value=max(default_max, min_val + 1.0), step=0.1)
            steps = st.slider("采样点数", min_value=10, max_value=100, value=50, step=10)
            
            st.markdown("---")
            st.markdown("### 📌 其他基准参数")
            st.caption("其他参数保持默认值或在此微调：")
            
            # 为了方便不把页面撑太长，用一个折叠面板设置基准值
            base_values = {}
            with st.expander("展开修改基准参数"):
                for f in raw_features:
                    if f != target_feature:
                        base_values[f] = st.number_input(f"{trans[f]}", value=float(default[f]), step=0.1)
            if not base_values:
                # 如果用户没展开，直接用默认值
                base_values = {f: default[f] for f in raw_features if f != target_feature}

        with col_chart:
            st.markdown(f"### 📈 {trans[target_feature]} 对屈曲载荷的影响")
            
            # 生成扫描数据
            x_vals = np.linspace(min_val, max_val, steps)
            # 构建 DataFrame
            plot_data = []
            for x in x_vals:
                row = base_values.copy()
                row[target_feature] = x
                plot_data.append(row)
            
            df_plot = pd.DataFrame(plot_data, columns=raw_features)
            
            # 调用模型预测
            try:
                with st.spinner("模型计算中..."):
                    y_vals = predict(df_plot)
                
                # 绘制结果
                df_result = pd.DataFrame({
                    trans[target_feature]: x_vals,
                    '预测屈曲载荷 (MPa)': y_vals
                })
                
                # 使用 Streamlit 自带的折线图
                st.line_chart(df_result, x=trans[target_feature], y='预测屈曲载荷 (MPa)', color="#00bcd4")
                
                # 添加一些结论性的文字
                st.success(f"""
                **规律洞察**：随着【{trans[target_feature]}】从 {min_val} 增加到 {max_val}，
                屈曲载荷从 **{y_vals[0]:.4f} MPa** 变化到了 **{y_vals[-1]:.4f} MPa**。
                变化斜率为 {(y_vals[-1] - y_vals[0]) / (max_val - min_val):.4f} MPa/单位。
                """)
            except Exception as e:
                st.error(f"计算出错: {e}")

    # --- Tab 2: 全局特征重要性 ---
    with tab2:
        st.markdown("### 全局特征重要性排名")
        st.markdown("基于随机森林模型内部评估机制，反映所有特征对屈曲载荷的整体影响权重。")
        
        # 提取特征重要性
        importances = model.feature_importances_
        feature_names_model = model.feature_names_in_
        
        # 构造 DataFrame 并排序
        df_imp = pd.DataFrame({
            '特征': [trans.get(f, f) for f in feature_names_model], # 转中文
            '重要性': importances
        }).sort_values(by='重要性', ascending=True) # 从小到大排，画图时大的在上面
        
        # 使用 Streamlit 的 bar_chart
        st.bar_chart(df_imp, x='特征', y='重要性', color="#28a745", horizontal=True)
        
        # 找出最重要的前两个特征
        top_2 = df_imp.iloc[-2:]['特征'].tolist()[::-1]
        st.info(f"💡 **工程建议**：从全局来看，影响封头屈曲载荷最关键的两个参数是 **{top_2[0]}** 和 **{top_2[1]}**。在实际制造和缺陷控制中，应优先严格把控这两个指标的加工公差。")
