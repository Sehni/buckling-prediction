import streamlit as st
import pandas as pd
import numpy as np
import io
import joblib
import altair as alt
from datap import Data

# ==========================================
# 0. 准备模型和特征
# ==========================================
st.set_page_config(page_title="屈曲载荷预测系统", page_icon="🛡️", layout="wide")

@st.cache_resource
def load_assets():
    model = joblib.load('best_model_xgb.pkl')
    data = Data()
    return model, data

model, data = load_assets()

feature_names = model.feature_names_in_
raw_features = data.features.copy()  
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
    'smax': '局部缺陷最大外凸（单位：mm）', 
    'top/d': '顶点内凹(相对)',
    'crown/d': '球冠区最大内凹(相对)',
    't/d': '壁厚(相对)',
    'knuckle/d': '过渡区整体缺陷最大外凸(相对)', 
    'smax/d': '局部缺陷最大外凸(相对)'
}
default = data.default

# ==========================================
# 💡 全局配置：校验边界与名称映射
# ==========================================
CHECK_BOUNDS = {
    'top/d': (-0.0064, 0.0), 
    'crown/d': (-0.0064, 0.0), 
    'fai': (17.0, 30.0), 
    'material': (205.0, 415.0), 
    'd/2h': (1.728, 2.25), 
    't/d': (0.0005, 0.0012), 
    'knuckle/d': (0.0, 0.0125), 
    'xita': (0.0, 15.0), 
    'smax/d': (0.0, 0.016)
}
CHECK_TRANS = {
    'top/d': '顶点内凹与直径比', 'crown/d': '球冠区最大内凹与直径比', 
    'fai': '过渡区外凸范围', 'material': '材料屈服强度', 
    'd/2h': '径高比', 't/d': '壁厚与直径比', 
    'knuckle/d': '过渡区整体缺陷最大外凸与直径比', 
    'xita': '过渡区局部缺陷环向范围', 'smax/d': '局部缺陷最大外凸与直径比'
}

# ==========================================
# 💡 真实无量纲样本空间字典
# ==========================================
DOT = {
    'top/d': np.array([0., -0.002, -0.0064]),
    'crown/d': np.array([-0.001, -0.002, -0.004, -0.0064, 0.]),
    'fai': np.array([20., 30., 24., 17.]),
    'material': np.array([205., 345., 415.]),
    'd/2h': np.array([2., 2.25, 1.75, 1.728, 2.2]),
    't/d': np.array([0.0005, 0.0006, 0.0008, 0.001, 0.0012]),
    'knuckle/d': np.array([0., 0.001, 0.002, 0.004, 0.006, 0.009, 0.0125]),
    'xita': np.array([10., 0., 15., 5., 2.5]),
    'smax/d': np.array([0., 0.001, 0.002, 0.003, 0.004, 0.006, 0.009, 0.0125, 0.016])
}

DOT_TO_RAW = {
    'top/d': 'top', 'crown/d': 'crown', 'fai': 'fai',
    'material': 'material', 'd/2h': 'd/2h', 't/d': 't',
    'knuckle/d': 'knuckle', 'xita': 'xita', 'smax/d': 'smax'
}

def check(df_processed):
    """校验输入数据是否严重偏离训练集物理边界"""
    for col, (min_v, max_v) in CHECK_BOUNDS.items():
        if col in df_processed.columns:
            if ((df_processed[col] < min_v) | (df_processed[col] > max_v)).any():
                raise ValueError(f"输入偏离常识：【{CHECK_TRANS[col]}】应在 {min_v} 到 {max_v} 之间。预测已阻断！")

def predict(input_df):
    dt = Data()
    input_df['buckling'] = 0 
    
    # 1. 进行无量纲转换
    dt.translate(input_df)
    df_processed = dt.good_dfs[0]
    df_processed = df_processed[model.feature_names_in_].copy()
    
    # 2. 边界校验
    check(df_processed)
    
    # 3. 返回预测结果
    return model.predict(df_processed)

# ==========================================
# 1. 侧边栏导航
# ==========================================
st.sidebar.title("🛡️ 导航菜单")
page = st.sidebar.radio("请选择功能", ["首页", "单次参数预测", "批量 Excel 预测", "敏感性分析", "连续插值敏感性分析"])

# ==========================================
# 2. 页面一：欢迎页
# ==========================================
if page == "首页":
    st.markdown("""
        <div style="text-align: center; margin-top: 100px;">
            <h1 style="color: #2c3e50; font-size: 40px; font-weight: bold; margin-bottom: 15px;">
                内压薄壁封头屈曲载荷预测系统
            </h1>
            <p style="color: #7f8c8d; font-size: 20px; margin-bottom: 60px;">
                基于 XGBoost 与贝叶斯优化
            </p>
        </div>
    """, unsafe_allow_html=True)
    st.info("👈 请在左侧侧边栏选择对应的功能开始使用。")

# ==========================================
# 3. 页面二：单次预测页
# ==========================================
elif page == "单次参数预测":
    st.title("单次参数预测")
    
    col_left, col_right = st.columns([1.2, 1])
    
    with col_left:
        st.markdown("### 输入特征参数")
        with st.form("single_predict_form"):
            input_values = {}
            col_a, col_b = st.columns(2)
            
            for i, name in enumerate(raw_features):
                with col_a if i % 2 == 0 else col_b:
                    input_values[name] = st.number_input(
                        label=f"{trans[name]}:",
                        value=float(default[name]),
                        step=0.1,
                        format="%.2f"
                    )
            
            submit_btn = st.form_submit_button("🚀 开始预测", use_container_width=True)
            
    with col_right:
        st.markdown("### 预测结果")
        result_placeholder = st.empty()
        
        if submit_btn:
            with st.spinner("正在计算中..."):
                try:
                    input_df = pd.DataFrame([input_values], columns=raw_features)
                    result = predict(input_df)[0]
                    
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
# 4. 页面三：批量预测页（逐条校验+界面报错版）
# ==========================================
elif page == "批量 Excel 预测":
    st.title("批量 Excel 数据预测")
    
    th_style = "border: 1px solid #ddd; padding: 10px; background-color: #f2f2f2; font-weight: bold;"
    td_style = "border: 1px solid #ddd; padding: 10px; text-align: center;"
    header_html = "".join([f"<th style='{th_style}'>{trans.get(col, col)}</th>" for col in raw_features])
    default_html = "".join([f"<td style='{td_style}'>{default.get(col, 0)}</td>" for col in raw_features])
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

    uploaded_file = st.file_uploader("📁 上传数据文件", type=["xlsx", "xls"])
    
    if uploaded_file is not None:
        if st.button("开始批量预测", type="primary"):
            with st.spinner("正在逐条校验并计算，请稍候..."):
                try:
                    df_upload = pd.read_excel(uploaded_file)
                    new_cols = df_upload.columns
                    missing_cols_count = len(raw_features) - len(new_cols)
                    if missing_cols_count != 0:
                        st.error(f"❌ 文件缺少 {missing_cols_count} 列。请确保特征数与示例一致！")
                        st.stop()
                        
                    # 强制替换列名，仅提取前10列特征
                    df_upload = df_upload[new_cols[:len(raw_features)]]
                    df_upload.columns = raw_features
                    
                    predictions = []
                    error_logs = [] # 收集错误信息用于界面展示
                    success_count = 0
                    fail_count = 0
                    
                    progress_bar = st.progress(0)
                    total_rows = len(df_upload)
                    
                    # 逐行遍历
                    for idx, row in df_upload.iterrows():
                        single_df = pd.DataFrame([row], columns=raw_features)
                        
                        # Excel 中的行号（表头是第1行，所以数据行从 2 开始）
                        excel_row_num = idx + 2 
                        
                        try:
                            pred_val = predict(single_df)[0]
                            predictions.append(pred_val)
                            success_count += 1
                        except Exception as e:
                            # 如果是越界报错，预测值填充 NaN，并记录行号和原因
                            predictions.append(np.nan) 
                            error_logs.append(f"第 {excel_row_num} 行：{str(e)}")
                            fail_count += 1
                        
                        progress_bar.progress((idx + 1) / total_rows)
                        
                    # 还原表头并追加预测结果列
                    df_upload.columns = new_cols[:len(raw_features)]
                    df_upload['预测屈曲载荷(MPa)'] = predictions
                    
                    # 根据是否成功给出提示
                    if fail_count > 0:
                        st.warning(f"⚠️ 处理完成！成功预测 {success_count} 条，因参数越界拦截 {fail_count} 条。拦截行的预测结果已留空，详情请查看下方报错说明。")
                        # 将报错信息放入折叠面板中，避免界面太乱
                        with st.expander("🔍 查看被拦截的具体行数与原因", expanded=True):
                            for log in error_logs:
                                st.write(log)
                    else:
                        st.success(f"✅ 全部成功！共处理了 {success_count} 条数据。")
                    
                    # 生成下载流
                    output_stream = io.BytesIO()
                    with pd.ExcelWriter(output_stream, engine='openpyxl') as writer:
                        df_upload.to_excel(writer, index=False)
                    
                    st.download_button(
                        label="📥 点击下载预测结果 (Excel)",
                        data=output_stream.getvalue(),
                        file_name="预测结果.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        type="primary"
                    )
                    
                except Exception as e:
                    st.error(f"❌ 文件读取或系统处理失败：{e}")

# ==========================================
# 5. 页面四：敏感性分析 (含离散分析与特征重要性)
# ==========================================
elif page == "敏感性分析":
    st.title("📊 缺陷参数敏感性分析")
    
    tab1, tab2 = st.tabs(["📈 真实离散样本敏感性曲线", "🏆 全局特征重要性"])
    
    # --- Tab 1: 真实离散样本敏感性曲线 ---
    with tab1:
        st.markdown("""
        **本模块完全基于真实训练样本空间。**
        """)
        
        col_ctrl, col_chart = st.columns([1, 2.5])
        
        with col_ctrl:
            st.markdown("### ⚙️ 实验设置")
            
            raw_scan_options = list(DOT_TO_RAW.values())
            target_feature = st.selectbox(
                "选择要扫描的参数", 
                options=raw_scan_options,
                format_func=lambda x: trans.get(x, x)
            )
            
            st.markdown("---")
            st.markdown("### 📌 基准参数设置")
            st.caption("除直径外，其余参数只能从真实样本库中选取：")
            
            base_values = {}
            
            if target_feature != 'd':
                base_d = st.number_input(
                    "基准直径 d (mm)", 
                    min_value=1000.0, 
                    max_value=10000.0, 
                    value=float(default.get('d', 5000.0)), 
                    step=100.0
                )
                base_values['d'] = base_d
            else:
                base_d = float(default.get('d', 5000.0))
            
            with st.expander("展开修改其他基准参数", expanded=True):
                for f in raw_features:
                    if f != target_feature and f != 'd':
                        dot_key = None
                        for k, v in DOT_TO_RAW.items():
                            if v == f:
                                dot_key = k
                                break
                        
                        if dot_key and dot_key in DOT:
                            options = DOT[dot_key]
                            if '/d' in dot_key:
                                display_options = [round(val * base_d, 4) for val in options]
                            else:
                                display_options = options
                            
                            base_values[f] = st.selectbox(
                                f"{trans.get(f, f)}",
                                options=display_options,
                                index=0
                            )
                        else:
                            base_values[f] = st.number_input(f"{trans.get(f, f)}", value=float(default.get(f, 0)))

        with col_chart:
            st.markdown(f"### 📈 {trans.get(target_feature, target_feature)} 对屈曲载荷的影响")
            
            try:
                target_dot_key = None
                for k, v in DOT_TO_RAW.items():
                    if v == target_feature:
                        target_dot_key = k
                        break
                        
                if target_dot_key and target_dot_key in DOT:
                    ratio_x_vals = DOT[target_dot_key]
                else:
                    st.error("未找到该参数的离散样本点！")
                    st.stop()
                    
                if '/d' in target_dot_key:
                    x_vals_physical = [round(val * base_d, 4) for val in ratio_x_vals]
                else:
                    x_vals_physical = ratio_x_vals
                    
                plot_data = []
                for x in x_vals_physical:
                    row = base_values.copy()
                    row[target_feature] = x
                    plot_data.append(row)
                
                df_plot = pd.DataFrame(plot_data, columns=raw_features)
                
                with st.spinner("模型计算中..."):
                    y_vals = predict(df_plot)
                
                df_result = pd.DataFrame({
                    trans.get(target_feature, target_feature): x_vals_physical,
                    '预测屈曲载荷 (MPa)': y_vals
                })
                
                y_min = float(min(y_vals)) * 0.95 if min(y_vals) > 0 else float(min(y_vals)) * 1.05
                y_max = float(max(y_vals)) * 1.05

                points = alt.Chart(df_result).mark_circle(
                    size=120, color='#e74c3c', opacity=0.9, stroke='white', strokeWidth=2
                ).encode(
                    x=alt.X(trans.get(target_feature, target_feature), title=trans.get(target_feature, target_feature), scale=alt.Scale(nice=True)),
                    y=alt.Y('预测屈曲载荷 (MPa)', title='屈曲载荷 (MPa)', scale=alt.Scale(domain=[y_min, y_max])),
                    tooltip=[trans.get(target_feature, target_feature), '预测屈曲载荷 (MPa)']
                )
                
                line = alt.Chart(df_result).mark_line(
                    color='#f39c12', strokeWidth=3, opacity=0.8
                ).encode(
                    x=trans.get(target_feature, target_feature),
                    y='预测屈曲载荷 (MPa)'
                )
                
                st.altair_chart((line + points).interactive(), use_container_width=True)
                
                st.success(f"""
                **真实离散响应：**
                *折线上的每一个红点均代表一组真实的训练工况，由无量纲样本反推得出。*
                """)
                
            except Exception as e:
                st.error(f"计算出错: {e}")

    # --- Tab 2: 全局特征重要性 ---
    with tab2:
        st.markdown("### 全局特征重要性排名")
        st.markdown("基于模型内部评估机制，反映所有特征对屈曲载荷的整体影响权重。")
        
        importances = model.feature_importances_
        feature_names_model = model.feature_names_in_
        
        df_imp = pd.DataFrame({
            '特征': [trans.get(f, f) for f in feature_names_model], 
            '重要性': importances
        }).sort_values(by='重要性', ascending=True) 
        
        chart = alt.Chart(df_imp).mark_bar(color='#28a745').encode(
            x=alt.X('重要性:Q', title='重要性得分'),
            y=alt.Y('特征:N', sort='-x', title=None, axis=alt.Axis(labelLimit=400)),
            tooltip=['特征', '重要性']
        ).properties(height=400)
        
        st.altair_chart(chart, use_container_width=True)

# ==========================================
# 6. 页面五：测试：连续敏感性分析
# ==========================================
elif page == "连续插值敏感性分析":
    st.title("📊 连续缺陷参数敏感性分析")
    st.markdown("""
    **基于离散样本的保守响应**
    由于使用基于决策树的模型，强行连续插值会导致曲线呈现阶梯状，结论仅供参考。
    """)
    
    col_ctrl, col_chart = st.columns([1, 2.5])
    
    with col_ctrl:
        st.markdown("### ⚙️ 实验设置")
        st.info("保持其他参数不变，单独调节一个参数，观察载荷变化。")
        
        scan_features = [f for f in raw_features if f != 'd']
        
        target_feature = st.selectbox(
            "选择要分析的参数", 
            options=scan_features,
            format_func=lambda x: trans.get(x, x), 
            index=scan_features.index('top') if 'top' in scan_features else 0
        )
        
        st.markdown("---")
        st.markdown("### 📌 其他基准参数")
        st.caption("其他参数保持默认值或在此微调：")
        
        base_values = {}
        with st.expander("展开修改基准参数"):
            for f in raw_features:
                if f != target_feature:
                    base_values[f] = st.number_input(f"{trans.get(f, f)}", value=float(default.get(f, 0)), step=0.1)
        if not base_values:
            base_values = {f: default.get(f, 0) for f in raw_features if f != target_feature}

        st.markdown("---")
        st.write("设置扫描范围：")
        
        base_d = base_values.get('d', default.get('d', 5000))
        feature_to_bound_key = {
            'top': 'top/d', 'crown': 'crown/d', 't': 't/d',
            'knuckle': 'knuckle/d', 'smax': 'smax/d',
            'material': 'material', 'd/2h': 'd/2h', 'fai': 'fai', 'xita': 'xita'
        }
        b_key = feature_to_bound_key.get(target_feature)
        
        if b_key and b_key in CHECK_BOUNDS:
            limit_min = float(CHECK_BOUNDS[b_key][0])
            limit_max = float(CHECK_BOUNDS[b_key][1])
            if target_feature in ['top', 'crown', 't', 'knuckle', 'smax']:
                limit_min *= base_d
                limit_max *= base_d
        else:
            limit_min, limit_max = 0.0, 10000.0

        limit_min = float(limit_min)
        limit_max = float(limit_max)
        if limit_max <= limit_min:
            limit_max = limit_min + 0.1

        range_gap = limit_max - limit_min
        auto_step = float(f"{range_gap / 50.0:.4f}") if range_gap < 10.0 else 0.1
        if auto_step <= 0:
            auto_step = 0.01
            
        min_val = st.number_input("最小值", min_value=limit_min, max_value=limit_max, value=limit_min, step=auto_step)
        max_val = st.number_input("最大值", min_value=min_val + auto_step, max_value=limit_max, value=limit_max, step=auto_step)
        
        steps = st.slider("采样点数", min_value=5, max_value=50, value=20, step=5)

    with col_chart:
        st.markdown(f"### 📈 {trans.get(target_feature, target_feature)} 对屈曲载荷的影响")
        
        x_vals = np.linspace(min_val, max_val, steps)
        plot_data = []
        for x in x_vals:
            row = base_values.copy()
            row[target_feature] = x
            plot_data.append(row)
        
        df_plot = pd.DataFrame(plot_data, columns=raw_features)
        
        try:
            with st.spinner("模型计算中..."):
                y_vals = predict(df_plot)
            
            df_result = pd.DataFrame({
                trans.get(target_feature, target_feature): x_vals,
                '预测屈曲载荷 (MPa)': y_vals
            })
            
            y_min_plot = float(min(y_vals)) * 0.9 if min(y_vals) > 0 else float(min(y_vals)) * 1.1
            y_max_plot = float(max(y_vals)) * 1.1

            points = alt.Chart(df_result).mark_circle(
                size=80, color='#007bff', opacity=0.8, stroke='white', strokeWidth=1.5
            ).encode(
                x=alt.X(trans.get(target_feature, target_feature), title=trans.get(target_feature, target_feature)),
                y=alt.Y('预测屈曲载荷 (MPa)', title='屈曲载荷 (MPa)', scale=alt.Scale(domain=[y_min_plot, y_max_plot])),
                tooltip=[trans.get(target_feature, target_feature), '预测屈曲载荷 (MPa)']
            )
            
            line = alt.Chart(df_result).mark_line(
                color='#00bcd4', strokeWidth=3, opacity=0.7
            ).encode(
                x=trans.get(target_feature, target_feature),
                y='预测屈曲载荷 (MPa)'
            )
            
            st.altair_chart((line + points).interactive(), use_container_width=True)
            
            st.success(f"""
            随着【{trans.get(target_feature, target_feature)}】从 **{min_val:.3g}** 增加到 **{max_val:.3g}**，
            屈曲载荷从 **{y_vals[0]:.4g} MPa** 变化到了 **{y_vals[-1]:.4g} MPa**。
            """)
        except Exception as e:
            st.error(f"计算出错: {e}")
