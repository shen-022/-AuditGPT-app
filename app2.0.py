import logging
import warnings
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 配置警告和日志
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# --- 配置和常量 ---
class Config:
    """配置类"""
    PAGE_TITLE = "智能审计AI平台"
    APP_NAME = "智鉴AuditGPT"
    VERSION = "v2.0"
    MAX_FILE_SIZE = 200  # MB
    SUPPORTED_FORMATS = ['csv', 'xlsx', 'xls']
    DEFAULT_CONTAMINATION = 0.02
    RANDOM_STATE = 42


# --- 页面配置 ---
st.set_page_config(
    page_title=Config.PAGE_TITLE,
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 自定义CSS样式 ---
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .anomaly-card {
        background: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .normal-card {
        background: #e8f5e8;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# --- 标题部分 ---
st.markdown(f"""
<div class="main-header">
    <h1>🔍 {Config.APP_NAME}</h1>
    <p>基于机器学习的智能审计异常检测平台 {Config.VERSION}</p>
</div>
""", unsafe_allow_html=True)


# --- 会话状态初始化 ---
def initialize_session_state():
    """初始化会话状态"""
    default_states = {
        'df': None,
        'processed_df': None,
        'anomalies_df': None,
        'model': None,
        'scaler': None,
        'X_scaled': None,
        'label_encoders': {},
        'feature_names': [],
        'analysis_complete': False,
        'file_name': None
    }

    for key, value in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_session_state()


# --- 数据处理函数 ---
class DataProcessor:
    """数据处理类"""

    @staticmethod
    def detect_data_types(df: pd.DataFrame) -> Dict[str, List[str]]:
        """自动检测数据类型"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        datetime_cols = []

        # 尝试检测日期时间列
        for col in categorical_cols:
            sample = df[col].dropna().head(100)
            try:
                pd.to_datetime(sample)
                datetime_cols.append(col)
            except:
                pass

        # 从分类列中移除日期时间列
        categorical_cols = [col for col in categorical_cols if col not in datetime_cols]

        return {
            'numeric': numeric_cols,
            'categorical': categorical_cols,
            'datetime': datetime_cols
        }

    @staticmethod
    def preprocess_data(df: pd.DataFrame, target_features: List[str] = None) -> Tuple[pd.DataFrame, Dict]:
        """数据预处理"""
        processed_df = df.copy()
        encoders = {}

        # 自动检测特征类型
        data_types = DataProcessor.detect_data_types(processed_df)

        # 如果没有指定目标特征，自动选择数值型特征
        if target_features is None:
            target_features = data_types['numeric']

        # 处理分类变量
        for col in data_types['categorical']:
            if col in target_features:
                le = LabelEncoder()
                processed_df[col] = le.fit_transform(processed_df[col].astype(str))
                encoders[col] = le

        # 处理日期时间变量
        for col in data_types['datetime']:
            if col in target_features:
                processed_df[col] = pd.to_datetime(processed_df[col])
                # 提取有用的时间特征
                processed_df[f'{col}_hour'] = processed_df[col].dt.hour
                processed_df[f'{col}_dayofweek'] = processed_df[col].dt.dayofweek
                processed_df[f'{col}_month'] = processed_df[col].dt.month

                # 更新目标特征列表
                target_features.extend([f'{col}_hour', f'{col}_dayofweek', f'{col}_month'])
                if col in target_features:
                    target_features.remove(col)

        # 处理缺失值
        processed_df = processed_df.fillna(processed_df.mean(numeric_only=True))
        processed_df = processed_df.fillna('Unknown')

        return processed_df, encoders


class ModelAnalyzer:
    """模型分析类"""

    @staticmethod
    def train_isolation_forest(X: pd.DataFrame, contamination: float = Config.DEFAULT_CONTAMINATION) -> Tuple[
        IsolationForest, StandardScaler]:
        """训练Isolation Forest模型"""
        # 数据标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 训练模型
        model = IsolationForest(
            contamination=contamination,
            random_state=Config.RANDOM_STATE,
            n_estimators=100,
            max_samples='auto',
            n_jobs=-1
        )
        model.fit(X_scaled)

        return model, scaler

    @staticmethod
    def generate_advanced_report(anomaly_score: float, feature_names: List[str],
                                 sample_data: pd.Series) -> Tuple[List[str], List[Dict]]:
        """生成高级解释报告（简化版，不使用SHAP）"""
        report = []

        # 异常得分解释
        risk_level = "高风险" if anomaly_score < -0.5 else "中风险" if anomaly_score < -0.2 else "低风险"
        risk_color = "🔴" if risk_level == "高风险" else "🟡" if risk_level == "中风险" else "🟢"

        report.append("### 📊 异常风险评估")
        report.append(f"{risk_color} **风险等级**: {risk_level}")
        report.append(f"**异常得分**: {anomaly_score:.4f} (越低越异常)")
        report.append("")

        # 特征重要性分析（简化版）
        feature_impacts = []

        # 这里使用简单的特征值分析作为替代
        for feature in feature_names:
            if feature in sample_data.index:
                value = sample_data[feature]
                # 简单的启发式规则：极端值可能更重要
                if isinstance(value, (int, float)):
                    # 假设数值型特征的极端值更可能异常
                    impact = abs(value - sample_data[feature].mean()) / sample_data[feature].std() if sample_data[
                                                                                                          feature].std() > 0 else 0
                    direction = "推异常" if abs(value) > 2 else "推正常"  # 假设绝对值大于2标准差为异常
                    color = "🔴" if direction == "推异常" else "🔵"

                    feature_impacts.append({
                        '特征名称': feature,
                        '特征值': round(value, 4),
                        '影响方向': direction,
                        '影响程度': "高" if impact > 2 else "中" if impact > 1 else "低",
                        'SHAP值': round(impact, 4),
                        '重要性排名': 0,
                        '标志': color
                    })

        # 按影响值排序并分配排名
        feature_impacts.sort(key=lambda x: x['SHAP值'], reverse=True)
        for i, impact in enumerate(feature_impacts):
            impact['重要性排名'] = i + 1

        # 生成关键发现
        report.append("### 🎯 关键发现")

        anomaly_drivers = [f for f in feature_impacts if f['影响方向'] == '推异常'][:3]
        normal_drivers = [f for f in feature_impacts if f['影响方向'] == '推正常'][:3]

        if anomaly_drivers:
            report.append("**🔍 主要异常驱动因素：**")
            for i, driver in enumerate(anomaly_drivers, 1):
                report.append(f"{i}. **{driver['特征名称']}** = {driver['特征值']}")
                report.append(f"   - 影响方向: {driver['标志']} {driver['影响方向']}")
                report.append(f"   - 影响程度: {driver['影响程度']}")
            report.append("")

        if normal_drivers:
            report.append("**✅ 主要正常驱动因素：**")
            for i, driver in enumerate(normal_drivers, 1):
                report.append(f"{i}. **{driver['特征名称']}** = {driver['特征值']}")
                report.append(f"   - 影响方向: {driver['标志']} {driver['影响方向']}")
                report.append(f"   - 影响程度: {driver['影响程度']}")
            report.append("")

        # 审计建议
        report.append("### 💡 智能审计建议")
        if risk_level == "高风险":
            report.append("🚨 **紧急关注**")
            report.append("1. 立即进行详细审查")
            report.append("2. 核实所有相关单据和凭证")
            report.append("3. 联系相关业务人员确认")
            if anomaly_drivers:
                report.append(f"4. 重点检查 **{anomaly_drivers[0]['特征名称']}** 相关业务")
        elif risk_level == "中风险":
            report.append("⚠️ **重点关注**")
            report.append("1. 进行抽样复核")
            report.append("2. 与历史数据进行对比")
            report.append("3. 必要时进行进一步调查")
        else:
            report.append("ℹ️ **常规处理**")
            report.append("1. 按常规审计程序处理")
            report.append("2. 可作为对比基准")

        return report, feature_impacts


# --- 侧边栏配置 ---
with st.sidebar:
    st.markdown("### ⚙️ 模型参数配置")

    contamination = st.slider(
        "异常比例 (%)",
        min_value=0.1,
        max_value=10.0,
        value=2.0,
        step=0.1,
        help="预期数据中异常样本的比例"
    ) / 100

    st.markdown("### 📈 分析选项")
    show_data_profile = st.checkbox("显示数据概览", value=True)
    show_correlation = st.checkbox("显示特征相关性", value=True)
    auto_feature_selection = st.checkbox("自动特征选择", value=True)

# --- 主要内容区域 ---
st.header("📤 1. 数据上传与预览")

# 文件上传
uploaded_file = st.file_uploader(
    "请上传审计数据文件",
    type=Config.SUPPORTED_FORMATS,
    help=f"支持格式: {', '.join(Config.SUPPORTED_FORMATS)}，最大文件大小: {Config.MAX_FILE_SIZE}MB"
)

if uploaded_file is not None:
    try:
        # 读取文件
        if st.session_state.file_name != uploaded_file.name:
            with st.spinner('正在读取文件...'):
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)

                st.session_state.df = df
                st.session_state.file_name = uploaded_file.name
                st.session_state.analysis_complete = False

            st.success(f"✅ 文件 '{uploaded_file.name}' 上传成功！")

        df = st.session_state.df

        # 数据基本信息
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("总记录数", len(df))
        with col2:
            st.metric("特征数量", len(df.columns))
        with col3:
            st.metric("缺失值", df.isnull().sum().sum())
        with col4:
            st.metric("重复行", df.duplicated().sum())

        # 数据预览
        st.subheader("📋 数据预览")
        preview_rows = st.selectbox("显示行数", [5, 10, 20, 50], index=0)
        st.dataframe(df.head(preview_rows), use_container_width=True)

        # 数据概览
        if show_data_profile:
            with st.expander("📊 数据概览分析", expanded=False):
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("数据类型分布")
                    data_types = DataProcessor.detect_data_types(df)
                    type_counts = {k: len(v) for k, v in data_types.items()}

                    fig = px.pie(
                        values=list(type_counts.values()),
                        names=list(type_counts.keys()),
                        title="特征类型分布"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.subheader("缺失值统计")
                    missing_data = df.isnull().sum()
                    missing_data = missing_data[missing_data > 0]

                    if not missing_data.empty:
                        fig = px.bar(
                            x=missing_data.index,
                            y=missing_data.values,
                            title="各特征缺失值数量"
                        )
                        fig.update_xaxis(tickangle=45)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("🎉 数据完整，无缺失值！")

        # 特征选择
        st.markdown("---")
        st.header("🎯 2. 特征选择与配置")

        if auto_feature_selection:
            # 自动特征选择
            data_types = DataProcessor.detect_data_types(df)
            suggested_features = data_types['numeric']

            # 添加一些可能的分类特征
            categorical_features = [col for col in data_types['categorical']
                                    if df[col].nunique() < 20]  # 限制分类数量
            suggested_features.extend(categorical_features[:3])  # 最多添加3个分类特征

            st.info(f"🤖 自动推荐特征: {', '.join(suggested_features)}")
        else:
            suggested_features = []

        # 手动特征选择
        available_columns = df.columns.tolist()
        selected_features = st.multiselect(
            "选择用于异常检测的特征",
            available_columns,
            default=suggested_features,
            help="选择数值型特征效果更好，分类特征会自动编码"
        )

        if len(selected_features) < 2:
            st.warning("⚠️ 请至少选择2个特征进行异常检测")
        else:
            # 特征相关性分析
            if show_correlation and len(selected_features) > 2:
                with st.expander("📈 特征相关性分析", expanded=False):
                    numeric_features = df[selected_features].select_dtypes(include=[np.number]).columns
                    if len(numeric_features) > 1:
                        corr_matrix = df[numeric_features].corr()

                        fig = px.imshow(
                            corr_matrix,
                            title="特征相关性热力图",
                            color_continuous_scale="RdBu_r",
                            aspect="auto"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # 高相关性提醒
                        high_corr = np.where(np.abs(corr_matrix) > 0.8)
                        high_corr_pairs = [(corr_matrix.index[x], corr_matrix.columns[y])
                                           for x, y in zip(*high_corr) if x != y]
                        if high_corr_pairs:
                            st.warning(f"⚠️ 发现高相关性特征对: {high_corr_pairs[:3]}")

            # 模型训练
            st.markdown("---")
            st.header("🤖 3. 异常检测模型训练")

            if st.button("🚀 开始智能分析", type="primary"):
                with st.spinner('🔄 AI模型正在进行异常检测分析...'):
                    try:
                        # 数据预处理
                        processed_df, encoders = DataProcessor.preprocess_data(df, selected_features)

                        # 准备特征矩阵
                        feature_cols = []
                        for feature in selected_features:
                            if feature in processed_df.columns:
                                feature_cols.append(feature)
                            else:
                                # 查找可能的派生特征（如时间特征）
                                derived = [col for col in processed_df.columns if col.startswith(f'{feature}_')]
                                feature_cols.extend(derived)

                        X = processed_df[feature_cols]

                        # 训练模型
                        model, scaler = ModelAnalyzer.train_isolation_forest(X, contamination)

                        # 预测
                        X_scaled = scaler.transform(X)
                        predictions = model.predict(X_scaled)
                        anomaly_scores = model.decision_function(X_scaled)

                        # 保存结果
                        processed_df['异常标识'] = predictions
                        processed_df['异常得分'] = anomaly_scores
                        processed_df['异常判定'] = processed_df['异常标识'].apply(
                            lambda x: "异常" if x == -1 else "正常")

                        # 更新会话状态
                        st.session_state.processed_df = processed_df
                        st.session_state.anomalies_df = processed_df[processed_df['异常判定'] == '异常'].copy()
                        st.session_state.model = model
                        st.session_state.scaler = scaler
                        st.session_state.X_scaled = X_scaled
                        st.session_state.label_encoders = encoders
                        st.session_state.feature_names = feature_cols
                        st.session_state.analysis_complete = True

                        st.success("✅ 分析完成！")
                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ 分析过程中出现错误: {str(e)}")
                        st.info("请检查数据格式或联系技术支持")

    except Exception as e:
        st.error(f"❌ 文件读取失败: {str(e)}")

# --- 结果展示 ---
if st.session_state.analysis_complete and st.session_state.anomalies_df is not None:
    st.markdown("---")
    st.header("📊 4. 异常检测结果")

    anomalies_df = st.session_state.anomalies_df
    total_records = len(st.session_state.processed_df)
    anomaly_count = len(anomalies_df)
    anomaly_rate = (anomaly_count / total_records) * 100

    # 结果统计
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("总记录数", total_records)
    with col2:
        st.metric("异常记录数", anomaly_count, delta=f"{anomaly_count}")
    with col3:
        st.metric("异常比例", f"{anomaly_rate:.2f}%")
    with col4:
        avg_score = anomalies_df['异常得分'].mean()
        st.metric("平均异常得分", f"{avg_score:.3f}")

    # 异常记录展示
    st.subheader("📋 异常交易列表")
    st.dataframe(anomalies_df.head(20), use_container_width=True)

    if len(anomalies_df) > 20:
        st.info(f"共发现 {len(anomalies_df)} 条异常记录，仅显示前20条")

    # 异常得分分布
    st.subheader("📈 异常得分分布")
    fig = px.histogram(
        st.session_state.processed_df,
        x='异常得分',
        color='异常判定',
        title='异常得分分布',
        nbins=50,
        color_discrete_map={'异常': 'red', '正常': 'green'}
    )
    st.plotly_chart(fig, use_container_width=True)

    # 详细分析
    st.markdown("---")
    st.header("🔍 5. 详细分析")

    # 选择要分析的异常记录
    if not anomalies_df.empty:
        selected_index = st.selectbox(
            "选择要详细分析的异常交易",
            anomalies_df.index,
            format_func=lambda x: f"记录 {x} (得分: {anomalies_df.loc[x, '异常得分']:.3f})"
        )

        if selected_index:
            selected_anomaly = anomalies_df.loc[selected_index]
            st.subheader(f"📝 异常交易详情 - 记录 {selected_index}")

            # 显示详细信息
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**交易特征值:**")
                for feature in st.session_state.feature_names:
                    if feature in selected_anomaly:
                        st.write(f"- {feature}: {selected_anomaly[feature]}")

            with col2:
                st.markdown("**异常信息:**")
                st.write(f"- 异常得分: {selected_anomaly['异常得分']:.4f}")
                st.write(f"- 异常标识: {selected_anomaly['异常判定']}")

            # 生成解释报告
            if st.button("📊 生成详细解释报告"):
                with st.spinner('正在生成解释报告...'):
                    report, feature_impacts = ModelAnalyzer.generate_advanced_report(
                        selected_anomaly['异常得分'],
                        st.session_state.feature_names,
                        selected_anomaly
                    )

                    # 显示报告
                    st.markdown("---")
                    st.header("📋 异常交易解释报告")

                    for line in report:
                        st.markdown(line)

                    # 显示特征影响表格
                    st.subheader("📈 特征影响汇总表")
                    impact_df = pd.DataFrame(feature_impacts)
                    if not impact_df.empty:
                        st.dataframe(impact_df[['特征名称', '特征值', '影响方向', '影响程度', 'SHAP值', '重要性排名']])
                    else:
                        st.info("无法生成详细的特征影响分析")

# 如果没有上传文件，显示指引
else:
    st.info("👆 请上传审计数据文件开始分析")


    # 示例数据下载
    @st.cache_data
    def generate_sample_data():
        """生成示例数据"""
        np.random.seed(42)
        n_samples = 1000

        # 正常数据
        normal_data = {
            '交易金额': np.random.normal(1000, 300, n_samples),
            '交易时间间隔': np.random.exponential(5, n_samples),
            '账户历史交易笔数': np.random.poisson(50, n_samples),
            '交易类型': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
            '商户类别': np.random.choice(['A', 'B', 'C'], n_samples, p=[0.5, 0.3, 0.2])
        }

        # 异常数据
        n_anomalies = 20
        anomaly_data = {
            '交易金额': np.concatenate([
                np.random.uniform(5000, 10000, n_anomalies // 2),
                np.random.uniform(10, 50, n_anomalies // 2)
            ]),
            '交易时间间隔': np.random.uniform(0.1, 1, n_anomalies),
            '账户历史交易笔数': np.random.randint(1, 5, n_anomalies),
            '交易类型': np.random.choice([0, 1], n_anomalies, p=[0.8, 0.2]),
            '商户类别': np.random.choice(['A', 'D'], n_anomalies, p=[0.3, 0.7])
        }

        # 合并数据
        df_normal = pd.DataFrame(normal_data)
        df_anomaly = pd.DataFrame(anomaly_data)

        return pd.concat([df_normal, df_anomaly], ignore_index=True)


    sample_df = generate_sample_data()
    csv = sample_df.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="📥 下载示例数据 (sample_audit_data.csv)",
        data=csv,
        file_name='sample_audit_data.csv',
        mime='text/csv',
        help="点击下载示例审计数据进行测试"
    )