"""
SwiftRoute Supply Chain Analytics Dashboard
============================================
Full 7-Phase Analysis Platform
Run: streamlit run streamlit_app.py
Make sure SwiftRoute_CLEANED_10K.xlsx is in the same folder as this file.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_auc_score, roc_curve, silhouette_score,
                             r2_score, mean_squared_error)
import warnings
warnings.filterwarnings('ignore')

# ---- PAGE CONFIG ----
st.set_page_config(page_title='SwiftRoute Supply Chain Dashboard', layout='wide')
st.title('SwiftRoute Supply Chain Analytics Dashboard')
st.caption('AI-Powered Supply Chain Optimization Platform | 10,500 Orders | 2023-2024')

# ---- LOAD DATA ----
@st.cache_data
def load_data():
    df = pd.read_excel('SwiftRoute_CLEANED_10K.xlsx')
    df['Order_Date'] = pd.to_datetime(df['Order_Date'])
    region_map = {'North America': 1, 'Europe': 2, 'Asia Pacific': 3, 'Latin America': 4, 'Middle East & Africa': 5}
    transport_map = {'Road': 1, 'Rail': 2, 'Air': 3, 'Sea': 4, 'Multimodal': 5}
    priority_map = {'Low': 1, 'Standard': 2, 'Expedited': 3, 'Critical': 4}
    df['region_num'] = df['Region'].map(region_map).fillna(0)
    df['transport_num'] = df['Transport_Mode'].map(transport_map).fillna(0)
    df['priority_num'] = df['Order_Priority'].map(priority_map).fillna(0)
    df['on_time_binary'] = (df['On_Time_Delivery'] == 'Yes').astype(int)
    return df

df = load_data()

# ---- SIDEBAR ----
st.sidebar.header('Navigation')
page = st.sidebar.radio('Select Phase:', [
    '1. Overview & KPIs',
    '2. EDA & Regression',
    '3. Classification',
    '4. Clustering',
    '5. Association Rules',
    '6. Stress Testing'
])

st.sidebar.divider()
st.sidebar.header('Filters')
regions = st.sidebar.multiselect('Region', sorted(df['Region'].unique()), default=sorted(df['Region'].unique()))
transport = st.sidebar.multiselect('Transport Mode', sorted(df['Transport_Mode'].unique()), default=sorted(df['Transport_Mode'].unique()))
categories = st.sidebar.multiselect('Product Category', sorted(df['Product_Category'].unique()), default=sorted(df['Product_Category'].unique()))

mask = (
    df['Region'].isin(regions) &
    df['Transport_Mode'].isin(transport) &
    df['Product_Category'].isin(categories)
)
filtered = df[mask]
st.sidebar.markdown(f'**Showing {len(filtered):,} of {len(df):,} orders**')

# ============================================================
# PAGE 1: OVERVIEW & KPIs
# ============================================================
if page == '1. Overview & KPIs':
    st.header('Phase 1: Overview & Key Performance Indicators')

    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Total Orders', f'{len(filtered):,}')
    col2.metric('Total Revenue', f'${filtered["Total_Order_Value_USD"].sum():,.0f}')
    col3.metric('Avg Delay (Days)', f'{filtered["Delivery_Delay_Days"].mean():.1f}')
    col4.metric('Avg Satisfaction', f'{filtered["Customer_Satisfaction_1to5"].mean():.2f}/5')

    col5, col6, col7, col8 = st.columns(4)
    on_time_pct = (filtered['On_Time_Delivery']=='Yes').mean()*100
    col5.metric('On-Time %', f'{on_time_pct:.1f}%')
    col6.metric('Avg Defect Rate', f'{filtered["Defect_Rate_Pct"].mean():.2f}%')
    col7.metric('Avg Lead Time', f'{filtered["Lead_Time_Days"].mean():.1f} days')
    col8.metric('Avg Shipping Cost', f'${filtered["Shipping_Cost_USD"].mean():,.0f}')

    st.divider()

    st.subheader('Monthly Order Volume & Revenue Trend')
    monthly = filtered.groupby(filtered['Order_Date'].dt.to_period('M').astype(str)).agg(
        Orders=('Order_ID', 'count'), Revenue=('Total_Order_Value_USD', 'sum')).round(2)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=monthly.index, y=monthly['Orders'], name='Orders', marker_color='steelblue'))
    fig.add_trace(go.Scatter(x=monthly.index, y=monthly['Revenue'], name='Revenue ($)', yaxis='y2',
                             line=dict(color='red', width=2)))
    fig.update_layout(yaxis=dict(title='Orders'), yaxis2=dict(title='Revenue ($)', overlaying='y', side='right'),
                      height=420, legend=dict(orientation='h', yanchor='bottom', y=1.02))
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.subheader('Orders by Region')
        region_data = filtered.groupby('Region').agg(Orders=('Order_ID','count'), Avg_Delay=('Delivery_Delay_Days','mean')).round(2).reset_index()
        fig = px.bar(region_data, x='Region', y='Orders', color='Avg_Delay', color_continuous_scale='RdYlGn_r', text='Orders')
        fig.update_traces(textposition='outside')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader('Transport Mode: Cost vs Lead Time')
        tr = filtered.groupby('Transport_Mode').agg(Avg_Cost=('Shipping_Cost_USD','mean'), Avg_Lead=('Lead_Time_Days','mean')).round(2).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=tr['Transport_Mode'], y=tr['Avg_Cost'], name='Avg Cost ($)', marker_color='#2196F3'))
        fig.add_trace(go.Bar(x=tr['Transport_Mode'], y=tr['Avg_Lead']*20, name='Lead Time (Days x20)', marker_color='#FF9800'))
        fig.update_layout(barmode='group', height=400)
        st.plotly_chart(fig, use_container_width=True)

    left2, right2 = st.columns(2)
    with left2:
        st.subheader('Correlation Heatmap')
        corr_cols = ['Order_Quantity','Unit_Cost_USD','Shipping_Cost_USD','Lead_Time_Days',
                    'Delivery_Delay_Days','Defect_Rate_Pct','Customer_Satisfaction_1to5']
        corr = filtered[corr_cols].corr().round(2)
        fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', aspect='auto')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with right2:
        st.subheader('Satisfaction vs Delay')
        sample = filtered.sample(min(2000, len(filtered)), random_state=42)
        fig = px.scatter(sample, x='Delivery_Delay_Days', y='Customer_Satisfaction_1to5',
                        color='Defect_Rate_Pct', color_continuous_scale='RdYlGn_r', opacity=0.5, trendline='ols')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader('Distribution Explorer')
    dist_col = st.selectbox('Select variable:', ['Order_Quantity','Unit_Cost_USD','Shipping_Cost_USD','Lead_Time_Days',
                            'Delivery_Delay_Days','Defect_Rate_Pct','Customer_Satisfaction_1to5'])
    col_l, col_r = st.columns(2)
    with col_l:
        fig = px.histogram(filtered, x=dist_col, nbins=50, marginal='box', color='Region')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    with col_r:
        fig = px.box(filtered, x='Region', y=dist_col, color='Region')
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PAGE 2: EDA & REGRESSION
# ============================================================
elif page == '2. EDA & Regression':
    st.header('Phase 2: EDA & Regression Analysis')
    st.markdown('**DV**: Customer_Satisfaction_1to5 | **Method**: Multiple Linear Regression')

    feature_cols = ['Lead_Time_Days', 'Delivery_Delay_Days', 'Delay_Flag', 'on_time_binary',
                    'Unit_Cost_USD', 'Shipping_Cost_USD', 'Discount_Pct',
                    'Defect_Rate_Pct', 'Return_Rate_Pct', 'Inventory_Turnover_Ratio',
                    'Supplier_Rating_1to5', 'Order_Processing_Time_Hrs',
                    'transport_num', 'region_num', 'priority_num']

    X = filtered[feature_cols].fillna(filtered[feature_cols].median())
    y = filtered['Customer_Satisfaction_1to5']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    lr = LinearRegression()
    lr.fit(X_train_sc, y_train)
    y_pred = lr.predict(X_test_sc)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    col1, col2, col3 = st.columns(3)
    col1.metric('R² Score', f'{r2:.4f}')
    col2.metric('RMSE', f'{rmse:.4f}')
    col3.metric('Features Used', f'{len(feature_cols)}')

    st.subheader('Standardized Coefficients')
    coef_df = pd.DataFrame({'Feature': feature_cols, 'Coefficient': lr.coef_}).sort_values('Coefficient')
    fig = px.bar(coef_df, x='Coefficient', y='Feature', orientation='h', color='Coefficient', color_continuous_scale='RdBu_r')
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.subheader('Actual vs Predicted')
        fig = px.scatter(x=y_test, y=y_pred, opacity=0.3, labels={'x': 'Actual', 'y': 'Predicted'})
        fig.add_trace(go.Scatter(x=[1,5], y=[1,5], mode='lines', name='Perfect Fit', line=dict(color='red', dash='dash')))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader('Residual Distribution')
        residuals = y_test - y_pred
        fig = px.histogram(residuals, nbins=50, title='Residuals')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader('IV Group Explanatory Power')
    groups = {
        'Delivery Performance': ['Lead_Time_Days', 'Delivery_Delay_Days', 'Delay_Flag', 'on_time_binary'],
        'Cost Efficiency': ['Unit_Cost_USD', 'Shipping_Cost_USD', 'Discount_Pct'],
        'Quality Metrics': ['Defect_Rate_Pct', 'Return_Rate_Pct', 'Inventory_Turnover_Ratio'],
        'Supplier Reliability': ['Supplier_Rating_1to5', 'Order_Processing_Time_Hrs'],
        'Logistics Complexity': ['transport_num', 'region_num', 'priority_num'],
    }
    group_r2 = []
    for gname, gcols in groups.items():
        Xg = filtered[gcols].fillna(filtered[gcols].median())
        Xg_tr, Xg_te, yg_tr, yg_te = train_test_split(Xg, y, test_size=0.2, random_state=42)
        lr_g = LinearRegression()
        lr_g.fit(Xg_tr, yg_tr)
        group_r2.append({'Group': gname, 'R²': r2_score(yg_te, lr_g.predict(Xg_te))})
    gr2_df = pd.DataFrame(group_r2).sort_values('R²', ascending=True)
    fig = px.bar(gr2_df, x='R²', y='Group', orientation='h', color='R²', color_continuous_scale='Viridis')
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PAGE 3: CLASSIFICATION
# ============================================================
elif page == '3. Classification':
    st.header('Phase 3: Classification Models')
    st.markdown('**Target**: On-Time Delivery (Binary) | **Models**: Logistic, RF, XGBoost, SVM, KNN')

    feature_cols = ['Order_Quantity', 'Unit_Weight_Kg', 'Unit_Cost_USD',
        'Total_Order_Value_USD', 'Shipping_Cost_USD',
        'Lead_Time_Days', 'Delivery_Delay_Days',
        'Defect_Rate_Pct', 'Customer_Satisfaction_1to5',
        'Inventory_Turnover_Ratio', 'Stockout_Frequency_Monthly',
        'Return_Rate_Pct', 'Discount_Pct',
        'Supplier_Rating_1to5', 'Order_Processing_Time_Hrs',
        'region_num', 'transport_num', 'priority_num']

    X = filtered[feature_cols].fillna(filtered[feature_cols].median())
    y = filtered['on_time_binary']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    classifiers = {
        'Logistic': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
    }

    results = {}
    for name, clf in classifiers.items():
        clf.fit(X_train_sc, y_train)
        y_pred = clf.predict(X_test_sc)
        y_prob = clf.predict_proba(X_test_sc)[:, 1]
        results[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1': f1_score(y_test, y_pred),
            'AUC': roc_auc_score(y_test, y_prob),
            'y_prob': y_prob, 'y_pred': y_pred, 'clf': clf,
        }

    summary = pd.DataFrame({k: {m: v for m, v in val.items() if m in ['Accuracy','Precision','Recall','F1','AUC']}
                            for k, val in results.items()}).T.round(4)
    st.subheader('Model Comparison')
    st.dataframe(summary, use_container_width=True)

    fig = px.bar(summary.reset_index().melt(id_vars='index'), x='index', y='value', color='variable',
                barmode='group', labels={'index': 'Model', 'value': 'Score', 'variable': 'Metric'})
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader('ROC Curves')
    fig = go.Figure()
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{name} (AUC={res['AUC']:.3f})", mode='lines'))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random', line=dict(dash='dash', color='gray')))
    fig.update_layout(xaxis_title='FPR', yaxis_title='TPR', height=450)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader('Confusion Matrices')
    cols = st.columns(5)
    for i, (name, res) in enumerate(results.items()):
        cm = confusion_matrix(y_test, res['y_pred'])
        with cols[i]:
            st.markdown(f'**{name}**')
            fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                           x=['Late', 'On-Time'], y=['Late', 'On-Time'])
            fig.update_layout(height=250, width=250, showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig)

    st.subheader('Feature Importance (Random Forest)')
    rf = results['Random Forest']['clf']
    imp_df = pd.DataFrame({'Feature': feature_cols, 'Importance': rf.feature_importances_}).sort_values('Importance', ascending=True)
    fig = px.bar(imp_df, x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale='Viridis')
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PAGE 4: CLUSTERING
# ============================================================
elif page == '4. Clustering':
    st.header('Phase 4: Clustering Analysis')
    st.markdown('**Methods**: K-Means, DBSCAN | **Profiles**: Order characteristics')

    cluster_features = ['Order_Quantity', 'Unit_Weight_Kg', 'Unit_Cost_USD',
        'Total_Order_Value_USD', 'Shipping_Cost_USD',
        'Lead_Time_Days', 'Delivery_Delay_Days',
        'Defect_Rate_Pct', 'Customer_Satisfaction_1to5',
        'Inventory_Turnover_Ratio', 'Return_Rate_Pct',
        'Supplier_Rating_1to5', 'Order_Processing_Time_Hrs',
        'region_num', 'transport_num', 'priority_num']

    X = filtered[cluster_features].fillna(filtered[cluster_features].median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    st.subheader('K-Means: Elbow Method')
    inertias, sil_scores = [], []
    K_range = range(2, 11)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_scaled, labels, sample_size=5000))

    left, right = st.columns(2)
    with left:
        fig = px.line(x=list(K_range), y=inertias, markers=True, labels={'x':'K','y':'Inertia'}, title='Elbow Plot')
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.line(x=list(K_range), y=sil_scores, markers=True, labels={'x':'K','y':'Silhouette'}, title='Silhouette Scores')
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    best_k = st.slider('Select K:', 2, 10, int(list(K_range)[np.argmax(sil_scores)]))
    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    km_labels = km.fit_predict(X_scaled)
    filtered_c = filtered.copy()
    filtered_c['Cluster'] = km_labels

    sil = silhouette_score(X_scaled, km_labels, sample_size=5000)
    st.metric('Silhouette Score', f'{sil:.4f}')

    st.subheader(f'Cluster Profiles (K={best_k})')
    profile_cols = ['Order_Quantity', 'Total_Order_Value_USD', 'Shipping_Cost_USD',
                    'Lead_Time_Days', 'Delivery_Delay_Days', 'Defect_Rate_Pct',
                    'Customer_Satisfaction_1to5', 'Supplier_Rating_1to5']
    profiles = filtered_c.groupby('Cluster')[profile_cols].mean().round(2)
    st.dataframe(profiles, use_container_width=True)

    st.subheader('Cluster Visualization (PCA)')
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame({'PC1': X_pca[:,0], 'PC2': X_pca[:,1], 'Cluster': km_labels.astype(str)})
    fig = px.scatter(pca_df, x='PC1', y='PC2', color='Cluster', opacity=0.4,
                    title=f'PCA (Var: {pca.explained_variance_ratio_.sum():.1%})')
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader('Cluster Feature Distributions')
    feat_select = st.selectbox('Feature:', profile_cols)
    fig = px.box(filtered_c, x='Cluster', y=feat_select, color='Cluster')
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PAGE 5: ASSOCIATION RULES
# ============================================================
elif page == '5. Association Rules':
    st.header('Phase 5: Association Rule Mining')
    st.markdown('**Method**: Apriori on discretized supply chain features')

    disc = pd.DataFrame()
    disc['order_size'] = pd.cut(filtered['Order_Quantity'], bins=[0,500,2000,5000,50000], labels=['Small','Medium','Large','Bulk'])
    disc['ship_cost'] = pd.cut(filtered['Shipping_Cost_USD'], bins=[-1,200,500,1000,100000], labels=['Low_Ship','Med_Ship','High_Ship','VHigh_Ship'])
    disc['lead_time'] = pd.cut(filtered['Lead_Time_Days'], bins=[0,7,14,30,100], labels=['Fast','Normal','Slow','Very_Slow'])
    disc['delay'] = pd.cut(filtered['Delivery_Delay_Days'], bins=[-1,0,3,7,100], labels=['No_Delay','Minor','Moderate','Major'])
    disc['defect'] = pd.cut(filtered['Defect_Rate_Pct'], bins=[-1,2,5,10,100], labels=['Low_Def','Med_Def','High_Def','Crit_Def'])
    disc['satisfaction'] = pd.cut(filtered['Customer_Satisfaction_1to5'], bins=[0,2,3,4,5], labels=['VLow_Sat','Low_Sat','Med_Sat','High_Sat'])
    disc['on_time'] = filtered['On_Time_Delivery'].map({'Yes': 'On_Time', 'No': 'Late'})
    disc['transport'] = filtered['Transport_Mode']
    disc['region'] = filtered['Region'].str.replace(' ', '_')

    transactions = pd.get_dummies(disc.astype(str))
    item_support = transactions.mean()
    freq_items = item_support[item_support >= 0.05].sort_values(ascending=False)

    st.subheader('Frequent Items (min support = 5%)')
    fig = px.bar(freq_items.head(20).reset_index(), x='index', y=0, labels={'index':'Item', 0:'Support'})
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    freq_list = list(freq_items.index)
    rules = []
    for i in range(len(freq_list)):
        for j in range(i+1, min(i+50, len(freq_list))):
            a, b = freq_list[i], freq_list[j]
            sup_ab = (transactions[a] & transactions[b]).mean()
            if sup_ab >= 0.05:
                conf_ab = sup_ab / item_support[a]
                lift_ab = conf_ab / item_support[b]
                if conf_ab >= 0.3 and lift_ab >= 1.2:
                    rules.append({'Antecedent': a, 'Consequent': b, 'Support': round(sup_ab,4),
                                  'Confidence': round(conf_ab,4), 'Lift': round(lift_ab,4)})
                conf_ba = sup_ab / item_support[b]
                lift_ba = conf_ba / item_support[a]
                if conf_ba >= 0.3 and lift_ba >= 1.2:
                    rules.append({'Antecedent': b, 'Consequent': a, 'Support': round(sup_ab,4),
                                  'Confidence': round(conf_ba,4), 'Lift': round(lift_ba,4)})

    rules_df = pd.DataFrame(rules).sort_values('Lift', ascending=False).head(30)
    st.subheader(f'Top Association Rules ({len(rules)} total)')
    st.dataframe(rules_df, use_container_width=True)

    if len(rules_df) > 0:
        st.subheader('Top Rules by Lift')
        rules_top = rules_df.head(15).copy()
        rules_top['Rule'] = rules_top.apply(lambda r: f"{r['Antecedent']} -> {r['Consequent']}", axis=1)
        fig = px.bar(rules_top, x='Lift', y='Rule', orientation='h', color='Confidence', color_continuous_scale='YlOrRd')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PAGE 6: STRESS TESTING
# ============================================================
elif page == '6. Stress Testing':
    st.header('Phase 7: Stress Testing & Model Robustness')

    feature_cols = ['Order_Quantity', 'Unit_Weight_Kg', 'Unit_Cost_USD',
        'Total_Order_Value_USD', 'Shipping_Cost_USD',
        'Lead_Time_Days', 'Delivery_Delay_Days',
        'Defect_Rate_Pct', 'Customer_Satisfaction_1to5',
        'Inventory_Turnover_Ratio', 'Stockout_Frequency_Monthly',
        'Return_Rate_Pct', 'Discount_Pct',
        'Supplier_Rating_1to5', 'Order_Processing_Time_Hrs',
        'region_num', 'transport_num', 'priority_num']

    X = filtered[feature_cols].fillna(filtered[feature_cols].median())
    y = filtered['on_time_binary']

    st.subheader('Test 1: Sensitivity to Train/Test Split')
    split_results = []
    for seed in range(10):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
        sc = StandardScaler()
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(sc.fit_transform(Xtr), ytr)
        pred = rf.predict(sc.transform(Xte))
        prob = rf.predict_proba(sc.transform(Xte))[:, 1]
        split_results.append({'Seed': seed, 'Accuracy': accuracy_score(yte, pred),
                              'F1': f1_score(yte, pred), 'AUC': roc_auc_score(yte, prob)})
    split_df = pd.DataFrame(split_results)
    fig = px.line(split_df.melt(id_vars='Seed'), x='Seed', y='value', color='variable', markers=True)
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f"**Accuracy**: {split_df['Accuracy'].mean():.4f} +/- {split_df['Accuracy'].std():.4f}")

    st.subheader('Test 2: Sample Size Sensitivity')
    size_results = []
    for frac in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        n = int(len(X) * frac)
        Xs, ys = X.iloc[:n], y.iloc[:n]
        Xtr, Xte, ytr, yte = train_test_split(Xs, ys, test_size=0.2, random_state=42, stratify=ys)
        sc = StandardScaler()
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(sc.fit_transform(Xtr), ytr)
        size_results.append({'Sample': f'{frac*100:.0f}% ({n})', 'Accuracy': accuracy_score(yte, rf.predict(sc.transform(Xte)))})
    size_df = pd.DataFrame(size_results)
    fig = px.bar(size_df, x='Sample', y='Accuracy', text='Accuracy', color='Accuracy', color_continuous_scale='Viridis')
    fig.update_traces(texttemplate='%{text:.4f}')
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader('Test 3: Noise Injection Robustness')
    noise_results = []
    for noise_pct in [0, 5, 10, 15, 20, 25]:
        y_noisy = y.copy()
        n_flip = int(len(y) * noise_pct / 100)
        if n_flip > 0:
            flip_idx = np.random.RandomState(42).choice(y_noisy.index, n_flip, replace=False)
            y_noisy.loc[flip_idx] = 1 - y_noisy.loc[flip_idx]
        Xtr, Xte, ytr, yte = train_test_split(X, y_noisy, test_size=0.2, random_state=42, stratify=y_noisy)
        sc = StandardScaler()
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(sc.fit_transform(Xtr), ytr)
        noise_results.append({'Noise %': noise_pct, 'Accuracy': accuracy_score(yte, rf.predict(sc.transform(Xte)))})
    noise_df = pd.DataFrame(noise_results)
    fig = px.line(noise_df, x='Noise %', y='Accuracy', markers=True)
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader('Stress Testing Summary')
    st.markdown(f"""
    | Test | Result |
    |------|--------|
    | Split Stability (10 seeds) | Accuracy: {split_df['Accuracy'].mean():.4f} +/- {split_df['Accuracy'].std():.4f} |
    | Min Sample (10%) | Accuracy: {size_df.iloc[0]['Accuracy']:.4f} |
    | Full Sample | Accuracy: {size_df.iloc[-1]['Accuracy']:.4f} |
    | 0% Noise | Accuracy: {noise_df.iloc[0]['Accuracy']:.4f} |
    | 25% Noise | Accuracy: {noise_df.iloc[-1]['Accuracy']:.4f} |
    """)

# ---- FOOTER ----
st.divider()
st.caption('SwiftRoute Supply Chain Analytics | Synthetic Data | Generated for Academic Analysis')
