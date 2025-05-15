import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import os  # 추가: NameError 해결
import geopandas as gpd
import requests
from io import BytesIO
from folium.plugins import MarkerCluster
import numpy as np

st.set_page_config(page_title="이상동기 범죄 경보 맵", page_icon="🚨", layout="wide", initial_sidebar_state="expanded")

@st.cache_data
def load_data(crime_path, indicator_path, prediction_path):
    # 최적화: 필요한 열만 로드, 결측치 처리 간소화
    if not os.path.exists(crime_path):
        st.error("범죄 데이터 파일 없음")
        st.stop()
    df_crime = pd.read_csv(crime_path, encoding='cp949', usecols=['날짜', '위도', '경도'])
    df_crime['date'] = pd.to_datetime(df_crime['날짜'], errors='co obiekcji')
    df_crime['위도'] = pd.to_numeric(df_crime['위도'], errors='coerce')
    df_crime['경도'] = pd.to_numeric(df_crime['경도'], errors='coerce')
    df_crime = df_crime.dropna(subset=['date', '위도', '경도'])
    if 'full_address' not in df_crime.columns:
        df_crime['full_address'] = df_crime[['위도', '경도']].apply(lambda x: f"위도: {x['위도']}, 경도: {x['경도']}", axis=1)
    crime_dates = sorted(df_crime['date'].dt.normalize().unique())
    
    if not os.path.exists(indicator_path):
        st.error("지표 데이터 파일 없음")
        st.stop()
    df_indicator = pd.read_csv(indicator_path, encoding='cp949')
    df_indicator['date'] = pd.to_datetime(df_indicator['date'], errors='coerce')
    df_indicator = df_indicator[df_indicator['date'].dt.year <= 2023].dropna(subset=['date'])
    indicator_dates = sorted(df_indicator['date'].dt.normalize().unique())
    
    if not os.path.exists(prediction_path):
        st.error("예측 데이터 파일 없음")
        st.stop()
    df_prediction = pd.read_csv(prediction_path, encoding='cp949', usecols=['date', '도단위', 'crime_probability'])
    df_prediction['date'] = pd.to_datetime(df_prediction['date'], errors='coerce')
    df_prediction = df_prediction.dropna(subset=['date', '도단위', 'crime_probability'])
    prediction_dates = sorted(df_prediction['date'].dt.normalize().unique())
    
    return df_crime, crime_dates, df_indicator, indicator_dates, df_prediction, prediction_dates

@st.cache_data(hash_funcs={gpd.GeoDataFrame: lambda x: str(x)})
def load_geojson():
    # 최적화: GeoJSON을 로컬 파일로 변경하거나 URL 캐싱
    geojson_url = "https://raw.githubusercontent.com/southkorea/southkorea-maps/master/gadm/json/skorea-provinces-geo.json"
    response = requests.get(geojson_url, timeout=5)
    geo_data = gpd.read_file(BytesIO(response.content))
    return geo_data

def calculate_risk_score(row, region):
    score = 0
    if f"기후스트레스:{region}" in row and pd.notna(row[f"기후스트레스:{region}"]) and row[f"기후스트레스:{region}"] > 13:
        score += 1
    if f"사회스트레스:{region}" in row and pd.notna(row[f"사회스트레스:{region}"]) and row[f"사회스트레스:{region}"] >= 0.7:
        score += 1
    if '금융스트레스' in row and pd.notna(row['금융스트레스']) and row['금융스트레스'] >= 2:
        score += 1
    return min(score, 3)

@st.cache_data
def create_risk_score_table(indicator_data, view_type, selected_year=None, selected_date=None):
    regions = ['서울특별시', '경기도', '강원도', '경상남도', '경상북도', '광주광역시', '대구광역시', '대전광역시', 
               '부산광역시', '세종특별자치시', '울산광역시', '인천광역시', '전라남도', '전라북도', '제주도', 
               '충청남도', '충청북도']
    
    # 최적화: 벡터화로 계산 속도 개선
    def compute_scores(data, region):
        climate_col = f"기후스트레스:{region}"
        social_col = f"사회스트레스:{region}"
        climate_score = (data[climate_col].notna() & (data[climate_col] > 13)).astype(float).mean() if climate_col in data else 0
        social_score = (data[social_col].notna() & (data[social_col] >= 0.7)).astype(float).mean() if social_col in data else 0
        financial_score = (data['금융스트레스'].notna() & (data['금융스트레스'] >= 2)).astype(float).mean() if '금융스트레스' in data else 0
        return climate_score, social_score, financial_score
    
    table_data = []
    for region in regions:
        region_data = indicator_data
        if view_type == "년도별":
            region_data = region_data[region_data['date'].dt.year == selected_year]
        elif view_type == "일별":
            region_data = region_data[region_data['date'].dt.normalize() == selected_date.normalize()]
        
        if region_data.empty:
            table_data.append({
                '지역': region,
                '기후스트레스 점수': 0,
                '사회스트레스 점수': 0,
                '금융스트레스 점수': 0,
                '총 점수': 0
            })
            continue
        
        if view_type in ["전체 데이터", "년도별"]:
            climate_score, social_score, financial_score = compute_scores(region_data, region)
        else:  # 일별
            row = region_data.iloc[0]
            climate_score = 1 if pd.notna(row.get(f"기후스트레스:{region}")) and row[f"기후스트레스:{region}"] > 13 else 0
            social_score = 1 if pd.notna(row.get(f"사회스트레스:{region}")) and row[f"사회스트레스:{region}"] >= 0.7 else 0
            financial_score = 1 if pd.notna(row.get('금융스트레스')) and row['금융스트레스'] >= 2 else 0
        
        total_score = min(climate_score + social_score + financial_score, 3)
        table_data.append({
            '지역': region,
            '기후스트레스 점수': round(climate_score, 2),
            '사회스트레스 점수': round(social_score, 2),
            '금융스트레스 점수': round(financial_score, 2),
            '총 점수': round(total_score, 2)
        })
    
    return pd.DataFrame(table_data)

@st.cache_data
def create_prediction_table(prediction_data, selected_year=None, selected_date=None, prediction_mode="년도별"):
    regions = ['서울특별시', '경기도', '강원도', '경상남도', '경상북도', '광주광역시', '대구광역시', '대전광역시', 
               '부산광역시', '세종특별자치시', '울산광역시', '인천광역시', '전라남도', '전라북도', '제주도', 
               '충청남도', '충청북도']
    
    # 최적화: groupby로 지역별 계산 간소화
    table_data = []
    for region in regions:
        region_data = prediction_data[prediction_data['도단위'] == region]
        if selected_year and prediction_mode == "년도별":
            region_data = region_data[region_data['date'].dt.year == selected_year]
        elif selected_date and prediction_mode == "일별":
            region_data = region_data[region_data['date'].dt.normalize() == selected_date.normalize()]
        
        if region_data.empty:
            table_data.append({'지역': region, '위험률': '데이터 없음'})
            continue
        
        prob = region_data['crime_probability'].mean() if prediction_mode == "년도별" else region_data['crime_probability'].iloc[0]
        table_data.append({'지역': region, '위험률': f"{prob:.3f}" if pd.notna(prob) else '데이터 없음'})
    
    return pd.DataFrame(table_data)

def get_prediction_color(prob):
    if prob is None:
        return 'gray'
    return 'green' if prob < 0.3 else 'lime' if prob < 0.5 else 'yellow' if prob < 0.7 else 'orange' if prob < 0.85 else 'red'

def create_map(view_type, selected_year=None, selected_date=None, df_crime=None, df_indicator=None, df_prediction=None, geo_data=None):
    # 최적화: 초기 줌 레벨 낮추고, 마커 수 제한 강화
    m = folium.Map(location=[36.5, 127.5], zoom_start=6, tiles='CartoDB Positron')
    crime_group = folium.FeatureGroup(name="범죄 마커", show=(view_type != "예측"))
    risk_group = folium.FeatureGroup(name="위험 코로플렛", show=True)
    marker_cluster = MarkerCluster(maxClusterRadius=30).add_to(crime_group)
    
    if view_type == "전체 데이터":
        crime_data = df_crime
        indicator_data = df_indicator
        title = "전체 데이터 맵"
    elif view_type == "년도별":
        crime_data = df_crime[df_crime['date'].dt.year == selected_year]
        indicator_data = df_indicator[df_indicator['date'].dt.year == selected_year]
        title = f"{selected_year}년 맵"
    elif view_type == "일별":
        selected_date_only = selected_date.normalize()
        crime_data = df_crime[df_crime['date'].dt.normalize() == selected_date_only]
        indicator_data = df_indicator[df_indicator['date'].dt.normalize() == selected_date_only]
        title = f"{selected_date_only.strftime('%Y-%m-%d')} 맵"
    else:  # 예측 모드
        crime_data = pd.DataFrame()
        if selected_date:
            indicator_data = df_prediction[df_prediction['date'].dt.normalize() == selected_date.normalize()]
        else:
            indicator_data = df_prediction[df_prediction['date'].dt.year == selected_year]
        title = f"{selected_year}년 예측 맵" if selected_year else f"{selected_date.strftime('%Y-%m-%d')} 예측 맵"
    
    if view_type != "예측" and not crime_data.empty:
        # 최적화: 마커 수 1000개로 제한
        crime_data_limited = crime_data.head(1000)
        point_count = len(crime_data)
        marker_color = 'green' if point_count < 100 else 'orange' if point_count < 500 else 'red'
        for _, row in crime_data_limited.iterrows():
            if pd.notna(row['위도']) and pd.notna(row['경도']):
                folium.Marker(
                    location=[float(row['위도']), float(row['경도'])],
                    icon=folium.Icon(color=marker_color, icon='exclamation-sign', prefix='glyphicon'),
                    popup=f"날짜: {row['date'].strftime('%Y-%m-%d')}<br>지역: {row['full_address']}"
                ).add_to(marker_cluster)
        if point_count > 1000:
            st.info(f"범죄 데이터 {point_count}건 중 1000건만 표시")
    
    regions = ['서울특별시', '경기도', '강원도', '경상남도', '경상북도', '광주광역시', '대구광역시', '대전광역시', 
               '부산광역시', '세종특별자치시', '울산광역시', '인천광역시', '전라남도', '전라북도', '제주도', 
               '충청남도', '충청북도']
    region_mapping = {
        'Seoul': '서울특별시', 'Gyeonggi-do': '경기도', 'Gangwon-do': '강원도', 'Gyeongsangnam-do': '경상남도', 
        'Gyeongsangbuk-do': '경상북도', 'Gwangju': '광주광역시', 'Daegu': '대구광역시', 'Daejeon': '대전광역시', 
        'Busan': '부산광역시', 'Sejong': '세종특별자치시', 'Ulsan': '울산광역시', 'Incheon': '인천광역시', 
        'Jeollanam-do': '전라남도', 'Jeollabuk-do': '전라북도', 'Jeju': '제주도', 
        'Chungcheongnam-do': '충청남도', 'Chungcheongbuk-do': '충청북도'
    }
    
    scores = {region: 0 for region in regions}
    probabilities = {region: None for region in regions}
    
    if not indicator_data.empty:
        if view_type == "예측":
            # 최적화: groupby로 평균 계산
            for region in regions:
                region_data = indicator_data[indicator_data['도단위'] == region]
                if not region_data.empty:
                    probabilities[region] = region_data['crime_probability'].iloc[0] if selected_date else region_data['crime_probability'].mean()
        else:
            for region in regions:
                if view_type in ["전체 데이터", "년도별"] and not selected_date:
                    region_scores = indicator_data.apply(lambda row: calculate_risk_score(row, region), axis=1)
                    scores[region] = round(region_scores.mean()) if not region_scores.empty else 0
                elif indicator_data.shape[0] > 0:
                    scores[region] = calculate_risk_score(indicator_data.iloc[0], region)
    
    def style_function(feature):
        region = region_mapping.get(feature['properties']['NAME_1'], feature['properties']['NAME_1'])
        prob = probabilities.get(region, None)
        color = get_prediction_color(prob) if view_type == "예측" else {0: 'green', 1: 'yellow', 2: 'orange', 3: 'red'}.get(scores.get(region, 0), 'green')
        return {'fillColor': color, 'color': 'black', 'weight': 1, 'fillOpacity': 0.3}
    
    def tooltip_function(feature):
        region = region_mapping.get(feature['properties']['NAME_1'], feature['properties']['NAME_1'])
        if view_type == "예측":
            prob = probabilities.get(region, None)
            prob_str = f"{prob:.3f}" if prob is not None else '없음'
            return folium.GeoJsonTooltip(fields=['NAME_1'], aliases=['지역'], extra_html=f'<br>위험률: {prob_str}')
        return folium.GeoJsonTooltip(fields=['NAME_1'], aliases=['지역'], extra_html=f'<br>위험 점수: {scores.get(region, 0)}')
    
    folium.GeoJson(geo_data, style_function=style_function, tooltip=tooltip_function).add_to(risk_group)
    crime_group.add_to(m)
    risk_group.add_to(m)
    folium.LayerControl().add_to(m)
    
    legend_html = '''
        <div style="position: fixed; bottom: 50px; right: 50px; z-index:9999; background-color:white; padding:10px; border:2px solid grey;">
            <p><strong>범례%s</strong></p>
            <p><strong>위험 코로플렛</strong></p>
            %s
            <p><strong>범죄 마커</strong>%s</p>
            <p><span style="color:green;">■</span> 소수 (<100건)</p>
            <p><span style="color:orange;">■</span> 보통 (<500건)</p>
            <p><span style="color:red;">■</span> 다수 (≥500건)</p>
        </div>
    ''' % (
        ' (예측)' if view_type == "예측" else '',
        '''
        <p><span style="color:green;">■</span> 안전 (<0.3)</p>
        <p><span style="color:lime;">■</span> 대비 (0.3~0.5)</p>
        <p><span style="color:yellow;">■</span> 주의 (0.5~0.7)</p>
        <p><span style="color:orange;">■</span> 경보 (0.7~0.85)</p>
        <p><span style="color:red;">■</span> 위험 (≥0.85)</p>
        ''' if view_type == "예측" else '''
        <p><span style="color:green;">■</span> 0점: 안전</p>
        <p><span style="color:yellow;">■</span> 1점: 주의</p>
        <p><span style="color:orange;">■</span> 2점: 경고</p>
        <p><span style="color:red;">■</span> 3점: 위험</p>
        ''',
        ': 표시되지 않음' if view_type == "예측" else ''
    )
    m.get_root().html.add_child(folium.Element(legend_html))
    
    if not crime_data.empty:
        avg_lat = crime_data['위도'].mean()
        avg_lon = crime_data['경도'].mean()
        if pd.notna(avg_lat) and pd.notna(avg_lon):
            m.location = [avg_lat, avg_lon]
            m.zoom_start = 8
    
    return m, title

st.title("이상동기 범죄 경보 맵")

# 경로: Streamlit 배포용
crime_path = "./data/15~25년도 이상동기(도단위추가)_with_coords_openai.csv"
indicator_path = "./data/지표데이터(4대범죄추가계산).csv"
prediction_path = "./data/crime_predictions_2024_2025_binary_risk.csv"

df_crime, crime_dates, df_indicator, indicator_dates, df_prediction, prediction_dates = load_data(crime_path, indicator_path, prediction_path)
geo_data = load_geojson()

with st.sidebar:
    st.title('🚨 대시보드')
    view_type = st.selectbox('보기 유형', ['전체 데이터', '년도별', '일별', '예측'])
    
    selected_year = None
    selected_date = None
    
    if view_type in ['년도별', '일별']:
        crime_years = [y for y in sorted(df_crime['date'].dt.year.unique()) if y <= 2023]
        selected_year = st.selectbox('년도', crime_years, index=len(crime_years)-1)
        if view_type == '일별':
            filtered_dates = [date for date in crime_dates if date.year == selected_year]
            if filtered_dates:
                selected_date = st.selectbox('날짜', filtered_dates, format_func=lambda x: x.strftime('%Y-%m-%d'))
                selected_date = pd.to_datetime(selected_date)
            else:
                st.warning(f"{selected_year}년 데이터 없음")
                view_type = "년도별"
    elif view_type == '예측':
        prediction_years = sorted(df_prediction['date'].dt.year.unique())
        selected_year = st.selectbox('예측 년도', prediction_years, index=len(prediction_years)-1)
        prediction_mode = st.radio('예측 모드', ['년도별', '일별'])
        if prediction_mode == '일별':
            filtered_dates = [date for date in prediction_dates if date.year == selected_year]
            if filtered_dates:
                selected_date = st.selectbox('예측 날짜', filtered_dates, format_func=lambda x: x.strftime('%Y-%m-%d'))
                selected_date = pd.to_datetime(selected_date)
            else:
                st.warning(f"{selected_year}년 예측 데이터 없음")
                prediction_mode = "년도별"

st.markdown("#### 통합 맵")
with st.spinner("맵을 로드하는 중..."):
    combined_map, combined_title = create_map(view_type, selected_year, selected_date, df_crime, df_indicator, df_prediction, geo_data)
    folium_static(combined_map, width=1000, height=600)

st.markdown("#### 지역별 위험 점수/예측 확률")
if view_type != "예측":
    risk_table = create_risk_score_table(df_indicator, view_type, selected_year, selected_date)
    st.dataframe(risk_table, use_container_width=True)
else:
    prediction_table = create_prediction_table(df_prediction, selected_year, selected_date, prediction_mode)
    st.dataframe(prediction_table, use_container_width=True)

if view_type != "예측":
    crime_count = len(df_crime[df_crime['date'].dt.year <= 2023]) if view_type == "전체 데이터" else len(df_crime[df_crime['date'].dt.year == selected_year]) if view_type == "년도별" else len(df_crime[df_crime['date'].dt.normalize() == selected_date.normalize()])
    st.write(f"범죄 건수: {crime_count}")
else:
    st.write("예측 모드: crime_probability 기반 코로플렛 및 표 표시")

with st.expander('대시보드 설명', expanded=False):
    st.write('''
        **통합 맵**: 범죄 마커와 지역별 위험도를 지도에 표시. 우측 상단에서 레이어를 조정할 수 있습니다.

        **범죄 마커** (2015~2023년, 예측 모드 제외):
        - 초록: 소수 (<100건)
        - 주황: 보통 (<500건)
        - 빨강: 다수 (≥500건)

        **위험 코로플렛**:
        - **2015~2023년** (전체 데이터, 년도별, 일별):
          - 위험도는 기후, 사회, 금융 스트레스 지표로 계산 (각 지표가 기준 초과 시 1점, 최대 3점).
          - 색상: 초록(0점, 안전), 노랑(1점, 주의), 주황(2점, 경고), 빨강(3점, 위험).
        - **2024~2025년** (예측 모드):
          - 위험률로 위험도 표시. 툴팁에 확률 값 제공.
          - 색상: 초록(<0.3, 안전), 연두(0.3 ~ 0.5, 대비), 노랑(0.5 ~ 0.7, 주의), 주황(0.7 ~ 0.85, 경보), 빨강(≥0.85, 위험).

        **지역별 표**:
        - **2015~2023년**: 지역별 기후, 사회, 금융 점수와 총점 표시.
        - **2024~2025년**: 지역별 위험률 표시 (년도별: 평균, 일별: 특정 날짜).

        **보기 유형**:
        - 전체 데이터/년도별/일별: 2015~2023년 실제 데이터.
        - 예측: 2024~2025년 범죄 확률 (년도별: 평균, 일별: 특정 날짜).
    ''')