import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import geopandas as gpd
import requests
from io import BytesIO
from folium.plugins import MarkerCluster

st.set_page_config(page_title="범죄 및 위험 대시보드", page_icon="🚨", layout="wide", initial_sidebar_state="expanded")

@st.cache_data
def load_data(crime_path, indicator_path, prediction_path):
    df_crime = pd.read_csv(crime_path, encoding='cp949', usecols=['날짜', '위도', '경도'])
    df_crime['date'] = pd.to_datetime(df_crime['날짜'], errors='coerce')
    df_crime['위도'] = pd.to_numeric(df_crime['위도'], errors='coerce')
    df_crime['경도'] = pd.to_numeric(df_crime['경도'], errors='coerce')
    df_crime = df_crime.dropna(subset=['date', '위도', '경도'])
    df_crime['full_address'] = df_crime.apply(lambda row: f"위도: {row['위도']}, 경도: {row['경도']}", axis=1)
    crime_dates = sorted(df_crime['date'].dt.normalize().unique())
    
    df_indicator = pd.read_csv(indicator_path, encoding='cp949')
    df_indicator['date'] = pd.to_datetime(df_indicator['date'], errors='coerce')
    df_indicator = df_indicator[df_indicator['date'].dt.year <= 2023].dropna(subset=['date'])
    indicator_dates = sorted(df_indicator['date'].dt.normalize().unique())
    
    df_prediction = pd.read_csv(prediction_path, encoding='cp949', usecols=['date', '도단위', 'crime_probability'])
    df_prediction['date'] = pd.to_datetime(df_prediction['date'], errors='coerce')
    df_prediction = df_prediction.dropna(subset=['date'])
    prediction_dates = sorted(df_prediction['date'].dt.normalize().unique())
    
    return df_crime, crime_dates, df_indicator, indicator_dates, df_prediction, prediction_dates

@st.cache_data
def load_geojson():
    url = "https://raw.githubusercontent.com/southkorea/southkorea-maps/master/gadm/json/skorea-provinces-geo.json"
    response = requests.get(url, timeout=5)
    return gpd.read_file(BytesIO(response.content))

def calculate_risk_score(df, region):
    score = pd.Series(0, index=df.index)
    climate_col = f"기후스트레스:{region}"
    social_col = f"사회스트레스:{region}"
    if climate_col in df.columns:
        score += (df[climate_col].notna() & (df[climate_col] > 13)).astype(int)
    if social_col in df.columns:
        score += (df[social_col].notna() & (df[social_col] >= 0.7)).astype(int)
    if '금융스트레스' in df.columns:
        score += (df['금융스트레스'].notna() & (df['금융스트레스'] >= 2)).astype(int)
    return score.clip(upper=3)

def get_prediction_color(prob):
    if prob is None:
        return 'gray'
    elif prob < 0.3:
        return 'green'
    elif prob < 0.5:
        return 'lime'
    elif prob < 0.7:
        return 'yellow'
    elif prob < 0.85:
        return 'orange'
    return 'red'

def create_map(view_type, selected_year=None, selected_date=None, crime_data=None, indicator_data=None, prediction_data=None, geo_data=None):
    m = folium.Map(location=[36.5, 127.5], zoom_start=7, tiles='CartoDB Positron')
    crime_group = folium.FeatureGroup(name="범죄 마커", show=(view_type != "예측"))
    risk_group = folium.FeatureGroup(name="위험 코로플렛", show=True)
    marker_cluster = MarkerCluster().add_to(crime_group)
    
    if view_type == "전체 데이터":
        crime_data = crime_data
        indicator_data = indicator_data
        title = "전체 데이터 맵"
    elif view_type == "년도별":
        crime_data = crime_data[crime_data['date'].dt.year == selected_year]
        indicator_data = indicator_data[indicator_data['date'].dt.year == selected_year]
        title = f"{selected_year}년 맵"
    elif view_type == "일별":
        selected_date_only = selected_date.normalize()
        crime_data = crime_data[crime_data['date'].dt.normalize() == selected_date_only]
        indicator_data = indicator_data[indicator_data['date'].dt.normalize() == selected_date_only]
        title = f"{selected_date_only.strftime('%Y-%m-%d')} 맵"
    else:
        crime_data = pd.DataFrame()
        indicator_data = prediction_data[prediction_data['date'].dt.year == selected_year] if selected_year else prediction_data[prediction_data['date'].dt.normalize() == selected_date.normalize()]
        title = f"{selected_year}년 예측 맵" if selected_year else f"{selected_date.strftime('%Y-%m-%d')} 예측 맵"
    
    if view_type != "예측" and not crime_data.empty:
        crime_data_limited = crime_data.head(1000)
        point_count = len(crime_data)
        marker_color = 'green' if point_count < 100 else 'orange' if point_count < 500 else 'red'
        for _, row in crime_data_limited.iterrows():
            folium.Marker(
                location=[float(row['위도']), float(row['경도'])],
                icon=folium.Icon(color=marker_color, icon='exclamation-sign', prefix='glyphicon'),
                popup=f"날짜: {row['date'].strftime('%Y-%m-%d')}<br>지역: {row['full_address']}"
            ).add_to(marker_cluster)
        if point_count > 1000:
            st.info(f"범죄 데이터 {point_count}건 중 1000건만 표시")
    
    regions = ['서울특별시', '경기도', '강원도', '경상남도', '경상북도', '광주광역시', '대구광역시', '대전광역시', '부산광역시', '세종특별자치시', '울산광역시', '인천광역시', '전라남도', '전라북도', '제주도', '충청남도', '충청북도']
    region_mapping = {
        'Seoul': '서울특별시', 'Gyeonggi-do': '경기도', 'Gangwon-do': '강원도', 'Gyeongsangnam-do': '경상남도', 'Gyeongsangbuk-do': '경상북도',
        'Gwangju': '광주광역시', 'Daegu': '대구광역시', 'Daejeon': '대전광역시', 'Busan': '부산광역시', 'Sejong': '세종특별자치시',
        'Ulsan': '울산광역시', 'Incheon': '인천광역시', 'Jeollanam-do': '전라남도', 'Jeollabuk-do': '전라북도', 'Jeju': '제주도',
        'Chungcheongnam-do': '충청남도', 'Chungcheongbuk-do': '충청북도'
    }
    
    scores = {region: 0 for region in regions}
    probabilities = {region: None for region in regions}
    
    if not indicator_data.empty:
        if view_type == "예측":
            for region in regions:
                region_data = indicator_data[indicator_data['도단위'] == region]
                if not region_data.empty:
                    probabilities[region] = region_data['crime_probability'].mean() if not selected_date else region_data.iloc[0]['crime_probability']
        else:
            for region in regions:
                if view_type in ["전체 데이터", "년도별"] and not selected_date:
                    region_scores = calculate_risk_score(indicator_data, region)
                    scores[region] = round(region_scores.mean()) if not region_scores.empty else 0
                elif indicator_data.shape[0] > 0:
                    scores[region] = calculate_risk_score(indicator_data, region).iloc[0]
    
    def style_function(feature):
        region = region_mapping.get(feature['properties']['NAME_1'], feature['properties']['NAME_1'])
        color = get_prediction_color(probabilities.get(region, None)) if view_type == "예측" else {0: 'green', 1: 'yellow', 2: 'orange', 3: 'red'}.get(scores.get(region, 0), 'green')
        return {'fillColor': color, 'color': 'black', 'weight': 1, 'fillOpacity': 0.3}
    
    def tooltip_function(feature):
        region = region_mapping.get(feature['properties']['NAME_1'], feature['properties']['NAME_1'])
        if view_type == "예측":
            prob = probabilities.get(region, None)
            prob_str = f"{prob:.3f}" if prob is not None else '없음'
            return folium.GeoJsonTooltip(fields=['NAME_1'], aliases=['지역'], extra_html=f'<br>범죄 확률: {prob_str}')
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

st.title("범죄 및 위험 대시보드")

crime_path = "data/15~25년도 이상동기(도단위추가)_with_coords_openai.csv"
indicator_path = "data/지표데이터(4대범죄추가계산).csv"
prediction_path = "data/crime_predictions_2024_2025_binary_risk.csv"

try:
    df_crime, crime_dates, df_indicator, indicator_dates, df_prediction, prediction_dates = load_data(crime_path, indicator_path, prediction_path)
except Exception:
    st.error("데이터 파일을 찾을 수 없습니다. 'data/' 폴더에 파일을 확인하세요.")
    st.stop()

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
    combined_map, combined_title = create_map(
        view_type, selected_year, selected_date,
        crime_data=df_crime, indicator_data=df_indicator, prediction_data=df_prediction, geo_data=geo_data
    )
    folium_static(combined_map, width=1000, height=600)

if view_type != "예측":
    crime_count = len(df_crime[df_crime['date'].dt.year <= 2023]) if view_type == "전체 데이터" else len(df_crime[df_crime['date'].dt.year == selected_year]) if view_type == "년도별" else len(df_crime[df_crime['date'].dt.normalize() == selected_date.normalize()])
    st.write(f"범죄 건수: {crime_count}")
else:
    st.write("예측 모드: crime_probability 기반 코로플렛 표시")

with st.expander('대시보드 설명', expanded=False):
    st.write('''
        **통합 맵**: 범죄 마커와 지역별 위험 코로플렛 표시.
        **범죄 마커**: 2015~2025년 실제 범죄 위치 (예측 모드 제외).
        - 색상: 소수(<100건, 초록), 보통(<500건, 주황), 다수(≥500건, 빨강).
        **위험 코로플렛**:
        - **2015~2023년**: 기후스트레스(>13, 1점), 사회스트레스(≥0.7, 1점), 금융스트레스(≥2, 1점).
          - 색상: 0점(초록), 1점(노랑), 2점(주황), 3점(빨강).
        - **2024~2025년** (예측): crime_probability 기준: <0.3(초록), 0.3~0.5(연두), 0.5~0.7(노랑), 0.7~0.85(주황), ≥0.85(빨강).
        **보기 유형**: 전체 데이터/년도별/일별(2015~2023년), 예측(2024~2025년).
    ''')