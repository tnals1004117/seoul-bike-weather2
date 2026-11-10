import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# 기본 화면 설정
# --------------------------------------------------
st.set_page_config(
    page_title="따릉이 날씨 예측 연구소",
    page_icon="🚲",
    layout="wide"
)

st.title("🚲 따릉이 날씨 예측 연구소")
st.caption("날씨에 따라 시간별 따릉이 대여량이 어떻게 달라지는지 살펴보고 예측해 봅니다.")


# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------
@st.cache_data(show_spinner="UCI에서 따릉이 데이터를 불러오는 중입니다...")
def load_data():
    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=560)
    features = dataset.data.features
    targets = dataset.data.targets

    # UCI 데이터는 입력 변수와 별도 target 표로 나뉘어 제공될 수 있으므로 합칩니다.
    data = pd.concat([features, targets], axis=1)
    data = data.loc[:, ~data.columns.duplicated()].copy()

    return data


def normalize_name(name):
    """열 이름에서 공백과 기호를 없애 열 이름을 비교하기 쉽게 만듭니다."""
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())


def find_column(data, possible_names):
    """실제 데이터의 열 이름이 조금 달라도 해당 열을 찾습니다."""
    normalized_columns = {normalize_name(col): col for col in data.columns}

    for name in possible_names:
        key = normalize_name(name)
        if key in normalized_columns:
            return normalized_columns[key]

    for col in data.columns:
        normalized_col = normalize_name(col)
        for name in possible_names:
            key = normalize_name(name)
            if normalized_col.startswith(key):
                return col
    return None


try:
    raw_data = load_data()
    if raw_data.empty:
        st.error("데이터가 비어 있습니다. 잠시 후 다시 실행해 주세요.")
        st.stop()
except Exception as error:
    st.error("데이터를 불러오지 못했습니다. 인터넷 연결이나 데이터 출처를 확인해 주세요.")
    st.caption(f"오류 내용: {error}")
    st.stop()


# 실제 데이터 열 이름 찾기
columns = {
    "date": find_column(raw_data, ["Date"]),
    "target": find_column(raw_data, ["Rented Bike Count", "Rented Bike count", "Rented"]),
    "hour": find_column(raw_data, ["Hour"]),
    "temperature": find_column(raw_data, ["Temperature"]),
    "humidity": find_column(raw_data, ["Humidity"]),
    "wind": find_column(raw_data, ["Wind speed", "Windspeed", "Wind"]),
    "rainfall": find_column(raw_data, ["Rainfall"]),
    "snowfall": find_column(raw_data, ["Snowfall"]),
    "solar": find_column(raw_data, ["Solar Radiation", "Solar"]),
    "season": find_column(raw_data, ["Seasons", "Season"]),
    "functioning": find_column(raw_data, ["Functioning Day", "Functioning_Day"]),
}

required_keys = ["target", "temperature", "humidity", "wind", "rainfall", "snowfall", "solar"]
missing_keys = [key for key in required_keys if columns[key] is None]
if missing_keys:
    st.error("데이터에서 필요한 열을 찾지 못했습니다.")
    st.write("찾지 못한 항목:", ", ".join(missing_keys))
    st.write("실제 데이터 열 이름:", list(raw_data.columns))
    st.stop()
