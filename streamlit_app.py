import streamlit as st
import pandas as pd
import statsmodels.api as sm
from datetime import datetime, timedelta
from io import BytesIO

st.title("Prediksi Fertilitas Palupi F.N")

# Sidebar inputs
st.sidebar.header("Upload Data & Settings")
uploaded_file = st.sidebar.file_uploader("Upload CSV with 'period_start' column", type="csv")
menstruation_days = st.sidebar.number_input("Menstruation Days", min_value=3, max_value=10, value=5)
predict_n = st.sidebar.number_input("Predict Next Cycles", min_value=1, max_value=12, value=1)

if uploaded_file:
    # Load historical data
    df_hist = pd.read_csv(uploaded_file).sort_values("period_start")
    df_hist['period_start'] = pd.to_datetime(df_hist['period_start'], errors='coerce')

    # Display invalid date warnings
    if df_hist['period_start'].isna().any():
        st.warning("⚠️ Beberapa tanggal tidak valid dan telah diabaikan. Periksa format tanggal pada file CSV Anda (contoh: 2024-12-01).")
    df_hist = df_hist.dropna(subset=['period_start'])

    st.subheader("Historical Period Start Dates")
    st.dataframe(df_hist)

    # Compute cycle lengths and fit AR(1)
    cycle_lens = df_hist['period_start'].diff().dt.days.dropna()
    model = sm.tsa.ARIMA(cycle_lens, order=(1,0,0)).fit()
    preds = model.forecast(steps=int(predict_n))

    # Forecast cycles
    last_start = df_hist['period_start'].max()
    summaries = []
    calendars = {}
    prob_map = {-5:0.10, -4:0.15, -3:0.20, -2:0.27, -1:0.30, 0:0.33, 1:0.10, 2:0.05}

    for i, length in enumerate(preds, start=1):
        L = int(round(length))
        start_i = last_start + timedelta(days=L)
        ov_offset = L - 14
        fert_start = ov_offset - 5
        fert_end = ov_offset + 1

        # Summary for this cycle
        summaries.append({
            'Cycle #': i,
            'Start Date': start_i.date(),
            'Length (days)': L,
            'Ovulation': (start_i + timedelta(days=ov_offset)).date(),
            'Fertile Start': (start_i + timedelta(days=fert_start)).date(),
            'Fertile End': (start_i + timedelta(days=fert_end)).date()
        })

        # Day-by-day calendar
        rows = []
        for d in range(L):
            date = start_i + timedelta(days=d)
            cd = d + 1
            if cd <= menstruation_days:
                status = "Menstruation"
                note = f"Day {cd}"
            elif fert_start <= cd <= fert_end:
                status = "Fertile"
                note = "Ovulation" if cd == ov_offset else ""
            else:
                status = "Safe"
                note = ""

            rel = cd - ov_offset
            base = prob_map.get(rel, 0)
            p0 = round(base * 100, 1) if status == "Fertile" else 0
            pc = round(p0 * 0.15, 1)
            pp = round(p0 * 0.10, 1)

            rows.append({
                'Date': date.date(),
                'Cycle Day': cd,
                'Status': status,
                'Note': note,
                'Prob % (No Prot.)': p0,
                'Prob % (Condom)': pc,
                'Prob % (Plan B)': pp
            })

        calendars[f"Cycle {i}"] = pd.DataFrame(rows)
        last_start = start_i

    df_summary = pd.DataFrame(summaries)
    st.subheader("Predicted Cycles Summary")
    st.dataframe(df_summary)

    st.subheader("Day-by-Day Calendar for Cycle 1")
    st.dataframe(calendars['Cycle 1'])

    # Prepare Excel download
    towrite = BytesIO()
    with pd.ExcelWriter(towrite, engine='xlsxwriter') as writer:
        df_summary.to_excel(writer, sheet_name='Summary', index=False)
        for name, cal in calendars.items():
            cal.to_excel(writer, sheet_name=name, index=False)
    towrite.seek(0)

    st.download_button(
        label="Download Excel",
        data=towrite,
        file_name="predicted_cycles.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
