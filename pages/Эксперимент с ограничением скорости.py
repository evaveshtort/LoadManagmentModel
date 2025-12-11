import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import exper_cloud as mdl
from scipy.stats import f_oneway

st.set_page_config(page_title="Эксперимент с rate_limit", layout="wide")
st.title("Зависимость среднего времени ответа, загруженности серверов и доли отклоненных запросов от максимальной скорости")

st.header("Параметры эксперимента")

# Параметры моделирования
sim_time = st.number_input("Время моделирования (сек)", value=60.0, min_value=1.0)
cfg = {"sim_time": sim_time, "strategy": "rate_limit"}

arrival_map = {
    "Экспоненциальное": "exponential",
    "Постоянное": "deterministic",
    "Равномерное": "uniform",
    "Пуассоновское (всплески)": "poisson_burst"
}
arrival_display = st.selectbox("Распределение входящей нагрузки", list(arrival_map.keys()))
arrival_dist = arrival_map[arrival_display] 
cfg["arrival_dist"] = arrival_dist

if arrival_dist == "exponential":
    arrival_rate = st.number_input("Среднее количество запросов в секунду", value=10.0)
    cfg["arrival_rate"] = arrival_rate
elif arrival_dist == "deterministic":
    arrival_interval = st.number_input("Интервал между запросами (сек)", value=0.1)
    cfg["arrival_interval"] = arrival_interval
elif arrival_dist == "uniform":
    arrival_low = st.number_input("Нижняя граница интервала между запросами (сек)", value=0.05)
    arrival_high = st.number_input("Верхняя граница интервала между запросами (сек)", value=0.2)
    cfg["arrival_low"] = arrival_low
    cfg["arrival_high"] = arrival_high
else:
    burst_size = st.number_input("Размер всплеска (количество запросов)", value=20, min_value=1)
    interburst_interval = st.number_input("Интервал между всплесками (сек)", value=5.0, min_value=0.1)
    cfg["burst_size"] = burst_size
    cfg["interburst_interval"] = interburst_interval

service_map = {
    "Экспоненциальное": "exponential",
    "Постоянное": "deterministic",
    "Равномерное": "uniform",
    "Нормальное": "normal"
}
service_display = st.selectbox("Распределение времени обработки", list(service_map.keys()))
service_dist = service_map[service_display] 
cfg["service_dist"] = service_dist

service_mean = st.number_input("Среднее время обработки (сек)", value=0.08)
cfg["service_mean"] = service_mean

if service_dist in ["normal", "uniform"]:
    service_std = st.number_input("Стандартное отклонение (сек)", value=0.02)
    cfg["service_std"] = service_std

num_servers = st.slider("Число параллельных серверов", 1, 20, 2)
cfg["num_servers"] = int(num_servers)

# Эксперимент по rate_limit
st.subheader("Диапазон максимальной скорости")
r_min = st.number_input("Минимальная скорость (запросы/сек)", value=1.0, min_value=0.1)
r_max = st.number_input("Максимальная скорость (апросы/сек)", value=50.0, min_value=0.1)
r_step = st.number_input("Шаг скорости (апросы/сек)", value=5.0, min_value=0.1)
replicas = st.number_input("Реплик на точку", value=10, min_value=1)
seed0 = st.number_input("Начальный seed", value=1000, min_value=0)

run_btn = st.button("Запустить эксперимент")

if run_btn:
    st.info("Запускаю эксперимент — это может занять время")
    rate_limits = np.arange(r_min, r_max + r_step, r_step)
    results = []
    all_resp = {r: [] for r in rate_limits}    
    all_util = {r: [] for r in rate_limits}     
    all_drops = {r: [] for r in rate_limits}  
    progress = st.progress(0)
    total_runs = len(rate_limits) * replicas
    run_count = 0

    for r_lim in rate_limits:
        means = []
        drops = []
        utils = []
        for rep in range(replicas):
            cfg.update({
                "rate_limit_rps": float(r_lim),
                "seed": int(seed0 + rep + int(r_lim*1000)),
            })
            res = mdl.model_env(cfg)
            means.append(res["avg_response_time"])
            drops.append(res["dropped"] / max(1, res["total_arrivals"]))
            utils.append(res["utilization"])

            all_resp[r_lim].append(res["avg_response_time"])
            all_util[r_lim].append(res["utilization"])
            all_drops[r_lim].append(res["dropped"] / max(1, res["total_arrivals"]))

            run_count += 1
            progress.progress(run_count / total_runs)
        
        arr = np.array(means)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        se = std / np.sqrt(len(arr) if len(arr)>0 else 1)
        ci95 = 1.96 * se

        utils_arr = np.array(utils)
        mean_util = float(np.mean(utils_arr))
        std_util = float(np.std(utils_arr, ddof=1)) if len(utils_arr) > 1 else 0.0
        se_util = std_util / np.sqrt(len(utils_arr) if len(utils_arr)>0 else 1)
        ci95_util = 1.96 * se_util

        drops_arr = np.array(drops)
        mean_drops = float(np.mean(drops_arr))
        std_drops = float(np.std(drops_arr, ddof=1)) if len(drops_arr) > 1 else 0.0
        se_drops = std_drops / np.sqrt(len(drops_arr) if len(drops_arr)>0 else 1)
        ci95_drops = 1.96 * se_drops

        results.append({
            "rate_limit_rps": r_lim,
            "mean_response": mean,
            "ci95": ci95,
            "mean_drop_rate": mean_drops,
            "ci95_drop_rate": ci95_drops,
            "mean_utilization": mean_util,
            "ci95_utilization": ci95_util
        })

    df = pd.DataFrame(results)
    
    st.header("Результаты эксперимента")

    resp_groups = [all_resp[r] for r in rate_limits]
    util_groups = [all_util[r] for r in rate_limits]
    drop_groups = [all_drops[r] for r in rate_limits]

    f_resp, p_resp = f_oneway(*resp_groups)
    if p_resp < 0.05:
        st.success(f"Максимальная скорость оказывает статистически значимое влияние на среднее время отклика (pvalue={p_resp:.2f})")
    else:
        st.info(f"Максимальная скорость НЕ оказывает статистически значимое влияние на среднее время отклика (pvalue={p_resp:.2f})")

    f_util, p_util = f_oneway(*util_groups)
    if p_util < 0.05:
        st.success(f"Максимальная скорость оказывает статистически значимое влияние на загрузку серверов (pvalue={p_util:.2f})")
    else:
        st.info(f"Максимальная скорость НЕ оказывает статистически значимое влияние на загрузку серверов (pvalue={p_util:.2f})")
    
    f_drop, p_drop = f_oneway(*drop_groups)
    if p_drop < 0.05:
        st.success(f"Максимальная скорость оказывает статистически значимое влияние на долю отклоненных (pvalue={p_drop:.2f})")
    else:
        st.info(f"Максимальная скорость НЕ оказывает статистически значимое влияние на долю отклоненных (pvalue={p_drop:.2f})")

    # График среднего времени отклика с 95% CI
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(df["rate_limit_rps"], df["mean_response"], marker='o', label="Среднее время отклика")
    ax.fill_between(df["rate_limit_rps"],
                    df["mean_response"] - df["ci95"],
                    df["mean_response"] + df["ci95"],
                    alpha=0.2, label="95% доверительный интервал")
    ax.set_xlabel("Максимальная скорость (запросы/сек)")
    ax.set_ylabel("Среднее время отклика (сек)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    # График средней загрузки серверов
    fig2, ax2 = plt.subplots(figsize=(8,4))
    ax2.plot(df["rate_limit_rps"], df["mean_utilization"]*100, marker='o', color='orange', label="Средняя загруженность серверов")
    ax2.fill_between(df["rate_limit_rps"],
                     (df["mean_utilization"] - df["ci95_utilization"])*100,
                     (df["mean_utilization"] + df["ci95_utilization"])*100,
                     alpha=0.2, label="95% доверительный интервал", color='orange')
    ax2.set_xlabel("Максимальная скорость (запросы/сек)")
    ax2.set_ylabel("Загруженность серверов (%)")
    ax2.grid(True)
    ax2.legend()
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
    st.pyplot(fig2)

    # График доли отклоненных
    fig3, ax3 = plt.subplots(figsize=(8,4))
    ax3.plot(df["rate_limit_rps"], df["mean_drop_rate"]*100, marker='o', color='green', label="Доля отклоненных запросов")
    ax3.fill_between(df["rate_limit_rps"],
                     (df["mean_drop_rate"] - df["ci95_drop_rate"])*100,
                     (df["mean_drop_rate"] + df["ci95_drop_rate"])*100,
                     alpha=0.2, label="95% доверительный интервал", color='green')
    ax3.set_xlabel("Максимальная скорость (запросы/сек)")
    ax3.set_ylabel("Доля отклоненных (%)")
    ax3.grid(True)
    ax3.legend()
    ax3.yaxis.set_major_formatter(mtick.PercentFormatter())
    st.pyplot(fig3)

    st.subheader("Таблица с результатами")
    st.dataframe(df[['rate_limit_rps', 'mean_response', 'mean_drop_rate', 'mean_utilization']].rename(
        columns={
            "rate_limit_rps": "Максимальная скорость (запросы/сек)", 
            "mean_response": "Среднее время отклика (сек)", 
            "mean_drop_rate": "Доля отклоненных запросов", 
            "mean_utilization": "Средняя загруженность серверов"
        }))
    st.download_button("Скачать результаты (CSV)", df.to_csv(index=False), file_name="exp_rate_limit_results.csv")

