import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import exper_cloud as mdl
import numpy as np

st.set_page_config(page_title="Модель регулирования нагрузки", layout="wide")
st.title("Модель регулирования нагрузки на облачное приложение")

st.header("Параметры моделирования")
# Выбор стратегии
st.subheader("Параметры стратегии")
strategy_map = {
    "Постановка в очередь": "queue",
    "Отклонение": "reject",
    "Ограничение скорости": "rate_limit"
}

strategy_display = st.selectbox("Стратегия регулирования", list(strategy_map.keys()))
strategy = strategy_map[strategy_display] 

# Параметры стратегии
params = {"strategy": strategy}

if strategy == "queue":
    queue_size = st.number_input("Размер очереди (-1 для неограниченной)", value=50)
    params["queue_size"] = int(queue_size) if queue_size >= 0 else None
elif strategy == "rate_limit":
    rate_limit_rps = st.number_input("Максимальная скорость (запросы/сек)", value=20.0)
    params["rate_limit_rps"] = rate_limit_rps

# Общие параметры моделирования
st.subheader("Общие параметры моделирования")
sim_time = st.number_input("Время моделирования (сек)", value=60.0, min_value=1.0)
num_servers = st.slider("Количество параллельных серверов", 1, 10, 2)
monitor_interval = st.number_input("Интервал мониторинга (сек)", value=0.5)
seed = st.number_input("Seed для генератора случайных чисел", value=1234)

# Распределение входящей нагрузки
st.subheader("Входящая нагрузка")

arrival_map = {
    "Экспоненциальное": "exponential",
    "Постоянное": "deterministic",
    "Равномерное": "uniform",
    "Пуассоновское (всплески)": "poisson_burst"
}
arrival_display = st.selectbox("Распределение входящей нагрузки", list(arrival_map.keys()))
arrival_dist = arrival_map[arrival_display] 

if arrival_dist == "exponential":
    arrival_rate = st.number_input("Среднее количество запросов в секунду", value=10.0)
    params["arrival_rate"] = arrival_rate
elif arrival_dist == "deterministic":
    arrival_interval = st.number_input("Интервал между запросами (сек)", value=0.1)
    params["arrival_interval"] = arrival_interval
elif arrival_dist == "uniform":
    arrival_low = st.number_input("Нижняя граница интервала между запросами (сек)", value=0.05)
    arrival_high = st.number_input("Верхняя граница интервала между запросами (сек)", value=0.2)

    params["arrival_low"] = arrival_low
    params["arrival_high"] = arrival_high
else:
    burst_size = st.number_input("Размер всплеска (количество запросов)", value=20, min_value=1)
    interburst_interval = st.number_input("Интервал между всплесками (сек)", value=5.0, min_value=0.1)
    params["burst_size"] = burst_size
    params["interburst_interval"] = interburst_interval

# Распределение времени обработки
st.subheader("Обработка запросов")

service_map = {
    "Экспоненциальное": "exponential",
    "Постоянное": "deterministic",
    "Равномерное": "uniform",
    "Нормальное": "normal"
}

service_display = st.selectbox("Распределение времени обработки", list(service_map.keys()))
service_dist = service_map[service_display] 

service_mean = st.number_input("Среднее время обработки (сек)", value=0.08)

if service_dist in ["normal", "uniform"]:
    service_std = st.number_input("Стандартное отклонение (сек)", value=0.02)
    params["service_std"] = service_std

# Кнопка запуска симуляции
if st.button("Запустить симуляцию"):
    # Формируем полный словарь параметров
    params.update({
        "sim_time": sim_time,
        "num_servers": num_servers,
        "monitor_interval": monitor_interval,
        "seed": int(seed),
        "arrival_dist": arrival_dist,
        "service_dist": service_dist,
        "service_mean": service_mean
    })

    # Запуск модели
    with st.spinner("Запуск модели..."):
        res = mdl.model_env(params)

    st.header("Результаты")
    # Итоговые метрики
    st.subheader("Итоговые метрики")
    st.metric("Всего запросов", res["total_arrivals"])
    st.metric("Обработано", res["processed"])
    st.metric("Отклонено", res["dropped"])
    st.metric("Среднее время отклика (сек)", round(res["avg_response_time"], 6))
    st.metric("Загруженность серверов", f"{round(res['utilization']*100,2)}%")

    # Скачать результаты
    st.subheader("Скачать результаты")
    df_metrics = pd.DataFrame([{
        "total": res["total_arrivals"],
        "processed": res["processed"],
        "dropped": res["dropped"],
        "avg_response": res["avg_response_time"],
        "utilization": res["utilization"],
    }])
    st.download_button("Скачать метрики (CSV)", df_metrics.to_csv(index=False), file_name="metrics.csv")

    # Графики приходящих и обрабатываемых запросов
    st.subheader("Динамика приходящих и обрабатываемых запросов по времени")

    if res["events"]:
        df_ev = pd.DataFrame(res["events"], columns=["time", "event", "id"])
        
        # Фильтруем события
        arrivals_times = df_ev[df_ev["event"]=="ARRIVAL"]["time"].values
        ended_times    = df_ev[df_ev["event"]=="SERVICE_END"]["time"].values
        dropped_times  = df_ev[df_ev["event"].isin(["DROPPED_QUEUE_FULL","DROPPED_RATE","DROPPED_RATE_TIMEOUT", "DROPPED_REJECT"])]["time"].values
        
        # Настройка интервалов
        bin_width = 1.0 
        t_min = 0.0
        t_max = max(np.max(arrivals_times, initial=0),
                    np.max(ended_times, initial=0),
                    np.max(dropped_times, initial=0),
                    res["config"]["sim_time"])
        
        bin_edges = np.arange(t_min, t_max + bin_width, bin_width)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Считаем количество событий в каждом интервале
        arrivals_counts, _ = np.histogram(arrivals_times, bins=bin_edges)
        ended_counts, _    = np.histogram(ended_times, bins=bin_edges)
        dropped_counts, _  = np.histogram(dropped_times, bins=bin_edges)

        # Рисуем линии
        fig4, ax4 = plt.subplots(figsize=(8,3))
        ax4.plot(bin_centers, arrivals_counts, label="Пришло запросов")
        ax4.plot(bin_centers, ended_counts, label="Обработано")
        ax4.plot(bin_centers, dropped_counts, label="Отклонено")
        ax4.set_xlabel("Время (сек)")
        ax4.set_ylabel("Количество запросов")
        ax4.legend()
        st.pyplot(fig4)
    else:
        st.write("Нет событий для построения графика.")


    st.subheader("Общее количество пришедших и обработанных запросов по времени")

    if res["events"]:
        df_ev = pd.DataFrame(res["events"], columns=["time", "event", "id"])
        
        arrivals = df_ev[df_ev["event"]=="ARRIVAL"].groupby("time").size().cumsum()
        ended = df_ev[df_ev["event"]=="SERVICE_END"].groupby("time").size().cumsum()
        dropped = df_ev[df_ev["event"].isin(["DROPPED_QUEUE_FULL","DROPPED_RATE","DROPPED_RATE_TIMEOUT", "DROPPED_REJECT"])].groupby("time").size().cumsum()
        
        fig3, ax3 = plt.subplots(figsize=(8,3))
        ax3.step(arrivals.index, arrivals.values, where='post', label="Пришло запросов")
        ax3.step(ended.index, ended.values, where='post', label="Обработано")
        ax3.step(dropped.index, dropped.values, where='post', label="Отклонено")
        ax3.set_xlabel("Время (с)")
        ax3.set_ylabel("Количество запросов")
        ax3.legend()
        st.pyplot(fig3)
    else:
        st.write("Нет событий для построения графика.")

    st.subheader("Занятость серверов во времени")
    if res["events"]:
        df_srv = pd.DataFrame(res["server_busy_time_series"], columns=["time", "busy"]).set_index("time")
        fig, ax5 = plt.subplots(figsize=(8,3))
        ax5.step(df_srv.index, df_srv["busy"], where='post')
        ax5.set_xlabel("Время (сек)")
        ax5.set_ylabel("Количество занятых серверов")
        ax5.set_ylim(0, params["num_servers"] + 0.5)
        st.pyplot(fig)
    else:
        st.write("Нет событий для построения графика.")

    # График размера очереди только для стратегии queue
    if strategy == "queue" and res["queue_time_series"]:
        st.subheader("Динамика длины очереди")
        df_q = pd.DataFrame(res["queue_time_series"], columns=["time", "qlen"]).set_index("time")
        fig, ax = plt.subplots(figsize=(8,3))
        ax.step(df_q.index, df_q["qlen"], where='post')
        ax.set_xlabel("Время (сек)")
        ax.set_ylabel("Длина очереди")
        st.pyplot(fig)
    elif strategy == "queue":
        st.write("Нет данных очереди.")

    # Гистограмма времени отклика
    st.subheader("Распределение времени отклика")
    rts = res["response_times"]
    if rts:
        fig2, ax2 = plt.subplots(figsize=(6,3))
        ax2.hist(rts, bins=min(50, max(5, int(len(rts)/5))))
        ax2.set_xlabel("Время отклика (сек)")
        ax2.set_ylabel("Количество запросов")
        st.pyplot(fig2)
    else:
        st.write("Нет обработанных запросов - нет данных о времени отклика.")

    # Журнал событий
    st.subheader("Журнал событий")
    ev = res["events"]
    if ev:
        df_ev = pd.DataFrame(ev, columns=["time", "event", "id"]).sort_values("time")
        df_ev = df_ev.rename(columns={"time": "Время (сек)", "event": "Событие", "id": "ID"})
        st.dataframe(df_ev)
    else:
        st.write("Нет событий")

    # Показать конфигурацию
    st.sidebar.subheader("Использованная конфигурация")
    st.sidebar.json(res["config"])
