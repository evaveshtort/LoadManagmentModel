import random as rd
import simpy
from collections import defaultdict
import math

# Параметры 
DEFAULTS = {
    "sim_time": 60.0,           # Время симуляции
    "arrival_dist": "exponential",  # Распределение входящей нагрузки (экспоненциальное | постоянный поток | равномерное распределение | пуассоновское)
    "arrival_rate": 10.0,      # Средняя нагрузка (запросы/сек) при экспоненциальном распределении
    "arrival_interval": 0.1,   # Интервал при постоянном потоке
    "arrival_low": 0.05,       # Нижняя граница интервала между запросами  при равномерном распределении
    "arrival_high": 0.2,       # Верхняя граница интервала между запросами  при равномерном распределении
    "burst_size": 20,          # Количество запросов во вспышке при пуассоновском распределении
    "interburst_interval": 5.0,# Количество секунд между вспышками при пуассоновском распределении
    "service_dist": "exponential", # Распределение времени обработки (экспоненциальное | нормальное | равномерное | постоянное)
    "service_mean": 0.08,      # Среднее время обработки
    "service_std": 0.02,       # Стандартное отклонение времени обработки
    "num_servers": 2,          # Количество серверов
    "strategy": "queue",       # Стратегия (отклонение | очередь |ограничение скорости)
    "queue_size": 50,          # Размер очереди (для стратегии "очередь"); None => бесконечная
    "rate_limit_rps": 20.0,    # Количество запросов в секунду для стратегии ограничения скорости
    "monitor_interval": 0.5,   
    "seed": 1234,
}


rd.seed(DEFAULTS["seed"])

def sample_interarrival(config):
    d = config["arrival_dist"]
    if d == "exponential":
        lam = max(1e-9, config["arrival_rate"])
        return rd.expovariate(lam)
    elif d == "deterministic":
        return config["arrival_interval"]
    elif d == "uniform":
        return rd.uniform(config["arrival_low"], config["arrival_high"])
    elif d == "poisson_burst":
        return None  
    else:
        return rd.expovariate(max(1e-9, config["arrival_rate"]))

def sample_service(config):
    d = config["service_dist"]
    if d == "exponential":
        mean = max(1e-9, config["service_mean"])
        return max(0.0, rd.expovariate(1.0/mean))
    elif d == "deterministic":
        return config["service_mean"]
    elif d == "normal":
        s = rd.gauss(config["service_mean"], config["service_std"])
        return max(0.0, s)
    elif d == "uniform":
        low = max(0.0, config["service_mean"] - config["service_std"])
        high = config["service_mean"] + config["service_std"]
        return rd.uniform(low, high)
    else:
        return max(0.0, rd.expovariate(1.0/config["service_mean"]))

def model_env(config=None):
    cfg = DEFAULTS.copy()
    if config:
        cfg.update(config)
    rd.seed(cfg.get("seed", DEFAULTS["seed"]))

    SIM_TIME = float(cfg["sim_time"])
    env = simpy.Environment()

    server = simpy.Resource(env, capacity=cfg["num_servers"])
    token_bucket = {"tokens": cfg["rate_limit_rps"], "last_time": 0.0}

    stats = {
        "total_arrivals": 0,
        "processed": 0,
        "dropped": 0,
        "response_times": [],
        "events": [],
        "queue_samples": [],
        "server_busy_samples": [],
    }

    server_busy_time = 0.0

    def refill_tokens(now):
        if cfg["rate_limit_rps"] <= 0:
            token_bucket["last_time"] = now
            return
        elapsed = now - token_bucket["last_time"]
        if elapsed <= 0:
            return
        token_bucket["tokens"] += elapsed * cfg["rate_limit_rps"]
        token_bucket["tokens"] = min(token_bucket["tokens"], cfg["rate_limit_rps"] * 2.0)
        token_bucket["last_time"] = now

    # Генерация поступающих запросов в систему
    def arrival_process(env):
        req_counter = 0  

        while env.now < SIM_TIME:
            if cfg["arrival_dist"] == "poisson_burst":

                yield env.timeout(cfg["interburst_interval"])

                for _ in range(cfg["burst_size"]):
                    req_counter += 1
                    env.process(handle_request(env, req_counter))
                continue


            req_counter += 1
            env.process(handle_request(env, req_counter))


            ia = sample_interarrival(cfg)
            if ia is None:
                ia = 0.0


            yield env.timeout(ia)



    # Обработка запроса
    def handle_request(env, req_id):
        nonlocal server_busy_time, stats

        arrival_time = env.now
        stats["total_arrivals"] += 1
        stats["events"].append((arrival_time, "ARRIVAL", req_id))

        strategy = cfg["strategy"]
        qlen = len(server.queue)
        in_service = server.count

        if strategy == "rate_limit":
            refill_tokens(env.now)
            if token_bucket["tokens"] >= 1.0:
                token_bucket["tokens"] -= 1.0
            else:
                stats["dropped"] += 1
                stats["events"].append((env.now, "DROPPED_RATE", req_id))
                return

        elif strategy == "reject":
            if in_service >= cfg["num_servers"]:
                stats["dropped"] += 1
                stats["events"].append((env.now, "DROPPED_REJECT", req_id))
                return

        elif strategy == "queue":
            if cfg["queue_size"] is not None:
                if (qlen + in_service) >= cfg["queue_size"] + cfg["num_servers"]:
                    stats["dropped"] += 1
                    stats["events"].append((env.now, "DROPPED_QUEUE_FULL", req_id))
                    return

        req = server.request()
        yield req

        start_service = env.now
        stats["events"].append((start_service, "SERVICE_START", req_id))

        service_time = sample_service(cfg)
        yield env.timeout(service_time)

        end_service = env.now
        server.release(req)

        server_busy_time += service_time
        stats["processed"] += 1
        stats["response_times"].append(end_service - arrival_time)
        stats["events"].append((end_service, "SERVICE_END", req_id))



    def monitor(env):
        while env.now < SIM_TIME:
            stats["queue_samples"].append((env.now, len(server.queue)))
            stats["server_busy_samples"].append((env.now, server.count))
            yield env.timeout(cfg["monitor_interval"])

    env.process(arrival_process(env))
    env.process(monitor(env))
    token_bucket["last_time"] = 0.0

    env.run(until=SIM_TIME)

    utilization = (server_busy_time / (SIM_TIME * max(1, cfg["num_servers"]))) if SIM_TIME > 0 else 0.0
    avg_response = sum(stats["response_times"]) / len(stats["response_times"]) if stats["response_times"] else 0.0

    results = {
        "total_arrivals": stats["total_arrivals"],
        "processed": stats["processed"],
        "dropped": stats["dropped"],
        "avg_response_time": avg_response,
        "utilization": utilization,
        "queue_time_series": stats["queue_samples"],
        "server_busy_time_series": stats["server_busy_samples"],
        "response_times": stats["response_times"],
        "events": stats["events"],
        "config": cfg,
    }
    return results

if __name__ == "__main__":
    cfg = {
        "sim_time": 30.0,
        "arrival_dist": "exponential",
        "arrival_rate": 15.0,
        "service_dist": "exponential",
        "service_mean": 0.05,
        "num_servers": 2,
        "strategy": "queue",
        "queue_size": 20,
    }
    res = model_env(cfg)
    print("Sanity run results:")
    for k, v in res.items():
        if k in ("queue_time_series", "response_times", "events"):
            print(k, "len:", len(v))
        else:
            print(k, ":", v)