# SPDX-FileCopyrightText: 2025 Doğu Kocatepe
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import subprocess
import sys
import json

def save_dataset(X, y):
    np.save('X.npy', X)
    np.save('y.npy', y)

def load_dataset():
    X = np.load('X.npy')
    y = np.load('y.npy')
    return X, y

def save_config(config):
    with open("config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

def load_config():
    with open("config.json", "r", encoding="utf-8") as f:
        return json.load(f)

def save_history(loss_wrt_gen, loss_wrt_time, time_hist):
    np.save('loss_wrt_gen.npy', loss_wrt_gen)
    np.save('loss_wrt_time.npy', loss_wrt_time)
    np.save('time_hist.npy', time_hist)

def load_history():
    loss_wrt_gen = np.load('loss_wrt_gen.npy')
    loss_wrt_time = np.load('loss_wrt_time.npy')
    time_hist = np.load('time_hist.npy')
    return loss_wrt_gen, loss_wrt_time, time_hist

def run_benchmark_script(script_name, config, X, y):
    X = np.array(X)
    y = np.array(y)

    save_dataset(X, y)
    save_config(config)

    result = subprocess.run(
        [sys.executable, script_name],
        capture_output=False,
        check=True
    )

    return load_history()

def run_benchmark(solver):
    X, y = load_dataset()
    config = load_config()
    save_history(*solver(config, X, y))
