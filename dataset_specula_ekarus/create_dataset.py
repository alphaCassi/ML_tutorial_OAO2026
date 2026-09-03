import os
import time
import gc
import queue
import logging
import logging.handlers
import argparse
import subprocess
import multiprocessing as mp
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import cupy as cp
import specula

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# ─────────────────────────────────────────────
#  DEFAULT CONFIG  (overridable via CLI)
# ─────────────────────────────────────────────
PRECISION       = 0           # 0 = fp32, 1 = fp64
GPU_THRESHOLD   = 0.8
OUTPUT_DIR      = "./output/parameter_sweep"

BASE_CONFIG     = "ekarus_main.yml"
PHASESCREEN_DIR = "./calibration/phasescreens/"
N_SAMPLES       = 100
QUEUE_MAX_SIZE  = 50


# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────

def setup_main_logging(output_dir: str, log_queue: mp.Queue) -> logging.handlers.QueueListener:
    os.makedirs(output_dir, exist_ok=True)
    fmt = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(os.path.join(output_dir, "sweep.log"), mode="a")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    listener = logging.handlers.QueueListener(log_queue, fh, ch, respect_handler_level=True)
    listener.start()
    logger = logging.getLogger("sweep")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    qh = logging.handlers.QueueHandler(log_queue)
    logger.addHandler(qh)
    return listener


def setup_worker_logging(log_queue: mp.Queue) -> None:
    logger = logging.getLogger("sweep")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    logger.addHandler(logging.handlers.QueueHandler(log_queue))
    logging.getLogger().handlers.clear()
    logging.getLogger().addHandler(logging.handlers.QueueHandler(log_queue))


# ─────────────────────────────────────────────
#  PARAMETER DISTRIBUTIONS
# ─────────────────────────────────────────────

param_distributions = {
    "main.zenithAngleInDeg": lambda: np.random.uniform(0, 45),
    "source_ngs.magnitude":  lambda: np.random.uniform(1,4),
    "seeing.constant":       lambda: np.random.uniform(0.4, 2.0),

    "atmo.Cn2": lambda: (
        lambda w: (w / w.sum()).tolist()
    )(
        np.clip(
            np.array([0.70, 0.06, 0.14, 0.10]) *
            (1 + np.random.randn(4) * np.array([0.10, 0.02, 0.02, 0.03])),
            1e-12, None,
        )
    ),

    "wind_speed.constant": lambda: [
        np.random.uniform(4,  16),
        np.random.uniform(5,  25),
        np.random.uniform(5,  25),
        np.random.uniform(20, 70),
    ],
    "wind_direction.constant": lambda: [
        np.random.uniform(0, 359.9) for _ in range(4)
    ],
    "atmo.L0": lambda: np.random.uniform(1., 40.),
}


# ─────────────────────────────────────────────
#  SAMPLE GENERATION
# ─────────────────────────────────────────────

def generate_samples(
    param_distributions: dict,
    n_samples: int,
    start_index: int = 0,
) -> pd.DataFrame:
    samples = {p: [f() for _ in range(n_samples)] for p, f in param_distributions.items()}
    df = pd.DataFrame(samples, index=range(start_index, start_index + n_samples))
    df.index.name = "sim_id"
    df["atmo.seed"] = df.index.astype(int) + 1
    return df


# ─────────────────────────────────────────────
#  GPU / MEMORY HELPERS
# ─────────────────────────────────────────────

def get_n_gpus() -> int:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    return len(result.stdout.strip().splitlines())


def get_gpu_memory() -> list:
    result = subprocess.run(
        ["nvidia-smi",
         "--query-gpu=memory.used,memory.total",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    fracs = []
    for line in result.stdout.strip().splitlines():
        used, total = line.split(",")
        fracs.append(int(used.strip()) / int(total.strip()))
    return fracs


def free_gpu_and_ram() -> None:
    gc.collect()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    time.sleep(0.2)


def wait_for_memory(gpu_index: int, threshold: float = GPU_THRESHOLD) -> None:
    usage = get_gpu_memory()
    if usage[gpu_index] >= threshold:
        log = logging.getLogger("sweep")
        log.warning(
            f"[GPU {gpu_index}] Memory {usage[gpu_index]*100:.1f}% > "
            f"{threshold*100:.0f}% — freeing ..."
        )
        free_gpu_and_ram()
        usage = get_gpu_memory()
        if usage[gpu_index] >= threshold:
            log.warning(
                f"[GPU {gpu_index}] Memory still high "
                f"({usage[gpu_index]*100:.1f}%) — waiting 1 s ..."
            )
            time.sleep(1)


# ─────────────────────────────────────────────
#  PHASE SCREEN CLEANUP
# ─────────────────────────────────────────────

def remove_files_phasescreens(folder: str = PHASESCREEN_DIR) -> None:
    if not os.path.isdir(folder):
        return
    for fname in os.listdir(folder):
        fp = os.path.join(folder, fname)
        if os.path.isfile(fp):
            os.remove(fp)


# ─────────────────────────────────────────────
#  SIMULATION
#  FIX: output in flat sim_XXXXX/ (no gpu subfolder)
#       so each sim_id exists exactly once on disk
#       regardless of which GPU ran it.
# ─────────────────────────────────────────────

def run_simulation(sim_id: int, params: dict, gpu_index: int, output_dir: str) -> None:
    override_str = ", ".join(f"{k}: {v}" for k, v in params.items())
    # ── FIX: cartella flat, senza gpu{N}/ ──
    sim_dir = os.path.join(output_dir, f"sim_{sim_id:05d}")
    overrides = (
        "{ " + override_str + ", "
        f"data_store.store_dir: {sim_dir}" + " }"
    )
    logging.disable(logging.CRITICAL)
    try:
        specula.main_simul(yml_files=[BASE_CONFIG], overrides=overrides)
    finally:
        logging.disable(logging.NOTSET)

    free_gpu_and_ram()


# ─────────────────────────────────────────────
#  CHECKPOINT
# ─────────────────────────────────────────────

def save_checkpoint(output_dir: str, sim_id: int, gpu_index: int) -> None:
    os.makedirs(output_dir, exist_ok=True)
    cp_file = os.path.join(output_dir, f"last_checkpoint_gpu{gpu_index}.txt")
    tmp     = cp_file + ".tmp"
    with open(tmp, "w") as f:
        f.write(str(sim_id))
    os.replace(tmp, cp_file)


def load_checkpoint(output_dir: str, gpu_list: list) -> int:
    ids = []
    for g in gpu_list:
        cp_file = os.path.join(output_dir, f"last_checkpoint_gpu{g}.txt")
        if os.path.exists(cp_file):
            with open(cp_file) as f:
                ids.append(int(f.read().strip()))
        else:
            ids.append(-1)
    return min(ids) if ids else -1


# ─────────────────────────────────────────────
#  PRODUCER
# ─────────────────────────────────────────────

def producer(df: pd.DataFrame, start_id: int, job_queue: mp.Queue) -> None:
    for sim_id, row in df.iterrows():
        if int(sim_id) > start_id:
            job_queue.put((int(sim_id), row.to_dict()))


# ─────────────────────────────────────────────
#  WORKER
# ─────────────────────────────────────────────

def worker(
    gpu_index:  int,
    job_queue:  mp.Queue,
    output_dir: str,
    counter:    "mp.Value",
    log_queue:  mp.Queue,
    precision:  int,
) -> None:
    setup_worker_logging(log_queue)
    log = logging.getLogger("sweep")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    specula.init(0, precision=precision)

    consecutive_empty = 0
    MAX_EMPTY = 3

    while True:
        try:
            item = job_queue.get(timeout=10)
            consecutive_empty = 0
        except queue.Empty:
            consecutive_empty += 1
            if consecutive_empty >= MAX_EMPTY:
                break
            continue

        if item is None:
            break

        sim_id, params = item
        # ── FIX: controlla cartella flat (no gpu subfolder) ──
        sim_dir = os.path.join(output_dir, f"sim_{sim_id:05d}")

        if os.path.isdir(sim_dir) and any(Path(sim_dir).iterdir()):
            log.info(f"[GPU {gpu_index}] sim {sim_id:05d} already exists, skip.")
            with counter.get_lock():
                counter.value += 1
            save_checkpoint(output_dir, sim_id, gpu_index)
            continue

        wait_for_memory(gpu_index)

        t0 = time.time()
        try:
            run_simulation(sim_id, params, gpu_index, output_dir)
            log.info(f"[GPU {gpu_index}] sim {sim_id:05d} done in {time.time()-t0:.1f}s")
        except Exception as e:
            log.error(f"[GPU {gpu_index}] sim {sim_id:05d} FAILED: {e}\n{traceback.format_exc()}")
        finally:
            free_gpu_and_ram()
            remove_files_phasescreens()

        save_checkpoint(output_dir, sim_id, gpu_index)
        with counter.get_lock():
            counter.value += 1


# ─────────────────────────────────────────────
#  PROGRESS MONITOR
# ─────────────────────────────────────────────

def progress_monitor(counter: "mp.Value", total: int, stop_event: mp.Event) -> None:
    if not HAS_TQDM:
        return
    bar = tqdm(total=total, unit="sim", dynamic_ncols=True)
    last = 0
    while not stop_event.is_set():
        current = counter.value
        if current > last:
            bar.update(current - last)
            last = current
        time.sleep(1)
    bar.update(counter.value - last)
    bar.close()


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SPECULA parameter-sweep runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--gpus", nargs="+", type=int, default=None,
        help="Physical GPU indices to use, e.g. --gpus 0 1.",
    )
    parser.add_argument(
        "--n-samples", type=int, default=N_SAMPLES,
        help="Number of NEW samples to add to the dataset (only used with --expand).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=OUTPUT_DIR,
        help="Root output directory.",
    )
    parser.add_argument(
        "--precision", type=int, choices=[0, 1], default=PRECISION,
        help="Float precision: 0=fp32, 1=fp64.",
    )
    # ── FIX: espansione CSV solo se richiesta esplicitamente ──
    parser.add_argument(
        "--expand", action="store_true",
        help=(
            "Expand the parameter CSV by adding --n-samples new rows. "
            "Without this flag the script only runs simulations already "
            "present in the CSV — it never adds new rows automatically."
        ),
    )
    return parser.parse_args()


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    mp.set_start_method("spawn", force=True)

    log_queue = mp.Queue()
    listener  = setup_main_logging(output_dir, log_queue)
    log       = logging.getLogger("sweep")

    # ── GPU selection ─────────────────────────────────────────────────────
    n_gpus_total = get_n_gpus()
    if args.gpus is None:
        gpu_list = list(range(n_gpus_total))
    else:
        invalid = [g for g in args.gpus if g >= n_gpus_total]
        if invalid:
            raise ValueError(
                f"Requested GPUs {invalid} do not exist "
                f"(system has {n_gpus_total} GPU(s))."
            )
        gpu_list = args.gpus

    log.info(f"Using GPUs: {gpu_list}  ({len(gpu_list)} worker(s))")

    # ── Parameter CSV ─────────────────────────────────────────────────────
    param_csv = os.path.join(output_dir, "parametri_simulazioni.csv")

    if os.path.exists(param_csv):
        df_existing = pd.read_csv(param_csv, index_col="sim_id")
        log.info(f"Existing CSV found with {len(df_existing)} rows.")

        if args.expand:
            # ── FIX: espansione SOLO con --expand ──
            next_id = int(df_existing.index.max()) + 1
            log.info(
                f"--expand flag set: appending {args.n_samples} new samples "
                f"starting at sim_id={next_id}."
            )
            df_new = generate_samples(param_distributions, args.n_samples, start_index=next_id)
            df_new.to_csv(param_csv, mode="a", header=False)
            df = pd.concat([df_existing, df_new])
            log.info(f"CSV now has {len(df)} total rows.")
        else:
            # Usa il CSV esistente senza modificarlo
            df = df_existing
            log.info(
                f"Running/resuming {len(df)} simulations from existing CSV. "
                f"Use --expand to add new samples."
            )
    else:
        # Prima volta: crea il CSV
        df = generate_samples(param_distributions, args.n_samples, start_index=0)
        df.to_csv(param_csv)
        log.info(f"Generated and saved {args.n_samples} parameter sets to {param_csv}")

    # ── Deduplicazione sim_id nel CSV (safety net) ────────────────────────
    dupes = df.index.duplicated().sum()
    if dupes > 0:
        log.warning(f"Found {dupes} duplicate sim_ids in CSV — keeping first occurrence.")
        df = df[~df.index.duplicated(keep="first")]

    # ── Checkpoint ───────────────────────────────────────────────────────
    start_id  = load_checkpoint(output_dir, gpu_list)
    remaining = len(df[df.index > start_id])

    if start_id >= 0:
        log.info(f"Resuming from sim_id {start_id}. Remaining: {remaining}")
    else:
        log.info(f"No checkpoint found. Running all {len(df)} simulations.")
        remaining = len(df)

    # ── Job queue & shared counter ────────────────────────────────────────
    job_queue = mp.Queue(maxsize=QUEUE_MAX_SIZE)
    counter   = mp.Value("i", 0)
    stop_evt  = mp.Event()

    prod = mp.Process(target=producer, args=(df, start_id, job_queue), daemon=True)
    prod.start()

    if HAS_TQDM:
        mon = mp.Process(
            target=progress_monitor,
            args=(counter, remaining, stop_evt),
            daemon=True,
        )
        mon.start()
    else:
        mon = None
        log.info("tqdm not installed — install it for a live progress bar.")

    processes = []
    for gpu_index in gpu_list:
        p = mp.Process(
            target=worker,
            args=(gpu_index, job_queue, output_dir, counter, log_queue, args.precision),
        )
        p.start()
        processes.append(p)

    prod.join()

    for _ in gpu_list:
        job_queue.put(None)

    for p in processes:
        p.join()

    stop_evt.set()
    if mon:
        mon.join(timeout=5)

    listener.stop()
    log.info("All simulations completed on all GPUs.")


if __name__ == "__main__":
    wall_t0 = time.time()
    main()
    logging.getLogger("sweep").info(f"Total wall time: {time.time()-wall_t0:.1f} s")