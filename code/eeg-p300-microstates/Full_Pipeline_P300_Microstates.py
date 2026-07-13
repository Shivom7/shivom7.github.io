###############################################################################
# FULL P300 + MICROSTATE PIPELINE FOR OPENNEURO ds002893 v2.0.0
#
# Core stack: MNE for EEGLAB/BIDS EEG loading and preprocessing, pycrostates for
# GFP peak extraction, ModKMeans model fitting, epoch segmentation, microstate
# parameters, transition matrices, expected transitions, entropy, and plots.
###############################################################################

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal
import scipy.stats
from scipy.optimize import linear_sum_assignment
from scipy.integrate import trapezoid
from pandas.errors import EmptyDataError

import mne
from mne.preprocessing import ICA

try:
    import pycrostates
    from pycrostates.cluster import ModKMeans
    from pycrostates.io import ChData
    from pycrostates.metrics import calinski_harabasz_score, davies_bouldin_score, dunn_score, silhouette_score
    from pycrostates.preprocessing import extract_gfp_peaks
except Exception as exc:
    pycrostates = None
    ModKMeans = None
    ChData = None
    calinski_harabasz_score = None
    davies_bouldin_score = None
    dunn_score = None
    silhouette_score = None
    extract_gfp_peaks = None
    PYCROSTATES_IMPORT_ERROR = exc
else:
    PYCROSTATES_IMPORT_ERROR = None

try:
    from statsmodels.formula.api import mixedlm
except Exception:
    mixedlm = None

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
mne.set_log_level("WARNING")
plt.rcParams["figure.dpi"] = 150


###############################################################################
# CONFIGURATION
###############################################################################

@dataclass
class Config:
    DATASET_DIR: Path = Path("data/ds002893-full")
    OUTPUT_DIR: Path = Path("results")
    CONDITIONS: Tuple[str, ...] = ("FA_Attended", "FA_Unattended", "FV_Attended", "FV_Unattended", "SF_Attended", "SF_Unattended")
    USE_TIMESTAMPED_RUN_DIR: bool = False
    RANDOM_STATE: int = 42
    L_FREQ: float = 0.1
    H_FREQ: float = 60.0
    NOTCH: float = 60.0
    TMIN: float = -0.20
    TMAX: float = 0.80
    BASELINE: Optional[Tuple[float, float]] = (-0.20, 0.00)
    P300_WINDOW: Tuple[float, float] = (0.30, 0.70)
    P300_WINDOWS: Tuple[Tuple[str, float, float], ...] = (("P300_300_600", 0.30, 0.60), ("P300_300_700", 0.30, 0.70))
    P300_CHANNELS: Tuple[str, ...] = ("Pz", "Cz", "CPz")
    ERP_CHANNELS: Tuple[str, ...] = ("Fz", "Cz", "CPz", "Pz", "Oz")
    ERP_WINDOWS: Tuple[Tuple[str, float, float, str], ...] = (("N100", 0.08, 0.15, "negative"), ("P200", 0.15, 0.25, "positive"), ("N200", 0.20, 0.35, "negative"), ("P300_300_600", 0.30, 0.60, "positive"), ("P300_300_700", 0.30, 0.70, "positive"), ("LatePositive", 0.50, 0.80, "positive"))
    SAVE_SINGLE_TRIAL_ERP: bool = True
    TARGET_EPOCH_SFREQ: Optional[float] = 250.0
    EPOCH_REJECT_PEAK_TO_PEAK_UV: Optional[float] = 100.0
    EPOCH_REJECT_MIN_KEEP_FRACTION: float = 0.20
    BAD_CHANNEL_DETECTION: bool = True
    INTERPOLATE_BAD_CHANNELS: bool = True
    BAD_CHANNEL_STD_Z: float = 6.0
    BAD_CHANNEL_PSD_Z: float = 6.0
    BAD_CHANNEL_FLAT_STD_UV: float = 0.5
    BAD_CHANNEL_HIGH_STD_UV: float = 1000.0
    BAD_CHANNEL_MAX_FRACTION: float = 0.33
    PREFER_PREPROCESSED_RAW_FIF: bool = True
    CROP_RAW_TO_SELECTED_EVENTS: bool = True
    RAW_CROP_PADDING_S: float = 5.0
    MEMORY_SAFE_EPOCH_FALLBACK: bool = True
    ICA_ENABLED: bool = True
    ICA_METHOD: str = "fastica"
    ICA_COMPONENTS: float = 0.99
    ICA_RANDOM_STATE: int = 42
    ICA_EOG_THRESHOLD: float = 3.0
    ICA_MAX_EXCLUDE: int = 2
    ICA_MIN_CLEAN_RMS_RATIO: float = 0.20
    N_CLUSTERS: int = 4
    AUTO_N_CLUSTERS: bool = True
    CLUSTER_RANGE: Tuple[int, int] = (3, 8)
    N_INIT: int = 100
    GFP_MIN_PEAK_DISTANCE: int = 10
    SEGMENT_FACTOR: int = 10
    SEGMENT_HALF_WINDOW_SIZE: int = 5
    SEGMENT_MIN_LENGTH: int = 5
    SEGMENT_REJECT_EDGES: bool = True
    ANALYSIS_MODE: str = "both"
    MICROSTATE_MODEL_MODE: str = "pooled_fa_fv"
    POOLED_MICROSTATE_MODEL_NAME: str = "Pooled_FA_FV_SF_Attended_Unattended"
    MICROSTATE_POOL_CONDITIONS: Tuple[str, ...] = ("FA_Attended", "FA_Unattended", "FV_Attended", "FV_Unattended", "SF_Attended", "SF_Unattended")
    EVENT_TYPE_FILTER: Optional[Tuple[str, ...]] = None
    TASK_ROLE_FILTER: Optional[Tuple[str, ...]] = None
    STIMULI_PER_EXPERIMENTAL_BLOCK: int = 240
    BLOCK_METADATA_COLUMN: str = "experimental_block"
    MICROSTATE_FIT_SOURCE: str = "erp_gfp_peaks"
    MICROSTATE_FIT_WINDOW: Optional[Tuple[float, float]] = (-0.20, 0.80)
    ERP_MICROSTATE_P300_WINDOW: Tuple[float, float] = (0.30, 0.60)
    BEHAVIOR_RESPONSE_WINDOW: Tuple[float, float] = (0.15, 1.50)
    TEMPLATE_DIR: Optional[Path] = None
    TEMPLATE_NAME: str = "MetaMaps_2023_06"
    TEMPLATE_MIN_CORR: float = 0.75
    APPLY_TEMPLATE_ALIGNMENT: bool = True
    TEMPLATE_INTERPOLATE_TO_MODEL: bool = True
    TEMPLATE_MIN_INTERPOLATION_CHANNELS: int = 8
    REORDER_TO_TEMPLATE: bool = True
    INVERT_TO_TEMPLATE_POLARITY: bool = True
    ENABLE_EPOCH_WINDOW_ANALYSIS: bool = True
    ENABLE_BLOCK_ANALYSIS: bool = True
    MICROSTATE_EPOCH_WINDOWS: Tuple[Tuple[str, float, float], ...] = (("Baseline", -0.20, 0.00), ("Stimulus_0_300ms", 0.00, 0.30), ("P300_300_700ms", 0.30, 0.70), ("Late_700_800ms", 0.70, 0.80))
    MAX_SEGMENTATION_PLOT_EPOCHS: int = 30
    SEGMENTATION_STEP_PLOT_EPOCHS: int = 8
    SAVE_TIME_SERIES_LONG: bool = True
    SAVE_TRANSITION_HEATMAPS: bool = True
    MIN_SF_BLOCK_EPOCHS: int = 1
    SAVE_FIF: bool = True
    SAVE_ANALYSIS_EPOCH_COPIES: bool = False
    SAVE_NPY: bool = True
    SAVE_CSV: bool = True
    SAVE_PNG: bool = True
    SAVE_PARAMETER_DISTRIBUTIONS: bool = True
    STAT_MIN_PER_GROUP: int = 3
    CACHE_VERSION: int = 5
    REUSE_PREPARED_CACHE: bool = True
    FORCE_PREPARE: bool = False
    PREPARE_ONLY: bool = False
    RUN_DIR: Path = field(init=False)
    COMMON_DIR: Path = field(init=False)
    SUBJECTS_DIR: Path = field(init=False)
    MASTER_DIR: Path = field(init=False)
    STATISTICS_DIR: Path = field(init=False)
    INTEGRATION_DIR: Path = field(init=False)
    FIGURES_DIR: Path = field(init=False)
    REPORTS_DIR: Path = field(init=False)
    LOGS_DIR: Path = field(init=False)
    CACHE_DIR: Path = field(init=False)

    def __post_init__(self) -> None:
        self.DATASET_DIR = Path(self.DATASET_DIR)
        self.OUTPUT_DIR = Path(self.OUTPUT_DIR)
        self.TEMPLATE_DIR = Path(self.TEMPLATE_DIR) if self.TEMPLATE_DIR is not None else self.DATASET_DIR
        stamp = datetime.now().strftime("Run_%Y%m%d_%H%M%S")
        self.RUN_DIR = self.OUTPUT_DIR / stamp if self.USE_TIMESTAMPED_RUN_DIR else self.OUTPUT_DIR
        self.COMMON_DIR = self.RUN_DIR / "Common_Models"
        self.SUBJECTS_DIR = self.RUN_DIR / "Subjects"
        self.MASTER_DIR = self.RUN_DIR / "Master"
        self.STATISTICS_DIR = self.RUN_DIR / "Statistics"
        self.INTEGRATION_DIR = self.RUN_DIR / "Integration"
        self.FIGURES_DIR = self.RUN_DIR / "Figures"
        self.REPORTS_DIR = self.RUN_DIR / "Reports"
        self.LOGS_DIR = self.RUN_DIR / "Logs"
        self.CACHE_DIR = self.RUN_DIR / "_Prepared_Cache"


CONDITION_MAP = {
    # Focus-auditory: auditory oddballs are attended; visual oddballs are unattended.
    "FA_Attended": {"label": "Focus Auditory - Attended Auditory", "base_condition": "FA", "condition": "attend_auditory", "target_events": ("high_tone",), "attention_status": "attended", "task_role": "infrequent_stimulus"},
    "FA_Unattended": {"label": "Focus Auditory - Unattended Visual", "base_condition": "FA", "condition": "attend_auditory", "target_events": ("light_bar",), "attention_status": "unattended", "task_role": "infrequent_stimulus"},
    "FA_All": {"label": "Focus Auditory - Attended and Unattended", "base_condition": "FA", "condition": "attend_auditory", "target_events": ("high_tone", "light_bar"), "attention_status": ("attended", "unattended"), "task_role": "infrequent_stimulus"},
    # Focus-visual: visual oddballs are attended; auditory oddballs are unattended.
    "FV_Attended": {"label": "Focus Visual - Attended Visual", "base_condition": "FV", "condition": "attend_visual", "target_events": ("light_bar",), "attention_status": "attended", "task_role": "infrequent_stimulus"},
    "FV_Unattended": {"label": "Focus Visual - Unattended Auditory", "base_condition": "FV", "condition": "attend_visual", "target_events": ("high_tone",), "attention_status": "unattended", "task_role": "infrequent_stimulus"},
    "FV_All": {"label": "Focus Visual - Attended and Unattended", "base_condition": "FV", "condition": "attend_visual", "target_events": ("high_tone", "light_bar"), "attention_status": ("attended", "unattended"), "task_role": "infrequent_stimulus"},
    # Shift-attention blocks retain both attended and unattended infrequent stimuli.
    "SF_Attended": {"label": "Shift Attention - Attended", "base_condition": "SF", "condition": "shift_attention", "target_events": ("high_tone", "light_bar"), "attention_status": "attended", "task_role": "infrequent_stimulus"},
    "SF_Unattended": {"label": "Shift Attention - Unattended", "base_condition": "SF", "condition": "shift_attention", "target_events": ("high_tone", "light_bar"), "attention_status": "unattended", "task_role": "infrequent_stimulus"},
    "SF_All": {"label": "Shift Attention - Attended and Unattended", "base_condition": "SF", "condition": "shift_attention", "target_events": ("high_tone", "light_bar"), "attention_status": ("attended", "unattended"), "task_role": "infrequent_stimulus"},
}

MICROSTATE_METRICS = ("mean_corr", "gev", "timecov", "meandurs", "occurrences", "dist_corr", "dist_gev", "dist_durs")
IDENTIFIER_COLUMNS = ("Participant", "Group", "Age", "Sex", "Condition", "Window", "Block")
TEMPLATE_FILES = {"Custo2017": "Custo2017.set", "Koenig2002": "Koenig2002.set", "MetaMaps_2023_06": "MetaMaps_2023_06.set"}
TEMPLATE_LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
# Legacy 10-20 names are still used in several published microstate templates.
# Match them to their modern equivalents so template correlations do not lose
# valid homologous temporal channels.
CHANNEL_NAME_ALIASES = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
METRIC_UNITS = {"mean_corr": "correlation", "gev": "proportion", "total_gev": "proportion", "timecov": "proportion", "meandurs": "seconds", "occurrences": "segments_per_second", "unlabeled_ratio": "proportion", "entropy_bits": "bits", "entropy_no_repetitions_bits": "bits", "lz_complexity": "count", "lz_complexity_normalized": "proportion", "markov_entropy_bits": "bits_per_transition", "switching_rate_hz": "switches_per_second", "mean_segment_duration_ms": "milliseconds", "valid_label_ratio": "proportion"}


###############################################################################
# LOGGING, PATHS, AND DATASET DISCOVERY
###############################################################################

def mkdirs(cfg: Config) -> None:
    for folder in (cfg.RUN_DIR, cfg.COMMON_DIR, cfg.SUBJECTS_DIR, cfg.MASTER_DIR, cfg.STATISTICS_DIR, cfg.INTEGRATION_DIR, cfg.FIGURES_DIR, cfg.REPORTS_DIR, cfg.LOGS_DIR, cfg.CACHE_DIR):
        folder.mkdir(parents=True, exist_ok=True)


def setup_logging(cfg: Config) -> None:
    mkdirs(cfg)
    log_file = cfg.LOGS_DIR / "Pipeline_Log.txt"
    handlers = [logging.FileHandler(log_file, mode="w", encoding="utf-8"), logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", handlers=handlers, force=True)
    logging.info("Pipeline log: %s", log_file)
    save_config(cfg)


def banner(title: str, char: str = "=") -> None:
    logging.info("\n%s\n%s\n%s", char * 88, title, char * 88)


def ensure_microstate_dependencies() -> None:
    if ModKMeans is None or extract_gfp_peaks is None or ChData is None:
        raise RuntimeError(f"pycrostates is required for ANALYSIS_MODE='both' or 'microstate'. Import error: {PYCROSTATES_IMPORT_ERROR}")


def json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    return value


def save_config(cfg: Config) -> None:
    config = {key: json_ready(value) for key, value in cfg.__dict__.items()}
    (cfg.RUN_DIR / "Analysis_Config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def load_participants(cfg: Config) -> pd.DataFrame:
    path = cfg.DATASET_DIR / "participants.tsv"
    if not path.exists():
        raise FileNotFoundError(f"Missing participants.tsv: {path}")
    participants = pd.read_csv(path, sep="\t", na_values=["n/a", "NA", "nan"]).sort_values("participant_id").reset_index(drop=True)
    participants["participant_id"] = participants["participant_id"].astype(str)
    participants["age"] = pd.to_numeric(participants.get("age"), errors="coerce")
    banner("PARTICIPANTS")
    logging.info("Total subjects: %s", len(participants))
    if "group" in participants.columns:
        logging.info("Group counts:\n%s", participants["group"].value_counts(dropna=False).to_string())
    return participants


@dataclass
class Recording:
    participant_id: str
    group: str
    age: float
    sex: str
    eeg_file: Path
    raw_fif_file: Path
    events_file: Path
    channels_file: Path
    eeg_json_file: Path
    events: pd.DataFrame
    metadata: Dict
    eog_channels: List[str]
    misc_channels: List[str]


class BIDSDataset:
    def __init__(self, root: Path):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.root}")

    def subject_dir(self, participant_id: str) -> Path:
        return self.root / participant_id / "eeg"

    def files_for(self, participant_id: str) -> Dict[str, Path]:
        eeg_dir = self.subject_dir(participant_id)
        eeg_files = sorted(eeg_dir.glob("*_eeg.set"))
        raw_fif_files = sorted(eeg_dir.glob("*_desc-preproc_raw.fif"))
        event_files = sorted(eeg_dir.glob("*_events.tsv"))
        channel_files = sorted(eeg_dir.glob("*_channels.tsv"))
        json_files = sorted(eeg_dir.glob("*_eeg.json"))
        if (not eeg_files and not raw_fif_files) or not event_files:
            raise FileNotFoundError(f"Missing EEG raw file or events.tsv for {participant_id} in {eeg_dir}")
        return {"eeg": eeg_files[0] if eeg_files else Path(), "raw_fif": raw_fif_files[0] if raw_fif_files else Path(), "events": event_files[0], "channels": channel_files[0] if channel_files else Path(), "eeg_json": json_files[0] if json_files else Path()}

    def load_recording(self, participant_row: pd.Series) -> Recording:
        participant_id = str(participant_row["participant_id"])
        files = self.files_for(participant_id)
        events = pd.read_csv(files["events"], sep="\t", na_values=["n/a", "NA", "nan"])
        metadata = read_json(files["eeg_json"])
        eog_channels, misc_channels = channel_groups(files["channels"])
        return Recording(participant_id=participant_id, group=str(participant_row.get("group", "unknown")), age=float(participant_row.get("age", np.nan)), sex=str(participant_row.get("sex", "unknown")), eeg_file=files["eeg"], raw_fif_file=files["raw_fif"], events_file=files["events"], channels_file=files["channels"], eeg_json_file=files["eeg_json"], events=events, metadata=metadata, eog_channels=eog_channels, misc_channels=misc_channels)


def channel_groups(channels_file: Path) -> Tuple[List[str], List[str]]:
    if not channels_file.exists():
        return [], []
    channels = pd.read_csv(channels_file, sep="\t")
    type_col = channels["type"].astype(str).str.lower()
    eog = channels.loc[type_col.eq("eog"), "name"].astype(str).tolist()
    misc = channels.loc[type_col.isin(["misc", "stim", "trigger"]), "name"].astype(str).tolist()
    return eog, misc


def subject_dir(cfg: Config, participant_id: str) -> Path:
    path = cfg.SUBJECTS_DIR / participant_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def split_filter_tokens(values: Optional[Sequence[str]]) -> Optional[Tuple[str, ...]]:
    if not values:
        return None
    tokens: List[str] = []
    for value in values:
        text = str(value).replace(",", " ").replace(";", " ").replace("|", " ")
        tokens.extend(item.strip() for item in text.split() if item.strip())
    if not tokens or any(item.casefold() in {"all", "*", "none", "default"} for item in tokens):
        return None
    return tuple(dict.fromkeys(tokens))


def scan_dataset_for_gui(dataset_dir: Path, max_event_files: int = 25, max_rows: int = 20000) -> Dict[str, object]:
    dataset_dir = Path(dataset_dir)
    result: Dict[str, object] = {
        "dataset": str(dataset_dir),
        "participants": 0,
        "groups": [],
        "subjects_by_group": {},
        "event_values": {},
        "templates": [],
    }
    participants_path = dataset_dir / "participants.tsv"
    if participants_path.exists():
        participants = pd.read_csv(participants_path, sep="\t", na_values=["n/a", "NA", "nan"])
        result["participants"] = int(len(participants))
        if "participant_id" in participants.columns and "group" in participants.columns:
            subjects_by_group = {}
            for group, part in participants.groupby(participants["group"].astype(str), dropna=False):
                subjects_by_group[str(group)] = part["participant_id"].dropna().astype(str).sort_values().tolist()
            result["subjects_by_group"] = subjects_by_group
            result["groups"] = sorted(subjects_by_group)
    value_columns = ("condition", "event_type", "attention_status", "task_role", "focus_modality")
    event_values: Dict[str, set] = {column: set() for column in value_columns}
    for events_file in sorted(dataset_dir.rglob("*_events.tsv"))[:max_event_files]:
        try:
            events = pd.read_csv(events_file, sep="\t", nrows=max_rows, na_values=["n/a", "NA", "nan"])
        except Exception:
            continue
        for column in value_columns:
            if column not in events.columns:
                continue
            values = events[column].dropna().astype(str).str.strip()
            values = values[(values != "") & (values.str.casefold() != "nan")]
            event_values[column].update(values.unique().tolist())
    result["event_values"] = {column: sorted(values) for column, values in event_values.items()}
    result["templates"] = sorted(path.name for path in dataset_dir.glob("*.set") if path.name in TEMPLATE_FILES.values())
    return result


def condition_dir(cfg: Config, participant_id: str, condition: str, window: str = "Whole") -> Path:
    if window == "Whole":
        path = subject_dir(cfg, participant_id) / condition
    elif condition == "SF" and window.startswith("Block_"):
        path = subject_dir(cfg, participant_id) / "SF_Blocks" / window
    else:
        path = subject_dir(cfg, participant_id) / condition / window
    path.mkdir(parents=True, exist_ok=True)
    return path


###############################################################################
# EEG LOADING, PREPROCESSING, AND EPOCHING
###############################################################################

def crop_raw_to_selected_events(raw: mne.io.BaseRaw, recording: Recording, cfg: Config) -> mne.io.BaseRaw:
    if not cfg.CROP_RAW_TO_SELECTED_EVENTS:
        return raw
    samples: List[int] = []
    for condition in cfg.CONDITIONS:
        try:
            df = filter_events_for_condition(recording.events, condition, cfg)
        except Exception:
            continue
        if "sample" in df.columns:
            samples.extend(pd.to_numeric(df["sample"], errors="coerce").dropna().astype(int).tolist())
    if not samples:
        return raw
    sfreq = float(raw.info["sfreq"])
    pad = max(0, int(round(float(cfg.RAW_CROP_PADDING_S) * sfreq)))
    start_sample = max(int(raw.first_samp), int(min(samples) + math.floor(cfg.TMIN * sfreq)) - pad)
    stop_sample = min(int(raw.first_samp + raw.n_times - 1), int(max(samples) + math.ceil(cfg.TMAX * sfreq)) + pad)
    if stop_sample <= start_sample:
        return raw
    tmin = max(0.0, (start_sample - raw.first_samp) / sfreq)
    tmax = min(float(raw.n_times - 1) / sfreq, (stop_sample - raw.first_samp) / sfreq)
    if tmax <= tmin:
        return raw
    logging.info("Cropping raw before preload to selected-event span %.2f..%.2f s with %.1f s padding.", tmin, tmax, float(cfg.RAW_CROP_PADDING_S))
    return raw.copy().crop(tmin=tmin, tmax=tmax)


def load_raw_eeg(recording: Recording, cfg: Config) -> mne.io.BaseRaw:
    banner(f"LOADING EEG: {recording.participant_id}", "-")
    candidates: List[Tuple[str, Path]] = []
    if cfg.PREFER_PREPROCESSED_RAW_FIF and recording.raw_fif_file.exists():
        candidates.append(("MNE FIF", recording.raw_fif_file))
    if recording.eeg_file.exists():
        candidates.append(("EEGLAB SET", recording.eeg_file))
    if not cfg.PREFER_PREPROCESSED_RAW_FIF and recording.raw_fif_file.exists():
        candidates.append(("MNE FIF", recording.raw_fif_file))
    if not candidates:
        raise FileNotFoundError(f"No readable EEG raw file was found for {recording.participant_id}.")
    last_error: Optional[BaseException] = None
    raw: Optional[mne.io.BaseRaw] = None
    for source_label, path in candidates:
        candidate = None
        try:
            logging.info("%s file: %s", source_label, path)
            if path.suffix.lower() == ".fif":
                candidate = mne.io.read_raw_fif(path, preload=False, verbose=False)
            else:
                candidate = mne.io.read_raw_eeglab(path, preload=False, verbose=False)
            candidate = crop_raw_to_selected_events(candidate, recording, cfg)
            candidate.load_data(verbose=False)
            raw = candidate
            logging.info("Loaded %s for %s", source_label, recording.participant_id)
            break
        except MemoryError as exc:
            last_error = exc
            logging.warning("%s failed with MemoryError for %s; trying next available raw source.", source_label, recording.participant_id)
        except Exception as exc:
            last_error = exc
            logging.warning("%s failed for %s: %s; trying next available raw source.", source_label, recording.participant_id, exc)
    if raw is None:
        raise RuntimeError(f"Could not load EEG for {recording.participant_id}; last error: {last_error}") from last_error
    expected_sfreq = recording.metadata.get("SamplingFrequency")
    if expected_sfreq is not None and abs(float(raw.info["sfreq"]) - float(expected_sfreq)) > 1e-3:
        raise RuntimeError(f"Sampling frequency mismatch for {recording.participant_id}: raw={raw.info['sfreq']} json={expected_sfreq}")
    duration = float(raw.n_times - 1) / float(raw.info["sfreq"]) if raw.n_times else 0.0
    logging.info("Sampling frequency: %.3f Hz | duration: %.2f sec | channels: %s", raw.info["sfreq"], duration, len(raw.ch_names))
    return raw


def assign_channel_types(raw: mne.io.BaseRaw, recording: Recording) -> mne.io.BaseRaw:
    mapping = {ch: "eog" for ch in recording.eog_channels if ch in raw.ch_names}
    mapping.update({ch: "misc" for ch in recording.misc_channels if ch in raw.ch_names})
    if mapping:
        raw.set_channel_types(mapping, verbose=False)
    try:
        raw.set_montage("standard_1020", match_case=False, on_missing="ignore", verbose=False)
    except Exception as exc:
        logging.warning("Could not set standard_1020 montage for %s: %s", recording.participant_id, exc)
    logging.info("Channel types assigned | EOG=%s | MISC=%s", len(recording.eog_channels), len(recording.misc_channels))
    return raw


def qc_table(raw: mne.io.BaseRaw, title: str, out_file: Optional[Path] = None) -> pd.DataFrame:
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    rows = []
    for pick in picks:
        idx = int(pick)
        ch = raw.ch_names[idx]
        data = np.asarray(raw._data[idx] if raw.preload else raw.get_data(picks=[idx])[0], dtype=float)
        rows.append({"Channel": ch, "STD_uV": float(np.nanstd(data) * 1e6), "RMS_uV": float(np.sqrt(np.nanmean(data * data)) * 1e6), "Min_uV": float(np.nanmin(data) * 1e6), "Max_uV": float(np.nanmax(data) * 1e6)})
    df = pd.DataFrame(rows)
    logging.info("%s | EEG channels=%s | mean STD=%.3f uV | max STD=%.3f uV", title, len(picks), df["STD_uV"].mean() if len(df) else np.nan, df["STD_uV"].max() if len(df) else np.nan)
    if out_file is not None:
        df.to_csv(out_file, index=False)
    return df


def robust_z(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    out = np.zeros(arr.shape, dtype=float)
    finite = np.isfinite(arr)
    if finite.sum() < 3:
        return out
    med = float(np.nanmedian(arr[finite]))
    mad = float(np.nanmedian(np.abs(arr[finite] - med)))
    if not np.isfinite(mad) or mad == 0:
        sd = float(np.nanstd(arr[finite]))
        return (arr - med) / sd if np.isfinite(sd) and sd > 0 else out
    out[finite] = 0.6745 * (arr[finite] - med) / mad
    out[~finite] = np.nan
    return out


def channelwise_bad_channel_metrics(raw: mne.io.BaseRaw, cfg: Config) -> pd.DataFrame:
    sfreq = float(raw.info["sfreq"])
    nperseg = max(8, min(int(round(sfreq * 4.0)), int(raw.n_times)))
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    rows = []
    for pick in picks:
        idx = int(pick)
        ch = raw.ch_names[idx]
        try:
            signal = raw._data[idx] if raw.preload else raw.get_data(picks=[idx], reject_by_annotation="omit")[0]
        except TypeError:
            signal = raw._data[idx] if raw.preload else raw.get_data(picks=[idx])[0]
        signal_uv = np.asarray(signal, dtype=np.float64) * 1e6
        signal_uv = signal_uv[np.isfinite(signal_uv)]
        if signal_uv.size == 0:
            rows.append({"Channel": ch, "STD_uV": np.nan, "RMS_uV": np.nan, "PeakToPeak_uV": np.nan, "BroadbandPower": np.nan, "LineNoiseRatio": np.nan})
            continue
        std_uv = float(np.nanstd(signal_uv))
        rms_uv = float(np.sqrt(np.nanmean(signal_uv * signal_uv)))
        ptp_uv = float(np.nanmax(signal_uv) - np.nanmin(signal_uv))
        try:
            freqs, psd = scipy.signal.welch(signal_uv, fs=sfreq, nperseg=min(nperseg, signal_uv.size), axis=-1)
            broad_mask = (freqs >= max(1.0, cfg.L_FREQ or 1.0)) & (freqs <= min(45.0, cfg.H_FREQ or 45.0))
            line_mask = (freqs >= max(0.0, cfg.NOTCH - 2.0)) & (freqs <= cfg.NOTCH + 2.0) if cfg.NOTCH else np.zeros(freqs.shape, dtype=bool)
            broad_power = float(np.nanmean(psd[broad_mask])) if broad_mask.any() else float(np.nanmean(psd))
            line_power = float(np.nanmean(psd[line_mask])) if line_mask.any() else 0.0
            line_ratio = line_power / max(broad_power, np.finfo(float).eps)
        except MemoryError as exc:
            logging.warning("Memory-safe PSD still failed for channel %s: %s", ch, exc)
            broad_power = np.nan
            line_ratio = np.nan
        except Exception as exc:
            logging.warning("PSD metric failed for channel %s: %s", ch, exc)
            broad_power = np.nan
            line_ratio = np.nan
        rows.append({"Channel": ch, "STD_uV": std_uv, "RMS_uV": rms_uv, "PeakToPeak_uV": ptp_uv, "BroadbandPower": broad_power, "LineNoiseRatio": line_ratio})
    return pd.DataFrame(rows)


def raw_eeg_rms(raw: mne.io.BaseRaw) -> float:
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    total = 0.0
    n_values = 0
    for pick in picks:
        idx = int(pick)
        data = np.asarray(raw._data[idx] if raw.preload else raw.get_data(picks=[idx])[0], dtype=float)
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            continue
        total += float(np.sum(finite * finite))
        n_values += int(finite.size)
    return float(math.sqrt(total / n_values)) if n_values else np.nan


def detect_bad_eeg_channels(raw: mne.io.BaseRaw, recording: Recording, cfg: Config, out_dir: Path) -> List[str]:
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    ch_names = [raw.ch_names[int(pick)] for pick in picks]
    if len(ch_names) == 0:
        return []
    metrics = channelwise_bad_channel_metrics(raw, cfg)
    std_uv = metrics["STD_uV"].to_numpy(dtype=float)
    rms_uv = metrics["RMS_uV"].to_numpy(dtype=float)
    ptp_uv = metrics["PeakToPeak_uV"].to_numpy(dtype=float)
    broad_power = metrics["BroadbandPower"].to_numpy(dtype=float)
    line_ratio = metrics["LineNoiseRatio"].to_numpy(dtype=float)
    log_std_z = robust_z(np.log10(np.maximum(std_uv, 1e-12)))
    log_rms_z = robust_z(np.log10(np.maximum(rms_uv, 1e-12)))
    log_power_z = robust_z(np.log10(np.maximum(broad_power, np.finfo(float).eps)))
    line_ratio_z = robust_z(np.log10(np.maximum(line_ratio, np.finfo(float).eps)))
    existing_bads = set(ch for ch in raw.info.get("bads", []) if ch in ch_names)
    rows = []
    detected = []
    for idx, ch in enumerate(ch_names):
        reasons = []
        if ch in existing_bads:
            reasons.append("pre_marked_bad")
        if np.isfinite(std_uv[idx]) and std_uv[idx] < cfg.BAD_CHANNEL_FLAT_STD_UV:
            reasons.append("flat_or_nearly_flat")
        if np.isfinite(std_uv[idx]) and std_uv[idx] > cfg.BAD_CHANNEL_HIGH_STD_UV:
            reasons.append("very_high_std")
        if np.isfinite(log_std_z[idx]) and abs(log_std_z[idx]) >= cfg.BAD_CHANNEL_STD_Z:
            reasons.append("std_robust_z")
        if np.isfinite(log_rms_z[idx]) and abs(log_rms_z[idx]) >= cfg.BAD_CHANNEL_STD_Z:
            reasons.append("rms_robust_z")
        if np.isfinite(log_power_z[idx]) and abs(log_power_z[idx]) >= cfg.BAD_CHANNEL_PSD_Z:
            reasons.append("psd_broadband_robust_z")
        if np.isfinite(line_ratio_z[idx]) and line_ratio_z[idx] >= cfg.BAD_CHANNEL_PSD_Z:
            reasons.append("psd_line_noise_robust_z")
        severity_values = np.asarray([log_std_z[idx], log_rms_z[idx], log_power_z[idx], line_ratio_z[idx]], dtype=float)
        severity_values = np.abs(severity_values[np.isfinite(severity_values)])
        severity = float(np.max(severity_values)) if severity_values.size else 0.0
        is_bad = bool(reasons)
        if is_bad:
            detected.append((ch, severity, reasons))
        rows.append({"Participant": recording.participant_id, "Channel": ch, "Is_Bad": is_bad, "Reasons": ";".join(reasons), "STD_uV": float(std_uv[idx]), "RMS_uV": float(rms_uv[idx]), "PeakToPeak_uV": float(ptp_uv[idx]), "STD_RobustZ": float(log_std_z[idx]), "RMS_RobustZ": float(log_rms_z[idx]), "BroadbandPSD_RobustZ": float(log_power_z[idx]), "LineNoiseRatio_RobustZ": float(line_ratio_z[idx]), "Severity": severity})
    detected = sorted(detected, key=lambda item: item[1], reverse=True)
    max_bad = max(1, int(math.floor(len(ch_names) * cfg.BAD_CHANNEL_MAX_FRACTION)))
    selected = {ch for ch, _, _ in detected[:max_bad]}
    if len(detected) > max_bad:
        logging.warning("%s: detected %s bad EEG channels; interpolating top %s by severity to avoid over-interpolation.", recording.participant_id, len(detected), max_bad)
    table = pd.DataFrame(rows)
    if not table.empty:
        table["Selected_For_Interpolation"] = table["Channel"].isin(selected)
        table.to_csv(out_dir / "Bad_Channel_Detection.csv", index=False)
    selected_list = [ch for ch, _, _ in detected if ch in selected]
    if selected_list:
        logging.info("%s: bad EEG channels selected for interpolation: %s", recording.participant_id, selected_list)
    else:
        logging.info("%s: no bad EEG channels selected for interpolation.", recording.participant_id)
    return selected_list


def interpolate_bad_eeg_channels(raw: mne.io.BaseRaw, bad_channels: Sequence[str], recording: Recording, cfg: Config, out_dir: Path) -> mne.io.BaseRaw:
    bads = [ch for ch in bad_channels if ch in raw.ch_names]
    if not bads:
        return raw
    raw.info["bads"] = sorted(set(raw.info.get("bads", [])) | set(bads))
    if not cfg.INTERPOLATE_BAD_CHANNELS:
        logging.info("%s: bad-channel interpolation disabled; marked bads retained: %s", recording.participant_id, raw.info["bads"])
        pd.DataFrame({"Participant": recording.participant_id, "Channel": bads, "Status": "marked_only", "Error": ""}).to_csv(out_dir / "Interpolated_Bad_Channels.csv", index=False)
        return raw
    success = False
    error = ""
    try:
        raw.interpolate_bads(reset_bads=True, mode="accurate", verbose=False)
        success = True
    except TypeError:
        try:
            raw.interpolate_bads(reset_bads=True, verbose=False)
            success = True
        except Exception as exc:
            error = str(exc)
            logging.warning("%s: bad-channel interpolation failed: %s", recording.participant_id, exc)
    except Exception as exc:
        error = str(exc)
        logging.warning("%s: bad-channel interpolation failed: %s", recording.participant_id, exc)
    status = "interpolated" if success else "failed"
    pd.DataFrame({"Participant": recording.participant_id, "Channel": bads, "Status": status, "Error": error}).to_csv(out_dir / "Interpolated_Bad_Channels.csv", index=False)
    if success:
        logging.info("%s: interpolated bad EEG channels: %s", recording.participant_id, bads)
    else:
        logging.warning("%s: bad EEG channels remain marked because interpolation did not complete: %s", recording.participant_id, raw.info.get("bads", []))
    return raw


def apply_notch_filter_memory_safe(raw: mne.io.BaseRaw, picks: Sequence[int], cfg: Config) -> None:
    if not cfg.NOTCH:
        return
    try:
        raw.notch_filter(freqs=cfg.NOTCH, picks=picks, method="iir", n_jobs=1, verbose=False)
        return
    except MemoryError as exc:
        logging.warning("MNE IIR notch filter hit MemoryError; falling back to channel-wise SciPy notch: %s", exc)
    except Exception as exc:
        logging.warning("MNE IIR notch filter failed; falling back to channel-wise SciPy notch: %s", exc)
    if not raw.preload:
        raw.load_data(verbose=False)
    sfreq = float(raw.info["sfreq"])
    if cfg.NOTCH <= 0 or cfg.NOTCH >= sfreq / 2.0:
        logging.warning("Skipping SciPy notch because %.3f Hz is outside valid range for sfreq %.3f Hz.", cfg.NOTCH, sfreq)
        return
    b, a = scipy.signal.iirnotch(w0=float(cfg.NOTCH), Q=30.0, fs=sfreq)
    for pick in picks:
        idx = int(pick)
        try:
            raw._data[idx] = scipy.signal.filtfilt(b, a, raw._data[idx], method="gust")
        except Exception as exc:
            logging.warning("Channel-wise notch failed for %s: %s", raw.ch_names[idx], exc)


def component_eog_score(scores, idx: int) -> float:
    try:
        arr = np.asarray(scores, dtype=float)
        if arr.ndim == 1 and idx < arr.size:
            return float(arr[idx])
        if arr.ndim == 2 and idx < arr.shape[-1]:
            return float(np.nanmax(np.abs(arr[..., idx])))
    except Exception:
        pass
    return np.nan


def select_ica_components(eog_inds: Sequence[int], scores, cfg: Config) -> List[int]:
    candidates = []
    for idx in sorted(set(int(x) for x in eog_inds)):
        score = component_eog_score(scores, idx)
        candidates.append((idx, abs(score) if np.isfinite(score) else np.inf, score))
    candidates.sort(key=lambda item: item[1], reverse=True)
    return [idx for idx, _, _ in candidates[:max(0, int(cfg.ICA_MAX_EXCLUDE))]]


def preprocess_recording(recording: Recording, cfg: Config) -> Tuple[mne.io.BaseRaw, Optional[ICA]]:
    raw = assign_channel_types(load_raw_eeg(recording, cfg), recording)
    out = subject_dir(cfg, recording.participant_id)
    qc_table(raw, "QC before preprocessing", out / "QC_Before_Preprocessing.csv")
    filter_picks = mne.pick_types(raw.info, eeg=True, eog=True, exclude=[])
    raw.filter(l_freq=cfg.L_FREQ, h_freq=cfg.H_FREQ, picks=filter_picks, verbose=False)
    apply_notch_filter_memory_safe(raw, filter_picks, cfg)
    if cfg.BAD_CHANNEL_DETECTION:
        bad_channels = detect_bad_eeg_channels(raw, recording, cfg, out)
        raw = interpolate_bad_eeg_channels(raw, bad_channels, recording, cfg, out)
        qc_table(raw, "QC after bad-channel interpolation", out / "QC_After_Bad_Channel_Interpolation.csv")
    raw.set_eeg_reference("average", projection=False, verbose=False)
    ica = None
    if cfg.ICA_ENABLED:
        raw, ica = run_ica(raw, recording, cfg)
    qc_table(raw, "QC after preprocessing", out / "QC_After_Preprocessing.csv")
    raw.pick("eeg")
    return raw, ica


def run_ica(raw: mne.io.BaseRaw, recording: Recording, cfg: Config) -> Tuple[mne.io.BaseRaw, ICA]:
    banner(f"ICA: {recording.participant_id}", "-")
    raw_ica = raw.copy().filter(l_freq=1.0, h_freq=None, picks="eeg", verbose=False)
    ica = ICA(n_components=cfg.ICA_COMPONENTS, method=cfg.ICA_METHOD, random_state=cfg.ICA_RANDOM_STATE, max_iter="auto")
    ica.fit(raw_ica, picks="eeg", verbose=False)
    eog_inds: List[int] = []
    eog_scores = None
    if recording.eog_channels:
        try:
            eog_chs = [ch for ch in recording.eog_channels if ch in raw_ica.ch_names]
            if eog_chs:
                eog_inds, eog_scores = ica.find_bads_eog(raw_ica, ch_name=eog_chs, threshold=cfg.ICA_EOG_THRESHOLD, verbose=False)
        except Exception as exc:
            logging.warning("EOG detection failed for %s: %s", recording.participant_id, exc)
    selected = select_ica_components(eog_inds, eog_scores, cfg)
    if len(eog_inds) > len(selected):
        logging.warning("%s: ICA EOG detection suggested %s components %s; limiting removal to %s component(s): %s", recording.participant_id, len(eog_inds), list(map(int, eog_inds)), cfg.ICA_MAX_EXCLUDE, selected)
    ica.exclude = selected
    score_rows = []
    for idx in sorted(set(int(x) for x in eog_inds) | set(selected)):
        score_rows.append({"Participant": recording.participant_id, "Component": idx, "EOG_Score": component_eog_score(eog_scores, idx), "Suggested_By_MNE": idx in set(map(int, eog_inds)), "Selected_For_Removal": idx in set(selected)})
    pd.DataFrame(score_rows).to_csv(subject_dir(cfg, recording.participant_id) / "ICA_Component_Rejection.csv", index=False)
    logging.info("Rejected ICA components for %s: %s", recording.participant_id, ica.exclude)
    clean = raw.copy()
    before_rms = raw_eeg_rms(raw)
    reverted = False
    if ica.exclude:
        ica.apply(clean, verbose=False)
    after_rms = raw_eeg_rms(clean)
    rms_ratio = after_rms / before_rms if np.isfinite(before_rms) and before_rms > 0 else np.nan
    if ica.exclude and np.isfinite(rms_ratio) and rms_ratio < cfg.ICA_MIN_CLEAN_RMS_RATIO:
        logging.warning("%s: ICA cleaning reduced EEG RMS ratio to %.3f below %.3f; reverting ICA removal to protect neural data.", recording.participant_id, rms_ratio, cfg.ICA_MIN_CLEAN_RMS_RATIO)
        clean = raw.copy()
        ica.exclude = []
        after_rms = before_rms
        rms_ratio = 1.0
        reverted = True
    pd.DataFrame([{"Participant": recording.participant_id, "N_Components": int(getattr(ica, "n_components_", 0) or 0), "Selected_Components": ",".join(map(str, ica.exclude)), "Before_RMS_uV": before_rms * 1e6 if np.isfinite(before_rms) else np.nan, "After_RMS_uV": after_rms * 1e6 if np.isfinite(after_rms) else np.nan, "After_Before_RMS_Ratio": rms_ratio, "Reverted_To_Uncleaned": reverted}]).to_csv(subject_dir(cfg, recording.participant_id) / "ICA_QC.csv", index=False)
    try:
        ica.save(subject_dir(cfg, recording.participant_id) / "ICA_Solution.fif", overwrite=True)
    except Exception as exc:
        logging.warning("Could not save ICA solution for %s: %s", recording.participant_id, exc)
    return clean, ica


def _is_missing_scalar(value) -> bool:
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(missing, (np.ndarray, pd.Series, pd.Index, list, tuple)):
        return False
    return bool(missing)


def _spec_options(value) -> Tuple[str, ...]:
    """Return condition-map values as normalized strings for pandas isin()."""
    if value is None or _is_missing_scalar(value):
        return tuple()
    if isinstance(value, (str, bytes)):
        values = (value,)
    elif isinstance(value, (list, tuple, set, frozenset, np.ndarray, pd.Index, pd.Series)):
        values = tuple(value)
    else:
        values = (value,)
    return tuple(str(item).strip().casefold() for item in values if not _is_missing_scalar(item))


def _event_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.casefold()


def _filter_events_by_options(df: pd.DataFrame, column: str, options) -> pd.DataFrame:
    allowed = _spec_options(options)
    if not allowed:
        return df
    return df.loc[_event_text(df[column]).isin(allowed)]


def add_experimental_block_labels(events: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = events.copy()
    block_column = cfg.BLOCK_METADATA_COLUMN
    if block_column not in df.columns:
        df[block_column] = np.nan
    if "condition" not in df.columns or "task_role" not in df.columns:
        return df
    stimuli_per_block = max(1, int(cfg.STIMULI_PER_EXPERIMENTAL_BLOCK))
    ordered = df.sort_values("sample" if "sample" in df.columns else "onset").copy()
    condition_values = ordered["condition"].astype(object).where(ordered["condition"].notna(), "__missing__").to_numpy()
    run_start = np.r_[True, condition_values[1:] != condition_values[:-1]]
    ordered["_condition_run"] = np.cumsum(run_start)
    block_offsets: Dict[str, int] = {}
    for _, run_df in ordered.groupby("_condition_run", sort=False):
        condition = str(run_df["condition"].dropna().iloc[0]) if run_df["condition"].notna().any() else "unknown"
        offset = block_offsets.get(condition, 0)
        task_roles = run_df["task_role"].astype(object).where(run_df["task_role"].notna(), "").map(lambda value: str(value).casefold())
        stim_mask = task_roles.isin(["frequent_stimulus", "infrequent_stimulus"]).to_numpy(dtype=bool)
        stim_indices = run_df.index[stim_mask].tolist()
        if not stim_indices:
            continue
        stim_order = np.arange(1, len(stim_indices) + 1)
        block_numbers = offset + ((stim_order - 1) // stimuli_per_block) + 1
        ordered.loc[stim_indices, block_column] = block_numbers.astype(int)
        block_offsets[condition] = int(offset + np.max(block_numbers - offset))
    df.loc[ordered.index, block_column] = ordered[block_column]
    return df


def filter_events_for_condition(events: pd.DataFrame, condition_name: str, cfg: Optional[Config] = None) -> pd.DataFrame:
    if condition_name not in CONDITION_MAP:
        raise KeyError(f"Unknown condition {condition_name!r}; expected one of {sorted(CONDITION_MAP)}")
    spec = CONDITION_MAP[condition_name]
    df = add_experimental_block_labels(events, cfg) if cfg is not None else events.copy()
    for col in ("condition", "attention_status", "task_role", "event_type", "sample", "onset", "sub_block", "trial", "focus_modality", "event_code", "cond_code", "experimental_block"):
        if col not in df.columns:
            df[col] = np.nan
    event_type_filter = cfg.EVENT_TYPE_FILTER if cfg is not None and cfg.EVENT_TYPE_FILTER else spec.get("target_events")
    task_role_filter = cfg.TASK_ROLE_FILTER if cfg is not None and cfg.TASK_ROLE_FILTER else spec.get("task_role")
    df = _filter_events_by_options(df, "condition", spec.get("condition"))
    df = _filter_events_by_options(df, "attention_status", spec.get("attention_status"))
    df = _filter_events_by_options(df, "task_role", task_role_filter)
    df = _filter_events_by_options(df, "event_type", event_type_filter)
    df["sample"] = pd.to_numeric(df["sample"], errors="coerce")
    df["onset"] = pd.to_numeric(df["onset"], errors="coerce")
    return df.dropna(subset=["sample"]).sort_values("sample").reset_index(drop=True)


def create_condition_epochs(raw: mne.io.BaseRaw, recording: Recording, condition_name: str, cfg: Config, baseline: bool = True) -> Optional[mne.Epochs]:
    df = filter_events_for_condition(recording.events, condition_name, cfg)
    if df.empty:
        logging.warning("%s %s: no target events", recording.participant_id, condition_name)
        return None
    samples = df["sample"].astype(int).to_numpy()
    sfreq = float(raw.info["sfreq"])
    valid = (samples + int(round(cfg.TMIN * sfreq)) >= raw.first_samp) & (samples + int(round(cfg.TMAX * sfreq)) < raw.first_samp + raw.n_times)
    df = df.loc[valid].reset_index(drop=True)
    samples = samples[valid]
    if len(samples) == 0:
        logging.warning("%s %s: all target events outside epoch bounds", recording.participant_id, condition_name)
        return None
    metadata_cols = ["condition", cfg.BLOCK_METADATA_COLUMN, "sub_block", "trial", "focus_modality", "event_type", "attention_status", "task_role", "event_code", "cond_code", "sample", "onset"]
    metadata = df[metadata_cols].copy()
    metadata.insert(0, "participant_id", recording.participant_id)
    metadata.insert(1, "condition_label", condition_name)
    metadata["epoch_id"] = np.arange(1, len(metadata) + 1)
    events = np.column_stack([samples, np.zeros(len(samples), dtype=int), np.ones(len(samples), dtype=int)])
    reject = None
    if cfg.EPOCH_REJECT_PEAK_TO_PEAK_UV is not None and float(cfg.EPOCH_REJECT_PEAK_TO_PEAK_UV) > 0:
        reject = {"eeg": float(cfg.EPOCH_REJECT_PEAK_TO_PEAK_UV) * 1e-6}
    epochs = mne.Epochs(raw, events=events, event_id={"Target": 1}, tmin=cfg.TMIN, tmax=cfg.TMAX, baseline=cfg.BASELINE if baseline else None, metadata=metadata, preload=True, reject=reject, reject_by_annotation=True, verbose=False)
    rejection_relaxed = False
    min_keep = int(math.ceil(len(samples) * max(0.0, min(1.0, float(cfg.EPOCH_REJECT_MIN_KEEP_FRACTION)))))
    if reject is not None and len(epochs) < max(1, min_keep):
        logging.warning("%s %s: epoch PTP rejection kept %s/%s epochs; relaxing PTP rejection to protect target data.", recording.participant_id, condition_name, len(epochs), len(samples))
        epochs = mne.Epochs(raw, events=events, event_id={"Target": 1}, tmin=cfg.TMIN, tmax=cfg.TMAX, baseline=cfg.BASELINE if baseline else None, metadata=metadata, preload=True, reject=None, reject_by_annotation=True, verbose=False)
        rejection_relaxed = True
    epochs = standardize_epoch_sfreq(epochs, cfg, context=f"{recording.participant_id} {condition_name}")
    dropped = int(sum(bool(reason) for reason in epochs.drop_log))
    pd.DataFrame([{
        "Participant": recording.participant_id,
        "Condition": condition_name,
        "Requested_Epochs": int(len(samples)),
        "Kept_Epochs": int(len(epochs)),
        "Dropped_Epochs": dropped,
        "Reject_PTP_uV": cfg.EPOCH_REJECT_PEAK_TO_PEAK_UV,
        "Reject_Relaxed_To_Protect_Data": rejection_relaxed,
        "Minimum_Keep_Fraction": cfg.EPOCH_REJECT_MIN_KEEP_FRACTION,
        "Baseline": str(cfg.BASELINE if baseline else None),
    }]).to_csv(subject_dir(cfg, recording.participant_id) / f"{condition_name}_Epoch_Rejection_QC.csv", index=False)
    logging.info("%s %s epochs: %s", recording.participant_id, condition_name, len(epochs))
    return epochs if len(epochs) else None


def response_event_table(events: pd.DataFrame) -> pd.DataFrame:
    df = events.copy()
    for col in ("sample", "onset", "event_type", "task_role"):
        if col not in df.columns:
            df[col] = np.nan
    role = _event_text(df["task_role"])
    event_type = _event_text(df["event_type"])
    response_labels = {"target_detected", "button_press", "response", "participant_response"}
    mask = role.isin(response_labels) | event_type.isin(response_labels)
    out = df.loc[mask].copy()
    out["sample"] = pd.to_numeric(out["sample"], errors="coerce")
    out["onset"] = pd.to_numeric(out["onset"], errors="coerce")
    return out.dropna(subset=["sample"]).sort_values("sample").reset_index(drop=True)


def filter_targets_for_behavior(recording: Recording, condition: str, block: str, cfg: Config) -> pd.DataFrame:
    targets = filter_events_for_condition(recording.events, condition, cfg)
    if block != "Whole" and cfg.BLOCK_METADATA_COLUMN in targets.columns:
        targets = targets.loc[targets[cfg.BLOCK_METADATA_COLUMN].astype(str).eq(str(block))]
    return targets.reset_index(drop=True)


def compute_behavior_metrics(recording: Recording, condition: str, window: str, block: str, cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    targets = filter_targets_for_behavior(recording, condition, block, cfg)
    responses = response_event_table(recording.events)
    sfreq = float(recording.metadata.get("SamplingFrequency", np.nan))
    if not np.isfinite(sfreq) or sfreq <= 0:
        sfreq = 250.0
    if targets.empty:
        return pd.DataFrame(), pd.DataFrame()
    response_samples = responses["sample"].astype(float).to_numpy()
    used = np.zeros(len(response_samples), dtype=bool)
    min_s, max_s = cfg.BEHAVIOR_RESPONSE_WINDOW
    rows = []
    for target_index, target in targets.iterrows():
        target_sample = float(target["sample"])
        lo = target_sample + float(min_s) * sfreq
        hi = target_sample + float(max_s) * sfreq
        candidates = np.where((~used) & (response_samples >= lo) & (response_samples <= hi))[0]
        response_idx = int(candidates[0]) if candidates.size else -1
        if response_idx >= 0:
            used[response_idx] = True
            response_sample = float(response_samples[response_idx])
            rt_ms = (response_sample - target_sample) / sfreq * 1000.0
        else:
            response_sample = np.nan
            rt_ms = np.nan
        rows.append({
            "Participant": recording.participant_id,
            "Group": recording.group,
            "Age": recording.age,
            "Sex": recording.sex,
            "Condition": condition,
            "Window": window,
            "Block": block,
            "Target_Index": int(target_index) + 1,
            "Target_Sample": target_sample,
            "Target_Onset": target.get("onset", np.nan),
            "Event_Type": target.get("event_type", ""),
            "Attention_Status": target.get("attention_status", ""),
            "Task_Role": target.get("task_role", ""),
            "Response_Sample": response_sample,
            "Hit": bool(response_idx >= 0),
            "RT_ms": rt_ms,
        })
    trial = pd.DataFrame(rows)
    rt = pd.to_numeric(trial.loc[trial["Hit"], "RT_ms"], errors="coerce").dropna()
    summary = pd.DataFrame([{
        "Participant": recording.participant_id,
        "Group": recording.group,
        "Age": recording.age,
        "Sex": recording.sex,
        "Condition": condition,
        "Window": window,
        "Block": block,
        "N_Targets": int(len(trial)),
        "Hits": int(trial["Hit"].sum()),
        "Misses": int((~trial["Hit"]).sum()),
        "HitRate": float(trial["Hit"].mean()) if len(trial) else np.nan,
        "MeanRT_ms": float(rt.mean()) if len(rt) else np.nan,
        "MedianRT_ms": float(rt.median()) if len(rt) else np.nan,
        "SDRT_ms": float(rt.std(ddof=1)) if len(rt) > 1 else np.nan,
        "Response_Window_Start_s": float(min_s),
        "Response_Window_End_s": float(max_s),
    }])
    return trial, summary


def save_behavior_outputs(recording: Recording, condition: str, window: str, block: str, out_dir: Path, cfg: Config) -> None:
    if window != "Whole" and not window.startswith("Block_"):
        return
    try:
        trial, summary = compute_behavior_metrics(recording, condition, window, block, cfg)
        trial.to_csv(out_dir / "Behavior_Trial_Metrics.csv", index=False)
        summary.to_csv(out_dir / "Behavior_Summary.csv", index=False)
    except Exception as exc:
        logging.warning("Behavior metrics failed for %s %s %s: %s", recording.participant_id, condition, window, exc)


def exception_contains_memory_error(exc: BaseException) -> bool:
    seen = set()
    current: Optional[BaseException] = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, MemoryError) or current.__class__.__name__ in {"MemoryError", "_ArrayMemoryError"}:
            return True
        current = current.__cause__ or current.__context__
    return False


def raw_source_candidates(recording: Recording, cfg: Config) -> List[Tuple[str, Path]]:
    candidates: List[Tuple[str, Path]] = []
    if cfg.PREFER_PREPROCESSED_RAW_FIF and recording.raw_fif_file.exists():
        candidates.append(("MNE FIF", recording.raw_fif_file))
    if recording.eeg_file.exists():
        candidates.append(("EEGLAB SET", recording.eeg_file))
    if not cfg.PREFER_PREPROCESSED_RAW_FIF and recording.raw_fif_file.exists():
        candidates.append(("MNE FIF", recording.raw_fif_file))
    return candidates


def open_raw_header_for_epoch_fallback(recording: Recording, cfg: Config) -> mne.io.BaseRaw:
    last_error: Optional[BaseException] = None
    for source_label, path in raw_source_candidates(recording, cfg):
        try:
            logging.info("Memory-safe epoch fallback opening %s file without preload: %s", source_label, path)
            raw = mne.io.read_raw_fif(path, preload=False, verbose=False) if path.suffix.lower() == ".fif" else mne.io.read_raw_eeglab(path, preload=False, verbose=False)
            raw = crop_raw_to_selected_events(raw, recording, cfg)
            return assign_channel_types(raw, recording)
        except Exception as exc:
            last_error = exc
            logging.warning("Memory-safe fallback could not open %s for %s: %s", source_label, recording.participant_id, exc)
    raise RuntimeError(f"Memory-safe epoch fallback could not open a raw header for {recording.participant_id}: {last_error}") from last_error


def detect_bad_eeg_channels_epochs(epochs: mne.Epochs, recording: Recording, cfg: Config, out_dir: Path) -> List[str]:
    picks = mne.pick_types(epochs.info, eeg=True, exclude=[])
    ch_names = [epochs.ch_names[int(pick)] for pick in picks]
    if not ch_names:
        return []
    data = epochs.get_data(picks=picks) * 1e6
    flat = data.transpose(1, 0, 2).reshape(len(ch_names), -1)
    std_uv = np.nanstd(flat, axis=1)
    rms_uv = np.sqrt(np.nanmean(flat * flat, axis=1))
    ptp_uv = np.nanmax(flat, axis=1) - np.nanmin(flat, axis=1)
    log_std_z = robust_z(np.log10(np.maximum(std_uv, 1e-12)))
    log_rms_z = robust_z(np.log10(np.maximum(rms_uv, 1e-12)))
    rows = []
    detected = []
    for idx, ch in enumerate(ch_names):
        reasons = []
        if ch in epochs.info.get("bads", []):
            reasons.append("pre_marked_bad")
        if np.isfinite(std_uv[idx]) and std_uv[idx] < cfg.BAD_CHANNEL_FLAT_STD_UV:
            reasons.append("flat_or_nearly_flat_epoch_fallback")
        if np.isfinite(std_uv[idx]) and std_uv[idx] > cfg.BAD_CHANNEL_HIGH_STD_UV:
            reasons.append("very_high_std_epoch_fallback")
        if np.isfinite(log_std_z[idx]) and abs(log_std_z[idx]) >= cfg.BAD_CHANNEL_STD_Z:
            reasons.append("std_robust_z_epoch_fallback")
        if np.isfinite(log_rms_z[idx]) and abs(log_rms_z[idx]) >= cfg.BAD_CHANNEL_STD_Z:
            reasons.append("rms_robust_z_epoch_fallback")
        severity_values = np.abs(np.asarray([log_std_z[idx], log_rms_z[idx]], dtype=float))
        severity_values = severity_values[np.isfinite(severity_values)]
        severity = float(np.max(severity_values)) if severity_values.size else 0.0
        if reasons:
            detected.append((ch, severity))
        rows.append({"Participant": recording.participant_id, "Channel": ch, "Is_Bad": bool(reasons), "Reasons": ";".join(reasons), "STD_uV": float(std_uv[idx]), "RMS_uV": float(rms_uv[idx]), "PeakToPeak_uV": float(ptp_uv[idx]), "STD_RobustZ": float(log_std_z[idx]), "RMS_RobustZ": float(log_rms_z[idx]), "Severity": severity, "Source": "memory_safe_epoch_fallback"})
    detected = sorted(detected, key=lambda item: item[1], reverse=True)
    max_bad = max(1, int(math.floor(len(ch_names) * cfg.BAD_CHANNEL_MAX_FRACTION)))
    selected = [ch for ch, _ in detected[:max_bad]]
    table = pd.DataFrame(rows)
    if not table.empty:
        table["Selected_For_Interpolation"] = table["Channel"].isin(selected)
        table.to_csv(out_dir / "Bad_Channel_Detection.csv", index=False)
    return selected


def interpolate_bad_epoch_channels(epochs: mne.Epochs, bad_channels: Sequence[str], recording: Recording, cfg: Config, out_dir: Path) -> mne.Epochs:
    bads = [ch for ch in bad_channels if ch in epochs.ch_names]
    if not bads:
        return epochs
    epochs.info["bads"] = sorted(set(epochs.info.get("bads", [])) | set(bads))
    if not cfg.INTERPOLATE_BAD_CHANNELS:
        pd.DataFrame({"Participant": recording.participant_id, "Channel": bads, "Status": "marked_only_epoch_fallback", "Error": ""}).to_csv(out_dir / "Interpolated_Bad_Channels.csv", index=False)
        return epochs
    try:
        epochs.interpolate_bads(reset_bads=True, verbose=False)
        status, error = "interpolated_epoch_fallback", ""
    except Exception as exc:
        status, error = "failed_epoch_fallback", str(exc)
        logging.warning("%s: epoch-level bad-channel interpolation failed: %s", recording.participant_id, exc)
    pd.DataFrame({"Participant": recording.participant_id, "Channel": bads, "Status": status, "Error": error}).to_csv(out_dir / "Interpolated_Bad_Channels.csv", index=False)
    return epochs


def create_condition_epochs_memory_safe(raw: mne.io.BaseRaw, recording: Recording, condition_name: str, cfg: Config, baseline: bool = True) -> Optional[mne.Epochs]:
    df = filter_events_for_condition(recording.events, condition_name, cfg)
    if df.empty:
        logging.warning("%s %s: no target events in memory-safe fallback", recording.participant_id, condition_name)
        return None
    samples = df["sample"].astype(int).to_numpy()
    sfreq = float(raw.info["sfreq"])
    valid = (samples + int(round(cfg.TMIN * sfreq)) >= raw.first_samp) & (samples + int(round(cfg.TMAX * sfreq)) < raw.first_samp + raw.n_times)
    df = df.loc[valid].reset_index(drop=True)
    samples = samples[valid]
    if len(samples) == 0:
        logging.warning("%s %s: all target events outside memory-safe fallback raw bounds", recording.participant_id, condition_name)
        return None
    metadata_cols = ["condition", cfg.BLOCK_METADATA_COLUMN, "sub_block", "trial", "focus_modality", "event_type", "attention_status", "task_role", "event_code", "cond_code", "sample", "onset"]
    metadata = df[metadata_cols].copy()
    metadata.insert(0, "participant_id", recording.participant_id)
    metadata.insert(1, "condition_label", condition_name)
    metadata["epoch_id"] = np.arange(1, len(metadata) + 1)
    events = np.column_stack([samples, np.zeros(len(samples), dtype=int), np.ones(len(samples), dtype=int)])
    epochs = mne.Epochs(raw, events=events, event_id={"Target": 1}, tmin=cfg.TMIN, tmax=cfg.TMAX, baseline=None, metadata=metadata, preload=True, reject=None, reject_by_annotation=True, verbose=False)
    picks = mne.pick_types(epochs.info, eeg=True, eog=True, exclude=[])
    epochs.filter(l_freq=cfg.L_FREQ, h_freq=cfg.H_FREQ, picks=picks, verbose=False)
    if cfg.NOTCH:
        try:
            epochs.notch_filter(freqs=cfg.NOTCH, picks=picks, method="iir", n_jobs=1, verbose=False)
        except Exception as exc:
            logging.warning("%s %s: epoch-level notch filter skipped after failure: %s", recording.participant_id, condition_name, exc)
    out = subject_dir(cfg, recording.participant_id)
    if cfg.BAD_CHANNEL_DETECTION:
        bad_channels = detect_bad_eeg_channels_epochs(epochs, recording, cfg, out)
        epochs = interpolate_bad_epoch_channels(epochs, bad_channels, recording, cfg, out)
    if baseline and cfg.BASELINE is not None:
        epochs.apply_baseline(cfg.BASELINE, verbose=False)
    reject = {"eeg": float(cfg.EPOCH_REJECT_PEAK_TO_PEAK_UV) * 1e-6} if cfg.EPOCH_REJECT_PEAK_TO_PEAK_UV is not None and float(cfg.EPOCH_REJECT_PEAK_TO_PEAK_UV) > 0 else None
    rejection_relaxed = False
    if reject:
        pre_reject = epochs.copy()
        epochs.drop_bad(reject=reject, verbose=False)
        min_keep = int(math.ceil(len(samples) * max(0.0, min(1.0, float(cfg.EPOCH_REJECT_MIN_KEEP_FRACTION)))))
        if len(epochs) < max(1, min_keep):
            logging.warning("%s %s: memory-safe epoch PTP rejection kept %s/%s epochs; relaxing PTP rejection to protect target data.", recording.participant_id, condition_name, len(epochs), len(samples))
            epochs = pre_reject
            rejection_relaxed = True
    epochs = standardize_epoch_sfreq(epochs, cfg, context=f"{recording.participant_id} {condition_name} memory-safe fallback")
    dropped = int(sum(bool(reason) for reason in epochs.drop_log))
    pd.DataFrame([{"Participant": recording.participant_id, "Condition": condition_name, "Requested_Epochs": int(len(samples)), "Kept_Epochs": int(len(epochs)), "Dropped_Epochs": dropped, "Reject_PTP_uV": cfg.EPOCH_REJECT_PEAK_TO_PEAK_UV, "Reject_Relaxed_To_Protect_Data": rejection_relaxed, "Minimum_Keep_Fraction": cfg.EPOCH_REJECT_MIN_KEEP_FRACTION, "Baseline": str(cfg.BASELINE if baseline else None), "Preprocessing_Mode": "memory_safe_epoch_fallback"}]).to_csv(out / f"{condition_name}_Epoch_Rejection_QC.csv", index=False)
    return epochs if len(epochs) else None


def prepare_subject_epochs_memory_safe(recording: Recording, cfg: Config, baseline: bool = True) -> Dict[str, Optional[mne.Epochs]]:
    banner(f"MEMORY-SAFE EPOCH FALLBACK: {recording.participant_id}", "-")
    raw = open_raw_header_for_epoch_fallback(recording, cfg)
    out = subject_dir(cfg, recording.participant_id)
    pd.DataFrame([{"Participant": recording.participant_id, "Skipped": True, "Reason": "Continuous raw preload failed with MemoryError; epoch-level fallback used.", "ICA_Removal_Applied": False}]).to_csv(out / "ICA_QC.csv", index=False)
    epochs = {condition: create_condition_epochs_memory_safe(raw, recording, condition, cfg, baseline=baseline) for condition in cfg.CONDITIONS}
    del raw
    gc.collect()
    return epochs


def standardize_epoch_sfreq(epochs: mne.Epochs, cfg: Config, context: str = "") -> mne.Epochs:
    target = cfg.TARGET_EPOCH_SFREQ
    if target is None:
        return epochs
    current = float(epochs.info["sfreq"])
    target = float(target)
    if abs(current - target) < 1e-6:
        return epochs
    logging.info("%s: resampling epochs from %.6f Hz to %.6f Hz for cross-subject comparability.", context or "Epochs", current, target)
    return epochs.copy().resample(target, npad="auto", verbose=False)


def load_preprocess_epoch_subject(dataset: BIDSDataset, row: pd.Series, cfg: Config, baseline: bool = True) -> Tuple[Recording, mne.io.BaseRaw, Dict[str, Optional[mne.Epochs]]]:
    recording = dataset.load_recording(row)
    raw, _ = preprocess_recording(recording, cfg)
    epochs = {condition: create_condition_epochs(raw, recording, condition, cfg, baseline=baseline) for condition in cfg.CONDITIONS}
    return recording, raw, epochs


###############################################################################
# P300 ANALYSIS
###############################################################################

def compute_p300(epochs: mne.Epochs, recording: Recording, condition: str, window_label: str, block: str, cfg: Config) -> Tuple[mne.Evoked, pd.DataFrame]:
    evoked = epochs.average()
    rows = []
    for ch in cfg.P300_CHANNELS:
        signal_uv = evoked_channel_signal_uv(evoked, ch)
        if signal_uv is None:
            logging.warning("%s not found for P300 in %s %s", ch, recording.participant_id, condition)
            continue
        times = evoked.times
        for component, start, stop in cfg.P300_WINDOWS:
            mask = (times >= start) & (times <= stop)
            win_times, win_signal = times[mask], signal_uv[mask]
            if len(win_signal) == 0:
                continue
            peaks, _ = scipy.signal.find_peaks(win_signal)
            peak_idx = int(peaks[np.argmax(win_signal[peaks])]) if len(peaks) else int(np.argmax(win_signal))
            rows.append({"Participant": recording.participant_id, "Group": recording.group, "Age": recording.age, "Sex": recording.sex, "Condition": condition, "Window": window_label, "Block": block, "Component": component, "Start_ms": start * 1000.0, "End_ms": stop * 1000.0, "Channel": ch, "N_Epochs": len(epochs), "PeakAmplitude_uV": float(win_signal[peak_idx]), "PeakLatency_ms": float(win_times[peak_idx] * 1000.0), "MeanAmplitude_uV": float(np.mean(win_signal)), "AUC_uV_s": float(trapezoid(win_signal, win_times))})
    return evoked, pd.DataFrame(rows)


def has_virtual_cpz(inst) -> bool:
    return "CPz" not in inst.ch_names and {"CP1", "CP2"}.issubset(set(inst.ch_names))


def available_channels(inst, requested: Sequence[str]) -> List[str]:
    channels = []
    for ch in requested:
        if ch in inst.ch_names or (ch == "CPz" and has_virtual_cpz(inst)):
            channels.append(ch)
    return channels


def evoked_channel_signal_uv(evoked: mne.Evoked, channel: str) -> Optional[np.ndarray]:
    if channel in evoked.ch_names:
        return evoked.data[evoked.ch_names.index(channel)] * 1e6
    if channel == "CPz" and has_virtual_cpz(evoked):
        cp1 = evoked.data[evoked.ch_names.index("CP1")]
        cp2 = evoked.data[evoked.ch_names.index("CP2")]
        return (cp1 + cp2) * 0.5 * 1e6
    return None


def epochs_channel_data_uv(epochs: mne.Epochs, channel: str) -> Optional[np.ndarray]:
    if channel in epochs.ch_names:
        return epochs.get_data(picks=[channel])[:, 0, :] * 1e6
    if channel == "CPz" and has_virtual_cpz(epochs):
        data = epochs.get_data(picks=["CP1", "CP2"])
        return data.mean(axis=1) * 1e6
    return None


def window_mask(times: np.ndarray, start: float, stop: float) -> np.ndarray:
    return (times >= start) & (times <= stop)


def component_peak(signal: np.ndarray, polarity: str) -> int:
    if polarity == "negative":
        peaks, _ = scipy.signal.find_peaks(-signal)
        return int(peaks[np.argmax(-signal[peaks])]) if len(peaks) else int(np.argmin(signal))
    peaks, _ = scipy.signal.find_peaks(signal)
    return int(peaks[np.argmax(signal[peaks])]) if len(peaks) else int(np.argmax(signal))


def compute_erp_window_metrics(epochs: mne.Epochs, evoked: mne.Evoked, recording: Recording, condition: str, window_label: str, block: str, cfg: Config) -> pd.DataFrame:
    rows = []
    channels = available_channels(evoked, cfg.ERP_CHANNELS)
    for component, start, stop, polarity in cfg.ERP_WINDOWS:
        mask = window_mask(evoked.times, start, stop)
        if not mask.any():
            continue
        for ch in channels:
            full_signal = evoked_channel_signal_uv(evoked, ch)
            if full_signal is None:
                continue
            signal = full_signal[mask]
            times = evoked.times[mask]
            peak_idx = component_peak(signal, polarity)
            abs_weight = np.abs(signal)
            centroid = np.nan if not np.isfinite(abs_weight).any() or np.nansum(abs_weight) == 0 else float(np.nansum(times * abs_weight) / np.nansum(abs_weight) * 1000.0)
            rows.append({"Participant": recording.participant_id, "Group": recording.group, "Age": recording.age, "Sex": recording.sex, "Condition": condition, "Window": window_label, "Block": block, "Component": component, "Channel": ch, "Polarity": polarity, "Start_ms": start * 1000.0, "End_ms": stop * 1000.0, "N_Epochs": len(epochs), "PeakAmplitude_uV": float(signal[peak_idx]), "PeakLatency_ms": float(times[peak_idx] * 1000.0), "MeanAmplitude_uV": float(np.nanmean(signal)), "MedianAmplitude_uV": float(np.nanmedian(signal)), "AUC_uV_ms": float(trapezoid(signal, times * 1000.0)), "FractionalAreaLatency_ms": centroid, "WindowMin_uV": float(np.nanmin(signal)), "WindowMax_uV": float(np.nanmax(signal)), "WindowStd_uV": float(np.nanstd(signal))})
    return pd.DataFrame(rows)


def compute_single_trial_erp_metrics(epochs: mne.Epochs, recording: Recording, condition: str, window_label: str, block: str, cfg: Config) -> pd.DataFrame:
    if not cfg.SAVE_SINGLE_TRIAL_ERP:
        return pd.DataFrame()
    channels = available_channels(epochs, cfg.P300_CHANNELS)
    if not channels:
        return pd.DataFrame()
    rows = []
    metadata = epochs.metadata.reset_index(drop=True) if epochs.metadata is not None else pd.DataFrame(index=np.arange(len(epochs)))
    for component, start, stop in cfg.P300_WINDOWS:
        polarity = "positive"
        mask = window_mask(epochs.times, start, stop)
        if not mask.any():
            continue
        times = epochs.times[mask]
        for ch in channels:
            channel_data = epochs_channel_data_uv(epochs, ch)
            if channel_data is None:
                continue
            for ep_idx in range(channel_data.shape[0]):
                epoch_id = metadata.iloc[ep_idx].get("epoch_id", ep_idx + 1) if len(metadata) > ep_idx else ep_idx + 1
                trial = metadata.iloc[ep_idx].get("trial", np.nan) if len(metadata) > ep_idx else np.nan
                signal = channel_data[ep_idx, mask]
                peak_idx = component_peak(signal, polarity)
                rows.append({"Participant": recording.participant_id, "Group": recording.group, "Age": recording.age, "Sex": recording.sex, "Condition": condition, "Window": window_label, "Block": block, "Epoch_ID": epoch_id, "Trial": trial, "Component": component, "Channel": ch, "PeakAmplitude_uV": float(signal[peak_idx]), "PeakLatency_ms": float(times[peak_idx] * 1000.0), "MeanAmplitude_uV": float(np.nanmean(signal)), "AUC_uV_ms": float(trapezoid(signal, times * 1000.0))})
    return pd.DataFrame(rows)


def safe_stem(value: object) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text).strip("_") or "value"


def save_p300_peak_topographies(evoked: mne.Evoked, metrics: pd.DataFrame, out_dir: Path, cfg: Config) -> None:
    if metrics.empty:
        return
    rows = []
    for _, row in metrics.iterrows():
        latency_ms = pd.to_numeric(pd.Series([row.get("PeakLatency_ms")]), errors="coerce").iloc[0]
        if not np.isfinite(latency_ms):
            continue
        time_s = float(latency_ms) / 1000.0
        sample_idx = int(np.argmin(np.abs(evoked.times - time_s)))
        time_s = float(evoked.times[sample_idx])
        base = {
            "Participant": row.get("Participant", ""),
            "Group": row.get("Group", ""),
            "Condition": row.get("Condition", ""),
            "Window": row.get("Window", ""),
            "Block": row.get("Block", ""),
            "Component": row.get("Component", ""),
            "Peak_Defining_Channel": row.get("Channel", ""),
            "PeakLatency_ms": time_s * 1000.0,
        }
        for ch in evoked.ch_names:
            rows.append({**base, "Topo_Channel": ch, "Voltage_uV": float(evoked.data[evoked.ch_names.index(ch), sample_idx] * 1e6)})
        if cfg.SAVE_PNG:
            try:
                fig = evoked.plot_topomap(times=[time_s], ch_type="eeg", scalings=1e6, units="uV", time_unit="s", show=False)
                fig.savefig(out_dir / f"P300_Topography_{safe_stem(row.get('Component', 'P300'))}_{safe_stem(row.get('Channel', 'channel'))}.png", dpi=300, bbox_inches="tight")
                plt.close(fig)
            except Exception as exc:
                logging.warning("Could not save P300 topography in %s: %s", out_dir, exc)
    pd.DataFrame(rows).to_csv(out_dir / "P300_Peak_Topography_Long.csv", index=False)


def save_p300_outputs(evoked: mne.Evoked, metrics: pd.DataFrame, out_dir: Path, cfg: Config) -> None:
    metrics.to_csv(out_dir / "P300_Metrics.csv", index=False)
    wave = pd.DataFrame({"Time_ms": evoked.times * 1000.0})
    if not metrics.empty:
        for col in [c for c in list(IDENTIFIER_COLUMNS) + ["N_Epochs"] if c in metrics.columns]:
            wave.insert(0 if col == "Participant" else len([x for x in wave.columns if x in IDENTIFIER_COLUMNS]), col, metrics.iloc[0][col])
    for ch in cfg.P300_CHANNELS:
        signal = evoked_channel_signal_uv(evoked, ch)
        if signal is not None:
            wave[ch + "_uV"] = signal
    wave.to_csv(out_dir / "P300_ERP_Waveforms.csv", index=False)
    save_p300_peak_topographies(evoked, metrics, out_dir, cfg)
    if cfg.SAVE_FIF:
        try:
            evoked.save(out_dir / "P300_Evoked-ave.fif", overwrite=True)
        except Exception as exc:
            logging.warning("Could not save evoked FIF in %s: %s", out_dir, exc)
    if cfg.SAVE_PNG and not wave.empty:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        for ch in cfg.P300_CHANNELS:
            col = ch + "_uV"
            if col in wave.columns:
                ax.plot(wave["Time_ms"], wave[col], linewidth=2, label=ch)
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        for _, start, stop in cfg.P300_WINDOWS:
            ax.axvspan(start * 1000.0, stop * 1000.0, alpha=0.10, color="tab:blue")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude (uV)")
        ax.set_title("P300 ERP")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / "P300_ERP.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


###############################################################################
# MICROSTATE MODELING, BACK-FITTING, MATRICES, AND FIGURES
###############################################################################

def cache_subject_dir(cfg: Config, participant_id: str, create: bool = True) -> Path:
    path = cfg.CACHE_DIR / participant_id
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def cache_manifest_path(cfg: Config, participant_id: str) -> Path:
    return cache_subject_dir(cfg, participant_id) / "Prepared_Manifest.json"


def cached_epoch_path(cfg: Config, participant_id: str, condition: str) -> Path:
    return cache_subject_dir(cfg, participant_id) / f"{condition}-epo.fif"


def preparation_signature(cfg: Config, baseline: bool = True) -> Dict:
    return json_ready({
        "cache_version": cfg.CACHE_VERSION,
        "l_freq": cfg.L_FREQ,
        "h_freq": cfg.H_FREQ,
        "notch": cfg.NOTCH,
        "tmin": cfg.TMIN,
        "tmax": cfg.TMAX,
        "baseline": cfg.BASELINE if baseline else None,
        "event_type_filter": cfg.EVENT_TYPE_FILTER,
        "task_role_filter": cfg.TASK_ROLE_FILTER,
        "target_epoch_sfreq": cfg.TARGET_EPOCH_SFREQ,
        "epoch_reject_peak_to_peak_uv": cfg.EPOCH_REJECT_PEAK_TO_PEAK_UV,
        "epoch_reject_min_keep_fraction": cfg.EPOCH_REJECT_MIN_KEEP_FRACTION,
        "prefer_preprocessed_raw_fif": cfg.PREFER_PREPROCESSED_RAW_FIF,
        "crop_raw_to_selected_events": cfg.CROP_RAW_TO_SELECTED_EVENTS,
        "raw_crop_padding_s": cfg.RAW_CROP_PADDING_S,
        "memory_safe_epoch_fallback": cfg.MEMORY_SAFE_EPOCH_FALLBACK,
        "bad_channel_detection": cfg.BAD_CHANNEL_DETECTION,
        "interpolate_bad_channels": cfg.INTERPOLATE_BAD_CHANNELS,
        "bad_channel_std_z": cfg.BAD_CHANNEL_STD_Z,
        "bad_channel_psd_z": cfg.BAD_CHANNEL_PSD_Z,
        "bad_channel_flat_std_uv": cfg.BAD_CHANNEL_FLAT_STD_UV,
        "bad_channel_high_std_uv": cfg.BAD_CHANNEL_HIGH_STD_UV,
        "bad_channel_max_fraction": cfg.BAD_CHANNEL_MAX_FRACTION,
        "ica_enabled": cfg.ICA_ENABLED,
        "ica_method": cfg.ICA_METHOD if cfg.ICA_ENABLED else None,
        "ica_components": cfg.ICA_COMPONENTS if cfg.ICA_ENABLED else None,
        "ica_random_state": cfg.ICA_RANDOM_STATE if cfg.ICA_ENABLED else None,
        "ica_eog_threshold": cfg.ICA_EOG_THRESHOLD if cfg.ICA_ENABLED else None,
        "ica_max_exclude": cfg.ICA_MAX_EXCLUDE if cfg.ICA_ENABLED else None,
        "ica_min_clean_rms_ratio": cfg.ICA_MIN_CLEAN_RMS_RATIO if cfg.ICA_ENABLED else None,
    })


def manifest_matches_preparation(manifest: Dict, cfg: Config, baseline: bool = True) -> bool:
    expected = preparation_signature(cfg, baseline=baseline)
    found = manifest.get("preprocessing")
    return json.dumps(found, sort_keys=True) == json.dumps(expected, sort_keys=True)


def cached_subject_is_usable(cfg: Config, participant_id: str, baseline: bool = True) -> Tuple[bool, str]:
    manifest_file = cache_manifest_path(cfg, participant_id)
    if not manifest_file.exists():
        return False, "missing manifest"
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"manifest unreadable: {exc}"
    if manifest.get("error"):
        return False, f"previous preparation error: {manifest.get('error')}"
    if manifest.get("cache_version") != cfg.CACHE_VERSION:
        return False, "cache version changed"
    if not manifest_matches_preparation(manifest, cfg, baseline=baseline):
        return False, "preprocessing or epoch settings changed"
    saved_specs = manifest.get("condition_specs", {})
    for condition in cfg.CONDITIONS:
        expected_spec = json_ready(CONDITION_MAP.get(condition, {}))
        found_spec = saved_specs.get(condition)
        if json.dumps(found_spec, sort_keys=True) != json.dumps(expected_spec, sort_keys=True):
            return False, f"condition definition changed for {condition}"
    conditions = manifest.get("conditions", {})
    missing_conditions = [condition for condition in cfg.CONDITIONS if condition not in conditions]
    if missing_conditions:
        return False, f"missing condition entries: {', '.join(missing_conditions)}"
    for condition in cfg.CONDITIONS:
        entry = conditions.get(condition)
        if entry is None:
            continue
        if not isinstance(entry, dict) or not entry.get("file"):
            return False, f"bad manifest entry for {condition}"
        if not (cache_subject_dir(cfg, participant_id) / str(entry["file"])).exists():
            return False, f"missing epoch file for {condition}"
    return True, "valid"


def prepare_subject_once(dataset: BIDSDataset, row: pd.Series, cfg: Config, baseline: bool = True) -> bool:
    """Run loading, filtering, referencing, ICA, and epoching exactly once per subject."""
    participant_id = str(row["participant_id"])
    out = cache_subject_dir(cfg, participant_id)
    if cfg.REUSE_PREPARED_CACHE and not cfg.FORCE_PREPARE:
        usable, reason = cached_subject_is_usable(cfg, participant_id, baseline=baseline)
        if usable:
            logging.info("%s: using prepared epoch cache in %s", participant_id, out)
            return True
        logging.info("%s: preparing epoch cache (%s)", participant_id, reason)
    elif cfg.FORCE_PREPARE:
        logging.info("%s: force-preparing epoch cache", participant_id)
    try:
        recording = dataset.load_recording(row)
        try:
            raw, _ = preprocess_recording(recording, cfg)
            epochs = {condition: create_condition_epochs(raw, recording, condition, cfg, baseline=baseline) for condition in cfg.CONDITIONS}
            del raw
            preprocessing_mode = "continuous_raw"
        except Exception as prep_exc:
            if cfg.MEMORY_SAFE_EPOCH_FALLBACK and exception_contains_memory_error(prep_exc):
                logging.warning("%s: continuous preprocessing failed with memory error; switching to epoch-level fallback: %s", participant_id, prep_exc)
                epochs = prepare_subject_epochs_memory_safe(recording, cfg, baseline=baseline)
                preprocessing_mode = "memory_safe_epoch_fallback"
            else:
                raise
        manifest = {
            "cache_version": cfg.CACHE_VERSION,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "participant_id": recording.participant_id,
            "group": recording.group,
            "age": recording.age,
            "sex": recording.sex,
            "preprocessing_mode": preprocessing_mode,
            "preprocessing": preparation_signature(cfg, baseline=baseline),
            "condition_specs": {condition: json_ready(CONDITION_MAP[condition]) for condition in cfg.CONDITIONS},
            "conditions": {},
        }
        for condition, ep in epochs.items():
            if ep is None or len(ep) == 0:
                manifest["conditions"][condition] = None
                continue
            path = cached_epoch_path(cfg, recording.participant_id, condition)
            ep.save(path, overwrite=True)
            manifest["conditions"][condition] = {
                "file": path.name,
                "n_epochs": int(len(ep)),
                "n_channels": int(len(ep.ch_names)),
                "sfreq": float(ep.info["sfreq"]),
                "tmin": float(ep.tmin),
                "tmax": float(ep.tmax),
                "metadata_rows": int(0 if ep.metadata is None else len(ep.metadata)),
            }
        cache_manifest_path(cfg, recording.participant_id).write_text(json.dumps(json_ready(manifest), indent=2), encoding="utf-8")
        del epochs
        gc.collect()
        return True
    except Exception as exc:
        logging.exception("Preparation failed for %s: %s", participant_id, exc)
        cache_manifest_path(cfg, participant_id).write_text(json.dumps({"participant_id": participant_id, "error": str(exc), "created_at": datetime.now().isoformat(timespec="seconds")}, indent=2), encoding="utf-8")
        return False


def prepare_dataset_once(dataset: BIDSDataset, participants: pd.DataFrame, cfg: Config, baseline: bool = True) -> pd.DataFrame:
    rows = []
    banner("ONE-PASS PREPROCESSING / ICA / EPOCHING")
    for index, row in participants.iterrows():
        participant_id = str(row["participant_id"])
        logging.info("Preparing subject %s/%s: %s", index + 1, len(participants), participant_id)
        prepared = prepare_subject_once(dataset, row, cfg, baseline)
        rows.append({"Participant": participant_id, "Prepared": prepared, "CacheDir": str(cache_subject_dir(cfg, participant_id))})
    result = pd.DataFrame(rows)
    result.to_csv(cfg.RUN_DIR / "Preparation_Summary.csv", index=False)
    return result


def recording_from_row_and_manifest(row: pd.Series) -> Recording:
    return Recording(participant_id=str(row["participant_id"]), group=str(row.get("group", "unknown")), age=float(row.get("age", np.nan)), sex=str(row.get("sex", "unknown")), eeg_file=Path(), raw_fif_file=Path(), events_file=Path(), channels_file=Path(), eeg_json_file=Path(), events=pd.DataFrame(), metadata={}, eog_channels=[], misc_channels=[])


def load_cached_epochs(cfg: Config, participant_id: str, condition: str) -> Optional[mne.Epochs]:
    path = cached_epoch_path(cfg, participant_id, condition)
    if not path.exists():
        return None
    try:
        epochs = mne.read_epochs(path, preload=True, verbose=False)
        return standardize_epoch_sfreq(epochs, cfg, context=f"{participant_id} {condition} cached epochs")
    except Exception as exc:
        logging.warning("Could not read cached epochs for %s %s from %s: %s", participant_id, condition, path, exc)
        return None


def microstate_training_conditions(cfg: Config) -> Tuple[str, ...]:
    requested = tuple(condition for condition in cfg.CONDITIONS if condition in CONDITION_MAP)
    if cfg.MICROSTATE_MODEL_MODE == "condition":
        return requested
    pooled = tuple(condition for condition in cfg.MICROSTATE_POOL_CONDITIONS if condition in requested)
    return pooled if pooled else requested


def microstate_model_key_for_condition(condition: str, cfg: Config) -> str:
    return condition if cfg.MICROSTATE_MODEL_MODE == "condition" else cfg.POOLED_MICROSTATE_MODEL_NAME


def build_common_models(participants: pd.DataFrame, cfg: Config, baseline: bool = True) -> Dict[str, ModKMeans]:
    banner("BUILDING COMMON MICROSTATE MODELS FROM PREPARED EPOCHS")
    training_conditions = microstate_training_conditions(cfg)
    epoch_bank: Dict[str, List[mne.Epochs]] = {condition: [] for condition in training_conditions}
    model_rows = []
    for _, row in participants.iterrows():
        participant_id = str(row["participant_id"])
        usable, reason = cached_subject_is_usable(cfg, participant_id, baseline=baseline)
        if not usable:
            logging.warning("Skipping %s for common-model building; prepared cache is not usable: %s", participant_id, reason)
            continue
        for condition in training_conditions:
            ep = load_cached_epochs(cfg, participant_id, condition)
            if ep is not None and len(ep):
                if cfg.MICROSTATE_FIT_SOURCE in ("erp_grand_average", "erp_gfp_peaks"):
                    fit_ep = crop_epochs_for_microstate_fit(ep, cfg)
                    evoked = fit_ep.average()
                    setattr(evoked, "pipeline_source_metadata", {"Participant": participant_id, "Condition": condition, "N_Epochs": int(len(ep)), "Fit_Tmin": float(fit_ep.tmin), "Fit_Tmax": float(fit_ep.tmax), "Fit_Source": cfg.MICROSTATE_FIT_SOURCE})
                    epoch_bank[condition].append(evoked)
                    del fit_ep, evoked
                else:
                    epoch_bank[condition].append(ep)
                model_rows.append({"Participant": participant_id, "Group": row.get("group", "unknown"), "Condition": condition, "Microstate_Model": microstate_model_key_for_condition(condition, cfg), "N_Epochs": len(ep), "Included_In_Pooled_Model": cfg.MICROSTATE_MODEL_MODE != "condition"})
                if cfg.MICROSTATE_FIT_SOURCE in ("erp_grand_average", "erp_gfp_peaks"):
                    del ep
                    gc.collect()
    pd.DataFrame(model_rows).to_csv(cfg.COMMON_DIR / "Model_Building_Epoch_Counts.csv", index=False)
    models: Dict[str, ModKMeans] = {}
    if cfg.MICROSTATE_MODEL_MODE == "condition":
        for condition, epoch_list in epoch_bank.items():
            if not epoch_list:
                logging.warning("No epochs available to build %s common model; condition will be skipped.", condition)
                continue
            models[condition] = fit_condition_model(condition, epoch_list, cfg, training_conditions=(condition,))
            del epoch_list
            gc.collect()
    else:
        pooled_epochs = [ep for condition in training_conditions for ep in epoch_bank.get(condition, [])]
        if not pooled_epochs:
            logging.warning("No epochs available to build pooled common microstate model: %s", cfg.POOLED_MICROSTATE_MODEL_NAME)
        else:
            pooled_model = fit_condition_model(cfg.POOLED_MICROSTATE_MODEL_NAME, pooled_epochs, cfg, training_conditions=training_conditions)
            models[cfg.POOLED_MICROSTATE_MODEL_NAME] = pooled_model
            for condition in cfg.CONDITIONS:
                models[condition] = pooled_model
            del pooled_epochs
        gc.collect()
    return models


def concatenate_epochs(epoch_list: List[mne.Epochs], cfg: Optional[Config] = None) -> mne.Epochs:
    epochs_to_concat = list(epoch_list)
    if cfg is not None and cfg.TARGET_EPOCH_SFREQ is not None:
        epochs_to_concat = [standardize_epoch_sfreq(ep, cfg, context=f"concatenate input {idx}") for idx, ep in enumerate(epochs_to_concat)]
    else:
        sfreqs = np.asarray([float(ep.info["sfreq"]) for ep in epochs_to_concat], dtype=float)
        if len(sfreqs) and np.nanmax(sfreqs) - np.nanmin(sfreqs) > 1e-6:
            target = float(np.nanmedian(sfreqs))
            logging.info("Harmonizing epoch sampling rates to %.6f Hz before concatenation: %s", target, sfreqs.tolist())
            epochs_to_concat = [ep if abs(float(ep.info["sfreq"]) - target) < 1e-6 else ep.copy().resample(target, npad="auto", verbose=False) for ep in epochs_to_concat]
    try:
        return mne.concatenate_epochs(epochs_to_concat, add_offset=True, on_mismatch="ignore")
    except TypeError:
        return mne.concatenate_epochs(epochs_to_concat, add_offset=True)


def info_channel_names(info) -> List[str]:
    if info is None:
        return []
    if isinstance(info, dict):
        return list(info.get("ch_names", []))
    try:
        return list(info["ch_names"])
    except Exception:
        return list(getattr(info, "ch_names", []))


def get_cluster_names(obj, n_clusters: Optional[int] = None) -> List[str]:
    for attr in ("cluster_names", "cluster_names_", "_cluster_names"):
        names = getattr(obj, attr, None)
        if names is not None:
            return [str(name) for name in list(names)]
    n = n_clusters
    centers = getattr(obj, "cluster_centers_", None)
    if n is None and centers is not None:
        n = int(np.asarray(centers).shape[0])
    n = int(n or 0)
    return [f"MS{i + 1}" for i in range(n)]


def rename_clusters_safe(model, names: Sequence[str]) -> None:
    try:
        model.rename_clusters(new_names=list(names))
    except TypeError:
        model.rename_clusters(list(names))


def reorder_clusters_safe(model, order: Sequence[int]) -> None:
    try:
        model.reorder_clusters(order=list(order))
    except TypeError:
        model.reorder_clusters(list(order))


def invert_polarity_safe(model, invert_flags: Sequence[bool]) -> None:
    try:
        model.invert_polarity(list(invert_flags))
    except TypeError:
        model.invert_polarity(invert=list(invert_flags))


def get_segmentation_labels(segmentation) -> np.ndarray:
    for attr in ("labels", "labels_", "_labels"):
        labels = getattr(segmentation, attr, None)
        if labels is not None:
            return np.asarray(labels, dtype=int)
    getter = getattr(segmentation, "get_labels", None)
    if callable(getter):
        return np.asarray(getter(), dtype=int)
    raise AttributeError("Could not find labels on pycrostates segmentation object.")


def labels_to_2d(labels: np.ndarray) -> np.ndarray:
    arr = np.asarray(labels, dtype=int)
    if arr.ndim == 1:
        return arr[np.newaxis, :]
    if arr.ndim == 2:
        return arr
    return arr.reshape(arr.shape[0], -1)


def segmentation_times(epochs: mne.Epochs, n_times: int) -> np.ndarray:
    if len(epochs.times) == n_times:
        return epochs.times
    return np.linspace(float(epochs.tmin), float(epochs.tmax), int(n_times))


def compute_segmentation_parameters(segmentation, norm_gfp: bool, return_dist: bool):
    try:
        return segmentation.compute_parameters(norm_gfp=norm_gfp, return_dist=return_dist)
    except TypeError:
        if return_dist:
            return segmentation.compute_parameters(norm_gfp=norm_gfp)
        return segmentation.compute_parameters(norm_gfp=norm_gfp)


def compute_transition_safe(segmentation, expected: bool, stat: str, ignore_repetitions: bool = True) -> np.ndarray:
    method_name = "compute_expected_transition_matrix" if expected else "compute_transition_matrix"
    method = getattr(segmentation, method_name)
    try:
        return method(stat=stat, ignore_repetitions=ignore_repetitions)
    except TypeError:
        return method(stat=stat)


def segmentation_entropy_bits(segmentation, ignore_repetitions: bool) -> float:
    for kwargs in ({"ignore_repetitions": ignore_repetitions, "log_base": 2}, {"ignore_repetitions": ignore_repetitions, "log_base": "bits"}, {"ignore_repetitions": ignore_repetitions}):
        try:
            value = segmentation.entropy(**kwargs)
            if kwargs.get("log_base") is None:
                value = float(value) / math.log(2)
            return float(value)
        except TypeError:
            continue
    return float(segmentation.entropy())


###############################################################################
# CANONICAL TEMPLATE ALIGNMENT AND MICROSTATE MAP QC
###############################################################################

def normalize_channel_name(name: str) -> str:
    cleaned = str(name).strip().upper().replace(" ", "")
    return CHANNEL_NAME_ALIASES.get(cleaned, cleaned)


def standardize_map(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float).ravel()
    arr = arr - np.nanmean(arr)
    denom = float(np.linalg.norm(arr))
    return arr / denom if np.isfinite(denom) and denom > 0 else np.full(arr.shape, np.nan)


def signed_spatial_correlation(a: Sequence[float], b: Sequence[float]) -> float:
    aa, bb = standardize_map(a), standardize_map(b)
    if np.any(~np.isfinite(aa)) or np.any(~np.isfinite(bb)) or aa.shape != bb.shape:
        return np.nan
    return float(np.dot(aa, bb))


def orient_template_maps(data: np.ndarray, n_labels: int, n_channels: int) -> np.ndarray:
    arr = np.asarray(data, dtype=float).squeeze()
    if arr.ndim == 3:
        arr = np.nanmean(arr, axis=-1)
    if arr.ndim != 2:
        raise ValueError(f"Template data must be 2D after squeezing, got shape {arr.shape}")
    if arr.shape[0] == n_channels and arr.shape[1] >= n_labels:
        arr = arr.T
    elif arr.shape[1] == n_channels and arr.shape[0] >= n_labels:
        pass
    elif arr.shape[0] < arr.shape[1] and arr.shape[0] >= n_labels:
        pass
    elif arr.shape[1] >= n_labels:
        arr = arr.T
    return arr[:n_labels]


def template_file_for(cfg: Config) -> Path:
    if cfg.TEMPLATE_NAME in TEMPLATE_FILES:
        return cfg.TEMPLATE_DIR / TEMPLATE_FILES[cfg.TEMPLATE_NAME]
    return cfg.TEMPLATE_DIR / cfg.TEMPLATE_NAME


def load_template_maps(cfg: Config, n_labels: Optional[int] = None) -> Tuple[Optional[np.ndarray], Optional[List[str]], Optional[List[str]], Optional[Path]]:
    template_file = template_file_for(cfg)
    if not template_file.exists():
        logging.warning("Canonical template file not found: %s", template_file)
        return None, None, None, template_file
    labels = list(TEMPLATE_LABELS[: int(n_labels or cfg.N_CLUSTERS)])
    try:
        from pymatreader import read_mat
        mat = read_mat(str(template_file))
        eeg = mat.get("EEG") if isinstance(mat, dict) else None
        if not isinstance(eeg, dict) or "data" not in eeg:
            eeg = mat
        data = np.asarray(eeg.get("data"), dtype=float)
        chanlocs = eeg.get("chanlocs", [])
        channels = extract_eeglab_channel_names(chanlocs, data.shape[0])
        maps = orient_template_maps(data, len(labels), len(channels))
        labels = labels[: maps.shape[0]]
        return maps, labels, channels, template_file
    except Exception as exc:
        logging.info("Direct template read failed for %s: %s", template_file, exc)
    try:
        raw = mne.io.read_raw_eeglab(template_file, preload=True, verbose=False)
        maps = orient_template_maps(raw.get_data(), len(labels), len(raw.ch_names))
        labels = labels[: maps.shape[0]]
        return maps, labels, list(raw.ch_names), template_file
    except Exception as exc:
        logging.info("Template raw load failed for %s: %s", template_file, exc)
    try:
        epochs_reader = getattr(mne.io, "read_epochs_eeglab", None) or getattr(mne, "read_epochs_eeglab", None)
        if epochs_reader is None:
            raise AttributeError("MNE has no EEGLAB epochs reader")
        epochs = epochs_reader(template_file, verbose=False)
        maps = orient_template_maps(epochs.get_data(), len(labels), len(epochs.ch_names))
        labels = labels[: maps.shape[0]]
        return maps, labels, list(epochs.ch_names), template_file
    except Exception as exc:
        logging.info("Template epochs load failed for %s: %s", template_file, exc)
    return None, None, None, template_file


def extract_eeglab_channel_names(chanlocs, n_channels: int) -> List[str]:
    names: List[str] = []
    if isinstance(chanlocs, dict) and "labels" in chanlocs:
        labels = chanlocs["labels"]
        names = [str(x) for x in labels] if isinstance(labels, (list, tuple, np.ndarray)) else [str(labels)]
    elif isinstance(chanlocs, (list, tuple, np.ndarray)):
        for item in chanlocs:
            label = item.get("labels") if isinstance(item, dict) else getattr(item, "labels", None)
            if label is not None:
                names.append(str(label))
    return names[:n_channels] if names else [f"ch{i + 1}" for i in range(n_channels)]


def create_standard_eeg_info(channels: Sequence[str], sfreq: float = 1.0):
    info = mne.create_info(list(channels), sfreq=sfreq, ch_types="eeg")
    try:
        info.set_montage("standard_1020", match_case=False, on_missing="ignore")
    except Exception as exc:
        logging.warning("Could not set standard_1020 montage for template interpolation: %s", exc)
    return info


def channels_with_valid_positions(info) -> set:
    valid = set()
    for ch in info.get("chs", []):
        loc = np.asarray(ch.get("loc", [])[:3], dtype=float)
        if loc.shape == (3,) and np.all(np.isfinite(loc)) and float(np.linalg.norm(loc)) > 0:
            valid.add(str(ch.get("ch_name", "")))
    return valid


def remap_template_channels_to_model(template_channels: Sequence[str], model_channels: Sequence[str]) -> List[Tuple[int, str, str]]:
    model_lookup = {normalize_channel_name(ch): str(ch) for ch in model_channels}
    remapped = []
    seen = set()
    for idx, ch in enumerate(template_channels):
        norm = normalize_channel_name(ch)
        if norm in seen:
            continue
        seen.add(norm)
        remapped.append((idx, model_lookup.get(norm, norm), norm))
    return remapped


def interpolate_template_maps_to_model_montage(model: ModKMeans, template_maps: np.ndarray, template_channels: List[str], cfg: Config) -> Tuple[np.ndarray, List[str], Dict[str, object]]:
    metadata: Dict[str, object] = {
        "template_interpolated_to_model": False,
        "template_interpolation_method": "mne.interpolate_bads_standard_1020_union_montage",
        "template_interpolation_reason": "disabled",
    }
    if not cfg.TEMPLATE_INTERPOLATE_TO_MODEL:
        return template_maps, template_channels, metadata
    model_channels = info_channel_names(getattr(model, "info", None))
    if not model_channels:
        metadata["template_interpolation_reason"] = "model_has_no_channel_names"
        return template_maps, template_channels, metadata
    remapped = remap_template_channels_to_model(template_channels, model_channels)
    if not remapped:
        metadata["template_interpolation_reason"] = "template_has_no_channel_names"
        return template_maps, template_channels, metadata
    source_channels_all = [name for _, name, _ in remapped]
    candidate_channels = list(dict.fromkeys([*source_channels_all, *model_channels]))
    candidate_info = create_standard_eeg_info(candidate_channels)
    valid_position_channels = channels_with_valid_positions(candidate_info)
    source_entries = [(template_idx, name, norm) for template_idx, name, norm in remapped if name in valid_position_channels]
    target_channels = [ch for ch in model_channels if ch in valid_position_channels]
    metadata.update({
        "template_original_channels": len(template_channels),
        "template_unique_channels": len(source_channels_all),
        "template_positioned_source_channels": len(source_entries),
        "model_channels": len(model_channels),
        "model_positioned_target_channels": len(target_channels),
        "template_min_interpolation_channels": cfg.TEMPLATE_MIN_INTERPOLATION_CHANNELS,
        "template_unpositioned_source_channels": ",".join([name for name in source_channels_all if name not in valid_position_channels]),
        "model_unpositioned_target_channels": ",".join([ch for ch in model_channels if ch not in valid_position_channels]),
    })
    if len(source_entries) < cfg.TEMPLATE_MIN_INTERPOLATION_CHANNELS or len(target_channels) < cfg.TEMPLATE_MIN_INTERPOLATION_CHANNELS:
        metadata["template_interpolation_reason"] = "too_few_positioned_channels"
        return template_maps, template_channels, metadata
    source_channels = [name for _, name, _ in source_entries]
    union_channels = list(dict.fromkeys([*source_channels, *target_channels]))
    info = create_standard_eeg_info(union_channels)
    data = np.zeros((len(union_channels), template_maps.shape[0]), dtype=float)
    for template_idx, name, _ in source_entries:
        data[union_channels.index(name), :] = template_maps[:, template_idx]
    raw = mne.io.RawArray(data, info, verbose=False)
    bad_targets = [ch for ch in target_channels if ch not in source_channels]
    raw.info["bads"] = bad_targets
    try:
        if bad_targets:
            try:
                raw.interpolate_bads(reset_bads=False, mode="accurate", verbose=False)
            except TypeError:
                raw.interpolate_bads(reset_bads=False, verbose=False)
        interpolated = raw.get_data(picks=target_channels).T
    except Exception as exc:
        metadata["template_interpolation_reason"] = f"interpolation_failed: {exc}"
        logging.warning("Template-to-model montage interpolation failed; falling back to common-channel matching: %s", exc)
        return template_maps, template_channels, metadata
    metadata.update({
        "template_interpolated_to_model": True,
        "template_interpolation_reason": "ok",
        "template_interpolation_source_channels": ",".join(source_channels),
        "template_interpolation_target_channels": ",".join(target_channels),
        "template_interpolated_target_channels": ",".join(bad_targets),
        "template_n_interpolated_target_channels": len(bad_targets),
    })
    return interpolated, target_channels, metadata


def template_correlation_matrices(model: ModKMeans, template_maps: np.ndarray, template_channels: List[str]) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    fitted = np.asarray(model.cluster_centers_, dtype=float)
    data_channels = info_channel_names(getattr(model, "info", None))
    template_lookup = {normalize_channel_name(ch): idx for idx, ch in enumerate(template_channels)}
    pairs = [(didx, template_lookup[normalize_channel_name(ch)]) for didx, ch in enumerate(data_channels) if normalize_channel_name(ch) in template_lookup]
    signed = np.full((fitted.shape[0], template_maps.shape[0]), np.nan)
    if len(pairs) < 5:
        logging.warning("Template alignment has only %s common channels; skipping correlations.", len(pairs))
        return signed, np.abs(signed), pairs
    for i in range(fitted.shape[0]):
        for j in range(template_maps.shape[0]):
            fmap = np.asarray([fitted[i, didx] for didx, _ in pairs], dtype=float)
            tmap = np.asarray([template_maps[j, tidx] for _, tidx in pairs], dtype=float)
            signed[i, j] = signed_spatial_correlation(fmap, tmap)
    return signed, np.abs(signed), pairs


def save_correlation_heatmap(matrix: np.ndarray, row_labels: Sequence[str], col_labels: Sequence[str], title: str, path: Path, cmap: str = "viridis", vmin: Optional[float] = None, vmax: Optional[float] = None) -> None:
    fig, ax = plt.subplots(figsize=(max(5, len(col_labels) * 1.2), max(4, len(row_labels) * 0.8)))
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("Canonical template")
    ax.set_ylabel("Fitted microstate")
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def align_model_to_template(model: ModKMeans, condition: str, out_dir: Path, cfg: Config) -> Dict[str, object]:
    n_clusters = int(getattr(model, "n_clusters", cfg.N_CLUSTERS))
    original_names = get_cluster_names(model, n_clusters)
    fallback_names = list(TEMPLATE_LABELS[:n_clusters]) if n_clusters <= len(TEMPLATE_LABELS) else [f"MS{i + 1}" for i in range(n_clusters)]
    if not cfg.APPLY_TEMPLATE_ALIGNMENT:
        rename_clusters_safe(model, fallback_names)
        return {"applied": False, "reason": "disabled", "assigned_labels": fallback_names}
    template_maps, template_labels, template_channels, template_file = load_template_maps(cfg, n_labels=n_clusters)
    if template_maps is None or template_labels is None or template_channels is None:
        rename_clusters_safe(model, fallback_names)
        return {"applied": False, "reason": "template_not_found_or_unreadable", "template_file": str(template_file), "assigned_labels": fallback_names}
    corr_template_maps, corr_template_channels, interpolation_metadata = interpolate_template_maps_to_model_montage(model, template_maps, template_channels, cfg)
    pd.DataFrame([interpolation_metadata]).to_csv(out_dir / f"{condition}_Template_Interpolation_Metadata.csv", index=False)
    signed, absolute, pairs = template_correlation_matrices(model, corr_template_maps, corr_template_channels)
    row_labels = [f"{name}({idx})" for idx, name in enumerate(original_names)]
    pd.DataFrame(signed, index=row_labels, columns=template_labels).to_csv(out_dir / f"{condition}_Template_Signed_Correlations.csv")
    pd.DataFrame(absolute, index=row_labels, columns=template_labels).to_csv(out_dir / f"{condition}_Template_Abs_Correlations.csv")
    save_correlation_heatmap(absolute, row_labels, template_labels, f"{condition} template abs correlation: {cfg.TEMPLATE_NAME}", out_dir / f"{condition}_Template_Abs_Correlation_Heatmap.png", cmap="viridis", vmin=0, vmax=1)
    save_correlation_heatmap(signed, row_labels, template_labels, f"{condition} template signed correlation: {cfg.TEMPLATE_NAME}", out_dir / f"{condition}_Template_Signed_Correlation_Heatmap.png", cmap="coolwarm", vmin=-1, vmax=1)
    cost = np.nan_to_num(absolute, nan=0.0)
    rows, cols = linear_sum_assignment(-cost)
    assignment = pd.DataFrame([{"Original_Index": int(r), "Original_Name": original_names[int(r)], "Template_Index": int(c), "Canonical_Label": template_labels[int(c)], "Signed_Correlation": float(signed[int(r), int(c)]), "Abs_Correlation": float(absolute[int(r), int(c)]), "Passes_Min_Corr": bool(absolute[int(r), int(c)] >= cfg.TEMPLATE_MIN_CORR), "Invert_Polarity": bool(signed[int(r), int(c)] < 0)} for r, c in zip(rows, cols)]).sort_values("Template_Index").reset_index(drop=True)
    assignment.to_csv(out_dir / f"{condition}_Template_Assignment.csv", index=False)
    data_channels = info_channel_names(getattr(model, "info", None))
    common_data_channels = [data_channels[didx] for didx, _ in pairs] if pairs else []
    common_template_channels = [corr_template_channels[tidx] for _, tidx in pairs] if pairs else []
    if common_data_channels:
        logging.info("%s template matching used %s common channels: %s", condition, len(common_data_channels), ", ".join(common_data_channels))
    reordered = False
    can_force_template_order = len(template_labels) >= n_clusters and len(assignment) >= n_clusters
    if cfg.REORDER_TO_TEMPLATE and can_force_template_order and set(range(n_clusters)).issubset(set(assignment["Template_Index"].astype(int))):
        order = [int(assignment.loc[assignment["Template_Index"].eq(j), "Original_Index"].iloc[0]) for j in range(n_clusters)]
        try:
            reorder_clusters_safe(model, order)
            reordered = True
        except Exception as exc:
            logging.warning("%s template reorder failed: %s", condition, exc)
    if reordered:
        rename_clusters_safe(model, template_labels[:n_clusters])
    elif len(assignment) == n_clusters:
        rename_clusters_safe(model, assignment.sort_values("Original_Index")["Canonical_Label"].astype(str).tolist())
    else:
        rename_clusters_safe(model, fallback_names)
    invert_flags = [False] * n_clusters
    if cfg.INVERT_TO_TEMPLATE_POLARITY:
        try:
            current_signed, _, _ = template_correlation_matrices(model, corr_template_maps, corr_template_channels)
            for idx in range(min(n_clusters, len(template_labels))):
                invert_flags[idx] = bool(current_signed[idx, idx] < 0) if reordered else bool(assignment.sort_values("Original_Index").iloc[idx]["Invert_Polarity"])
            invert_polarity_safe(model, invert_flags)
        except Exception as exc:
            logging.warning("%s polarity inversion failed: %s", condition, exc)
    low_confidence = pd.to_numeric(assignment["Abs_Correlation"], errors="coerce").fillna(-np.inf) < cfg.TEMPLATE_MIN_CORR
    return {**interpolation_metadata, "applied": True, "template_name": cfg.TEMPLATE_NAME, "template_file": str(template_file), "n_common_channels": len(pairs), "common_data_channels": ",".join(common_data_channels), "common_template_channels": ",".join(common_template_channels), "channel_aliases": ",".join(f"{k}->{v}" for k, v in CHANNEL_NAME_ALIASES.items()), "min_corr": cfg.TEMPLATE_MIN_CORR, "reordered": reordered, "template_count": len(template_labels), "force_template_order_supported": bool(can_force_template_order), "invert_flags": ",".join(map(str, invert_flags)), "mean_abs_correlation": float(np.nanmean([absolute[int(r), int(c)] for r, c in zip(rows, cols)])), "assigned_labels": ",".join(get_cluster_names(model, n_clusters)), "low_confidence_assignments": int(low_confidence.sum())}


def compute_topographic_metrics(model: ModKMeans) -> pd.DataFrame:
    centers = np.asarray(model.cluster_centers_, dtype=float)
    names = get_cluster_names(model, centers.shape[0])
    rows = []
    for idx, name in enumerate(names):
        values = centers[idx]
        rows.extend([{"State": name, "Metric": "map_variance", "Value": float(np.nanvar(values)), "Unit": "a.u.^2"}, {"State": name, "Metric": "map_l2_norm", "Value": float(np.linalg.norm(values)), "Unit": "a.u."}, {"State": name, "Metric": "map_peak_to_peak", "Value": float(np.nanmax(values) - np.nanmin(values)), "Unit": "a.u."}, {"State": name, "Metric": "map_abs_max", "Value": float(np.nanmax(np.abs(values))), "Unit": "a.u."}])
    return pd.DataFrame(rows)


def save_individual_topomaps(model: ModKMeans, out_dir: Path, cfg: Config) -> None:
    if not cfg.SAVE_PNG:
        return
    centers = np.asarray(model.cluster_centers_, dtype=float)
    names = get_cluster_names(model, centers.shape[0])
    for idx, name in enumerate(names):
        try:
            fig, ax = plt.subplots(figsize=(3.2, 3.2))
            mne.viz.plot_topomap(centers[idx], model.info, axes=ax, show=False)
            ax.set_title(str(name))
            fig.tight_layout()
            fig.savefig(out_dir / f"Microstate_Map_{name}.png", dpi=300, bbox_inches="tight")
            plt.close(fig)
        except Exception as exc:
            logging.warning("Could not save individual topomap %s: %s", name, exc)


def model_training_gev_table(model_name: str, model: ModKMeans, all_epochs: mne.Epochs, cfg: Config, training_conditions: Sequence[str]) -> pd.DataFrame:
    rows = []
    try:
        segmentation = backfit_microstates(model, all_epochs, cfg)
        params = params_to_dict(compute_segmentation_parameters(segmentation, norm_gfp=True, return_dist=False))
        cluster_names = get_cluster_names(model, int(getattr(model, "n_clusters", cfg.N_CLUSTERS)))
        state_values = []
        for state in cluster_names:
            value = scalar_or_nan(params.get(f"{state}_gev", np.nan))
            state_values.append(value)
            rows.append({"Model": model_name, "Training_Conditions": ",".join(training_conditions), "State": state, "Metric": "training_state_gev", "Value": value, "Unit": "proportion"})
        total = scalar_or_nan(params.get("total_gev", np.nan))
        if not np.isfinite(total) and any(np.isfinite(state_values)):
            total = float(np.nansum(state_values))
        rows.append({"Model": model_name, "Training_Conditions": ",".join(training_conditions), "State": "all", "Metric": "training_total_gev", "Value": total, "Unit": "proportion"})
        model_total = float(getattr(model, "GEV_", np.nan))
        rows.append({"Model": model_name, "Training_Conditions": ",".join(training_conditions), "State": "all", "Metric": "model_fit_gev", "Value": model_total, "Unit": "proportion"})
    except Exception as exc:
        logging.warning("Could not compute training GEV table for %s: %s", model_name, exc)
    return pd.DataFrame(rows)


def model_fit_source_gev_table(model_name: str, model: ModKMeans, training_conditions: Sequence[str], source: str) -> pd.DataFrame:
    rows = []
    cluster_names = get_cluster_names(model, int(getattr(model, "n_clusters", 0)))
    note = "Per-state GEV is saved during subject/condition backfitting; this compact model-fit source table reports the global ModKMeans fit GEV."
    for state in cluster_names:
        rows.append({"Model": model_name, "Training_Conditions": ",".join(training_conditions), "State": state, "Metric": "training_state_gev", "Value": np.nan, "Unit": "proportion", "Source": source, "Note": note})
    rows.append({"Model": model_name, "Training_Conditions": ",".join(training_conditions), "State": "all", "Metric": "model_fit_gev", "Value": float(getattr(model, "GEV_", np.nan)), "Unit": "proportion", "Source": source, "Note": "GEV reported by ModKMeans on the fit source."})
    return pd.DataFrame(rows)


def crop_epochs_for_microstate_fit(epochs: mne.Epochs, cfg: Config) -> mne.Epochs:
    if cfg.MICROSTATE_FIT_WINDOW is None:
        return epochs
    start, stop = cfg.MICROSTATE_FIT_WINDOW
    if start >= stop:
        logging.warning("Ignoring invalid microstate fit window %.3f..%.3f sec.", start, stop)
        return epochs
    tmin = max(float(start), float(epochs.tmin))
    tmax = min(float(stop), float(epochs.tmax))
    if tmin >= tmax:
        logging.warning("Microstate fit window %.3f..%.3f sec is outside epoch range %.3f..%.3f sec; using full epochs.", start, stop, epochs.tmin, epochs.tmax)
        return epochs
    return epochs.copy().crop(tmin=tmin, tmax=tmax, include_tmax=True)


def crop_evoked_for_microstate_fit(evoked: mne.Evoked, cfg: Config) -> mne.Evoked:
    if cfg.MICROSTATE_FIT_WINDOW is None:
        return evoked
    start, stop = cfg.MICROSTATE_FIT_WINDOW
    if start >= stop:
        return evoked
    tmin = max(float(start), float(evoked.times[0]))
    tmax = min(float(stop), float(evoked.times[-1]))
    if tmin >= tmax:
        return evoked
    return evoked.copy().crop(tmin=tmin, tmax=tmax, include_tmax=True)


def source_epoch_count(source) -> int:
    if isinstance(source, mne.BaseEpochs):
        return int(len(source))
    return int(getattr(source, "nave", 0) or getattr(source, "pipeline_source_metadata", {}).get("N_Epochs", 0) or 0)


def microstate_cluster_candidates(cfg: Config) -> List[int]:
    if not cfg.AUTO_N_CLUSTERS:
        return [int(cfg.N_CLUSTERS)]
    lo, hi = sorted((int(cfg.CLUSTER_RANGE[0]), int(cfg.CLUSTER_RANGE[1])))
    lo = max(2, lo)
    hi = max(lo, hi)
    return list(range(lo, hi + 1))


def safe_metric(func, model: ModKMeans) -> float:
    if func is None:
        return np.nan
    try:
        return float(func(model))
    except Exception as exc:
        logging.warning("Cluster metric %s failed for k=%s: %s", getattr(func, "__name__", func), getattr(model, "n_clusters", "?"), exc)
        return np.nan


def model_cluster_score_row(model: ModKMeans) -> Dict[str, float]:
    return {
        "N_Clusters": int(getattr(model, "n_clusters", np.nan)),
        "GEV": float(getattr(model, "GEV_", np.nan)),
        "Silhouette": safe_metric(silhouette_score, model),
        "Calinski_Harabasz": safe_metric(calinski_harabasz_score, model),
        "Davies_Bouldin": safe_metric(davies_bouldin_score, model),
        "Dunn": safe_metric(dunn_score, model),
    }


def add_cluster_score_ranks(scores: pd.DataFrame) -> pd.DataFrame:
    if scores.empty:
        return scores
    ranked = scores.copy()
    directions = {"GEV": False, "Silhouette": False, "Calinski_Harabasz": False, "Davies_Bouldin": True, "Dunn": False}
    rank_cols = []
    for metric, ascending in directions.items():
        if metric not in ranked.columns:
            continue
        values = pd.to_numeric(ranked[metric], errors="coerce")
        fill = np.inf if ascending else -np.inf
        rank_col = f"{metric}_Rank"
        ranked[rank_col] = values.fillna(fill).rank(ascending=ascending, method="min")
        rank_cols.append(rank_col)
    ranked["Composite_Rank"] = ranked[rank_cols].mean(axis=1) if rank_cols else np.nan
    best_idx = ranked.sort_values(["Composite_Rank", "GEV", "N_Clusters"], ascending=[True, False, True]).index[0]
    ranked["Selected"] = False
    ranked.loc[best_idx, "Selected"] = True
    return ranked


def fit_modkmeans_candidates(fit_inst, cfg: Config, condition: str, out_dir: Path) -> ModKMeans:
    models: Dict[int, ModKMeans] = {}
    rows = []
    for n_clusters in microstate_cluster_candidates(cfg):
        logging.info("%s: fitting ModKMeans with k=%s on %s source.", condition, n_clusters, cfg.MICROSTATE_FIT_SOURCE)
        model = ModKMeans(n_clusters=int(n_clusters), n_init=cfg.N_INIT, random_state=cfg.RANDOM_STATE)
        model.fit(fit_inst)
        rows.append(model_cluster_score_row(model))
        models[int(n_clusters)] = model
    scores = add_cluster_score_ranks(pd.DataFrame(rows))
    if not scores.empty:
        scores.to_csv(out_dir / f"{condition}_Cluster_Selection_Scores.csv", index=False)
        selected_k = int(scores.loc[scores["Selected"].eq(True), "N_Clusters"].iloc[0])
    else:
        selected_k = int(cfg.N_CLUSTERS)
    selected = models[selected_k]
    try:
        setattr(selected, "pipeline_selected_n_clusters", selected_k)
        setattr(selected, "pipeline_cluster_selection_scores", scores.to_dict("records"))
    except Exception:
        pass
    logging.info("%s: selected k=%s | GEV=%.4f", condition, selected_k, float(getattr(selected, "GEV_", np.nan)))
    return selected


def evoked_metadata_from_epochs(epochs: mne.Epochs, index: int) -> Dict[str, object]:
    metadata = epochs.metadata
    row = metadata.iloc[0].to_dict() if metadata is not None and len(metadata) else {}
    return {
        "Evoked_Index": index,
        "Participant": row.get("participant_id", ""),
        "Condition": row.get("condition_label", ""),
        "N_Epochs": len(epochs),
        "Tmin": float(epochs.tmin),
        "Tmax": float(epochs.tmax),
        "N_Timepoints": int(len(epochs.times)),
    }


def evoked_metadata_from_source(source, index: int) -> Dict[str, object]:
    if isinstance(source, mne.BaseEpochs):
        return evoked_metadata_from_epochs(source, index)
    metadata = getattr(source, "pipeline_source_metadata", {}) or {}
    return {
        "Evoked_Index": index,
        "Participant": metadata.get("Participant", ""),
        "Condition": metadata.get("Condition", ""),
        "N_Epochs": int(metadata.get("N_Epochs", getattr(source, "nave", 0) or 0)),
        "Tmin": float(source.times[0]),
        "Tmax": float(source.times[-1]),
        "N_Timepoints": int(len(source.times)),
    }


def build_erp_grand_average_fit_data(epoch_list: List, cfg: Config) -> Tuple[ChData, pd.DataFrame]:
    evokeds = []
    rows = []
    for idx, source in enumerate(epoch_list, start=1):
        if isinstance(source, mne.BaseEpochs):
            fit_source = crop_epochs_for_microstate_fit(source, cfg)
            evoked = fit_source.average()
        else:
            evoked = crop_evoked_for_microstate_fit(source, cfg)
            fit_source = evoked
        evokeds.append(evoked)
        row = evoked_metadata_from_source(source, idx)
        row.update({"Fit_Tmin": float(fit_source.times[0]), "Fit_Tmax": float(fit_source.times[-1]), "Fit_Source": cfg.MICROSTATE_FIT_SOURCE})
        rows.append(row)
    if not evokeds:
        raise RuntimeError("No evoked ERP maps were available for ERP microstate fitting.")
    common_channels = list(evokeds[0].ch_names)
    for evoked in evokeds[1:]:
        common_channels = [ch for ch in common_channels if ch in evoked.ch_names]
    if not common_channels:
        raise RuntimeError("No common channels across evoked ERP maps for ERP microstate fitting.")
    data_blocks = []
    info = None
    for evoked in evokeds:
        ev = evoked.copy().pick(common_channels)
        if info is None:
            info = ev.info.copy()
        data_blocks.append(ev.data)
    data = np.concatenate(data_blocks, axis=1)
    return ChData(data, info), pd.DataFrame(rows)


def build_erp_gfp_peak_fit_data(epoch_list: List, cfg: Config) -> Tuple[ChData, pd.DataFrame]:
    evokeds = []
    rows = []
    for idx, source in enumerate(epoch_list, start=1):
        evoked = crop_evoked_for_microstate_fit(source.average(), cfg) if isinstance(source, mne.BaseEpochs) else crop_evoked_for_microstate_fit(source, cfg)
        evokeds.append(evoked)
        row = evoked_metadata_from_source(source, idx)
        row.update({"Fit_Tmin": float(evoked.times[0]), "Fit_Tmax": float(evoked.times[-1]), "Fit_Source": cfg.MICROSTATE_FIT_SOURCE})
        rows.append(row)
    if not evokeds:
        raise RuntimeError("No evoked ERP maps were available for ERP-GFP-peak microstate fitting.")
    common_channels = list(evokeds[0].ch_names)
    for evoked in evokeds[1:]:
        common_channels = [ch for ch in common_channels if ch in evoked.ch_names]
    if not common_channels:
        raise RuntimeError("No common channels across ERP maps for ERP-GFP-peak microstate fitting.")
    data_blocks = []
    info = None
    for row, evoked in zip(rows, evokeds):
        ev = evoked.copy().pick(common_channels)
        data = ev.data
        gfp = np.nanstd(data, axis=0)
        peaks, _ = scipy.signal.find_peaks(gfp, distance=max(1, int(cfg.GFP_MIN_PEAK_DISTANCE)))
        if peaks.size == 0 and gfp.size:
            peaks = np.asarray([int(np.nanargmax(gfp))])
        row["N_GFP_Peak_Maps"] = int(peaks.size)
        row["Peak_Times_ms"] = ";".join(f"{ev.times[int(peak)] * 1000.0:.1f}" for peak in peaks)
        if peaks.size:
            if info is None:
                info = ev.info.copy()
            data_blocks.append(data[:, peaks])
    if not data_blocks or info is None:
        raise RuntimeError("ERP-GFP-peak fitting found no valid GFP peaks.")
    return ChData(np.concatenate(data_blocks, axis=1), info), pd.DataFrame(rows)


def build_single_trial_gfp_fit_data(epoch_list: List[mne.Epochs], cfg: Config) -> Tuple[ChData, pd.DataFrame]:
    peak_sets = []
    rows = []
    for idx, epochs in enumerate(epoch_list, start=1):
        fit_epochs = crop_epochs_for_microstate_fit(epochs, cfg)
        peaks = extract_gfp_peaks(fit_epochs, min_peak_distance=cfg.GFP_MIN_PEAK_DISTANCE)
        peak_sets.append(peaks)
        rows.append({
            "Source_Index": idx,
            "N_Epochs": int(len(epochs)),
            "Fit_Tmin": float(fit_epochs.tmin),
            "Fit_Tmax": float(fit_epochs.tmax),
            "N_GFP_Peak_Maps": int(peaks.get_data().shape[-1]),
            "Fit_Source": cfg.MICROSTATE_FIT_SOURCE,
        })
        del fit_epochs, peaks
        gc.collect()
    if not peak_sets:
        raise RuntimeError("No GFP peak maps were available for single-trial microstate fitting.")
    common_channels = list(peak_sets[0].ch_names)
    for peaks in peak_sets[1:]:
        common_channels = [ch for ch in common_channels if ch in peaks.ch_names]
    if not common_channels:
        raise RuntimeError("No common channels across GFP peak maps for single-trial microstate fitting.")
    data_blocks = []
    info = None
    for peaks in peak_sets:
        pk = peaks.copy().pick(common_channels)
        if info is None:
            info = pk.info.copy()
        data_blocks.append(pk.get_data())
    data = np.concatenate(data_blocks, axis=1)
    return ChData(data, info), pd.DataFrame(rows)


def fit_condition_model(condition: str, epoch_list: List[mne.Epochs], cfg: Config, training_conditions: Optional[Sequence[str]] = None) -> ModKMeans:
    banner(f"COMMON MODEL: {condition}", "-")
    ensure_microstate_dependencies()
    training_conditions = tuple(training_conditions or (condition,))
    out = cfg.COMMON_DIR / condition
    out.mkdir(parents=True, exist_ok=True)
    n_training_epochs = int(sum(source_epoch_count(source) for source in epoch_list))
    template_epochs = epoch_list[0]
    fit_tmin = float(template_epochs.tmin)
    fit_tmax = float(template_epochs.tmax)
    fit_source_rows = pd.DataFrame()
    if cfg.MICROSTATE_FIT_SOURCE == "erp_grand_average":
        fit_inst, fit_source_rows = build_erp_grand_average_fit_data(epoch_list, cfg)
        fit_source_rows.to_csv(out / f"{condition}_ERP_Microstate_Fit_Source_Evokeds.csv", index=False)
        if not fit_source_rows.empty:
            fit_tmin = float(pd.to_numeric(fit_source_rows["Fit_Tmin"], errors="coerce").min())
            fit_tmax = float(pd.to_numeric(fit_source_rows["Fit_Tmax"], errors="coerce").max())
    elif cfg.MICROSTATE_FIT_SOURCE == "erp_gfp_peaks":
        fit_inst, fit_source_rows = build_erp_gfp_peak_fit_data(epoch_list, cfg)
        fit_source_rows.to_csv(out / f"{condition}_ERP_GFP_Peak_Fit_Source_Evokeds.csv", index=False)
        if not fit_source_rows.empty:
            fit_tmin = float(pd.to_numeric(fit_source_rows["Fit_Tmin"], errors="coerce").min())
            fit_tmax = float(pd.to_numeric(fit_source_rows["Fit_Tmax"], errors="coerce").max())
    else:
        fit_inst, fit_source_rows = build_single_trial_gfp_fit_data(epoch_list, cfg)
        fit_source_rows.to_csv(out / f"{condition}_Single_Trial_GFP_Fit_Source_Peaks.csv", index=False)
        if not fit_source_rows.empty:
            fit_tmin = float(pd.to_numeric(fit_source_rows["Fit_Tmin"], errors="coerce").min())
            fit_tmax = float(pd.to_numeric(fit_source_rows["Fit_Tmax"], errors="coerce").max())
    logging.info("%s training epochs: %s | fit source: %s | fit window: %.3f..%.3f sec", condition, n_training_epochs, cfg.MICROSTATE_FIT_SOURCE, fit_tmin, fit_tmax)
    model = fit_modkmeans_candidates(fit_inst, cfg, condition, out)
    try:
        setattr(model, "pipeline_model_name", condition)
        setattr(model, "pipeline_training_conditions", ",".join(training_conditions))
        setattr(model, "pipeline_fit_source", cfg.MICROSTATE_FIT_SOURCE)
    except Exception:
        pass
    alignment = align_model_to_template(model, condition, out, cfg)
    centers = getattr(model, "cluster_centers_", None)
    actual_n_clusters = int(getattr(model, "n_clusters", cfg.N_CLUSTERS))
    cluster_names = get_cluster_names(model, actual_n_clusters)
    n_fit_samples = int(getattr(fit_inst, "get_data")().shape[-1]) if hasattr(fit_inst, "get_data") else np.nan
    metadata = pd.DataFrame([{**{"Condition": condition, "Model_Name": condition, "Microstate_Model_Mode": cfg.MICROSTATE_MODEL_MODE, "Fit_Source": cfg.MICROSTATE_FIT_SOURCE, "Auto_N_Clusters": cfg.AUTO_N_CLUSTERS, "Cluster_Range": f"{cfg.CLUSTER_RANGE[0]}-{cfg.CLUSTER_RANGE[1]}", "Training_Conditions": ",".join(training_conditions), "N_SubjectEpochSets": len(epoch_list), "N_Epochs": n_training_epochs, "N_Fit_Samples": n_fit_samples, "Fit_Tmin": fit_tmin, "Fit_Tmax": fit_tmax, "N_Clusters": actual_n_clusters, "N_Init": cfg.N_INIT, "GEV": float(getattr(model, "GEV_", np.nan)), "Cluster_Names": ",".join(cluster_names)}, **{f"Template_{k}": v for k, v in alignment.items()}}])
    metadata.to_csv(out / f"{condition}_Model_Metadata.csv", index=False)
    if cfg.MICROSTATE_FIT_SOURCE == "erp_grand_average":
        training_gev = model_fit_source_gev_table(condition, model, training_conditions, cfg.MICROSTATE_FIT_SOURCE)
    else:
        training_gev = model_fit_source_gev_table(condition, model, training_conditions, cfg.MICROSTATE_FIT_SOURCE)
    training_gev.to_csv(out / f"{condition}_Training_GEV_By_Topomap.csv", index=False)
    if centers is not None:
        np.save(out / f"{condition}_Cluster_Centers.npy", centers)
        center_columns = info_channel_names(getattr(model, "info", None)) or template_epochs.ch_names
        pd.DataFrame(centers, index=cluster_names, columns=center_columns).to_csv(out / f"{condition}_Cluster_Centers.csv")
        compute_topographic_metrics(model).to_csv(out / f"{condition}_Topographic_Map_Metrics.csv", index=False)
    try:
        model_file = out / f"{condition}_ModKMeans.fif"
        if model_file.exists():
            model_file = out / f"{condition}_ModKMeans_{datetime.now().strftime('%Y%m%d_%H%M%S')}.fif"
        model.save(model_file)
    except Exception as exc:
        logging.warning("Could not save pycrostates model for %s: %s", condition, exc)
    save_model_topomap(model, out / f"{condition}_Microstate_Maps.png", cfg)
    save_individual_topomaps(model, out, cfg)
    return model


def save_model_topomap(model: ModKMeans, path: Path, cfg: Config) -> None:
    if not cfg.SAVE_PNG:
        return
    try:
        result = model.plot(show=False)
        fig = result.figure if hasattr(result, "figure") else result
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        logging.warning("Could not save model topomap %s: %s", path, exc)


def backfit_microstates(model: ModKMeans, epochs: mne.Epochs, cfg: Config):
    attempts = (
        {"factor": cfg.SEGMENT_FACTOR, "half_window_size": cfg.SEGMENT_HALF_WINDOW_SIZE, "min_segment_length": cfg.SEGMENT_MIN_LENGTH, "reject_edges": cfg.SEGMENT_REJECT_EDGES, "verbose": False},
        {"factor": cfg.SEGMENT_FACTOR, "half_window_size": cfg.SEGMENT_HALF_WINDOW_SIZE, "min_segment_length": cfg.SEGMENT_MIN_LENGTH, "verbose": False},
        {"factor": cfg.SEGMENT_FACTOR, "half_window_size": cfg.SEGMENT_HALF_WINDOW_SIZE, "verbose": False},
        {"verbose": False},
    )
    last_error = None
    for kwargs in attempts:
        try:
            return model.predict(epochs, **kwargs)
        except TypeError as exc:
            last_error = exc
    raise last_error


def parameters_to_wide(params: Dict, recording: Recording, condition: str, window: str, block: str, model: ModKMeans, n_epochs: int) -> pd.DataFrame:
    row = {"Participant": recording.participant_id, "Group": recording.group, "Age": recording.age, "Sex": recording.sex, "Condition": condition, "Window": window, "Block": block, "N_Epochs": n_epochs, "Model_GEV": float(getattr(model, "GEV_", np.nan))}
    row.update({k: scalar_or_nan(v) for k, v in params.items() if not is_distribution(v)})
    return pd.DataFrame([row])


def parameters_to_long(params: Dict, recording: Recording, condition: str, window: str, block: str, model: ModKMeans, n_epochs: int) -> pd.DataFrame:
    rows = []
    for key, value in params.items():
        if is_distribution(value):
            continue
        state, metric = split_microstate_parameter_key(key)
        rows.append({"Participant": recording.participant_id, "Group": recording.group, "Age": recording.age, "Sex": recording.sex, "Condition": condition, "Window": window, "Block": block, "N_Epochs": n_epochs, "Model_GEV": float(getattr(model, "GEV_", np.nan)), "State": state, "Metric": metric, "Value": scalar_or_nan(value), "Unit": METRIC_UNITS.get(metric, "")})
    return pd.DataFrame(rows)


def split_microstate_parameter_key(key: str) -> Tuple[str, str]:
    if key == "unlabeled":
        return "unlabeled", "unlabeled_ratio"
    if key == "total_gev":
        return "all", "total_gev"
    for metric in sorted(MICROSTATE_METRICS, key=len, reverse=True):
        suffix = "_" + metric
        if key.endswith(suffix):
            return key[:-len(suffix)], metric
    return key, "value"


def is_distribution(value) -> bool:
    return isinstance(value, (list, tuple, np.ndarray)) and np.asarray(value, dtype=object).shape not in [(), (1,)]


def scalar_or_nan(value) -> float:
    try:
        arr = np.asarray(value)
        if arr.shape == ():
            return float(arr)
        if arr.size == 1:
            return float(arr.ravel()[0])
    except Exception:
        pass
    return np.nan


def save_parameter_distributions(params: Dict, out_dir: Path) -> None:
    params = params_to_dict(params)
    dist_dir = out_dir / "Parameter_Distributions"
    dist_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for key, value in params.items():
        if not is_distribution(value):
            continue
        arr = np.asarray(value, dtype=object)
        safe_key = key.replace("/", "_").replace("\\", "_")
        np.save(dist_dir / f"{safe_key}.npy", arr)
        manifest.append({"Parameter": key, "Shape": str(arr.shape), "File": f"Parameter_Distributions/{safe_key}.npy"})
    pd.DataFrame(manifest).to_csv(out_dir / "Microstate_Parameter_Distributions_Manifest.csv", index=False)


def params_to_dict(params) -> Dict:
    if isinstance(params, pd.DataFrame):
        return params.iloc[0].to_dict() if len(params) else {}
    if isinstance(params, pd.Series):
        return params.to_dict()
    return dict(params)


def add_global_parameter_rows(params: Dict, cluster_names: Sequence[str], recording: Recording, condition: str, window: str, block: str, model: ModKMeans, n_epochs: int) -> pd.DataFrame:
    rows = []
    gev_values = [scalar_or_nan(params.get(f"{state}_gev", np.nan)) for state in cluster_names]
    if not any(np.isfinite(gev_values)):
        gev_values = [scalar_or_nan(value) for key, value in params.items() if str(key).endswith("_gev") and not is_distribution(value)]
    if any(np.isfinite(gev_values)):
        rows.append({"Participant": recording.participant_id, "Group": recording.group, "Age": recording.age, "Sex": recording.sex, "Condition": condition, "Window": window, "Block": block, "N_Epochs": n_epochs, "Model_GEV": float(getattr(model, "GEV_", np.nan)), "State": "all", "Metric": "total_gev", "Value": float(np.nansum(gev_values)), "Unit": METRIC_UNITS.get("total_gev", "")})
    if "unlabeled" in params:
        rows.append({"Participant": recording.participant_id, "Group": recording.group, "Age": recording.age, "Sex": recording.sex, "Condition": condition, "Window": window, "Block": block, "N_Epochs": n_epochs, "Model_GEV": float(getattr(model, "GEV_", np.nan)), "State": "unlabeled", "Metric": "unlabeled_ratio", "Value": scalar_or_nan(params["unlabeled"]), "Unit": METRIC_UNITS.get("unlabeled_ratio", "")})
    return pd.DataFrame(rows)


def distribution_summary(values) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0, "mean": np.nan, "std": np.nan, "median": np.nan, "q25": np.nan, "q75": np.nan, "min": np.nan, "max": np.nan}
    return {"n": int(arr.size), "mean": float(np.mean(arr)), "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0, "median": float(np.median(arr)), "q25": float(np.percentile(arr, 25)), "q75": float(np.percentile(arr, 75)), "min": float(np.min(arr)), "max": float(np.max(arr))}


def distribution_summaries_to_long(params: Dict, recording: Recording, condition: str, window: str, block: str, model: ModKMeans, n_epochs: int) -> pd.DataFrame:
    params = params_to_dict(params)
    rows = []
    readable = {"dist_corr": "correlation_distribution", "dist_gev": "gev_contribution_distribution", "dist_durs": "duration_distribution"}
    unit = {"correlation_distribution": "correlation", "gev_contribution_distribution": "proportion", "duration_distribution": "seconds"}
    for key, value in params.items():
        if not is_distribution(value):
            continue
        state, metric = split_microstate_parameter_key(key)
        metric_prefix = readable.get(metric, metric)
        for stat, stat_value in distribution_summary(value).items():
            rows.append({"Participant": recording.participant_id, "Group": recording.group, "Age": recording.age, "Sex": recording.sex, "Condition": condition, "Window": window, "Block": block, "N_Epochs": n_epochs, "Model_GEV": float(getattr(model, "GEV_", np.nan)), "State": state, "Metric": f"{metric_prefix}_{stat}", "Value": float(stat_value), "Unit": "count" if stat == "n" else unit.get(metric_prefix, "")})
    return pd.DataFrame(rows)


def flatten_label_sequence(labels: np.ndarray, collapse_repetitions: bool = False) -> np.ndarray:
    arr = np.asarray(labels, dtype=int).ravel()
    arr = arr[arr >= 0]
    if collapse_repetitions and arr.size:
        arr = arr[np.r_[True, arr[1:] != arr[:-1]]]
    return arr


def lempel_ziv_complexity(labels: np.ndarray) -> int:
    seq = "".join(chr(65 + int(label)) for label in labels if int(label) >= 0)
    if not seq:
        return 0
    phrases, phrase, complexity = set(), "", 0
    for symbol in seq:
        phrase += symbol
        if phrase not in phrases:
            phrases.add(phrase)
            complexity += 1
            phrase = ""
    return complexity + (1 if phrase else 0)


def count_runs_and_switches(labels_2d: np.ndarray) -> Tuple[int, int, int]:
    run_count = 0
    switch_count = 0
    valid_samples = 0
    for row in labels_to_2d(labels_2d):
        valid = row[row >= 0]
        valid_samples += int(valid.size)
        if valid.size == 0:
            continue
        changes = valid[1:] != valid[:-1]
        switch_count += int(np.sum(changes))
        run_count += int(np.sum(changes) + 1)
    return run_count, switch_count, valid_samples


def markov_entropy(labels: np.ndarray, n_states: int) -> float:
    counts = np.zeros((n_states, n_states), dtype=float)
    for row in labels_to_2d(labels):
        valid = row[row >= 0]
        if valid.size < 2:
            continue
        for a, b in zip(valid[:-1], valid[1:]):
            if 0 <= a < n_states and 0 <= b < n_states:
                counts[a, b] += 1.0
    outgoing = counts.sum(axis=1)
    total = float(outgoing.sum())
    if total == 0:
        return np.nan
    entropy = 0.0
    for idx in range(n_states):
        if outgoing[idx] == 0:
            continue
        probs = counts[idx] / outgoing[idx]
        probs = probs[probs > 0]
        entropy += (outgoing[idx] / total) * float(-np.sum(probs * np.log2(probs)))
    return float(entropy)


def sequence_metrics(labels: np.ndarray, sfreq: float, recording: Recording, condition: str, window: str, block: str, model: ModKMeans, n_epochs: int, cluster_names: Sequence[str]) -> pd.DataFrame:
    labels_2d = labels_to_2d(labels)
    flat = flatten_label_sequence(labels_2d, collapse_repetitions=False)
    run_count, switches, valid_samples = count_runs_and_switches(labels_2d)
    duration_sec = valid_samples / sfreq if sfreq else np.nan
    lz = lempel_ziv_complexity(flat)
    metrics = {"lz_complexity": float(lz), "lz_complexity_normalized": float(lz / len(flat)) if len(flat) else np.nan, "markov_entropy_bits": markov_entropy(labels_2d, len(cluster_names)), "switching_rate_hz": float(switches / duration_sec) if duration_sec and np.isfinite(duration_sec) and duration_sec > 0 else np.nan, "mean_segment_duration_ms": float(duration_sec * 1000.0 / run_count) if run_count and np.isfinite(duration_sec) else np.nan, "valid_label_ratio": float(valid_samples / labels_2d.size) if labels_2d.size else np.nan}
    return pd.DataFrame([{"Participant": recording.participant_id, "Group": recording.group, "Age": recording.age, "Sex": recording.sex, "Condition": condition, "Window": window, "Block": block, "Metric": key, "Value": value, "Unit": METRIC_UNITS.get(key, "proportion" if key == "valid_label_ratio" else ""), "N_Epochs": n_epochs, "Model_GEV": float(getattr(model, "GEV_", np.nan))} for key, value in metrics.items()])


def matrix_to_frame(matrix: np.ndarray, names: Sequence[str], recording: Recording, condition: str, window: str, block: str, matrix_name: str, statistic: str) -> pd.DataFrame:
    rows = []
    for i, from_state in enumerate(names):
        for j, to_state in enumerate(names):
            rows.append({"Participant": recording.participant_id, "Group": recording.group, "Age": recording.age, "Sex": recording.sex, "Condition": condition, "Window": window, "Block": block, "Matrix": matrix_name, "Statistic": statistic, "From": from_state, "To": to_state, "Value": float(matrix[i, j])})
    return pd.DataFrame(rows)


def save_matrix(matrix: np.ndarray, names: Sequence[str], out_dir: Path, stem: str, cfg: Optional[Config] = None) -> None:
    pd.DataFrame(matrix, index=names, columns=names).to_csv(out_dir / f"{stem}.csv")
    np.save(out_dir / f"{stem}.npy", matrix)
    if cfg is not None and cfg.SAVE_TRANSITION_HEATMAPS and cfg.SAVE_PNG:
        cmap = "coolwarm" if "Delta" in stem or "delta" in stem else "Blues"
        fig, ax = plt.subplots(figsize=(5.2, 4.4))
        if cmap == "coolwarm":
            vmax = float(np.nanmax(np.abs(matrix))) if np.isfinite(matrix).any() else 1.0
            im = ax.imshow(matrix, cmap=cmap, vmin=-vmax, vmax=vmax)
        else:
            im = ax.imshow(matrix, cmap=cmap)
        ax.set_xticks(np.arange(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel("To microstate")
        ax.set_ylabel("From microstate")
        ax.set_title(stem.replace("_", " "))
        fig.colorbar(im, ax=ax, shrink=0.85)
        fig.tight_layout()
        fig.savefig(out_dir / f"{stem}.png", dpi=250, bbox_inches="tight")
        plt.close(fig)


def standardize_columns(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data, dtype=float).copy()
    arr -= np.nanmean(arr, axis=0, keepdims=True)
    denom = np.nanstd(arr, axis=0, keepdims=True)
    denom[~np.isfinite(denom) | (denom == 0)] = 1.0
    return arr / denom


def standardize_rows(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data, dtype=float).copy()
    arr -= np.nanmean(arr, axis=1, keepdims=True)
    denom = np.nanstd(arr, axis=1, keepdims=True)
    denom[~np.isfinite(denom) | (denom == 0)] = 1.0
    return arr / denom


def evoked_model_channels(model: ModKMeans) -> List[str]:
    return info_channel_names(getattr(model, "info", None))


def segment_evoked_with_model(evoked: mne.Evoked, model: ModKMeans) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model_channels = evoked_model_channels(model) or list(evoked.ch_names)
    missing = [ch for ch in model_channels if ch not in evoked.ch_names]
    if missing:
        raise ValueError(f"Evoked is missing channels used by model: {missing}")
    ev = evoked.copy().pick(model_channels)
    data_uv = ev.data * 1e6
    states = np.asarray(model.cluster_centers_, dtype=float)
    data_z = standardize_columns(data_uv)
    states_z = standardize_rows(states)
    corr = (states_z @ data_z) / max(1, data_z.shape[0])
    labels = np.argmax(np.abs(corr), axis=0)
    best_corr = corr[labels, np.arange(corr.shape[1])]
    gfp_uv = np.nanstd(data_uv, axis=0)
    explained_uV2 = (gfp_uv * np.abs(best_corr)) ** 2
    return labels.astype(int), best_corr.astype(float), gfp_uv.astype(float), explained_uV2.astype(float)


def erp_microstate_p300_temporal_parameters(epochs: mne.Epochs, model: ModKMeans, recording: Recording, condition: str, window: str, block: str, cfg: Config, out_dir: Path) -> pd.DataFrame:
    evoked = epochs.average()
    labels, corr, gfp_uv, explained_uV2 = segment_evoked_with_model(evoked, model)
    cluster_names = get_cluster_names(model, int(getattr(model, "n_clusters", cfg.N_CLUSTERS)))
    times_ms = evoked.times * 1000.0
    start, stop = cfg.ERP_MICROSTATE_P300_WINDOW
    mask = (evoked.times >= start) & (evoked.times <= stop)
    label_table = pd.DataFrame({"Time_ms": times_ms, "State_Index": labels, "State": [cluster_names[idx] if 0 <= idx < len(cluster_names) else "" for idx in labels], "Signed_Correlation": corr, "GFP_uV": gfp_uv, "Explained_uV2": explained_uV2})
    label_table.to_csv(out_dir / "ERP_Microstate_Evoked_Label_Times.csv", index=False)
    if not mask.any():
        return pd.DataFrame()
    rows = []
    dt_ms = 1000.0 / float(evoked.info["sfreq"]) if evoked.info["sfreq"] else np.nan
    p300_times = times_ms[mask]
    p300_labels = labels[mask]
    p300_corr = corr[mask]
    p300_gfp = gfp_uv[mask]
    p300_explained = explained_uV2[mask]
    model_gev = float(getattr(model, "GEV_", np.nan))
    for idx, state in enumerate(cluster_names):
        state_mask = p300_labels == idx
        if state_mask.any():
            state_times = p300_times[state_mask]
            state_explained = p300_explained[state_mask]
            weight_sum = float(np.nansum(state_explained))
            metrics = {
                "onset_ms": float(np.nanmin(state_times)),
                "offset_ms": float(np.nanmax(state_times)),
                "duration_ms": float(state_mask.sum() * dt_ms) if np.isfinite(dt_ms) else np.nan,
                "explained_auc_uV2_ms": float(trapezoid(state_explained, state_times)) if state_explained.size > 1 else 0.0,
                "center_of_gravity_ms": float(np.nansum(state_times * state_explained) / weight_sum) if weight_sum > 0 else np.nan,
                "mean_gfp_uV": float(np.nanmean(p300_gfp[state_mask])),
                "mean_abs_correlation": float(np.nanmean(np.abs(p300_corr[state_mask]))),
            }
        else:
            metrics = {"onset_ms": np.nan, "offset_ms": np.nan, "duration_ms": 0.0, "explained_auc_uV2_ms": 0.0, "center_of_gravity_ms": np.nan, "mean_gfp_uV": np.nan, "mean_abs_correlation": np.nan}
        for metric, value in metrics.items():
            unit = "ms" if metric.endswith("_ms") else "uV^2_ms" if metric == "explained_auc_uV2_ms" else "uV" if metric == "mean_gfp_uV" else "correlation"
            rows.append({"Participant": recording.participant_id, "Group": recording.group, "Age": recording.age, "Sex": recording.sex, "Condition": condition, "Window": window, "Block": block, "Analysis_Window": "P300", "Analysis_Tmin_s": start, "Analysis_Tmax_s": stop, "N_Epochs": len(epochs), "Model_GEV": model_gev, "State": state, "Metric": metric, "Value": value, "Unit": unit})
    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "ERP_Microstate_P300_Temporal_Parameters.csv", index=False)
    if cfg.SAVE_PNG:
        fig, ax = plt.subplots(figsize=(9, 3.2))
        ax.plot(times_ms, gfp_uv, color="black", linewidth=1.2)
        y0, _ = ax.get_ylim()
        ax.scatter(times_ms, np.full_like(times_ms, y0), c=labels, cmap="tab20", s=10, marker="s")
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.axvspan(start * 1000.0, stop * 1000.0, color="tab:blue", alpha=0.12)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("GFP (uV)")
        ax.set_title("ERP microstate segmentation on averaged ERP")
        fig.tight_layout()
        fig.savefig(out_dir / "ERP_Microstate_Evoked_Segmentation.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    return result


def segmentation_base_row(recording: Recording, condition: str, window: str, block: str) -> Dict[str, object]:
    return {"Participant": recording.participant_id, "Group": recording.group, "Age": recording.age, "Sex": recording.sex, "Condition": condition, "Window": window, "Block": block}


def save_microstate_segments(labels: np.ndarray, epochs: mne.Epochs, out_dir: Path, cluster_names: Sequence[str], recording: Recording, condition: str, window: str, block: str) -> pd.DataFrame:
    labels_2d = labels_to_2d(labels)
    times = segmentation_times(epochs, labels_2d.shape[1])
    metadata = epochs.metadata.reset_index(drop=True) if epochs.metadata is not None else pd.DataFrame(index=np.arange(labels_2d.shape[0]))
    base = segmentation_base_row(recording, condition, window, block)
    sfreq = float(epochs.info["sfreq"])
    rows = []
    for ep_idx, row_labels in enumerate(labels_2d):
        epoch_id = metadata.iloc[ep_idx].get("epoch_id", ep_idx + 1) if len(metadata) > ep_idx else ep_idx + 1
        if row_labels.size == 0:
            continue
        starts = np.r_[0, np.where(row_labels[1:] != row_labels[:-1])[0] + 1]
        stops = np.r_[starts[1:] - 1, row_labels.size - 1]
        for segment_idx, (start_idx, stop_idx) in enumerate(zip(starts, stops), start=1):
            label = int(row_labels[int(start_idx)])
            name = cluster_names[label] if 0 <= label < len(cluster_names) else "unlabeled"
            start_time = float(times[int(start_idx)])
            end_time = float(times[int(stop_idx)])
            n_samples = int(stop_idx - start_idx + 1)
            duration_ms = float(n_samples / sfreq * 1000.0) if sfreq else np.nan
            rows.append({**base, "Epoch_Index": ep_idx, "Epoch_ID": epoch_id, "Segment_Index": segment_idx, "Microstate_Label": label, "Microstate_Name": name, "Start_Sample": int(start_idx), "End_Sample": int(stop_idx), "Start_Time_s": start_time, "End_Time_s": end_time, "Start_Time_ms": start_time * 1000.0, "End_Time_ms": end_time * 1000.0, "Duration_ms": duration_ms, "N_Samples": n_samples})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "Microstate_Segments.csv", index=False)
    return df


def save_epoch_state_summary(labels: np.ndarray, epochs: mne.Epochs, out_dir: Path, cluster_names: Sequence[str], recording: Recording, condition: str, window: str, block: str) -> pd.DataFrame:
    labels_2d = labels_to_2d(labels)
    metadata = epochs.metadata.reset_index(drop=True) if epochs.metadata is not None else pd.DataFrame(index=np.arange(labels_2d.shape[0]))
    base = segmentation_base_row(recording, condition, window, block)
    sfreq = float(epochs.info["sfreq"])
    rows = []
    all_names = list(cluster_names) + ["unlabeled"]
    for ep_idx, row_labels in enumerate(labels_2d):
        epoch_id = metadata.iloc[ep_idx].get("epoch_id", ep_idx + 1) if len(metadata) > ep_idx else ep_idx + 1
        epoch_samples = int(row_labels.size)
        starts = np.r_[0, np.where(row_labels[1:] != row_labels[:-1])[0] + 1] if row_labels.size else np.array([], dtype=int)
        stops = np.r_[starts[1:] - 1, row_labels.size - 1] if starts.size else np.array([], dtype=int)
        segment_labels = row_labels[starts] if starts.size else np.array([], dtype=int)
        segment_lengths = stops - starts + 1 if starts.size else np.array([], dtype=int)
        for label_idx, state_name in enumerate(all_names):
            label_value = label_idx if state_name != "unlabeled" else -1
            if state_name == "unlabeled":
                sample_mask = row_labels < 0
                segment_mask = segment_labels < 0
            else:
                sample_mask = row_labels == label_value
                segment_mask = segment_labels == label_value
            n_samples = int(np.sum(sample_mask))
            n_segments = int(np.sum(segment_mask))
            rows.append({**base, "Epoch_Index": ep_idx, "Epoch_ID": epoch_id, "Microstate_Label": label_value, "Microstate_Name": state_name, "N_Samples": n_samples, "TimeCoverage": float(n_samples / epoch_samples) if epoch_samples else np.nan, "TotalDuration_ms": float(n_samples / sfreq * 1000.0) if sfreq else np.nan, "N_Segments": n_segments, "MeanDuration_ms": float(np.mean(segment_lengths[segment_mask]) / sfreq * 1000.0) if n_segments and sfreq else np.nan})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "Microstate_Epoch_State_Summary.csv", index=False)
    return df


def save_segmentation_labels(segmentation, epochs: mne.Epochs, out_dir: Path, cluster_names: Sequence[str], cfg: Config) -> np.ndarray:
    labels = get_segmentation_labels(segmentation)
    np.save(out_dir / "Microstate_Labels.npy", labels)
    labels_2d = labels_to_2d(labels)
    pd.DataFrame(labels_2d).to_csv(out_dir / "Microstate_Labels.csv", index_label="Epoch_Index")
    times = segmentation_times(epochs, labels_2d.shape[1])
    pd.DataFrame({"Time_s": times, "Time_ms": times * 1000.0}).to_csv(out_dir / "Microstate_Label_Times.csv", index=False)
    if cfg.SAVE_TIME_SERIES_LONG:
        n_epochs, n_times = labels_2d.shape
        metadata = epochs.metadata.reset_index(drop=True) if epochs.metadata is not None else pd.DataFrame(index=np.arange(n_epochs))
        epoch_ids = metadata["epoch_id"].to_numpy() if "epoch_id" in metadata.columns and len(metadata) == n_epochs else np.arange(1, n_epochs + 1)
        flat_labels = labels_2d.ravel()
        names = np.asarray([cluster_names[int(label)] if 0 <= int(label) < len(cluster_names) else "unlabeled" for label in flat_labels], dtype=object)
        ts = pd.DataFrame({"Epoch_Index": np.repeat(np.arange(n_epochs), n_times), "Epoch_ID": np.repeat(epoch_ids, n_times), "Time_s": np.tile(times, n_epochs), "Time_ms": np.tile(times * 1000.0, n_epochs), "Microstate_Label": flat_labels, "Microstate_Name": names})
        ts.to_csv(out_dir / "Microstate_TimeSeries_Long.csv", index=False)
        ts[ts["Epoch_Index"].eq(0)].to_csv(out_dir / "Microstate_TimeSeries_FirstEpoch.csv", index=False)
    return labels


def save_segmentation_figure(labels: np.ndarray, epochs: mne.Epochs, out_dir: Path, cfg: Config) -> None:
    if not cfg.SAVE_PNG:
        return
    try:
        labels_2d = labels_to_2d(labels)
        times = segmentation_times(epochs, labels_2d.shape[1])
        n_show = min(labels_2d.shape[0], cfg.MAX_SEGMENTATION_PLOT_EPOCHS)
        fig, ax = plt.subplots(figsize=(10, max(3.5, n_show * 0.22)))
        image = ax.imshow(labels_2d[:n_show], aspect="auto", interpolation="nearest", extent=[times[0] * 1000, times[-1] * 1000, n_show, 0])
        ax.axvline(0, color="white", linestyle="--", linewidth=1)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Epoch")
        ax.set_title(f"Microstate segmentation, first {n_show} epochs")
        fig.colorbar(image, ax=ax, label="State label")
        fig.tight_layout()
        fig.savefig(out_dir / "Microstate_Segmentation_Heatmap.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        n_step = min(labels_2d.shape[0], cfg.SEGMENTATION_STEP_PLOT_EPOCHS)
        fig, ax = plt.subplots(figsize=(10, max(3.5, n_step * 0.45)))
        finite_labels = labels_2d[np.isfinite(labels_2d)]
        n_states = max(int(cfg.N_CLUSTERS), int(np.nanmax(finite_labels)) + 1 if finite_labels.size else int(cfg.N_CLUSTERS))
        for idx in range(n_step):
            ax.step(times * 1000.0, labels_2d[idx] + idx * (n_states + 1), where="post", linewidth=1.2, label=f"Epoch {idx + 1}" if idx < 5 else None)
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Microstate label + epoch offset")
        ax.set_title(f"Microstate time-series segmentation, first {n_step} epochs")
        ax.grid(True, axis="x", alpha=0.2)
        if n_step <= 5:
            ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(out_dir / "Microstate_TimeSeries_StepPlot.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        logging.warning("Could not save custom segmentation heatmap in %s: %s", out_dir, exc)


def save_pycrostates_segmentation_plot(segmentation, out_dir: Path, cfg: Config) -> None:
    if not cfg.SAVE_PNG:
        return
    try:
        result = segmentation.plot(show=False)
        fig = result.figure if hasattr(result, "figure") else result
        fig.savefig(out_dir / "Microstate_Segmentation_Pycrostates.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        logging.warning("Could not save pycrostates segmentation plot in %s: %s", out_dir, exc)


def compute_and_save_microstates(epochs: mne.Epochs, model: ModKMeans, recording: Recording, condition: str, window: str, block: str, out_dir: Path, cfg: Config) -> Dict[str, pd.DataFrame]:
    segmentation = backfit_microstates(model, epochs, cfg)
    params = compute_segmentation_parameters(segmentation, norm_gfp=True, return_dist=False)
    params = params_to_dict(params)
    params_wide = parameters_to_wide(params, recording, condition, window, block, model, len(epochs))
    params_long = parameters_to_long(params, recording, condition, window, block, model, len(epochs))
    actual_n_clusters = int(getattr(model, "n_clusters", cfg.N_CLUSTERS))
    cluster_names = get_cluster_names(segmentation, actual_n_clusters)
    pd.DataFrame([{"Participant": recording.participant_id, "Condition": condition, "Window": window, "Block": block, "Microstate_Model": str(getattr(model, "pipeline_model_name", "")), "Model_Training_Conditions": str(getattr(model, "pipeline_training_conditions", "")), "Model_GEV": float(getattr(model, "GEV_", np.nan)), "Cluster_Names": ",".join(cluster_names)}]).to_csv(out_dir / "Microstate_Model_Used.csv", index=False)
    global_params = add_global_parameter_rows(params, cluster_names, recording, condition, window, block, model, len(epochs))
    if not global_params.empty:
        params_long = pd.concat([params_long, global_params], ignore_index=True)
    params_wide.to_csv(out_dir / "Microstate_Parameters_Wide.csv", index=False)
    dist_summary = pd.DataFrame()
    if cfg.SAVE_PARAMETER_DISTRIBUTIONS:
        try:
            dist_params = params_to_dict(compute_segmentation_parameters(segmentation, norm_gfp=True, return_dist=True))
            save_parameter_distributions(dist_params, out_dir)
            dist_summary = distribution_summaries_to_long(dist_params, recording, condition, window, block, model, len(epochs))
            dist_summary.to_csv(out_dir / "Microstate_Distribution_Summaries.csv", index=False)
            if not dist_summary.empty:
                params_long = pd.concat([params_long, dist_summary], ignore_index=True)
        except Exception as exc:
            logging.warning("Parameter distributions failed in %s: %s", out_dir, exc)
    params_long.to_csv(out_dir / "Microstate_Parameters_Long.csv", index=False)
    matrix_frames = []
    observed_probability = None
    expected_probability = None
    for stat in ("count", "probability", "percent"):
        try:
            observed = compute_transition_safe(segmentation, expected=False, stat=stat, ignore_repetitions=True)
            save_matrix(observed, cluster_names, out_dir, f"Observed_Transition_{stat}", cfg)
            if stat == "probability":
                observed_probability = observed
            matrix_frames.append(matrix_to_frame(observed, cluster_names, recording, condition, window, block, "Observed_Transition", stat))
        except Exception as exc:
            logging.warning("Observed transition %s failed in %s: %s", stat, out_dir, exc)
    for stat in ("probability", "percent"):
        try:
            expected = compute_transition_safe(segmentation, expected=True, stat=stat, ignore_repetitions=True)
            save_matrix(expected, cluster_names, out_dir, f"Expected_Transition_{stat}", cfg)
            if stat == "probability":
                expected_probability = expected
            matrix_frames.append(matrix_to_frame(expected, cluster_names, recording, condition, window, block, "Expected_Transition", stat))
        except Exception as exc:
            logging.warning("Expected transition %s failed in %s: %s", stat, out_dir, exc)
    if observed_probability is not None and expected_probability is not None:
        delta = observed_probability - expected_probability
        save_matrix(delta, cluster_names, out_dir, "Delta_Transition_probability_observed_minus_expected", cfg)
        matrix_frames.append(matrix_to_frame(delta, cluster_names, recording, condition, window, block, "Delta_Transition_ObservedMinusExpected", "probability"))
    matrix_long = pd.concat(matrix_frames, ignore_index=True) if matrix_frames else pd.DataFrame()
    matrix_long.to_csv(out_dir / "Microstate_Matrices_Long.csv", index=False)
    entropy_rows = []
    for ignore_reps, label in [(False, "Entropy"), (True, "Entropy_NoRepeats")]:
        try:
            entropy_rows.append({"Participant": recording.participant_id, "Group": recording.group, "Age": recording.age, "Sex": recording.sex, "Condition": condition, "Window": window, "Block": block, "Metric": label, "Value": segmentation_entropy_bits(segmentation, ignore_repetitions=ignore_reps), "N_Epochs": len(epochs), "Model_GEV": float(getattr(model, "GEV_", np.nan))})
        except Exception as exc:
            logging.warning("Entropy failed in %s: %s", out_dir, exc)
    entropy = pd.DataFrame(entropy_rows)
    entropy.to_csv(out_dir / "Microstate_Entropy.csv", index=False)
    labels = save_segmentation_labels(segmentation, epochs, out_dir, cluster_names, cfg)
    segment_table = save_microstate_segments(labels, epochs, out_dir, cluster_names, recording, condition, window, block)
    epoch_state_summary = save_epoch_state_summary(labels, epochs, out_dir, cluster_names, recording, condition, window, block)
    seq_metrics = sequence_metrics(labels, float(epochs.info["sfreq"]), recording, condition, window, block, model, len(epochs), cluster_names)
    seq_metrics.to_csv(out_dir / "Microstate_Sequence_Metrics.csv", index=False)
    save_segmentation_figure(labels, epochs, out_dir, cfg)
    save_pycrostates_segmentation_plot(segmentation, out_dir, cfg)
    return {"parameters_wide": params_wide, "parameters_long": params_long, "distribution_summaries": dist_summary, "matrices_long": matrix_long, "entropy": entropy, "sequence_metrics": seq_metrics, "segments": segment_table, "epoch_state_summary": epoch_state_summary}


###############################################################################
# SUBJECT-WISE PROCESSING, SF BLOCK-WISE ANALYSIS, AND MASTER TABLES
###############################################################################

def crop_epochs_window(epochs: mne.Epochs, start: float, stop: float) -> Optional[mne.Epochs]:
    tmin = max(float(start), float(epochs.tmin))
    tmax = min(float(stop), float(epochs.tmax))
    if tmax <= tmin:
        return None
    cropped = epochs.copy().crop(tmin=tmin, tmax=tmax)
    return cropped if len(cropped.times) > 2 else None


def process_microstate_epoch_windows(recording: Recording, epochs: mne.Epochs, model: ModKMeans, condition: str, block: str, parent_dir: Path, cfg: Config) -> None:
    if not cfg.ENABLE_EPOCH_WINDOW_ANALYSIS:
        return
    windows_dir = parent_dir / "Microstate_Epoch_Windows"
    windows_dir.mkdir(parents=True, exist_ok=True)
    for name, start, stop in cfg.MICROSTATE_EPOCH_WINDOWS:
        cropped = crop_epochs_window(epochs, start, stop)
        if cropped is None:
            continue
        out = windows_dir / name
        out.mkdir(parents=True, exist_ok=True)
        compute_and_save_microstates(cropped, model, recording, condition, f"EpochWindow_{name}", block, out, cfg)


def process_condition(recording: Recording, epochs: mne.Epochs, model: Optional[ModKMeans], condition: str, window: str, block: str, out_dir: Path, cfg: Config) -> None:
    banner(f"{recording.participant_id} | {condition} | {window}", "-")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_behavior_outputs(recording, condition, window, block, out_dir, cfg)
    if cfg.ANALYSIS_MODE in ("both", "erp"):
        evoked, p300 = compute_p300(epochs, recording, condition, window, block, cfg)
        save_p300_outputs(evoked, p300, out_dir, cfg)
        compute_erp_window_metrics(epochs, evoked, recording, condition, window, block, cfg).to_csv(out_dir / "ERP_Window_Metrics.csv", index=False)
        compute_single_trial_erp_metrics(epochs, recording, condition, window, block, cfg).to_csv(out_dir / "Single_Trial_P300_Metrics.csv", index=False)
    if cfg.ANALYSIS_MODE in ("both", "microstate") and model is not None:
        compute_and_save_microstates(epochs, model, recording, condition, window, block, out_dir, cfg)
        if window == "Whole":
            try:
                erp_microstate_p300_temporal_parameters(epochs, model, recording, condition, window, block, cfg, out_dir)
            except Exception as exc:
                logging.warning("ERP microstate P300 temporal parameters failed in %s: %s", out_dir, exc)
            process_microstate_epoch_windows(recording, epochs, model, condition, block, out_dir, cfg)
    metadata = epochs.metadata if epochs.metadata is not None else pd.DataFrame(index=np.arange(len(epochs)))
    metadata.to_csv(out_dir / "Epoch_Metadata.csv", index=False)
    if window == "Whole" and block == "Whole":
        try:
            (out_dir / "Epoch_Cache_Path.txt").write_text(str(cached_epoch_path(cfg, recording.participant_id, condition)), encoding="utf-8")
        except Exception as exc:
            logging.warning("Could not write epoch cache pointer in %s: %s", out_dir, exc)
    if cfg.SAVE_FIF and cfg.SAVE_ANALYSIS_EPOCH_COPIES:
        try:
            epochs.save(out_dir / "Epochs-epo.fif", overwrite=True)
        except Exception as exc:
            logging.warning("Could not save epochs FIF in %s: %s", out_dir, exc)


def process_condition_blocks(recording: Recording, epochs: mne.Epochs, model: Optional[ModKMeans], cfg: Config, condition: str) -> None:
    block_column = ""
    if epochs is not None and epochs.metadata is not None and cfg.BLOCK_METADATA_COLUMN in epochs.metadata.columns:
        block_column = cfg.BLOCK_METADATA_COLUMN
    if not block_column and epochs is not None and epochs.metadata is not None and "sub_block" in epochs.metadata.columns:
        block_column = "sub_block"
    if not cfg.ENABLE_BLOCK_ANALYSIS or epochs is None or epochs.metadata is None or not block_column:
        return
    blocks = sorted(pd.Series(epochs.metadata[block_column]).dropna().unique(), key=lambda x: float(x) if str(x).replace(".", "", 1).isdigit() else str(x))
    block_p300_frames = []
    for block_value in blocks:
        idx = np.where(epochs.metadata[block_column].astype(str).to_numpy() == str(block_value))[0]
        if len(idx) < cfg.MIN_SF_BLOCK_EPOCHS:
            continue
        block_label = f"Block_{int(float(block_value)):03d}" if str(block_value).replace(".", "", 1).isdigit() else f"Block_{block_value}"
        block_epochs = epochs[idx]
        out_dir = condition_dir(cfg, recording.participant_id, condition, block_label)
        process_condition(recording, block_epochs, model, condition, block_label, str(block_value), out_dir, cfg)
        p300_path = out_dir / "P300_Metrics.csv"
        if p300_path.exists():
            try:
                block_p300_frames.append(pd.read_csv(p300_path))
            except Exception as exc:
                logging.warning("Could not read block P300 metrics from %s: %s", p300_path, exc)
    if block_p300_frames:
        summary = pd.concat(block_p300_frames, ignore_index=True)
        summary.to_csv(condition_dir(cfg, recording.participant_id, condition) / "Block_P300_Summary.csv", index=False)


def process_subject(row: pd.Series, models: Dict[str, ModKMeans], cfg: Config, baseline: bool = True, dataset: Optional[BIDSDataset] = None) -> bool:
    participant_id = str(row["participant_id"])
    try:
        usable, reason = cached_subject_is_usable(cfg, participant_id, baseline=baseline)
        if not usable:
            logging.warning("Subject skipped: %s | prepared cache is not usable: %s", participant_id, reason)
            out = subject_dir(cfg, participant_id)
            pd.DataFrame([{"Participant": participant_id, "Completed": False, "Error": f"Prepared cache is not usable: {reason}"}]).to_csv(out / "Subject_Metadata.csv", index=False)
            return False
        recording = dataset.load_recording(row) if dataset is not None else recording_from_row_and_manifest(row)
        summary_rows = []
        for condition in cfg.CONDITIONS:
            ep = load_cached_epochs(cfg, participant_id, condition)
            summary_rows.append({"Participant": participant_id, "Group": recording.group, "Condition": condition, "N_Epochs": 0 if ep is None else len(ep)})
            if ep is None or len(ep) == 0:
                continue
            model = models.get(condition)
            process_condition(recording, ep, model, condition, "Whole", "Whole", condition_dir(cfg, participant_id, condition), cfg)
            process_condition_blocks(recording, ep, model, cfg, condition)
            del ep
            gc.collect()
        pd.DataFrame(summary_rows).to_csv(subject_dir(cfg, participant_id) / "Subject_Epoch_Counts.csv", index=False)
        pd.DataFrame([{"Participant": participant_id, "Group": recording.group, "Age": recording.age, "Sex": recording.sex, "Completed": True}]).to_csv(subject_dir(cfg, participant_id) / "Subject_Metadata.csv", index=False)
        return True
    except Exception as exc:
        logging.exception("Subject failed: %s | %s", participant_id, exc)
        out = subject_dir(cfg, participant_id)
        pd.DataFrame([{"Participant": participant_id, "Completed": False, "Error": str(exc)}]).to_csv(out / "Subject_Metadata.csv", index=False)
        return False


def process_dataset(dataset: BIDSDataset, participants: pd.DataFrame, cfg: Config, baseline: bool = True) -> None:
    prepare_dataset_once(dataset, participants, cfg, baseline=baseline)
    if cfg.PREPARE_ONLY:
        logging.info("Prepare-only requested; prepared epochs are available in: %s", cfg.CACHE_DIR)
        return
    if cfg.ANALYSIS_MODE in ("both", "microstate"):
        ensure_microstate_dependencies()
    models = build_common_models(participants, cfg, baseline=baseline) if cfg.ANALYSIS_MODE in ("both", "microstate") else {}
    rows = []
    for index, row in participants.iterrows():
        participant_id = str(row["participant_id"])
        banner(f"PROCESSING SUBJECT {index + 1}/{len(participants)}: {participant_id}")
        ok = process_subject(row, models, cfg, baseline=baseline, dataset=dataset)
        rows.append({"Participant": participant_id, "Completed": ok})
    pd.DataFrame(rows).to_csv(cfg.RUN_DIR / "Processing_Summary.csv", index=False)
    build_master_tables(cfg)
    run_integration_analyses(cfg)
    run_group_statistics(cfg)
    make_publication_figures(cfg)
    generate_final_report(cfg, participants)


def concatenate_csvs(files: Iterable[Path], out_file: Path) -> pd.DataFrame:
    frames = []
    for file in files:
        try:
            frames.append(pd.read_csv(file))
        except Exception as exc:
            logging.warning("Could not read %s: %s", file, exc)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    df.to_csv(out_file, index=False)
    return df


def safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path) if path.exists() else pd.DataFrame()
    except EmptyDataError:
        return pd.DataFrame()


def subject_output_manifest(cfg: Config, pattern: str, out_file: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(cfg.SUBJECTS_DIR.rglob(pattern)):
        try:
            rel = path.relative_to(cfg.SUBJECTS_DIR)
            parts = rel.parts
        except Exception:
            parts = path.parts
        rows.append({"File": str(path), "Participant": parts[0] if len(parts) > 0 else "", "ConditionOrFolder": parts[1] if len(parts) > 1 else "", "Subfolder": "/".join(parts[2:-1]) if len(parts) > 3 else "", "Rows": safe_count_csv_rows(path) if path.suffix.lower() == ".csv" else np.nan, "Bytes": path.stat().st_size if path.exists() else np.nan})
    df = pd.DataFrame(rows)
    df.to_csv(out_file, index=False)
    return df


def summarize_single_trial_subjects(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    group_cols = [c for c in list(IDENTIFIER_COLUMNS) + ["Component", "Channel"] if c in df.columns]
    value_cols = [c for c in ("PeakAmplitude_uV", "PeakLatency_ms", "MeanAmplitude_uV", "AUC_uV_ms") if c in df.columns]
    if not group_cols or not value_cols:
        return pd.DataFrame()
    work = df.copy()
    for col in value_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    grouped = work.groupby(group_cols, dropna=False)[value_cols]
    summary = grouped.agg(["count", "mean", "std", "median", "min", "max"]).reset_index()
    summary.columns = ["_".join([str(x) for x in col if str(x)]) if isinstance(col, tuple) else str(col) for col in summary.columns]
    return summary


def build_master_tables(cfg: Config) -> Dict[str, pd.DataFrame]:
    banner("BUILDING MASTER TABLES")
    single_trial = concatenate_csvs(cfg.SUBJECTS_DIR.rglob("Single_Trial_P300_Metrics.csv"), cfg.MASTER_DIR / "Master_Single_Trial_P300_Metrics.csv")
    single_trial_subject_summary = summarize_single_trial_subjects(single_trial)
    single_trial_subject_summary.to_csv(cfg.MASTER_DIR / "Master_Single_Trial_P300_Subject_Summary.csv", index=False)
    tables = {
        "p300": concatenate_csvs(cfg.SUBJECTS_DIR.rglob("P300_Metrics.csv"), cfg.MASTER_DIR / "Master_P300.csv"),
        "p300_topographies": concatenate_csvs(cfg.SUBJECTS_DIR.rglob("P300_Peak_Topography_Long.csv"), cfg.MASTER_DIR / "Master_P300_Peak_Topography_Long.csv"),
        "block_p300_summary": concatenate_csvs(cfg.SUBJECTS_DIR.rglob("Block_P300_Summary.csv"), cfg.MASTER_DIR / "Master_Block_P300_Summary.csv"),
        "erp_windows": concatenate_csvs(cfg.SUBJECTS_DIR.rglob("ERP_Window_Metrics.csv"), cfg.MASTER_DIR / "Master_ERP_Window_Metrics.csv"),
        "single_trial_p300": single_trial,
        "single_trial_p300_subject_summary": single_trial_subject_summary,
        "behavior_trial": concatenate_csvs(cfg.SUBJECTS_DIR.rglob("Behavior_Trial_Metrics.csv"), cfg.MASTER_DIR / "Master_Behavior_Trial_Metrics.csv"),
        "behavior_summary": concatenate_csvs(cfg.SUBJECTS_DIR.rglob("Behavior_Summary.csv"), cfg.MASTER_DIR / "Master_Behavior_Summary.csv"),
        "epoch_rejection_qc": concatenate_csvs(cfg.SUBJECTS_DIR.rglob("*_Epoch_Rejection_QC.csv"), cfg.MASTER_DIR / "Master_Epoch_Rejection_QC.csv"),
        "erp_waveforms": concatenate_csvs(cfg.SUBJECTS_DIR.rglob("P300_ERP_Waveforms.csv"), cfg.MASTER_DIR / "Master_ERP_Waveforms.csv"),
        "microstate_wide": concatenate_csvs(cfg.SUBJECTS_DIR.rglob("Microstate_Parameters_Wide.csv"), cfg.MASTER_DIR / "Master_Microstate_Parameters_Wide.csv"),
        "microstate_long": concatenate_csvs(cfg.SUBJECTS_DIR.rglob("Microstate_Parameters_Long.csv"), cfg.MASTER_DIR / "Master_Microstate_Parameters_Long.csv"),
        "microstate_model_used": concatenate_csvs(cfg.SUBJECTS_DIR.rglob("Microstate_Model_Used.csv"), cfg.MASTER_DIR / "Master_Microstate_Model_Used.csv"),
        "microstate_distributions": concatenate_csvs(cfg.SUBJECTS_DIR.rglob("Microstate_Distribution_Summaries.csv"), cfg.MASTER_DIR / "Master_Microstate_Distribution_Summaries.csv"),
        "entropy": concatenate_csvs(cfg.SUBJECTS_DIR.rglob("Microstate_Entropy.csv"), cfg.MASTER_DIR / "Master_Microstate_Entropy.csv"),
        "matrices": concatenate_csvs(cfg.SUBJECTS_DIR.rglob("Microstate_Matrices_Long.csv"), cfg.MASTER_DIR / "Master_Microstate_Matrices_Long.csv"),
        "sequence": concatenate_csvs(cfg.SUBJECTS_DIR.rglob("Microstate_Sequence_Metrics.csv"), cfg.MASTER_DIR / "Master_Microstate_Sequence_Metrics.csv"),
        "segments": concatenate_csvs(cfg.SUBJECTS_DIR.rglob("Microstate_Segments.csv"), cfg.MASTER_DIR / "Master_Microstate_Segments.csv"),
        "epoch_state_summary": concatenate_csvs(cfg.SUBJECTS_DIR.rglob("Microstate_Epoch_State_Summary.csv"), cfg.MASTER_DIR / "Master_Microstate_Epoch_State_Summary.csv"),
        "erp_microstate_p300_temporal": concatenate_csvs(cfg.SUBJECTS_DIR.rglob("ERP_Microstate_P300_Temporal_Parameters.csv"), cfg.MASTER_DIR / "Master_ERP_Microstate_P300_Temporal_Parameters.csv"),
        "erp_microstate_evoked_labels": concatenate_csvs(cfg.SUBJECTS_DIR.rglob("ERP_Microstate_Evoked_Label_Times.csv"), cfg.MASTER_DIR / "Master_ERP_Microstate_Evoked_Label_Times.csv"),
        "common_model_training_gev": concatenate_csvs(cfg.COMMON_DIR.rglob("*_Training_GEV_By_Topomap.csv"), cfg.MASTER_DIR / "Master_Common_Model_Training_GEV_By_Topomap.csv"),
        "common_model_cluster_selection": concatenate_csvs(cfg.COMMON_DIR.rglob("*_Cluster_Selection_Scores.csv"), cfg.MASTER_DIR / "Master_Common_Model_Cluster_Selection_Scores.csv"),
        "erp_microstate_fit_source_evokeds": concatenate_csvs(list(cfg.COMMON_DIR.rglob("*_ERP_Microstate_Fit_Source_Evokeds.csv")) + list(cfg.COMMON_DIR.rglob("*_ERP_GFP_Peak_Fit_Source_Evokeds.csv")), cfg.MASTER_DIR / "Master_ERP_Microstate_Fit_Source_Evokeds.csv"),
    }
    subject_output_manifest(cfg, "Microstate_TimeSeries_Long.csv", cfg.MASTER_DIR / "Master_Microstate_TimeSeries_Manifest.csv")
    subject_output_manifest(cfg, "Microstate_Labels.npy", cfg.MASTER_DIR / "Master_Microstate_Label_NPY_Manifest.csv")
    logging.info("Master tables saved to %s", cfg.MASTER_DIR)
    return tables


def safe_count_csv_rows(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            return max(0, sum(1 for _ in handle) - 1)
    except Exception:
        return 0


###############################################################################
# STATISTICS, MIXED-EFFECTS MODELS, EFFECT SIZES, FDR, AND FIGURES
###############################################################################

def p_adjust_bh(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return out
    order = np.argsort(p[ok])
    ranked = p[ok][order]
    n = len(ranked)
    adjusted = np.minimum.accumulate((ranked * n / np.arange(1, n + 1))[::-1])[::-1]
    restored = np.empty(n)
    restored[order] = np.minimum(adjusted, 1.0)
    out[np.where(ok)[0]] = restored
    return out


def hedges_g(young: np.ndarray, older: np.ndarray) -> float:
    young = young[np.isfinite(young)]
    older = older[np.isfinite(older)]
    n1, n2 = len(young), len(older)
    if n1 < 2 or n2 < 2:
        return np.nan
    s1, s2 = np.var(young, ddof=1), np.var(older, ddof=1)
    pooled = math.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2)) if (n1 + n2 - 2) > 0 else np.nan
    if not np.isfinite(pooled) or pooled == 0:
        return np.nan
    correction = 1.0 - (3.0 / (4.0 * (n1 + n2) - 9.0))
    return float(((np.mean(older) - np.mean(young)) / pooled) * correction)


def group_statistics(long_df: pd.DataFrame, group_cols: List[str], table_name: str, cfg: Config) -> pd.DataFrame:
    if long_df.empty or "Group" not in long_df.columns or "Value" not in long_df.columns:
        return pd.DataFrame()
    rows = []
    for keys, part in long_df.groupby(group_cols, dropna=False):
        young = pd.to_numeric(part.loc[part["Group"].astype(str).eq("younger"), "Value"], errors="coerce").dropna().to_numpy()
        older = pd.to_numeric(part.loc[part["Group"].astype(str).eq("older"), "Value"], errors="coerce").dropna().to_numpy()
        if len(young) < cfg.STAT_MIN_PER_GROUP or len(older) < cfg.STAT_MIN_PER_GROUP:
            continue
        t_stat, p_value = scipy.stats.ttest_ind(older, young, equal_var=False, nan_policy="omit")
        key_values = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_cols, key_values))
        row.update({"Table": table_name, "N_Young": len(young), "N_Older": len(older), "Mean_Young": float(np.mean(young)), "Mean_Older": float(np.mean(older)), "SD_Young": float(np.std(young, ddof=1)), "SD_Older": float(np.std(older, ddof=1)), "Older_minus_Young_Hedges_g": hedges_g(young, older), "Welch_t": float(t_stat), "p_uncorrected": float(p_value)})
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_fdr"] = p_adjust_bh(out["p_uncorrected"].to_numpy())
    return out


def mixed_effects(long_df: pd.DataFrame, group_cols: List[str], table_name: str, cfg: Config) -> pd.DataFrame:
    if mixedlm is None or long_df.empty or "Participant" not in long_df.columns or "Group" not in long_df.columns or "Value" not in long_df.columns:
        return pd.DataFrame()
    rows = []
    df = long_df.copy()
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df["GroupCode"] = df["Group"].astype(str).map({"younger": 0.0, "older": 1.0})
    df = df.dropna(subset=["Value", "GroupCode", "Participant"])
    for keys, part in df.groupby(group_cols, dropna=False):
        if part["Participant"].nunique() < 6 or part["GroupCode"].nunique() < 2:
            continue
        try:
            fit = mixedlm("Value ~ GroupCode", part, groups=part["Participant"]).fit(reml=False, method="lbfgs", disp=False)
            key_values = keys if isinstance(keys, tuple) else (keys,)
            row = dict(zip(group_cols, key_values))
            row.update({"Table": table_name, "N": len(part), "N_Subjects": part["Participant"].nunique(), "Beta_Older_vs_Young": float(fit.params.get("GroupCode", np.nan)), "SE": float(fit.bse.get("GroupCode", np.nan)), "p_uncorrected": float(fit.pvalues.get("GroupCode", np.nan)), "Converged": bool(getattr(fit, "converged", True))})
            rows.append(row)
        except Exception:
            continue
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_fdr"] = p_adjust_bh(out["p_uncorrected"].to_numpy())
    return out


def factorial_mixed_effects(long_df: pd.DataFrame, group_cols: Sequence[str], formula: str, table_name: str, cfg: Config) -> pd.DataFrame:
    if mixedlm is None or long_df.empty or "Participant" not in long_df.columns or "Value" not in long_df.columns:
        return pd.DataFrame()
    rows = []
    df = long_df.copy()
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df.dropna(subset=["Value", "Participant"])
    needed = set(group_cols)
    if not needed.issubset(df.columns):
        return pd.DataFrame()
    grouped = df.groupby(list(group_cols), dropna=False) if group_cols else [((), df)]
    for keys, part in grouped:
        if part["Participant"].nunique() < 6:
            continue
        try:
            fit = mixedlm(formula, part, groups=part["Participant"]).fit(reml=False, method="lbfgs", disp=False)
        except Exception as exc:
            logging.debug("Factorial mixed model failed for %s %s: %s", table_name, keys, exc)
            continue
        key_values = keys if isinstance(keys, tuple) else (keys,)
        for term, beta in fit.params.items():
            row = dict(zip(group_cols, key_values))
            row.update({
                "Table": table_name,
                "Term": term,
                "Beta": float(beta),
                "SE": float(fit.bse.get(term, np.nan)),
                "p_uncorrected": float(fit.pvalues.get(term, np.nan)),
                "N": int(len(part)),
                "N_Subjects": int(part["Participant"].nunique()),
                "Formula": formula,
                "Converged": bool(getattr(fit, "converged", True)),
            })
            rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_fdr"] = p_adjust_bh(out["p_uncorrected"].to_numpy())
    return out


def p300_long(df: pd.DataFrame) -> pd.DataFrame:
    value_cols = [c for c in ("PeakAmplitude_uV", "PeakLatency_ms", "MeanAmplitude_uV", "AUC_uV_s") if c in df.columns]
    id_cols = [c for c in list(IDENTIFIER_COLUMNS) + ["Component", "Channel", "Start_ms", "End_ms", "N_Epochs"] if c in df.columns]
    return df.melt(id_vars=id_cols, value_vars=value_cols, var_name="Metric", value_name="Value") if value_cols else pd.DataFrame()


def erp_windows_long(df: pd.DataFrame) -> pd.DataFrame:
    value_cols = [c for c in ("PeakAmplitude_uV", "PeakLatency_ms", "MeanAmplitude_uV", "MedianAmplitude_uV", "AUC_uV_ms", "FractionalAreaLatency_ms", "WindowMin_uV", "WindowMax_uV", "WindowStd_uV") if c in df.columns]
    id_cols = [c for c in list(IDENTIFIER_COLUMNS) + ["Component", "Channel", "Polarity", "Start_ms", "End_ms", "N_Epochs"] if c in df.columns]
    return df.melt(id_vars=id_cols, value_vars=value_cols, var_name="Metric", value_name="Value") if value_cols else pd.DataFrame()


def single_trial_subject_summary_long(df: pd.DataFrame) -> pd.DataFrame:
    value_cols = [c for c in df.columns if c.endswith("_mean") or c.endswith("_median") or c.endswith("_std")]
    id_cols = [c for c in list(IDENTIFIER_COLUMNS) + ["Component", "Channel"] if c in df.columns]
    return df.melt(id_vars=id_cols, value_vars=value_cols, var_name="Metric", value_name="Value") if value_cols else pd.DataFrame()


def behavior_long(df: pd.DataFrame) -> pd.DataFrame:
    value_cols = [c for c in ("N_Targets", "Hits", "Misses", "HitRate", "MeanRT_ms", "MedianRT_ms", "SDRT_ms") if c in df.columns]
    id_cols = [c for c in list(IDENTIFIER_COLUMNS) if c in df.columns]
    return df.melt(id_vars=id_cols, value_vars=value_cols, var_name="Metric", value_name="Value") if value_cols else pd.DataFrame()


def _corr_kind(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return {"N": int(ok.sum()), "Pearson_r": np.nan, "Pearson_p": np.nan, "Spearman_r": np.nan, "Spearman_p": np.nan}
    pearson_r, pearson_p = scipy.stats.pearsonr(x[ok], y[ok])
    spearman_r, spearman_p = scipy.stats.spearmanr(x[ok], y[ok])
    return {"N": int(ok.sum()), "Pearson_r": float(pearson_r), "Pearson_p": float(pearson_p), "Spearman_r": float(spearman_r), "Spearman_p": float(spearman_p)}


def _correlate_pairs(df: pd.DataFrame, x_cols: Sequence[str], y_cols: Sequence[str], group_cols: Sequence[str], table: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    group_cols = [col for col in group_cols if col in df.columns]
    grouped = [((), df)] if not group_cols else df.groupby(group_cols, dropna=False)
    for keys, part in grouped:
        key_values = keys if isinstance(keys, tuple) else (keys,)
        group_dict = dict(zip(group_cols, key_values))
        for x_col in x_cols:
            if x_col not in part.columns:
                continue
            x = pd.to_numeric(part[x_col], errors="coerce").to_numpy(dtype=float)
            for y_col in y_cols:
                if y_col not in part.columns:
                    continue
                y = pd.to_numeric(part[y_col], errors="coerce").to_numpy(dtype=float)
                row = {"Table": table, **group_dict, "X": x_col, "Y": y_col}
                row.update(_corr_kind(x, y))
                rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["Pearson_p_fdr"] = p_adjust_bh(out["Pearson_p"].to_numpy())
        out["Spearman_p_fdr"] = p_adjust_bh(out["Spearman_p"].to_numpy())
    return out


def pivot_metric_table(df: pd.DataFrame, prefix: str, index_cols: Sequence[str], label_cols: Sequence[str], metrics: Sequence[str]) -> pd.DataFrame:
    required = set(index_cols) | set(label_cols) | {"Metric", "Value"}
    if df.empty or not required.issubset(df.columns):
        return pd.DataFrame()
    work = df[df["Metric"].astype(str).isin(metrics)].copy()
    if work.empty:
        return pd.DataFrame()
    work["MetricLabel"] = work[[*label_cols, "Metric"]].astype(str).agg("_".join, axis=1)
    wide = work.pivot_table(index=list(index_cols), columns="MetricLabel", values="Value", aggfunc="mean").reset_index()
    wide.columns = [str(col) if col in index_cols else f"{prefix}_{col}" for col in wide.columns]
    return wide


def run_integration_analyses(cfg: Config) -> None:
    banner("P300 + MICROSTATE INTEGRATION")
    p300 = p300_long(safe_read_csv(cfg.MASTER_DIR / "Master_P300.csv"))
    micro = safe_read_csv(cfg.MASTER_DIR / "Master_Microstate_Parameters_Long.csv")
    behavior = behavior_long(safe_read_csv(cfg.MASTER_DIR / "Master_Behavior_Summary.csv"))
    key = ["Participant", "Group", "Age", "Sex", "Condition", "Window", "Block"]
    p300 = p300[p300["Window"].astype(str).eq("Whole")] if "Window" in p300.columns else p300
    micro = micro[micro["Window"].astype(str).eq("Whole")] if "Window" in micro.columns else micro
    behavior = behavior[behavior["Window"].astype(str).eq("Whole")] if "Window" in behavior.columns else behavior
    p300_wide = pivot_metric_table(p300, "P300", key, ["Component", "Channel"], ["PeakAmplitude_uV", "PeakLatency_ms", "MeanAmplitude_uV"])
    micro_wide = pivot_metric_table(micro, "Microstate", key, ["State"], ["mean_corr", "gev", "timecov", "meandurs", "occurrences", "total_gev"])
    behavior_wide = pivot_metric_table(behavior, "Behavior", key, [], ["HitRate", "MeanRT_ms", "MedianRT_ms"])
    correlation_frames = []
    if not p300_wide.empty and not micro_wide.empty:
        merged = p300_wide.merge(micro_wide, on=key, how="inner")
        merged.to_csv(cfg.INTEGRATION_DIR / "Integration_P300_Microstate_Wide.csv", index=False)
        p300_cols = [col for col in merged.columns if col.startswith("P300_")]
        micro_cols = [col for col in merged.columns if col.startswith("Microstate_")]
        correlation_frames.append(_correlate_pairs(merged, p300_cols, micro_cols, ["Condition"], "P300_vs_Microstate"))
    if not behavior_wide.empty and not p300_wide.empty:
        merged = behavior_wide.merge(p300_wide, on=key, how="inner")
        merged.to_csv(cfg.INTEGRATION_DIR / "Integration_Behavior_P300_Wide.csv", index=False)
        behavior_cols = [col for col in merged.columns if col.startswith("Behavior_")]
        p300_cols = [col for col in merged.columns if col.startswith("P300_")]
        correlation_frames.append(_correlate_pairs(merged, behavior_cols, p300_cols, ["Condition"], "Behavior_vs_P300"))
    if not behavior_wide.empty and not micro_wide.empty:
        merged = behavior_wide.merge(micro_wide, on=key, how="inner")
        merged.to_csv(cfg.INTEGRATION_DIR / "Integration_Behavior_Microstate_Wide.csv", index=False)
        behavior_cols = [col for col in merged.columns if col.startswith("Behavior_")]
        micro_cols = [col for col in merged.columns if col.startswith("Microstate_")]
        correlation_frames.append(_correlate_pairs(merged, behavior_cols, micro_cols, ["Condition"], "Behavior_vs_Microstate"))
    correlations = pd.concat([frame for frame in correlation_frames if not frame.empty], ignore_index=True) if correlation_frames else pd.DataFrame()
    correlations.to_csv(cfg.INTEGRATION_DIR / "Integration_Correlations.csv", index=False)


def run_group_statistics(cfg: Config) -> None:
    banner("GROUP STATISTICS")
    p300 = p300_long(safe_read_csv(cfg.MASTER_DIR / "Master_P300.csv"))
    erp = erp_windows_long(safe_read_csv(cfg.MASTER_DIR / "Master_ERP_Window_Metrics.csv"))
    single_trial_summary = single_trial_subject_summary_long(safe_read_csv(cfg.MASTER_DIR / "Master_Single_Trial_P300_Subject_Summary.csv"))
    micro = safe_read_csv(cfg.MASTER_DIR / "Master_Microstate_Parameters_Long.csv")
    entropy = safe_read_csv(cfg.MASTER_DIR / "Master_Microstate_Entropy.csv")
    matrices = safe_read_csv(cfg.MASTER_DIR / "Master_Microstate_Matrices_Long.csv")
    sequence = safe_read_csv(cfg.MASTER_DIR / "Master_Microstate_Sequence_Metrics.csv")
    behavior = behavior_long(safe_read_csv(cfg.MASTER_DIR / "Master_Behavior_Summary.csv"))
    stat_specs = [
        ("P300", p300, [c for c in ["Condition", "Window", "Block", "Component", "Channel", "Metric"] if c in p300.columns]),
        ("ERP_Window_Metrics", erp, [c for c in ["Condition", "Window", "Block", "Component", "Channel", "Metric"] if c in erp.columns]),
        ("Single_Trial_P300_Subject_Summary", single_trial_summary, [c for c in ["Condition", "Window", "Block", "Component", "Channel", "Metric"] if c in single_trial_summary.columns]),
        ("Behavior_Summary", behavior, [c for c in ["Condition", "Window", "Block", "Metric"] if c in behavior.columns]),
        ("Microstate_Parameters", micro, [c for c in ["Condition", "Window", "Block", "State", "Metric"] if c in micro.columns]),
        ("Microstate_Entropy", entropy, [c for c in ["Condition", "Window", "Block", "Metric"] if c in entropy.columns]),
        ("Microstate_Matrices", matrices, [c for c in ["Condition", "Window", "Block", "Matrix", "Statistic", "From", "To"] if c in matrices.columns]),
        ("Microstate_Sequence_Metrics", sequence, [c for c in ["Condition", "Window", "Block", "Metric"] if c in sequence.columns]),
    ]
    stat_frames, mixed_frames = [], []
    for name, df, cols in stat_specs:
        if df.empty or not cols:
            continue
        stat = group_statistics(df, cols, name, cfg)
        mix = mixed_effects(df, cols, name, cfg)
        stat.to_csv(cfg.STATISTICS_DIR / f"{name}_Young_vs_Older_Welch_Hedges_FDR.csv", index=False)
        mix.to_csv(cfg.STATISTICS_DIR / f"{name}_Mixed_Effects.csv", index=False)
        stat_frames.append(stat)
        mixed_frames.append(mix)
    pd.concat(stat_frames, ignore_index=True).to_csv(cfg.STATISTICS_DIR / "All_Young_vs_Older_Welch_Hedges_FDR.csv", index=False) if stat_frames else pd.DataFrame().to_csv(cfg.STATISTICS_DIR / "All_Young_vs_Older_Welch_Hedges_FDR.csv", index=False)
    pd.concat(mixed_frames, ignore_index=True).to_csv(cfg.STATISTICS_DIR / "All_Mixed_Effects.csv", index=False) if mixed_frames else pd.DataFrame().to_csv(cfg.STATISTICS_DIR / "All_Mixed_Effects.csv", index=False)
    whole_p300 = p300[p300["Window"].astype(str).eq("Whole")] if "Window" in p300.columns else p300
    p300_factorial = factorial_mixed_effects(whole_p300, [c for c in ["Component", "Metric"] if c in whole_p300.columns], "Value ~ C(Group) * C(Condition) * C(Channel)", "P300_Group_x_Condition_x_Channel", cfg)
    p300_factorial.to_csv(cfg.STATISTICS_DIR / "P300_Factorial_Mixed_Effects.csv", index=False)
    whole_micro = micro[micro["Window"].astype(str).eq("Whole")] if "Window" in micro.columns else micro
    whole_micro = whole_micro[whole_micro["Metric"].astype(str).isin(["mean_corr", "gev", "timecov", "meandurs", "occurrences", "total_gev"])] if "Metric" in whole_micro.columns else whole_micro
    micro_factorial = factorial_mixed_effects(whole_micro, [c for c in ["Metric"] if c in whole_micro.columns], "Value ~ C(Group) * C(Condition) * C(State)", "Microstate_Group_x_Condition_x_State", cfg)
    micro_factorial.to_csv(cfg.STATISTICS_DIR / "Microstate_Factorial_Mixed_Effects.csv", index=False)
    whole_behavior = behavior[behavior["Window"].astype(str).eq("Whole")] if "Window" in behavior.columns else behavior
    behavior_factorial = factorial_mixed_effects(whole_behavior, [c for c in ["Metric"] if c in whole_behavior.columns], "Value ~ C(Group) * C(Condition)", "Behavior_Group_x_Condition", cfg)
    behavior_factorial.to_csv(cfg.STATISTICS_DIR / "Behavior_Factorial_Mixed_Effects.csv", index=False)


def make_boxplot(df: pd.DataFrame, value_col: str, title: str, out_file: Path, category_cols: List[str]) -> None:
    if df.empty or value_col not in df.columns:
        return
    plot_df = df.copy()
    plot_df["Category"] = plot_df[category_cols].astype(str).agg(" | ".join, axis=1)
    cats = list(plot_df["Category"].dropna().unique())
    if not cats:
        return
    fig, ax = plt.subplots(figsize=(max(8, len(cats) * 0.65), 5))
    data = [pd.to_numeric(plot_df.loc[(plot_df["Category"] == cat) & (plot_df["Group"].astype(str) == grp), value_col], errors="coerce").dropna().to_numpy() for cat in cats for grp in ("younger", "older")]
    positions = []
    labels = []
    for i, cat in enumerate(cats):
        positions.extend([i * 3 + 1, i * 3 + 2])
        labels.extend([cat + "\nY", cat + "\nO"])
    ax.boxplot(data, positions=positions, widths=0.65, showfliers=False)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_title(title)
    ax.set_ylabel(value_col)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_publication_figures(cfg: Config) -> None:
    banner("PUBLICATION FIGURES")
    p300_file = cfg.MASTER_DIR / "Master_P300.csv"
    erp_file = cfg.MASTER_DIR / "Master_ERP_Window_Metrics.csv"
    micro_file = cfg.MASTER_DIR / "Master_Microstate_Parameters_Long.csv"
    entropy_file = cfg.MASTER_DIR / "Master_Microstate_Entropy.csv"
    sequence_file = cfg.MASTER_DIR / "Master_Microstate_Sequence_Metrics.csv"
    if p300_file.exists():
        p300 = safe_read_csv(p300_file)
        whole = p300[p300["Window"].astype(str).eq("Whole")] if "Window" in p300.columns else p300
        make_boxplot(whole, "PeakAmplitude_uV", "P300 Peak Amplitude by Group", cfg.FIGURES_DIR / "P300_PeakAmplitude_Group_Boxplot.png", ["Condition", "Channel"])
        make_boxplot(whole, "PeakLatency_ms", "P300 Peak Latency by Group", cfg.FIGURES_DIR / "P300_PeakLatency_Group_Boxplot.png", ["Condition", "Channel"])
    if erp_file.exists():
        erp = safe_read_csv(erp_file)
        whole = erp[(erp["Window"].astype(str).eq("Whole")) & (erp["Component"].astype(str).isin(["N100", "P200", "N200", "P300_300_600", "P300_300_700"]))] if {"Window", "Component"}.issubset(erp.columns) else pd.DataFrame()
        make_boxplot(whole, "MeanAmplitude_uV", "ERP Window Mean Amplitude by Group", cfg.FIGURES_DIR / "ERP_Window_MeanAmplitude_Group_Boxplot.png", ["Condition", "Component", "Channel"])
    if micro_file.exists():
        micro = safe_read_csv(micro_file)
        whole = micro[(micro["Window"].astype(str).eq("Whole")) & (micro["Metric"].astype(str).isin(["mean_corr", "gev", "timecov", "meandurs", "occurrences"]))] if {"Window", "Metric"}.issubset(micro.columns) else pd.DataFrame()
        for metric in ["mean_corr", "gev", "timecov", "meandurs", "occurrences"]:
            part = whole[whole["Metric"].eq(metric)].rename(columns={"Value": metric})
            make_boxplot(part, metric, f"Microstate {metric} by Group", cfg.FIGURES_DIR / f"Microstate_{metric}_Group_Boxplot.png", ["Condition", "State"])
    if entropy_file.exists():
        entropy = safe_read_csv(entropy_file)
        whole = entropy[(entropy["Window"].astype(str).eq("Whole")) & (entropy["Metric"].astype(str).eq("Entropy"))].rename(columns={"Value": "Entropy"}) if {"Window", "Metric"}.issubset(entropy.columns) else pd.DataFrame()
        make_boxplot(whole, "Entropy", "Microstate Entropy by Group", cfg.FIGURES_DIR / "Microstate_Entropy_Group_Boxplot.png", ["Condition"])
    if sequence_file.exists():
        sequence = safe_read_csv(sequence_file)
        whole = sequence[(sequence["Window"].astype(str).eq("Whole")) & (sequence["Metric"].astype(str).isin(["switching_rate_hz", "markov_entropy_bits", "lz_complexity_normalized"]))] if {"Window", "Metric"}.issubset(sequence.columns) else pd.DataFrame()
        for metric in ["switching_rate_hz", "markov_entropy_bits", "lz_complexity_normalized"]:
            part = whole[whole["Metric"].eq(metric)].rename(columns={"Value": metric})
            make_boxplot(part, metric, f"Microstate {metric} by Group", cfg.FIGURES_DIR / f"Microstate_{metric}_Group_Boxplot.png", ["Condition"])


###############################################################################
# FINAL REPORT AND COMMAND LINE ENTRYPOINT
###############################################################################

def generate_final_report(cfg: Config, participants: pd.DataFrame) -> None:
    banner("FINAL REPORT")
    dataset_description = read_json(cfg.DATASET_DIR / "dataset_description.json")
    processing_summary = safe_read_csv(cfg.RUN_DIR / "Processing_Summary.csv")
    model_files = sorted(cfg.COMMON_DIR.glob("*/*_Model_Metadata.csv"))
    model_rows = [pd.read_csv(file) for file in model_files]
    models = pd.concat(model_rows, ignore_index=True) if model_rows else pd.DataFrame()
    versions = {"mne": getattr(mne, "__version__", "unknown"), "pycrostates": getattr(pycrostates, "__version__", "unknown") if pycrostates else "unknown", "numpy": np.__version__, "pandas": pd.__version__}
    lines = [
        "# P300 + Microstate Pipeline Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Dataset: {dataset_description.get('Name', cfg.DATASET_DIR.name)}",
        f"Dataset DOI: {dataset_description.get('DatasetDOI', 'n/a')}",
        f"Dataset directory: `{cfg.DATASET_DIR}`",
        f"Output directory: `{cfg.RUN_DIR}`",
        "",
        "## Software Versions",
        "",
    ]
    lines += [f"- {key}: {value}" for key, value in versions.items()]
    lines += ["", "## Participants", "", f"- Total participants in participants.tsv: {len(participants)}"]
    if "group" in participants.columns:
        for group, count in participants["group"].value_counts(dropna=False).items():
            lines.append(f"- {group}: {count}")
    if not processing_summary.empty:
        completed = processing_summary["Completed"].astype(str).str.lower().isin(["true", "1", "yes"]) if "Completed" in processing_summary.columns else pd.Series(dtype=bool)
        lines += ["", "## Processing", "", f"- Completed subjects: {int(completed.sum()) if len(completed) else 'n/a'}", f"- Failed subjects: {int((~completed).sum()) if len(completed) else 'n/a'}"]
    if not models.empty:
        lines += ["", "## Common Microstate Models", ""]
        for _, row in models.iterrows():
            lines.append(f"- {row.get('Condition')}: epochs={row.get('N_Epochs')}, clusters={row.get('N_Clusters')}, GEV={row.get('GEV')}")
    lines += [
        "",
        "## Saved Outputs",
        "",
        f"- Analysis mode: `{cfg.ANALYSIS_MODE}`.",
        f"- Event type override: `{cfg.EVENT_TYPE_FILTER}`; task role override: `{cfg.TASK_ROLE_FILTER}`.",
        f"- Microstate common-model mode: `{cfg.MICROSTATE_MODEL_MODE}`; pooled training conditions: `{','.join(microstate_training_conditions(cfg))}`.",
        f"- Microstate fit source: `{cfg.MICROSTATE_FIT_SOURCE}`; fit window: `{cfg.MICROSTATE_FIT_WINDOW}`.",
        f"- Auto cluster selection: `{cfg.AUTO_N_CLUSTERS}`; cluster range: `{cfg.CLUSTER_RANGE}`; fixed clusters: `{cfg.N_CLUSTERS}`.",
        f"- ERP microstate P300 summary window: `{cfg.ERP_MICROSTATE_P300_WINDOW}`.",
        f"- Block-level analysis by `{cfg.BLOCK_METADATA_COLUMN}`: `{cfg.ENABLE_BLOCK_ANALYSIS}`; stimuli per reconstructed block: `{cfg.STIMULI_PER_EXPERIMENTAL_BLOCK}`.",
        f"- Canonical template alignment: `{cfg.APPLY_TEMPLATE_ALIGNMENT}` using `{cfg.TEMPLATE_NAME}` from `{cfg.TEMPLATE_DIR}`.",
        f"- Template maps interpolated to model montage before matching: `{cfg.TEMPLATE_INTERPOLATE_TO_MODEL}`; minimum positioned channels: `{cfg.TEMPLATE_MIN_INTERPOLATION_CHANNELS}`.",
        "- Effective run settings are saved in `Analysis_Config.json`.",
        "- Subject-wise P300 metrics, block-average P300 summaries, ERP component-window metrics, single-trial P300 summaries, ERP waveforms, evoked FIF files, and epoch metadata.",
        "- Both attended and unattended event classes are processed for FA, FV, and SF where present in the events table.",
        "- Preprocessing, re-referencing, ICA, and epoching are cached once per participant in `_Prepared_Cache/` and reused across later analyses when settings match.",
        f"- Epoch baseline correction: `{cfg.BASELINE}`. `None` means no baseline correction was applied.",
        f"- Bad-channel detection/interpolation: `{cfg.BAD_CHANNEL_DETECTION}` / `{cfg.INTERPOLATE_BAD_CHANNELS}`; ICA max excluded components: `{cfg.ICA_MAX_EXCLUDE}`; target epoch sampling rate: `{cfg.TARGET_EPOCH_SFREQ}`.",
        f"- Duplicate analysis-folder epoch FIF files: `{cfg.SAVE_ANALYSIS_EPOCH_COPIES}`. Whole-condition folders include `Epoch_Cache_Path.txt` pointers to the canonical cached epochs.",
        "- Microstate labels, long time-series manifests, segment/run tables, per-epoch state summaries, parameter tables, entropy, sequence metrics, observed/expected/delta transitions, segmentation figures, auto-k score tables, and ERP/P300 microstate temporal summaries.",
        "- Master tables are in `Master/`.",
        "- Young vs older Welch tests, Hedges' g, FDR-corrected p-values, and mixed-effects model tables are in `Statistics/`.",
        "- Publication-ready summary figures are in `Figures/`.",
    ]
    report = "\n".join(lines) + "\n"
    (cfg.REPORTS_DIR / "Final_Report.md").write_text(report, encoding="utf-8")
    logging.info("Report saved: %s", cfg.REPORTS_DIR / "Final_Report.md")


def launch_gui(defaults: argparse.Namespace) -> Optional[argparse.Namespace]:
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except Exception as exc:
        raise RuntimeError(f"Tkinter GUI is not available in this Python environment: {exc}") from exc

    result: Dict[str, Optional[argparse.Namespace]] = {"args": None}
    root = tk.Tk()
    root.title("P300 + Microstate Pipeline Launcher")
    root.geometry("900x780")

    dataset_var = tk.StringVar(value=str(defaults.dataset))
    output_var = tk.StringVar(value=str(defaults.output))
    template_dir_var = tk.StringVar(value=str(defaults.template_dir or ""))
    scan_summary_var = tk.StringVar(value="Click Scan dataset to list groups, event types, task roles, and template files.")
    default_conditions = tuple(defaults.conditions or Config.MICROSTATE_POOL_CONDITIONS)
    condition_vars = {name: tk.BooleanVar(value=name in default_conditions) for name in CONDITION_MAP}
    subjects_var = tk.StringVar(value=" ".join(defaults.subjects or ["sub-001", "sub-003", "sub-020", "sub-021"]))
    event_types_var = tk.StringVar(value=" ".join(defaults.event_types or []))
    task_roles_var = tk.StringVar(value=" ".join(defaults.task_roles or []))
    stimuli_per_block_var = tk.StringVar(value=str(defaults.stimuli_per_block))
    analysis_mode_var = tk.StringVar(value=defaults.analysis_mode)
    microstate_model_mode_var = tk.StringVar(value=defaults.microstate_model_mode)
    microstate_fit_source_var = tk.StringVar(value=defaults.microstate_fit_source)
    fit_start_var = tk.StringVar(value="-0.20" if defaults.microstate_fit_window is None else str(defaults.microstate_fit_window[0]))
    fit_stop_var = tk.StringVar(value="0.80" if defaults.microstate_fit_window is None else str(defaults.microstate_fit_window[1]))
    baseline_var = tk.BooleanVar(value=True)
    baseline_start_var = tk.StringVar(value=str(defaults.baseline_window[0]))
    baseline_stop_var = tk.StringVar(value=str(defaults.baseline_window[1]))
    p300_start_var = tk.StringVar(value=str(defaults.erp_microstate_p300_window[0]))
    p300_stop_var = tk.StringVar(value=str(defaults.erp_microstate_p300_window[1]))
    auto_clusters_var = tk.BooleanVar(value=True)
    cluster_min_var = tk.StringVar(value=str(defaults.cluster_range[0]))
    cluster_max_var = tk.StringVar(value=str(defaults.cluster_range[1]))
    n_clusters_var = tk.StringVar(value=str(defaults.n_clusters))
    n_init_var = tk.StringVar(value=str(defaults.n_init))
    template_name_var = tk.StringVar(value="MetaMaps_2023_06")
    template_min_corr_var = tk.StringVar(value=str(defaults.template_min_corr))
    template_alignment_var = tk.BooleanVar(value=not defaults.no_template_alignment)
    template_interpolation_var = tk.BooleanVar(value=not defaults.no_template_interpolation)
    template_min_interpolation_channels_var = tk.StringVar(value=str(defaults.template_min_interpolation_channels))
    template_reorder_var = tk.BooleanVar(value=not defaults.no_template_reorder)
    template_polarity_var = tk.BooleanVar(value=not defaults.no_template_polarity_inversion)
    ica_var = tk.BooleanVar(value=not defaults.no_ica)
    ica_max_exclude_var = tk.StringVar(value=str(defaults.ica_max_exclude))
    ica_eog_threshold_var = tk.StringVar(value=str(defaults.ica_eog_threshold))
    bad_detection_var = tk.BooleanVar(value=not defaults.no_bad_channel_detection)
    bad_interpolation_var = tk.BooleanVar(value=not defaults.no_bad_channel_interpolation)
    target_sfreq_var = tk.StringVar(value=str(defaults.target_epoch_sfreq))
    epoch_reject_uv_var = tk.StringVar(value=str(defaults.epoch_reject_uv))
    timestamped_var = tk.BooleanVar(value=True)
    force_prepare_var = tk.BooleanVar(value=bool(defaults.force_prepare))
    cache_reuse_var = tk.BooleanVar(value=not defaults.no_cache_reuse)
    save_analysis_epoch_copies_var = tk.BooleanVar(value=bool(defaults.save_analysis_epoch_copies))
    save_fif_var = tk.BooleanVar(value=not defaults.no_fif)
    save_png_var = tk.BooleanVar(value=not defaults.no_png)
    save_time_series_long_var = tk.BooleanVar(value=not defaults.no_time_series_long)
    save_distributions_var = tk.BooleanVar(value=not defaults.no_parameter_distributions)
    save_transition_heatmaps_var = tk.BooleanVar(value=not defaults.no_transition_heatmaps)
    epoch_window_var = tk.BooleanVar(value=not defaults.no_epoch_window_analysis)
    block_analysis_var = tk.BooleanVar(value=not defaults.no_block_analysis)
    skip_statistics_var = tk.BooleanVar(value=bool(defaults.skip_statistics))

    def browse_directory(var: tk.StringVar) -> None:
        path = filedialog.askdirectory(initialdir=var.get() or str(Path.cwd()))
        if path:
            var.set(path)

    def add_path_row(parent, row: int, label: str, var: tk.StringVar) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=6, pady=5)
        ttk.Entry(parent, textvariable=var, width=84).grid(row=row, column=1, sticky="ew", padx=6, pady=5)
        ttk.Button(parent, text="Browse", command=lambda: browse_directory(var)).grid(row=row, column=2, padx=6, pady=5)

    def add_entry(parent, row: int, label: str, var: tk.StringVar, width: int = 16) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=6, pady=5)
        ttk.Entry(parent, textvariable=var, width=width).grid(row=row, column=1, sticky="w", padx=6, pady=5)

    def use_two_younger_two_older() -> None:
        try:
            participants_file = Path(dataset_var.get()) / "participants.tsv"
            participants = pd.read_csv(participants_file, sep="\t")
            younger = participants.loc[participants["group"].astype(str).str.lower().eq("younger"), "participant_id"].dropna().astype(str).head(2).tolist()
            older = participants.loc[participants["group"].astype(str).str.lower().eq("older"), "participant_id"].dropna().astype(str).head(2).tolist()
            if len(younger) < 2 or len(older) < 2:
                raise ValueError("Could not find at least two younger and two older participants.")
            subjects_var.set(" ".join([*younger, *older]))
        except Exception as exc:
            messagebox.showerror("Subject selection failed", str(exc))

    def scan_dataset_clicked() -> None:
        try:
            scan = scan_dataset_for_gui(Path(dataset_var.get()))
            event_values = scan.get("event_values", {})
            groups = ", ".join(scan.get("groups", [])) or "none"
            templates = ", ".join(scan.get("templates", [])) or "none"
            event_types = ", ".join(event_values.get("event_type", [])) or "none"
            task_roles = ", ".join(event_values.get("task_role", [])) or "none"
            attention = ", ".join(event_values.get("attention_status", [])) or "none"
            scan_summary_var.set(
                f"Participants: {scan.get('participants', 0)} | Groups: {groups}\n"
                f"Event types: {event_types}\n"
                f"Task roles: {task_roles}\n"
                f"Attention: {attention}\n"
                f"Templates in dataset folder: {templates}"
            )
        except Exception as exc:
            messagebox.showerror("Dataset scan failed", str(exc))

    def use_condition_event_defaults() -> None:
        event_types_var.set("")
        task_roles_var.set("")

    def use_oddball_event_defaults() -> None:
        event_types_var.set("high_tone light_bar")
        task_roles_var.set("infrequent_stimulus")

    def apply_onset_preset() -> None:
        onset_conditions = ("FA_Attended", "FA_Unattended", "FV_Attended", "FV_Unattended")
        for name, var in condition_vars.items():
            var.set(name in onset_conditions)
        analysis_mode_var.set("both")
        microstate_model_mode_var.set("pooled_fa_fv")
        microstate_fit_source_var.set("erp_gfp_peaks")
        use_condition_event_defaults()
        fit_start_var.set("-0.20")
        fit_stop_var.set("0.80")
        p300_start_var.set("0.30")
        p300_stop_var.set("0.60")
        auto_clusters_var.set(True)
        cluster_min_var.set("3")
        cluster_max_var.set("8")
        baseline_var.set(True)
        template_name_var.set("MetaMaps_2023_06")
        timestamped_var.set(True)

    def selected_conditions() -> List[str]:
        return [name for name, var in condition_vars.items() if var.get()]

    def as_float(var: tk.StringVar, label: str) -> float:
        try:
            return float(var.get())
        except ValueError as exc:
            raise ValueError(f"{label} must be numeric.") from exc

    def as_int(var: tk.StringVar, label: str) -> int:
        try:
            return int(var.get())
        except ValueError as exc:
            raise ValueError(f"{label} must be an integer.") from exc

    def build_args() -> argparse.Namespace:
        conditions = selected_conditions()
        if not conditions:
            raise ValueError("Select at least one condition.")
        subjects = [item.strip() for item in subjects_var.get().replace(",", " ").split() if item.strip()]
        if not subjects:
            subjects = None
        fit_window = [as_float(fit_start_var, "Microstate fit start"), as_float(fit_stop_var, "Microstate fit stop")]
        if fit_window[0] >= fit_window[1]:
            raise ValueError("Microstate fit start must be less than stop.")
        p300_window = [as_float(p300_start_var, "P300 start"), as_float(p300_stop_var, "P300 stop")]
        if p300_window[0] >= p300_window[1]:
            raise ValueError("P300 start must be less than stop.")
        baseline_window = [as_float(baseline_start_var, "Baseline start"), as_float(baseline_stop_var, "Baseline stop")]
        cluster_range = [as_int(cluster_min_var, "Cluster minimum"), as_int(cluster_max_var, "Cluster maximum")]
        if cluster_range[0] < 2 or cluster_range[1] < cluster_range[0]:
            raise ValueError("Cluster range must satisfy MIN >= 2 and MAX >= MIN.")
        target_sfreq = as_float(target_sfreq_var, "Target epoch sampling frequency")
        return argparse.Namespace(
            gui=False,
            dataset=Path(dataset_var.get()),
            output=Path(output_var.get()),
            subjects=subjects,
            conditions=conditions,
            include_combined_attention=False,
            combined_attention_only=False,
            event_types=split_filter_tokens([event_types_var.get()]),
            task_roles=split_filter_tokens([task_roles_var.get()]),
            stimuli_per_block=as_int(stimuli_per_block_var, "Stimuli per block"),
            analysis_mode=analysis_mode_var.get(),
            microstate_model_mode=microstate_model_mode_var.get(),
            microstate_fit_source=microstate_fit_source_var.get(),
            microstate_fit_window=fit_window,
            baseline_correction=baseline_var.get(),
            baseline_window=baseline_window,
            no_ica=not ica_var.get(),
            ica_max_exclude=as_int(ica_max_exclude_var, "ICA max exclude"),
            ica_eog_threshold=as_float(ica_eog_threshold_var, "ICA EOG threshold"),
            target_epoch_sfreq=target_sfreq,
            epoch_reject_uv=as_float(epoch_reject_uv_var, "Epoch rejection PTP uV"),
            epoch_reject_min_keep_fraction=Config.EPOCH_REJECT_MIN_KEEP_FRACTION,
            no_bad_channel_detection=not bad_detection_var.get(),
            no_bad_channel_interpolation=not bad_interpolation_var.get(),
            prefer_eeglab_set=False,
            no_raw_event_crop=False,
            raw_crop_padding=Config.RAW_CROP_PADDING_S,
            no_memory_safe_epoch_fallback=False,
            no_fif=not save_fif_var.get(),
            no_png=not save_png_var.get(),
            force_prepare=force_prepare_var.get(),
            no_cache_reuse=not cache_reuse_var.get(),
            prepare_only=False,
            save_analysis_epoch_copies=save_analysis_epoch_copies_var.get(),
            timestamped_run_dir=timestamped_var.get(),
            n_clusters=as_int(n_clusters_var, "Fixed clusters"),
            auto_n_clusters=auto_clusters_var.get(),
            cluster_range=cluster_range,
            n_init=as_int(n_init_var, "N init"),
            erp_microstate_p300_window=p300_window,
            template_dir=Path(template_dir_var.get()) if template_dir_var.get().strip() else None,
            template_name=template_name_var.get(),
            template_min_corr=as_float(template_min_corr_var, "Template minimum correlation"),
            no_template_interpolation=not template_interpolation_var.get(),
            template_min_interpolation_channels=as_int(template_min_interpolation_channels_var, "Template minimum interpolation channels"),
            no_template_alignment=not template_alignment_var.get(),
            no_template_reorder=not template_reorder_var.get(),
            no_template_polarity_inversion=not template_polarity_var.get(),
            no_epoch_window_analysis=not epoch_window_var.get(),
            no_block_analysis=not block_analysis_var.get(),
            no_time_series_long=not save_time_series_long_var.get(),
            no_parameter_distributions=not save_distributions_var.get(),
            no_transition_heatmaps=not save_transition_heatmaps_var.get(),
            skip_statistics=skip_statistics_var.get(),
        )

    def on_run() -> None:
        try:
            result["args"] = build_args()
        except Exception as exc:
            messagebox.showerror("Invalid settings", str(exc))
            return
        root.destroy()

    def on_cancel() -> None:
        result["args"] = None
        root.destroy()

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True, padx=10, pady=10)
    input_tab = ttk.Frame(notebook, padding=10)
    analysis_tab = ttk.Frame(notebook, padding=10)
    advanced_tab = ttk.Frame(notebook, padding=10)
    notebook.add(input_tab, text="Input")
    notebook.add(analysis_tab, text="Analysis")
    notebook.add(advanced_tab, text="Advanced")

    input_tab.columnconfigure(1, weight=1)
    add_path_row(input_tab, 0, "Dataset", dataset_var)
    add_path_row(input_tab, 1, "Output", output_var)
    add_path_row(input_tab, 2, "Template dir", template_dir_var)
    ttk.Button(input_tab, text="Scan dataset", command=scan_dataset_clicked).grid(row=3, column=0, sticky="w", padx=6, pady=5)
    ttk.Label(input_tab, textvariable=scan_summary_var, wraplength=760, justify="left").grid(row=3, column=1, columnspan=2, sticky="ew", padx=6, pady=5)
    ttk.Label(input_tab, text="Subjects").grid(row=4, column=0, sticky="w", padx=6, pady=5)
    ttk.Entry(input_tab, textvariable=subjects_var, width=84).grid(row=4, column=1, sticky="ew", padx=6, pady=5)
    ttk.Button(input_tab, text="2 younger + 2 older", command=use_two_younger_two_older).grid(row=4, column=2, padx=6, pady=5)
    condition_frame = ttk.LabelFrame(input_tab, text="Conditions")
    condition_frame.grid(row=5, column=0, columnspan=3, sticky="ew", padx=6, pady=10)
    for idx, name in enumerate(CONDITION_MAP):
        ttk.Checkbutton(condition_frame, text=name, variable=condition_vars[name]).grid(row=idx // 3, column=idx % 3, sticky="w", padx=8, pady=3)
    ttk.Button(input_tab, text="Apply onset FA/FV preset", command=apply_onset_preset).grid(row=6, column=1, sticky="e", padx=6, pady=8)

    for col in range(4):
        analysis_tab.columnconfigure(col, weight=1)
    ttk.Label(analysis_tab, text="Analysis mode").grid(row=0, column=0, sticky="w", padx=6, pady=5)
    ttk.Combobox(analysis_tab, textvariable=analysis_mode_var, values=["both", "erp", "microstate"], width=20, state="readonly").grid(row=0, column=1, sticky="w", padx=6, pady=5)
    ttk.Label(analysis_tab, text="Model mode").grid(row=1, column=0, sticky="w", padx=6, pady=5)
    ttk.Combobox(analysis_tab, textvariable=microstate_model_mode_var, values=["pooled_fa_fv", "condition"], width=20, state="readonly").grid(row=1, column=1, sticky="w", padx=6, pady=5)
    ttk.Label(analysis_tab, text="Fit source").grid(row=2, column=0, sticky="w", padx=6, pady=5)
    ttk.Combobox(analysis_tab, textvariable=microstate_fit_source_var, values=["erp_gfp_peaks", "erp_grand_average", "single_trial_gfp"], width=20, state="readonly").grid(row=2, column=1, sticky="w", padx=6, pady=5)
    ttk.Label(analysis_tab, text="Fit window start/stop").grid(row=3, column=0, sticky="w", padx=6, pady=5)
    ttk.Entry(analysis_tab, textvariable=fit_start_var, width=10).grid(row=3, column=1, sticky="w", padx=6, pady=5)
    ttk.Entry(analysis_tab, textvariable=fit_stop_var, width=10).grid(row=3, column=2, sticky="w", padx=6, pady=5)
    ttk.Label(analysis_tab, text="P300 window start/stop").grid(row=4, column=0, sticky="w", padx=6, pady=5)
    ttk.Entry(analysis_tab, textvariable=p300_start_var, width=10).grid(row=4, column=1, sticky="w", padx=6, pady=5)
    ttk.Entry(analysis_tab, textvariable=p300_stop_var, width=10).grid(row=4, column=2, sticky="w", padx=6, pady=5)
    ttk.Checkbutton(analysis_tab, text="Baseline correction", variable=baseline_var).grid(row=5, column=0, sticky="w", padx=6, pady=5)
    ttk.Entry(analysis_tab, textvariable=baseline_start_var, width=10).grid(row=5, column=1, sticky="w", padx=6, pady=5)
    ttk.Entry(analysis_tab, textvariable=baseline_stop_var, width=10).grid(row=5, column=2, sticky="w", padx=6, pady=5)
    ttk.Checkbutton(analysis_tab, text="Auto clusters", variable=auto_clusters_var).grid(row=6, column=0, sticky="w", padx=6, pady=5)
    ttk.Entry(analysis_tab, textvariable=cluster_min_var, width=10).grid(row=6, column=1, sticky="w", padx=6, pady=5)
    ttk.Entry(analysis_tab, textvariable=cluster_max_var, width=10).grid(row=6, column=2, sticky="w", padx=6, pady=5)
    add_entry(analysis_tab, 7, "Fixed clusters", n_clusters_var)
    add_entry(analysis_tab, 8, "N init", n_init_var)
    ttk.Label(analysis_tab, text="Template").grid(row=9, column=0, sticky="w", padx=6, pady=5)
    ttk.Combobox(analysis_tab, textvariable=template_name_var, values=["MetaMaps_2023_06", "Koenig2002", "Custo2017"], width=22).grid(row=9, column=1, sticky="w", padx=6, pady=5)
    add_entry(analysis_tab, 10, "Template min corr", template_min_corr_var)
    ttk.Label(analysis_tab, text="Stimulus/event types").grid(row=11, column=0, sticky="w", padx=6, pady=5)
    ttk.Entry(analysis_tab, textvariable=event_types_var, width=30).grid(row=11, column=1, columnspan=2, sticky="ew", padx=6, pady=5)
    ttk.Label(analysis_tab, text="Task roles").grid(row=12, column=0, sticky="w", padx=6, pady=5)
    ttk.Entry(analysis_tab, textvariable=task_roles_var, width=30).grid(row=12, column=1, columnspan=2, sticky="ew", padx=6, pady=5)
    add_entry(analysis_tab, 13, "Stimuli per block", stimuli_per_block_var)
    ttk.Button(analysis_tab, text="Use condition defaults", command=use_condition_event_defaults).grid(row=14, column=1, sticky="w", padx=6, pady=5)
    ttk.Button(analysis_tab, text="Oddball only", command=use_oddball_event_defaults).grid(row=14, column=2, sticky="w", padx=6, pady=5)

    ttk.Checkbutton(advanced_tab, text="Use ICA", variable=ica_var).grid(row=0, column=0, sticky="w", padx=6, pady=5)
    add_entry(advanced_tab, 1, "ICA max exclude", ica_max_exclude_var)
    add_entry(advanced_tab, 2, "ICA EOG threshold", ica_eog_threshold_var)
    ttk.Checkbutton(advanced_tab, text="Bad-channel detection", variable=bad_detection_var).grid(row=3, column=0, sticky="w", padx=6, pady=5)
    ttk.Checkbutton(advanced_tab, text="Interpolate bad channels", variable=bad_interpolation_var).grid(row=4, column=0, sticky="w", padx=6, pady=5)
    add_entry(advanced_tab, 5, "Target epoch sfreq", target_sfreq_var)
    add_entry(advanced_tab, 6, "Epoch reject PTP uV", epoch_reject_uv_var)
    ttk.Checkbutton(advanced_tab, text="Timestamped run directory", variable=timestamped_var).grid(row=7, column=0, sticky="w", padx=6, pady=5)
    ttk.Checkbutton(advanced_tab, text="Force prepare cache", variable=force_prepare_var).grid(row=8, column=0, sticky="w", padx=6, pady=5)
    ttk.Checkbutton(advanced_tab, text="Reuse prepared cache", variable=cache_reuse_var).grid(row=9, column=0, sticky="w", padx=6, pady=5)
    ttk.Checkbutton(advanced_tab, text="Save analysis epoch copies", variable=save_analysis_epoch_copies_var).grid(row=10, column=0, sticky="w", padx=6, pady=5)
    ttk.Checkbutton(advanced_tab, text="Save FIF outputs", variable=save_fif_var).grid(row=11, column=0, sticky="w", padx=6, pady=5)
    ttk.Checkbutton(advanced_tab, text="Save PNG figures", variable=save_png_var).grid(row=12, column=0, sticky="w", padx=6, pady=5)
    ttk.Checkbutton(advanced_tab, text="Save long time series", variable=save_time_series_long_var).grid(row=13, column=0, sticky="w", padx=6, pady=5)
    ttk.Checkbutton(advanced_tab, text="Save parameter distributions", variable=save_distributions_var).grid(row=14, column=0, sticky="w", padx=6, pady=5)
    ttk.Checkbutton(advanced_tab, text="Save transition heatmaps", variable=save_transition_heatmaps_var).grid(row=15, column=0, sticky="w", padx=6, pady=5)
    ttk.Checkbutton(advanced_tab, text="Epoch-window microstate analysis", variable=epoch_window_var).grid(row=16, column=0, sticky="w", padx=6, pady=5)
    ttk.Checkbutton(advanced_tab, text="Block-level ERP/P300 and microstate analysis", variable=block_analysis_var).grid(row=17, column=0, sticky="w", padx=6, pady=5)
    ttk.Checkbutton(advanced_tab, text="Template alignment", variable=template_alignment_var).grid(row=18, column=0, sticky="w", padx=6, pady=5)
    ttk.Checkbutton(advanced_tab, text="Interpolate template to model montage", variable=template_interpolation_var).grid(row=19, column=0, sticky="w", padx=6, pady=5)
    add_entry(advanced_tab, 20, "Min interpolation channels", template_min_interpolation_channels_var)
    ttk.Checkbutton(advanced_tab, text="Template reorder", variable=template_reorder_var).grid(row=21, column=0, sticky="w", padx=6, pady=5)
    ttk.Checkbutton(advanced_tab, text="Template polarity inversion", variable=template_polarity_var).grid(row=22, column=0, sticky="w", padx=6, pady=5)
    ttk.Checkbutton(advanced_tab, text="Skip group statistics", variable=skip_statistics_var).grid(row=23, column=0, sticky="w", padx=6, pady=5)

    button_bar = ttk.Frame(root)
    button_bar.pack(fill="x", padx=10, pady=(0, 10))
    ttk.Button(button_bar, text="Run pipeline", command=on_run).pack(side="right", padx=6)
    ttk.Button(button_bar, text="Cancel", command=on_cancel).pack(side="right", padx=6)
    root.protocol("WM_DELETE_WINDOW", on_cancel)
    root.mainloop()
    return result["args"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full P300 + pycrostates microstate pipeline for OpenNeuro ds002893.")
    parser.add_argument("--gui", action="store_true", help="Open a GUI launcher to choose subjects, conditions, onset windows, and analysis parameters.")
    parser.add_argument("--dataset", type=Path, default=Config.DATASET_DIR, help="Path to ds002893 BIDS root.")
    parser.add_argument("--output", type=Path, default=Config.OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--subjects", nargs="*", default=None, help="Optional participant IDs, for example sub-001 sub-002.")
    parser.add_argument("--conditions", nargs="*", choices=list(CONDITION_MAP.keys()), default=None, help="Optional condition subset.")
    parser.add_argument("--include-combined-attention", action="store_true", help="Also run FA_All, FV_All, and SF_All combined attended+unattended conditions.")
    parser.add_argument("--combined-attention-only", action="store_true", help="Run only FA_All, FV_All, and SF_All combined attended+unattended conditions.")
    parser.add_argument("--event-types", nargs="*", default=None, help="Optional event_type override for epoching, for example high_tone light_bar. Omit to use each condition's default oddball/infrequent event types.")
    parser.add_argument("--task-roles", nargs="*", default=None, help="Optional task_role override for epoching, for example infrequent_stimulus or frequent_stimulus. Omit to use each condition's default task_role.")
    parser.add_argument("--stimuli-per-block", type=int, default=Config.STIMULI_PER_EXPERIMENTAL_BLOCK, help="Number of stimulus rows used to reconstruct experimental blocks for block-average P300 summaries.")
    parser.add_argument("--analysis-mode", choices=["both", "erp", "microstate"], default=Config.ANALYSIS_MODE, help="Run ERP only, microstate only, or both.")
    parser.add_argument("--microstate-model-mode", choices=["pooled_fa_fv", "condition"], default=Config.MICROSTATE_MODEL_MODE, help="Use one pooled attended/unattended model from the selected training conditions, or fit separate condition-wise models.")
    parser.add_argument("--microstate-fit-source", choices=["erp_gfp_peaks", "erp_grand_average", "single_trial_gfp"], default=Config.MICROSTATE_FIT_SOURCE, help="Fit microstate maps from ERP GFP peaks, concatenated ERP grand-average maps, or single-trial GFP peaks.")
    parser.add_argument("--microstate-fit-window", nargs=2, type=float, metavar=("START", "END"), default=Config.MICROSTATE_FIT_WINDOW, help="Crop epochs to this time window before GFP peak extraction and common-model fitting; backfitting still uses each requested analysis window.")
    parser.add_argument("--baseline-correction", dest="baseline_correction", action="store_true", default=True, help="Apply baseline correction to epochs. This is the default.")
    parser.add_argument("--no-baseline-correction", dest="baseline_correction", action="store_false", help="Disable epoch baseline correction.")
    parser.add_argument("--baseline-window", nargs=2, type=float, metavar=("START", "END"), default=Config.BASELINE or (-0.20, 0.0), help="Baseline window in seconds when baseline correction is enabled.")
    parser.add_argument("--no-ica", action="store_true", help="Disable ICA artifact correction.")
    parser.add_argument("--ica-max-exclude", type=int, default=Config.ICA_MAX_EXCLUDE, help="Maximum ICA components allowed to be removed per subject.")
    parser.add_argument("--ica-eog-threshold", type=float, default=Config.ICA_EOG_THRESHOLD, help="EOG threshold passed to MNE ICA.find_bads_eog.")
    parser.add_argument("--target-epoch-sfreq", type=float, default=Config.TARGET_EPOCH_SFREQ, help="Resample epochs to this frequency after epoching; use 0 to keep native rates.")
    parser.add_argument("--epoch-reject-uv", type=float, default=Config.EPOCH_REJECT_PEAK_TO_PEAK_UV, help="Reject epochs exceeding this EEG peak-to-peak amplitude in microvolts; use 0 to disable epoch rejection.")
    parser.add_argument("--epoch-reject-min-keep-fraction", type=float, default=Config.EPOCH_REJECT_MIN_KEEP_FRACTION, help="Relax epoch PTP rejection if fewer than this fraction of requested epochs would remain.")
    parser.add_argument("--no-bad-channel-detection", action="store_true", help="Disable automatic EEG bad-channel detection.")
    parser.add_argument("--no-bad-channel-interpolation", action="store_true", help="Mark detected bad EEG channels but do not interpolate them.")
    parser.add_argument("--prefer-eeglab-set", action="store_true", help="Try the original EEGLAB .set file before the MNE desc-preproc_raw.fif fallback. Default prefers FIF to avoid MATLAB v7.3/HDF5 memory errors.")
    parser.add_argument("--no-raw-event-crop", action="store_true", help="Disable memory-saving crop of raw data to the selected event span before preload.")
    parser.add_argument("--raw-crop-padding", type=float, default=Config.RAW_CROP_PADDING_S, help="Seconds of padding added before the first and after the last selected event when raw event cropping is enabled.")
    parser.add_argument("--no-memory-safe-epoch-fallback", action="store_true", help="Disable epoch-level fallback when continuous raw preprocessing fails with MemoryError.")
    parser.add_argument("--no-fif", action="store_true", help="Do not save analysis FIF outputs; the prepared epoch cache is still saved.")
    parser.add_argument("--no-png", action="store_true", help="Do not save PNG figures.")
    parser.add_argument("--force-prepare", action="store_true", help="Ignore existing prepared epochs and rebuild preprocessing/epoch cache.")
    parser.add_argument("--no-cache-reuse", action="store_true", help="Do not reuse existing prepared epochs during this run.")
    parser.add_argument("--prepare-only", action="store_true", help="Create or validate the prepared epoch cache, then stop before ERP/microstate analysis.")
    parser.add_argument("--save-analysis-epoch-copies", action="store_true", help="Also save duplicate Epochs-epo.fif files inside each analysis output folder.")
    parser.add_argument("--timestamped-run-dir", action="store_true", help="Write outputs into Results/Run_YYYYMMDD_HHMMSS.")
    parser.add_argument("--n-clusters", type=int, default=Config.N_CLUSTERS, help="Number of microstate clusters.")
    parser.add_argument("--auto-n-clusters", dest="auto_n_clusters", action="store_true", default=Config.AUTO_N_CLUSTERS, help="Fit every cluster count in --cluster-range and select one using GEV plus pycrostates cluster-quality scores. This is the default.")
    parser.add_argument("--no-auto-n-clusters", dest="auto_n_clusters", action="store_false", help="Disable score-based cluster selection and use --n-clusters.")
    parser.add_argument("--cluster-range", nargs=2, type=int, metavar=("MIN", "MAX"), default=Config.CLUSTER_RANGE, help="Inclusive cluster-count range used when --auto-n-clusters is set.")
    parser.add_argument("--n-init", type=int, default=Config.N_INIT, help="Number of ModKMeans initializations.")
    parser.add_argument("--erp-microstate-p300-window", nargs=2, type=float, metavar=("START", "END"), default=Config.ERP_MICROSTATE_P300_WINDOW, help="P300 window in seconds used for ERP microstate onset/offset/duration/AUC/center-of-gravity summaries.")
    parser.add_argument("--template-dir", type=Path, default=None, help="Directory containing canonical template .set files.")
    parser.add_argument("--template-name", default=Config.TEMPLATE_NAME, help="Template name or filename. Built-ins: Custo2017, Koenig2002, MetaMaps_2023_06.")
    parser.add_argument("--template-min-corr", type=float, default=Config.TEMPLATE_MIN_CORR, help="Minimum absolute spatial correlation used to flag low-confidence template assignments.")
    parser.add_argument("--no-template-interpolation", action="store_true", help="Disable interpolation of template maps onto the fitted model montage before template matching.")
    parser.add_argument("--template-min-interpolation-channels", type=int, default=Config.TEMPLATE_MIN_INTERPOLATION_CHANNELS, help="Minimum number of positioned source/target channels required for template-to-model montage interpolation.")
    parser.add_argument("--no-template-alignment", action="store_true", help="Rename clusters A/B/C/D without template matching.")
    parser.add_argument("--no-template-reorder", action="store_true", help="Do not reorder fitted clusters into template order.")
    parser.add_argument("--no-template-polarity-inversion", action="store_true", help="Do not invert fitted map polarity to match template signs for visualization.")
    parser.add_argument("--no-epoch-window-analysis", action="store_true", help="Disable microstate analysis of configured epoch subwindows.")
    parser.add_argument("--no-block-analysis", action="store_true", help="Disable per-sub_block block-level ERP/P300 and microstate outputs.")
    parser.add_argument("--no-time-series-long", action="store_true", help="Do not save per-sample Microstate_TimeSeries_Long.csv files.")
    parser.add_argument("--no-parameter-distributions", action="store_true", help="Do not save pycrostates distribution arrays/summaries.")
    parser.add_argument("--no-transition-heatmaps", action="store_true", help="Do not save PNG heatmaps for transition matrices.")
    parser.add_argument("--skip-statistics", action="store_true", help="Run analysis but skip group statistics and figures.")
    return parser.parse_args()


def run_pipeline_from_args(args: argparse.Namespace) -> None:
    cfg = Config(DATASET_DIR=args.dataset, OUTPUT_DIR=args.output, USE_TIMESTAMPED_RUN_DIR=args.timestamped_run_dir, TEMPLATE_DIR=args.template_dir)
    if args.conditions:
        cfg.CONDITIONS = tuple(args.conditions)
    if args.combined_attention_only:
        cfg.CONDITIONS = ("FA_All", "FV_All", "SF_All")
    elif args.include_combined_attention:
        cfg.CONDITIONS = tuple(dict.fromkeys((*cfg.CONDITIONS, "FA_All", "FV_All", "SF_All")))
    cfg.ANALYSIS_MODE = args.analysis_mode
    cfg.EVENT_TYPE_FILTER = split_filter_tokens(args.event_types)
    cfg.TASK_ROLE_FILTER = split_filter_tokens(args.task_roles)
    cfg.STIMULI_PER_EXPERIMENTAL_BLOCK = int(args.stimuli_per_block)
    if cfg.STIMULI_PER_EXPERIMENTAL_BLOCK < 1:
        raise ValueError("--stimuli-per-block must be at least 1.")
    cfg.MICROSTATE_MODEL_MODE = args.microstate_model_mode
    cfg.MICROSTATE_FIT_SOURCE = args.microstate_fit_source
    cfg.MICROSTATE_FIT_WINDOW = tuple(args.microstate_fit_window) if args.microstate_fit_window is not None else None
    cfg.BASELINE = tuple(args.baseline_window) if args.baseline_correction else None
    cfg.N_CLUSTERS = args.n_clusters
    cfg.AUTO_N_CLUSTERS = bool(args.auto_n_clusters)
    cfg.CLUSTER_RANGE = tuple(args.cluster_range)
    if cfg.N_CLUSTERS < 2:
        raise ValueError("--n-clusters must be at least 2.")
    if cfg.CLUSTER_RANGE[0] < 2 or cfg.CLUSTER_RANGE[1] < cfg.CLUSTER_RANGE[0]:
        raise ValueError("--cluster-range must be two integers with MIN >= 2 and MAX >= MIN.")
    cfg.N_INIT = args.n_init
    cfg.ERP_MICROSTATE_P300_WINDOW = tuple(args.erp_microstate_p300_window)
    cfg.TEMPLATE_NAME = args.template_name
    cfg.TEMPLATE_MIN_CORR = args.template_min_corr
    cfg.TEMPLATE_INTERPOLATE_TO_MODEL = not args.no_template_interpolation
    cfg.TEMPLATE_MIN_INTERPOLATION_CHANNELS = args.template_min_interpolation_channels
    if cfg.TEMPLATE_MIN_INTERPOLATION_CHANNELS < 3:
        raise ValueError("--template-min-interpolation-channels must be at least 3.")
    if args.no_ica:
        cfg.ICA_ENABLED = False
    cfg.ICA_MAX_EXCLUDE = args.ica_max_exclude
    cfg.ICA_EOG_THRESHOLD = args.ica_eog_threshold
    cfg.TARGET_EPOCH_SFREQ = None if args.target_epoch_sfreq == 0 else args.target_epoch_sfreq
    cfg.EPOCH_REJECT_PEAK_TO_PEAK_UV = None if args.epoch_reject_uv == 0 else float(args.epoch_reject_uv)
    cfg.EPOCH_REJECT_MIN_KEEP_FRACTION = float(args.epoch_reject_min_keep_fraction)
    if args.no_bad_channel_detection:
        cfg.BAD_CHANNEL_DETECTION = False
    if args.no_bad_channel_interpolation:
        cfg.INTERPOLATE_BAD_CHANNELS = False
    if args.prefer_eeglab_set:
        cfg.PREFER_PREPROCESSED_RAW_FIF = False
    if args.no_raw_event_crop:
        cfg.CROP_RAW_TO_SELECTED_EVENTS = False
    cfg.RAW_CROP_PADDING_S = float(args.raw_crop_padding)
    if args.no_memory_safe_epoch_fallback:
        cfg.MEMORY_SAFE_EPOCH_FALLBACK = False
    if args.no_fif:
        cfg.SAVE_FIF = False
    if args.save_analysis_epoch_copies:
        cfg.SAVE_ANALYSIS_EPOCH_COPIES = True
    cfg.FORCE_PREPARE = args.force_prepare
    cfg.REUSE_PREPARED_CACHE = not args.no_cache_reuse
    cfg.PREPARE_ONLY = args.prepare_only
    if args.no_png:
        cfg.SAVE_PNG = False
    if args.no_template_alignment:
        cfg.APPLY_TEMPLATE_ALIGNMENT = False
    if args.no_template_reorder:
        cfg.REORDER_TO_TEMPLATE = False
    if args.no_template_polarity_inversion:
        cfg.INVERT_TO_TEMPLATE_POLARITY = False
    if args.no_epoch_window_analysis:
        cfg.ENABLE_EPOCH_WINDOW_ANALYSIS = False
    if args.no_block_analysis:
        cfg.ENABLE_BLOCK_ANALYSIS = False
    if args.no_time_series_long:
        cfg.SAVE_TIME_SERIES_LONG = False
    if args.no_parameter_distributions:
        cfg.SAVE_PARAMETER_DISTRIBUTIONS = False
    if args.no_transition_heatmaps:
        cfg.SAVE_TRANSITION_HEATMAPS = False
    setup_logging(cfg)
    banner("FULL P300 + MICROSTATE PIPELINE")
    participants = load_participants(cfg)
    if args.subjects:
        participants = participants[participants["participant_id"].isin(args.subjects)].reset_index(drop=True)
        if participants.empty:
            raise RuntimeError("No requested subjects were found in participants.tsv")
    dataset = BIDSDataset(cfg.DATASET_DIR)
    if args.skip_statistics:
        original_run_group_statistics = globals()["run_group_statistics"]
        original_make_publication_figures = globals()["make_publication_figures"]
        globals()["run_group_statistics"] = lambda _cfg: logging.info("Skipping group statistics by request.")
        globals()["make_publication_figures"] = lambda _cfg: logging.info("Skipping publication figures by request.")
        try:
            process_dataset(dataset, participants, cfg, baseline=True)
        finally:
            globals()["run_group_statistics"] = original_run_group_statistics
            globals()["make_publication_figures"] = original_make_publication_figures
    else:
        process_dataset(dataset, participants, cfg, baseline=True)
    banner("PIPELINE COMPLETE")
    logging.info("Results saved in: %s", cfg.RUN_DIR)


def main() -> None:
    args = parse_args()
    if args.gui:
        gui_args = launch_gui(args)
        if gui_args is None:
            print("Pipeline GUI cancelled.")
            return
        args = gui_args
    run_pipeline_from_args(args)


if __name__ == "__main__":
    main()
