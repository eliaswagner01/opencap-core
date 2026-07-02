"""
Compare available LabValidation generic and modified model results.

The script discovers staged lab_subject* sessions in opencap-processing/Data,
loads every available OpenSimAD optimaltrajectories.npy file, normalizes
selected waveforms to 101 points, computes kinematic and activation metrics,
and writes CSV/plot/report outputs. Missing or non-converged trials are
skipped and recorded in result_status.csv.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import numpy as np


REPO_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = REPO_DIR.parent
PROCESSING_DIR = WORKSPACE_DIR / "opencap-processing"

MODEL_CASES = ("generic", "modified")
TRIAL_KINDS = ("walking", "squats1")
KINEMATIC_VARIABLES = [
    "hip_flexion_l",
    "hip_adduction_l",
    "hip_rotation_l",
    "knee_angle_l",
    "hip_flexion_r",
    "hip_adduction_r",
    "hip_rotation_r",
    "knee_angle_r",
]
MUSCLE_GROUPS = {
    "gluteus": [
        "glmax1", "glmax2", "glmax3", "glmed1", "glmed2", "glmed3",
    ],
    "vasti": ["vaslat", "vasmed", "vasint"],
    "hamstrings": ["bflh", "bfsh", "semimem", "semiten"],
}
NORM_N = 101


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare available LabValidation OpenSimAD results."
    )
    parser.add_argument(
        "--processing-data",
        type=Path,
        default=PROCESSING_DIR / "Data",
        help="OpenSimAD data root containing staged lab_subject* sessions.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSING_DIR / "Data" / "LabValidationPilotComparison",
        help="Folder for comparison CSV, plots, and report.",
    )
    parser.add_argument(
        "--trial",
        choices=["walking", "walking1", "squats1", "both"],
        default="both",
        help=(
            "Trial(s) to compare. walking/walking1 uses walking1 when present "
            "and otherwise uses the available non-treadmill walking trial."
        ),
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=["all"],
        help=(
            "Subjects to compare, for example: --subjects subject2 subject3. "
            "Use 'all' to discover every staged lab_subject* session."
        ),
    )
    parser.add_argument(
        "--require-paired",
        action="store_true",
        help=(
            "Only compute metrics for subject/trial combinations where both "
            "generic and modified optimal trajectories are available."
        ),
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Write CSV/report outputs without waveform plots.",
    )
    return parser.parse_args()


def selected_trials(trial_arg: str) -> list[str]:
    if trial_arg == "both":
        return list(TRIAL_KINDS)
    if trial_arg == "walking1":
        return ["walking"]
    return [trial_arg]


def subject_sort_key(subject: str) -> tuple[str, int | str]:
    prefix = "".join(ch for ch in subject if not ch.isdigit())
    suffix = "".join(ch for ch in subject if ch.isdigit())
    return (prefix, int(suffix) if suffix else subject)


def staged_session_name(subject: str, model_case: str) -> str:
    return f"lab_{subject}_{model_case}"


def parse_staged_session_name(name: str) -> tuple[str, str] | None:
    for model_case in MODEL_CASES:
        suffix = f"_{model_case}"
        if name.startswith("lab_subject") and name.endswith(suffix):
            return name[4:-len(suffix)], model_case
    return None


def discover_subjects(processing_data: Path, subject_args: list[str]) -> list[str]:
    if len(subject_args) == 1 and subject_args[0].lower() == "all":
        subjects = set()
        for path in processing_data.glob("lab_subject*_*"):
            if not path.is_dir():
                continue
            parsed = parse_staged_session_name(path.name)
            if parsed is not None:
                subjects.add(parsed[0])
        if not subjects:
            raise FileNotFoundError(
                f"No staged lab_subject* sessions found in {processing_data}"
            )
        return sorted(subjects, key=subject_sort_key)

    subjects = sorted(dict.fromkeys(subject_args), key=subject_sort_key)
    missing = []
    for subject in subjects:
        if not any(
            (processing_data / staged_session_name(subject, case)).exists()
            for case in MODEL_CASES
        ):
            missing.append(subject)
    if missing:
        raise FileNotFoundError(
            "No staged generic/modified session found for: " +
            ", ".join(missing)
        )
    return subjects


def normalize_1d(values: np.ndarray, n_points: int = NORM_N) -> np.ndarray:
    y = np.asarray(values, dtype=float).reshape(-1)
    if y.size == 0:
        return np.full(n_points, np.nan)
    if y.size == 1:
        return np.full(n_points, y[0])
    x_old = np.linspace(0.0, 100.0, y.size)
    x_new = np.linspace(0.0, 100.0, n_points)
    finite = np.isfinite(y)
    if finite.sum() < 2:
        return np.full(n_points, np.nan)
    return np.interp(x_new, x_old[finite], y[finite])


def normalize_rows(values: np.ndarray, n_points: int = NORM_N) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return np.vstack([normalize_1d(row, n_points) for row in arr])


def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return math.nan
    aa = a[mask]
    bb = b[mask]
    if np.std(aa) == 0 or np.std(bb) == 0:
        return math.nan
    return float(np.corrcoef(aa, bb)[0, 1])


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if not np.any(mask):
        return math.nan
    return float(np.sqrt(np.mean((a[mask] - b[mask]) ** 2)))


def mean_bias(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if not np.any(mask):
        return math.nan
    return float(np.mean(a[mask] - b[mask]))


def peak_error(a: np.ndarray, b: np.ndarray) -> float:
    if not np.any(np.isfinite(a)) or not np.any(np.isfinite(b)):
        return math.nan
    return float(np.nanmax(a) - np.nanmax(b))


def rom_error(a: np.ndarray, b: np.ndarray) -> float:
    if not np.any(np.isfinite(a)) or not np.any(np.isfinite(b)):
        return math.nan
    return float((np.nanmax(a) - np.nanmin(a)) - (np.nanmax(b) - np.nanmin(b)))


def first_dict_key(data: dict[str, Any], preferred: str | None = None) -> str:
    if preferred is not None and preferred in data:
        return preferred
    return next(iter(data.keys()))


class TrajectoryView:
    def __init__(self, data: dict[str, Any]):
        self.data = data
        self.old_layout = "joints" in data and "muscles" in data
        if self.old_layout:
            self.record = data
            self.case_key = self._infer_old_case_key()
        else:
            self.case_key = first_dict_key(data, "pilot")
            self.record = data[self.case_key]

        self.joints = list(self.record.get("joints", self.record.get("coordinates", [])))
        self.rotational_joints = list(
            self.record.get(
                "rotationalJoints",
                self.record.get("rotationalCoordinates", []),
            )
        )
        self.muscles = list(self.record.get("muscles", []))

    def _infer_old_case_key(self) -> str:
        for value in self.data.values():
            if isinstance(value, dict) and value:
                return first_dict_key(value)
        raise ValueError("Could not infer case key from old optimaltrajectories.")

    def get(self, *fields: str) -> np.ndarray | None:
        for field in fields:
            if field not in self.record:
                continue
            value = self.record[field]
            if isinstance(value, dict):
                key = first_dict_key(value, self.case_key)
                return np.asarray(value[key], dtype=float)
            return np.asarray(value, dtype=float)
        return None


def load_view(path: Path) -> TrajectoryView:
    return TrajectoryView(np.load(path, allow_pickle=True).item())


def is_walking_folder(name: str) -> bool:
    return name.startswith("walking") and "TS" not in name


def result_candidates(dynamics_dir: Path, trial_kind: str) -> list[Path]:
    if trial_kind == "walking":
        walking_folders = [
            path for path in dynamics_dir.glob("walking*")
            if path.is_dir() and is_walking_folder(path.name)
        ]
        return sorted(
            walking_folders,
            key=lambda path: (path.name != "walking1", subject_sort_key(path.name)),
        )
    return [
        dynamics_dir / "squats1_rep0",
        dynamics_dir / "squats1",
    ]


def resolve_result(processing_data: Path, subject: str, model_case: str,
                   trial_kind: str) -> dict[str, Any]:
    session_dir = processing_data / staged_session_name(subject, model_case)
    dynamics_dir = session_dir / "OpenSimData" / "Dynamics"
    row = {
        "subject": subject,
        "movement": trial_kind,
        "trial": "",
        "model": model_case,
        "status": "",
        "path": "",
    }
    if not session_dir.exists():
        row["status"] = "missing_staged_session"
        row["path"] = str(session_dir)
        return row
    if not dynamics_dir.exists():
        row["status"] = "missing_dynamics_folder"
        row["path"] = str(dynamics_dir)
        return row

    candidates = result_candidates(dynamics_dir, trial_kind)
    existing = [path for path in candidates if path.exists()]
    for folder in existing:
        result = folder / "optimaltrajectories.npy"
        if result.exists():
            row["trial"] = folder.name
            row["status"] = "available"
            row["path"] = str(result)
            return row

    if existing:
        row["trial"] = existing[0].name
        row["status"] = "missing_optimaltrajectories"
        row["path"] = str(existing[0])
    else:
        row["status"] = "missing_trial_folder"
        row["path"] = str(candidates[0] if candidates else dynamics_dir)
    return row


def coordinate_matrix(view: TrajectoryView, *fields: str) -> np.ndarray | None:
    matrix = view.get(*fields)
    if matrix is None:
        return None
    return normalize_rows(matrix)


def muscle_matrix(view: TrajectoryView, *fields: str) -> np.ndarray | None:
    matrix = view.get(*fields)
    if matrix is None:
        return None
    return normalize_rows(matrix)


def kinematic_scale(view: TrajectoryView, variable: str) -> float:
    if variable in view.rotational_joints or variable.startswith(("hip_", "knee_")):
        return 180.0 / np.pi
    return 1.0


def add_kinematic_rows(rows: list[dict[str, Any]], subject: str,
                       movement: str, trial_name: str, model_case: str,
                       view: TrajectoryView) -> None:
    reference = coordinate_matrix(
        view, "coordinate_values_mocap", "coordinate_values_ref"
    )
    simulation = coordinate_matrix(view, "coordinate_values")
    to_track = coordinate_matrix(view, "coordinate_values_toTrack")
    if reference is None or simulation is None:
        return

    for variable in KINEMATIC_VARIABLES:
        if variable not in view.joints:
            continue
        idx = view.joints.index(variable)
        scale = kinematic_scale(view, variable)
        ref = reference[idx] * scale
        sim = simulation[idx] * scale
        rows.append(
            metric_row(
                subject, movement, trial_name, "kinematics", variable,
                model_case, "sim-vs-mocap", sim, ref,
            )
        )
        if to_track is not None:
            rows.append(
                metric_row(
                    subject, movement, trial_name, "kinematics", variable,
                    model_case, "videoIK-vs-mocap", to_track[idx] * scale, ref,
                )
            )


def group_indices(view: TrajectoryView, group: str, side: str) -> list[int]:
    indices = []
    for base_name in MUSCLE_GROUPS[group]:
        muscle = f"{base_name}_{side}"
        if muscle in view.muscles:
            indices.append(view.muscles.index(muscle))
    return indices


def add_activation_rows(rows: list[dict[str, Any]], subject: str,
                        movement: str, trial_name: str, model_case: str,
                        view: TrajectoryView) -> None:
    simulation = muscle_matrix(view, "muscle_activations")
    reference = muscle_matrix(
        view, "muscle_activations_emg", "muscle_activations_ref"
    )
    if simulation is None:
        return

    for group in MUSCLE_GROUPS:
        for side in ("l", "r"):
            indices = group_indices(view, group, side)
            if not indices:
                continue
            sim = np.nanmean(simulation[indices], axis=0)
            ref = None
            if reference is not None:
                ref_candidate = np.nanmean(reference[indices], axis=0)
                if np.isfinite(ref_candidate).sum() >= 3:
                    ref = ref_candidate
            rows.append(
                activation_row(
                    subject, movement, trial_name, f"{group}_{side}",
                    model_case, sim, ref,
                )
            )


def metric_row(subject: str, movement: str, trial_name: str,
               variable_type: str, variable: str, model_case: str,
               comparison: str, signal: np.ndarray,
               reference: np.ndarray) -> dict[str, Any]:
    return {
        "subject": subject,
        "movement": movement,
        "trial": trial_name,
        "variable_type": variable_type,
        "variable": variable,
        "model": model_case,
        "comparison": comparison,
        "rmse": rmse(signal, reference),
        "mean_bias": mean_bias(signal, reference),
        "peak_error": peak_error(signal, reference),
        "rom_error": rom_error(signal, reference),
        "pearson_r": pearson_r(signal, reference),
        "mean_activation": "",
        "peak_activation": "",
        "peak_timing_percent": "",
        "emg_rmse": "",
        "emg_pearson_r": "",
    }


def activation_row(subject: str, movement: str, trial_name: str,
                   variable: str, model_case: str, signal: np.ndarray,
                   reference: np.ndarray | None) -> dict[str, Any]:
    if np.any(np.isfinite(signal)):
        peak_idx = int(np.nanargmax(signal))
        peak_time = peak_idx * 100.0 / (len(signal) - 1)
        peak_value = float(np.nanmax(signal))
        mean_value = float(np.nanmean(signal))
    else:
        peak_time = math.nan
        peak_value = math.nan
        mean_value = math.nan

    return {
        "subject": subject,
        "movement": movement,
        "trial": trial_name,
        "variable_type": "activation",
        "variable": variable,
        "model": model_case,
        "comparison": "activation",
        "rmse": "",
        "mean_bias": "",
        "peak_error": "",
        "rom_error": "",
        "pearson_r": "",
        "mean_activation": mean_value,
        "peak_activation": peak_value,
        "peak_timing_percent": peak_time,
        "emg_rmse": "" if reference is None else rmse(signal, reference),
        "emg_pearson_r": "" if reference is None else pearson_r(signal, reference),
    }


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "subject", "movement", "trial", "variable_type", "variable",
        "model", "comparison", "rmse", "mean_bias", "peak_error",
        "rom_error", "pearson_r", "mean_activation", "peak_activation",
        "peak_timing_percent", "emg_rmse", "emg_pearson_r",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_status_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["subject", "movement", "trial", "model", "status", "path"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def first_available_view(views: dict[str, TrajectoryView]) -> TrajectoryView:
    if "generic" in views:
        return views["generic"]
    return next(iter(views.values()))


def plot_kinematics(output_dir: Path, subject: str, movement: str,
                    views: dict[str, TrajectoryView],
                    trial_names: dict[str, str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.linspace(0.0, 100.0, NORM_N)
    plot_dir = output_dir / "plots" / subject / movement / "kinematics"
    plot_dir.mkdir(parents=True, exist_ok=True)
    colors = {"generic": "#2f65b0", "modified": "#c73e3a"}
    reference_view = first_available_view(views)
    reference = coordinate_matrix(
        reference_view, "coordinate_values_mocap", "coordinate_values_ref"
    )

    for variable in KINEMATIC_VARIABLES:
        if not any(variable in view.joints for view in views.values()):
            continue
        plt.figure(figsize=(6, 4))
        if reference is not None and variable in reference_view.joints:
            idx_ref = reference_view.joints.index(variable)
            scale_ref = kinematic_scale(reference_view, variable)
            plt.plot(x, reference[idx_ref] * scale_ref, color="black",
                     linewidth=2, label="mocap IK")
        for model_case, view in views.items():
            if variable not in view.joints:
                continue
            idx = view.joints.index(variable)
            scale = kinematic_scale(view, variable)
            sim = coordinate_matrix(view, "coordinate_values")
            to_track = coordinate_matrix(view, "coordinate_values_toTrack")
            label_prefix = f"{model_case} {trial_names[model_case]}"
            if to_track is not None:
                plt.plot(x, to_track[idx] * scale, color=colors[model_case],
                         linestyle="--", label=f"{label_prefix} video IK")
            if sim is not None:
                plt.plot(x, sim[idx] * scale, color=colors[model_case],
                         label=f"{label_prefix} sim")
        plt.xlabel("Movement cycle (%)")
        plt.ylabel("Angle (deg)")
        plt.title(f"{subject} {movement} {variable}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / f"{variable}.png", dpi=200)
        plt.close()


def plot_activations(output_dir: Path, subject: str, movement: str,
                     views: dict[str, TrajectoryView],
                     trial_names: dict[str, str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.linspace(0.0, 100.0, NORM_N)
    plot_dir = output_dir / "plots" / subject / movement / "activations"
    plot_dir.mkdir(parents=True, exist_ok=True)
    colors = {"generic": "#2f65b0", "modified": "#c73e3a"}
    reference_view = first_available_view(views)
    reference = muscle_matrix(
        reference_view, "muscle_activations_emg", "muscle_activations_ref"
    )

    for group in MUSCLE_GROUPS:
        for side in ("l", "r"):
            plt.figure(figsize=(6, 4))
            ref_indices = group_indices(reference_view, group, side)
            if reference is not None and ref_indices:
                ref = np.nanmean(reference[ref_indices], axis=0)
                if np.isfinite(ref).sum() >= 3:
                    plt.plot(x, ref, color="black", linestyle="--",
                             linewidth=2, label="EMG")
            for model_case, view in views.items():
                sim = muscle_matrix(view, "muscle_activations")
                indices = group_indices(view, group, side)
                if sim is None or not indices:
                    continue
                plt.plot(x, np.nanmean(sim[indices], axis=0),
                         color=colors[model_case],
                         label=f"{model_case} {trial_names[model_case]}")
            plt.xlabel("Movement cycle (%)")
            plt.ylabel("Activation")
            plt.ylim(0, 1)
            plt.title(f"{subject} {movement} {group}_{side}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(plot_dir / f"{group}_{side}.png", dpi=200)
            plt.close()


def finite_values(rows: list[dict[str, Any]], field: str) -> list[float]:
    values = []
    for row in rows:
        value = row[field]
        if value == "":
            continue
        value = float(value)
        if math.isfinite(value):
            values.append(value)
    return values


def write_paired_delta_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    kin_rows = [
        row for row in rows
        if row["variable_type"] == "kinematics" and
        row["comparison"] == "sim-vs-mocap"
    ]
    grouped: dict[tuple[str, str, str, str], dict[str, dict[str, Any]]] = {}
    for row in kin_rows:
        key = (
            row["subject"], row["movement"], row["trial"], row["variable"],
        )
        grouped.setdefault(key, {})[row["model"]] = row

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "subject", "movement", "trial", "variable", "generic_rmse",
        "modified_rmse", "modified_minus_generic_rmse",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key, model_rows in sorted(grouped.items()):
            if not all(case in model_rows for case in MODEL_CASES):
                continue
            generic_rmse = float(model_rows["generic"]["rmse"])
            modified_rmse = float(model_rows["modified"]["rmse"])
            writer.writerow({
                "subject": key[0],
                "movement": key[1],
                "trial": key[2],
                "variable": key[3],
                "generic_rmse": generic_rmse,
                "modified_rmse": modified_rmse,
                "modified_minus_generic_rmse": modified_rmse - generic_rmse,
            })


def write_report(rows: list[dict[str, Any]], status_rows: list[dict[str, Any]],
                 output_path: Path) -> None:
    kin_rows = [
        r for r in rows
        if r["variable_type"] == "kinematics" and r["comparison"] == "sim-vs-mocap"
    ]
    lines = ["# LabValidation Comparison", ""]
    lines.append(f"Loaded result files: {sum(r['status'] == 'available' for r in status_rows)}")
    skipped = [r for r in status_rows if r["status"] != "available"]
    lines.append(f"Skipped/missing result files: {len(skipped)}")
    lines.append("")

    for movement in sorted({r["movement"] for r in rows}):
        lines.append(f"## {movement}")
        for model_case in MODEL_CASES:
            model_kin = [
                r for r in kin_rows
                if r["movement"] == movement and r["model"] == model_case
            ]
            values = finite_values(model_kin, "rmse")
            subjects = sorted({r["subject"] for r in model_kin}, key=subject_sort_key)
            if values:
                lines.append(
                    f"- {model_case}: mean hip/knee angle RMSE "
                    f"{np.mean(values):.3f} deg across {len(subjects)} subjects"
                )
        act_rows = [
            r for r in rows
            if r["movement"] == movement and r["variable_type"] == "activation"
            and r["emg_rmse"] != ""
        ]
        for model_case in MODEL_CASES:
            values = finite_values(
                [r for r in act_rows if r["model"] == model_case],
                "emg_rmse",
            )
            if values:
                lines.append(
                    f"- {model_case}: mean activation EMG RMSE "
                    f"{np.mean(values):.3f}"
                )

        paired_deltas = []
        keys = {
            (r["subject"], r["trial"], r["variable"]) for r in kin_rows
            if r["movement"] == movement
        }
        for subject, trial, variable in keys:
            generic = [
                r for r in kin_rows
                if r["subject"] == subject and r["movement"] == movement and
                r["trial"] == trial and r["variable"] == variable and
                r["model"] == "generic"
            ]
            modified = [
                r for r in kin_rows
                if r["subject"] == subject and r["movement"] == movement and
                r["trial"] == trial and r["variable"] == variable and
                r["model"] == "modified"
            ]
            if generic and modified:
                paired_deltas.append(float(modified[0]["rmse"]) - float(generic[0]["rmse"]))
        if paired_deltas:
            lines.append(
                f"- paired modified-minus-generic hip/knee RMSE: "
                f"{np.mean(paired_deltas):.3f} deg"
            )
        lines.append("")

    if skipped:
        lines.append("## Skipped Results")
        for row in skipped:
            lines.append(
                f"- {row['subject']} {row['movement']} {row['model']}: "
                f"{row['status']}"
            )
        lines.append("")

    lines.append(
        "Note: applying one modified model to LabValidation subjects is a "
        "numerical/model-sensitivity check, not subject-anatomy validation."
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


def main_cli() -> None:
    args = parse_args()
    trials = selected_trials(args.trial)
    subjects = discover_subjects(args.processing_data, args.subjects)
    rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []

    print(f"Subjects: {', '.join(subjects)}")
    for subject in subjects:
        for movement in trials:
            views: dict[str, TrajectoryView] = {}
            trial_names: dict[str, str] = {}
            for model_case in MODEL_CASES:
                status = resolve_result(
                    args.processing_data, subject, model_case, movement
                )
                status_rows.append(status)
                if status["status"] != "available":
                    continue
                try:
                    views[model_case] = load_view(Path(status["path"]))
                    trial_names[model_case] = str(status["trial"])
                except Exception as exc:  # noqa: BLE001 - record and continue.
                    status["status"] = f"load_failed: {exc}"
                    views.pop(model_case, None)
                    trial_names.pop(model_case, None)

            if args.require_paired and len(views) != len(MODEL_CASES):
                continue
            if not views:
                continue

            for model_case, view in views.items():
                add_kinematic_rows(
                    rows, subject, movement, trial_names[model_case],
                    model_case, view,
                )
                add_activation_rows(
                    rows, subject, movement, trial_names[model_case],
                    model_case, view,
                )
            if not args.skip_plots:
                plot_kinematics(args.output_dir, subject, movement, views, trial_names)
                plot_activations(args.output_dir, subject, movement, views, trial_names)

    write_csv(rows, args.output_dir / "comparison_summary.csv")
    write_status_csv(status_rows, args.output_dir / "result_status.csv")
    write_paired_delta_csv(rows, args.output_dir / "paired_kinematic_rmse_delta.csv")
    write_report(rows, status_rows, args.output_dir / "comparison_report.md")
    print(f"Wrote comparison outputs to {args.output_dir}")
    print(
        "Available results: "
        f"{sum(row['status'] == 'available' for row in status_rows)}; "
        f"skipped/missing: {sum(row['status'] != 'available' for row in status_rows)}"
    )


if __name__ == "__main__":
    main_cli()
