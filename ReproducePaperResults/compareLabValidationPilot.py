"""
Compare LabValidation pilot generic and modified model results.

The script reads OpenSimAD optimaltrajectories.npy files, normalizes selected
waveforms to 101 points, computes kinematic and activation metrics, and writes
CSV/plot/report outputs.
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

STAGED_SESSIONS = {
    "generic": "lab_subject10_generic",
    "modified": "lab_subject10_modified",
}
TRIAL_FOLDERS = {
    "walking1": "walking1",
    "squats1": "squats1_rep0",
}
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
        description="Compare LabValidation pilot OpenSimAD results."
    )
    parser.add_argument(
        "--processing-data",
        type=Path,
        default=PROCESSING_DIR / "Data",
        help="OpenSimAD data root containing staged lab_subject10_* sessions.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSING_DIR / "Data" / "LabValidationPilotComparison",
        help="Folder for comparison CSV, plots, and report.",
    )
    parser.add_argument(
        "--trial",
        choices=["walking1", "squats1", "both"],
        default="both",
        help="Trial(s) to compare.",
    )
    return parser.parse_args()


def selected_trials(trial_arg: str) -> list[str]:
    return ["walking1", "squats1"] if trial_arg == "both" else [trial_arg]


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

        self.joints = list(self.record.get("joints", []))
        self.rotational_joints = list(self.record.get("rotationalJoints", []))
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
    if not path.exists():
        raise FileNotFoundError(f"Missing optimal trajectories: {path}")
    return TrajectoryView(np.load(path, allow_pickle=True).item())


def result_path(processing_data: Path, model_case: str, trial_name: str) -> Path:
    session = STAGED_SESSIONS[model_case]
    folder = TRIAL_FOLDERS[trial_name]
    primary = (
        processing_data / session / "OpenSimData" / "Dynamics" /
        folder / "optimaltrajectories.npy"
    )
    if primary.exists():
        return primary
    fallback = (
        processing_data / session / "OpenSimData" / "Dynamics" /
        trial_name / "optimaltrajectories.npy"
    )
    return fallback


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


def add_kinematic_rows(rows: list[dict[str, Any]], trial_name: str,
                       model_case: str, view: TrajectoryView) -> None:
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
                trial_name, "kinematics", variable, model_case,
                "sim-vs-mocap", sim, ref,
            )
        )
        if to_track is not None:
            rows.append(
                metric_row(
                    trial_name, "kinematics", variable, model_case,
                    "videoIK-vs-mocap", to_track[idx] * scale, ref,
                )
            )


def group_indices(view: TrajectoryView, group: str,
                  side: str) -> list[int]:
    indices = []
    for base_name in MUSCLE_GROUPS[group]:
        muscle = f"{base_name}_{side}"
        if muscle in view.muscles:
            indices.append(view.muscles.index(muscle))
    return indices


def add_activation_rows(rows: list[dict[str, Any]], trial_name: str,
                        model_case: str, view: TrajectoryView) -> None:
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
                    trial_name, f"{group}_{side}", model_case, sim, ref
                )
            )


def metric_row(trial_name: str, variable_type: str, variable: str,
               model_case: str, comparison: str, signal: np.ndarray,
               reference: np.ndarray) -> dict[str, Any]:
    return {
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


def activation_row(trial_name: str, variable: str, model_case: str,
                   signal: np.ndarray, reference: np.ndarray | None) -> dict[str, Any]:
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
        "trial", "variable_type", "variable", "model", "comparison",
        "rmse", "mean_bias", "peak_error", "rom_error", "pearson_r",
        "mean_activation", "peak_activation", "peak_timing_percent",
        "emg_rmse", "emg_pearson_r",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_kinematics(output_dir: Path, trial_name: str,
                    views: dict[str, TrajectoryView]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.linspace(0.0, 100.0, NORM_N)
    plot_dir = output_dir / "plots" / trial_name / "kinematics"
    plot_dir.mkdir(parents=True, exist_ok=True)
    colors = {"generic": "#2f65b0", "modified": "#c73e3a"}

    reference = coordinate_matrix(
        views["generic"], "coordinate_values_mocap", "coordinate_values_ref"
    )
    for variable in KINEMATIC_VARIABLES:
        if variable not in views["generic"].joints:
            continue
        plt.figure(figsize=(6, 4))
        idx_ref = views["generic"].joints.index(variable)
        scale_ref = kinematic_scale(views["generic"], variable)
        if reference is not None:
            plt.plot(x, reference[idx_ref] * scale_ref, color="black",
                     linewidth=2, label="mocap IK")
        for model_case, view in views.items():
            if variable not in view.joints:
                continue
            idx = view.joints.index(variable)
            scale = kinematic_scale(view, variable)
            sim = coordinate_matrix(view, "coordinate_values")
            to_track = coordinate_matrix(view, "coordinate_values_toTrack")
            if to_track is not None:
                plt.plot(x, to_track[idx] * scale, color=colors[model_case],
                         linestyle="--", label=f"{model_case} video IK")
            if sim is not None:
                plt.plot(x, sim[idx] * scale, color=colors[model_case],
                         label=f"{model_case} sim")
        plt.xlabel("Movement cycle (%)")
        plt.ylabel("Angle (deg)")
        plt.title(f"{trial_name} {variable}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / f"{variable}.png", dpi=200)
        plt.close()


def plot_activations(output_dir: Path, trial_name: str,
                     views: dict[str, TrajectoryView]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.linspace(0.0, 100.0, NORM_N)
    plot_dir = output_dir / "plots" / trial_name / "activations"
    plot_dir.mkdir(parents=True, exist_ok=True)
    colors = {"generic": "#2f65b0", "modified": "#c73e3a"}

    reference = muscle_matrix(
        views["generic"], "muscle_activations_emg", "muscle_activations_ref"
    )
    for group in MUSCLE_GROUPS:
        for side in ("l", "r"):
            plt.figure(figsize=(6, 4))
            ref_indices = group_indices(views["generic"], group, side)
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
                         color=colors[model_case], label=model_case)
            plt.xlabel("Movement cycle (%)")
            plt.ylabel("Activation")
            plt.ylim(0, 1)
            plt.title(f"{trial_name} {group}_{side}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(plot_dir / f"{group}_{side}.png", dpi=200)
            plt.close()


def write_report(rows: list[dict[str, Any]], output_path: Path) -> None:
    kin_rows = [
        r for r in rows
        if r["variable_type"] == "kinematics" and r["comparison"] == "sim-vs-mocap"
    ]
    lines = ["# LabValidation Pilot Comparison", ""]
    for trial in sorted({r["trial"] for r in rows}):
        lines.append(f"## {trial}")
        for model_case in ("generic", "modified"):
            values = [
                float(r["rmse"]) for r in kin_rows
                if r["trial"] == trial and r["model"] == model_case and r["rmse"] != ""
            ]
            if values:
                lines.append(
                    f"- {model_case} mean hip/knee angle RMSE: "
                    f"{np.mean(values):.3f} deg"
                )
        act_rows = [
            r for r in rows
            if r["trial"] == trial and r["variable_type"] == "activation"
            and r["emg_rmse"] != ""
        ]
        if act_rows:
            for model_case in ("generic", "modified"):
                values = [
                    float(r["emg_rmse"]) for r in act_rows
                    if r["model"] == model_case
                ]
                if values:
                    lines.append(
                        f"- {model_case} mean activation EMG RMSE: "
                        f"{np.mean(values):.3f}"
                    )
        else:
            lines.append("- No non-NaN EMG references found for activation RMSE.")
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
    rows: list[dict[str, Any]] = []

    for trial_name in trials:
        views = {
            case: load_view(result_path(args.processing_data, case, trial_name))
            for case in ("generic", "modified")
        }
        for model_case, view in views.items():
            add_kinematic_rows(rows, trial_name, model_case, view)
            add_activation_rows(rows, trial_name, model_case, view)
        plot_kinematics(args.output_dir, trial_name, views)
        plot_activations(args.output_dir, trial_name, views)

    write_csv(rows, args.output_dir / "comparison_summary.csv")
    write_report(rows, args.output_dir / "pilot_report.md")
    print(f"Wrote comparison outputs to {args.output_dir}")


if __name__ == "__main__":
    main_cli()
