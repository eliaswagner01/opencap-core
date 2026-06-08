"""
Stage and run OpenSimAD for the LabValidation pilot comparison.

Default pilot:
    subject10, generic and modified cases, walking1 and squats1 repetition 0.

The script can stage files only, run OpenSimAD only, or do both:

    python labValidationPilotOpenSimAD.py --mode stage
    python labValidationPilotOpenSimAD.py --mode run --case generic --trial walking1
    python labValidationPilotOpenSimAD.py --mode all
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = REPO_DIR.parent
PROCESSING_DIR = WORKSPACE_DIR / "opencap-processing"
OPENSIM_AD_DIR = (
    PROCESSING_DIR / "UtilsDynamicSimulations" / "OpenSimAD"
)
for path in (PROCESSING_DIR, OPENSIM_AD_DIR):
    if str(path) not in sys.path:
        sys.path.append(str(path))


CASES = {
    "generic": "LaiUhlrich2022",
    "modified": "LaiUhlrich2022_adjusted",
}
SUBJECT = "subject10"
STAGED_SESSIONS = {
    "generic": f"lab_{SUBJECT}_generic",
    "modified": f"lab_{SUBJECT}_modified",
}
TRIALS = {
    "walking1": {
        "motion_type": "walking",
        "source_session": f"{SUBJECT}_Session1",
        "repetition": None,
        "dynamics_folder": "walking1",
    },
    "squats1": {
        "motion_type": "squats",
        "source_session": f"{SUBJECT}_Session0",
        "repetition": 0,
        "dynamics_folder": "squats1_rep0",
    },
}
POSE_FOLDER = "OpenPose_default"
CAMERA_SETUP = "2-cameras"
RUN_CASE_NAME = "pilot"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage and run OpenSimAD for LabValidation pilot cases."
    )
    parser.add_argument(
        "--mode",
        choices=["stage", "run", "all"],
        default="stage",
        help="Stage inputs, run OpenSimAD, or do both.",
    )
    parser.add_argument(
        "--case",
        choices=["generic", "modified", "both"],
        default="both",
        help="Which staged case to process.",
    )
    parser.add_argument(
        "--trial",
        choices=["walking1", "squats1", "both"],
        default="both",
        help="Which pilot trial to process.",
    )
    parser.add_argument(
        "--kinematics-root",
        type=Path,
        default=REPO_DIR / "Data" / "LabValidationPilot",
        help="Root containing kinematics output from labValidationPilotKinematics.py.",
    )
    parser.add_argument(
        "--lab-data",
        type=Path,
        default=WORKSPACE_DIR / "LabValidation_withVideos",
        help="Original LabValidation_withVideos folder.",
    )
    parser.add_argument(
        "--processing-data",
        type=Path,
        default=PROCESSING_DIR / "Data",
        help="OpenSimAD data root.",
    )
    parser.add_argument(
        "--overwrite-stage",
        action="store_true",
        help="Overwrite staged files if they already exist.",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Skip solving and only analyze existing OpenSimAD solutions.",
    )
    parser.add_argument(
        "--overwrite-opensimad-inputs",
        action="store_true",
        help="Force OpenSimAD to regenerate adjusted/contact models and functions.",
    )
    parser.add_argument(
        "--polynomial-sample-count",
        type=int,
        default=0,
        help=(
            "Use an evenly subsampled polynomial-fitting dummy motion with this "
            "many rows. Leave at 0 to use OpenSimAD's default DummyMotion.mot."
        ),
    )
    return parser.parse_args()


def selected_cases(case_arg: str) -> list[str]:
    return ["generic", "modified"] if case_arg == "both" else [case_arg]


def selected_trials(trial_arg: str) -> list[str]:
    return ["walking1", "squats1"] if trial_arg == "both" else [trial_arg]


def copy_file(source: Path, target: Path, overwrite: bool) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Missing required source file: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        return
    shutil.copy2(source, target)


def copy_tree(source: Path, target: Path, overwrite: bool) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Missing required source folder: {source}")
    if target.exists() and overwrite:
        shutil.rmtree(target)
    shutil.copytree(source, target, dirs_exist_ok=True)


def find_kinematics_mot(kinematics_dir: Path, trial_name: str) -> Path:
    candidates = [
        kinematics_dir / f"{trial_name}_LSTM.mot",
        kinematics_dir / f"{trial_name}.mot",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    candidate_list = "\n  ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        "Missing required kinematics file. Expected one of:\n"
        f"  {candidate_list}"
    )


def staged_session_dir(args: argparse.Namespace, case_name: str) -> Path:
    return args.processing_data / STAGED_SESSIONS[case_name]


def source_case_session(args: argparse.Namespace, case_name: str,
                        session_name: str) -> Path:
    return args.kinematics_root / case_name / "Data" / session_name


def write_staged_metadata(source_metadata: Path, target_metadata: Path,
                          case_name: str, overwrite: bool) -> None:
    if target_metadata.exists() and not overwrite:
        return
    if not source_metadata.exists():
        raise FileNotFoundError(f"Missing metadata: {source_metadata}")

    replacements = {
        "openSimModel": CASES[case_name],
        "subjectID": STAGED_SESSIONS[case_name],
    }
    found = set()
    lines = source_metadata.read_text().splitlines()
    for idx, line in enumerate(lines):
        key = line.split(":", 1)[0].strip()
        if key in replacements and not line.startswith(" "):
            lines[idx] = f"{key}: {replacements[key]}"
            found.add(key)
    for key, value in replacements.items():
        if key not in found:
            lines.append(f"{key}: {value}")

    target_metadata.parent.mkdir(parents=True, exist_ok=True)
    target_metadata.write_text("\n".join(lines) + "\n")


def stage_case(args: argparse.Namespace, case_name: str,
               trial_names: list[str]) -> None:
    stage_dir = staged_session_dir(args, case_name)
    session0 = source_case_session(args, case_name, f"{SUBJECT}_Session0")
    source_model_dir = (
        session0 / "OpenSimData" / POSE_FOLDER / CAMERA_SETUP / "Model"
    )
    source_metadata = session0 / "sessionMetadata.yaml"
    model_name = CASES[case_name]
    scaled_model = source_model_dir / f"{model_name}_scaled.osim"
    if not scaled_model.exists():
        raise FileNotFoundError(
            f"Missing scaled model for {case_name}: {scaled_model}. "
            "Run labValidationPilotKinematics.py first."
        )

    print(f"Staging {case_name} into {stage_dir}")
    write_staged_metadata(
        source_metadata,
        stage_dir / "sessionMetadata.yaml",
        case_name,
        args.overwrite_stage,
    )
    copy_tree(
        source_model_dir,
        stage_dir / "OpenSimData" / "Model",
        args.overwrite_stage,
    )

    for trial_name in trial_names:
        trial_info = TRIALS[trial_name]
        source_session = source_case_session(
            args, case_name, trial_info["source_session"]
        )
        source_ik = find_kinematics_mot(
            source_session / "OpenSimData" / POSE_FOLDER / CAMERA_SETUP /
            "Kinematics",
            trial_name,
        )
        print(f"  {trial_name}: staging kinematics from {source_ik.name}")
        copy_file(
            source_ik,
            stage_dir / "OpenSimData" / "Kinematics" / f"{trial_name}.mot",
            args.overwrite_stage,
        )

        lab_subject = args.lab_data / SUBJECT
        copy_file(
            lab_subject / "ForceData" / f"{trial_name}_forces.mot",
            stage_dir / "ForceData" / f"{trial_name}.mot",
            args.overwrite_stage,
        )
        copy_file(
            lab_subject / "EMGData" / f"{trial_name}_EMG.sto",
            stage_dir / "EMGData" / f"{trial_name}.sto",
            args.overwrite_stage,
        )
        copy_file(
            lab_subject / "OpenSimData" / "Mocap" / "IK" / f"{trial_name}.mot",
            stage_dir / "OpenSimDataMocap" / "InverseKinematics" /
            f"{trial_name}.mot",
            args.overwrite_stage,
        )
        copy_file(
            lab_subject / "OpenSimData" / "Mocap" / "ID" / f"{trial_name}.sto",
            stage_dir / "OpenSimDataMocap" / "InverseDynamics" /
            f"{trial_name}.sto",
            args.overwrite_stage,
        )


def walking_time_window(stage_dir: Path, trial_name: str) -> list[float]:
    from utilsDataPostprocessing import segmentWalkStance

    force_path = stage_dir / "ForceData" / f"{trial_name}.mot"
    _, times = segmentWalkStance(str(force_path))
    return [float(times[0]), float(times[1])]


def make_subsampled_polynomial_motion(stage_dir: Path,
                                      sample_count: int) -> Path:
    source = (
        PROCESSING_DIR / "OpenSimPipeline" / "MuscleAnalysis" /
        "DummyMotion.mot"
    )
    if sample_count <= 0:
        raise ValueError("sample_count must be positive.")
    if not source.exists():
        raise FileNotFoundError(f"Missing default DummyMotion.mot: {source}")

    lines = source.read_text().splitlines()
    endheader_idx = next(
        (idx for idx, line in enumerate(lines)
         if line.strip().lower() == "endheader"),
        None,
    )
    if endheader_idx is None or endheader_idx + 1 >= len(lines):
        raise ValueError(f"Could not parse OpenSim motion header: {source}")

    header = lines[:endheader_idx + 2]
    rows = [line for line in lines[endheader_idx + 2:] if line.strip()]
    if sample_count > len(rows):
        raise ValueError(
            f"Requested {sample_count} polynomial samples, but {source} only "
            f"contains {len(rows)} rows."
        )

    if sample_count == len(rows):
        selected_rows = rows
    else:
        indices = np.linspace(0, len(rows) - 1, sample_count, dtype=int)
        selected_rows = [rows[idx] for idx in indices]

    for idx, line in enumerate(header):
        if line.strip().startswith("nRows="):
            header[idx] = f"nRows={len(selected_rows)}"

    target = (
        stage_dir / "OpenSimData" / "Model" /
        f"DummyMotion_polynomial_{sample_count}.mot"
    )
    target.write_text("\n".join(header + selected_rows) + "\n")
    return target


def run_case_trial(args: argparse.Namespace, case_name: str,
                   trial_name: str) -> None:
    from UtilsDynamicSimulations.OpenSimAD.mainOpenSimAD import run_tracking
    from UtilsDynamicSimulations.OpenSimAD.utilsOpenSimAD import (
        processInputsOpenSimAD,
    )

    stage_dir = staged_session_dir(args, case_name)
    if not stage_dir.exists():
        raise FileNotFoundError(
            f"Missing staged session {stage_dir}; run --mode stage first."
        )

    trial_info = TRIALS[trial_name]
    if trial_name == "walking1":
        time_window = walking_time_window(stage_dir, trial_name)
        repetition = None
    else:
        time_window = []
        repetition = trial_info["repetition"]

    print(
        f"Running OpenSimAD {case_name} {trial_name}: "
        f"motion_type={trial_info['motion_type']}, "
        f"time_window={time_window}, repetition={repetition}"
    )
    settings = processInputsOpenSimAD(
        str(PROCESSING_DIR),
        str(args.processing_data),
        STAGED_SESSIONS[case_name],
        trial_name,
        trial_info["motion_type"],
        time_window,
        repetition,
        0,
        "all",
        overwrite=args.overwrite_opensimad_inputs,
    )
    if args.polynomial_sample_count:
        path_dummy_motion = make_subsampled_polynomial_motion(
            stage_dir, args.polynomial_sample_count)
        settings["pathDummyMotion"] = str(path_dummy_motion)
        print(
            "Using subsampled polynomial-fitting motion: "
            f"{path_dummy_motion}"
        )
    run_tracking(
        str(PROCESSING_DIR),
        str(args.processing_data),
        STAGED_SESSIONS[case_name],
        settings,
        case=RUN_CASE_NAME,
        solveProblem=not args.analyze_only,
        analyzeResults=True,
    )


def validate_processing_model(case_name: str) -> None:
    model = PROCESSING_DIR / "OpenSimPipeline" / "Models" / (
        CASES[case_name] + ".osim"
    )
    if not model.exists():
        raise FileNotFoundError(
            f"Missing OpenSimAD model for {case_name}: {model}. "
            "Copy the same model into opencap-processing/OpenSimPipeline/Models."
        )


def main_cli() -> None:
    args = parse_args()
    cases = selected_cases(args.case)
    trials = selected_trials(args.trial)
    for case_name in cases:
        validate_processing_model(case_name)

    if args.mode in ("stage", "all"):
        for case_name in cases:
            stage_case(args, case_name, trials)

    if args.mode in ("run", "all"):
        for case_name in cases:
            for trial_name in trials:
                run_case_trial(args, case_name, trial_name)


if __name__ == "__main__":
    main_cli()
