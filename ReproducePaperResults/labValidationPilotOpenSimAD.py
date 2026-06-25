"""
Stage and run OpenSimAD for the LabValidation pilot comparison.

Default:
    all staged LabValidation pilot subjects, generic and modified cases, and
    all staged walking/squats trials.

The script can stage files only, run OpenSimAD only, or do both:

    python labValidationPilotOpenSimAD.py --mode stage
    python labValidationPilotOpenSimAD.py --mode functions
    python labValidationPilotOpenSimAD.py --mode run --subjects subject10 --case generic --trial walking1
    python labValidationPilotOpenSimAD.py --mode all
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Optional, Union

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
POSE_FOLDER = "OpenPose_default"
CAMERA_SETUP = "2-cameras"
RUN_CASE_NAME = "pilot"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage and run OpenSimAD for LabValidation pilot cases."
    )
    parser.add_argument(
        "--mode",
        choices=["stage", "functions", "run", "all"],
        default="stage",
        help=(
            "Stage inputs, generate OpenSimAD external functions, run "
            "OpenSimAD, or do stage+run."
        ),
    )
    parser.add_argument(
        "--case",
        choices=["generic", "modified", "both"],
        default="both",
        help="Which staged case to process.",
    )
    parser.add_argument(
        "--trial",
        nargs="+",
        default=["all"],
        help=(
            "Trial names to process, for example --trial walking1 squats1. "
            "Use 'all' or 'both' to process every staged walking/squats trial."
        ),
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=["all"],
        help=(
            "Subjects to process, for example --subjects subject2 subject3. "
            "Use 'all' to process every subject staged by kinematics."
        ),
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


def subject_sort_key(subject: str) -> tuple[str, Union[int, str]]:
    prefix = "".join(ch for ch in subject if not ch.isdigit())
    suffix = "".join(ch for ch in subject if ch.isdigit())
    return (prefix, int(suffix) if suffix else subject)


def session_sort_key(session_name: str) -> tuple[str, Union[int, str], int]:
    subject, session = session_name.split("_Session", 1)
    return (*subject_sort_key(subject), int(session))


def selected_subjects(args: argparse.Namespace, cases: list[str]) -> list[str]:
    if len(args.subjects) == 1 and args.subjects[0].lower() == "all":
        subject_sets = []
        for case_name in cases:
            data_root = args.kinematics_root / case_name / "Data"
            if not data_root.exists():
                raise FileNotFoundError(
                    f"Missing kinematics data root for {case_name}: {data_root}"
                )
            subjects = {
                path.name.split("_Session", 1)[0]
                for path in data_root.iterdir()
                if path.is_dir() and "_Session" in path.name
            }
            subject_sets.append(subjects)
        selected = sorted(set.intersection(*subject_sets), key=subject_sort_key)
    else:
        selected = sorted(dict.fromkeys(args.subjects), key=subject_sort_key)

    if not selected:
        raise FileNotFoundError("No subjects selected.")

    for case_name in cases:
        for subject in selected:
            session0 = (
                args.kinematics_root / case_name / "Data" /
                f"{subject}_Session0"
            )
            if not session0.exists():
                raise FileNotFoundError(
                    f"Missing staged kinematics session: {session0}"
                )
    return selected


def staged_session_name(subject: str, case_name: str) -> str:
    return f"lab_{subject}_{case_name}"


def trial_motion_type(trial_name: str) -> str:
    lowered = trial_name.lower()
    if lowered.startswith("walking"):
        return "walking"
    if lowered.startswith("squats"):
        return "squats"
    raise ValueError(
        f"Unsupported OpenSimAD trial '{trial_name}'. This staging script "
        "currently supports walking* and squats* trials."
    )


def trial_repetition(trial_name: str) -> Optional[int]:
    return 0 if trial_motion_type(trial_name) == "squats" else None


def discover_trial_infos(args: argparse.Namespace, case_name: str,
                         subject: str) -> dict[str, dict[str, object]]:
    subject_sessions = sorted(
        (
            path for path in (args.kinematics_root / case_name / "Data").glob(
                f"{subject}_Session*"
            )
            if path.is_dir()
        ),
        key=lambda path: session_sort_key(path.name),
    )
    trial_infos: dict[str, dict[str, object]] = {}
    for session_dir in subject_sessions:
        kinematics_dir = (
            session_dir / "OpenSimData" / POSE_FOLDER / CAMERA_SETUP /
            "Kinematics"
        )
        if not kinematics_dir.exists():
            continue
        for motion_file in sorted(kinematics_dir.glob("*_LSTM.mot")):
            trial_name = motion_file.name.removesuffix("_LSTM.mot")
            try:
                motion_type = trial_motion_type(trial_name)
            except ValueError:
                continue
            trial_infos[trial_name] = {
                "motion_type": motion_type,
                "source_session": session_dir.name,
                "repetition": trial_repetition(trial_name),
            }

    requested = [trial for trial in args.trial]
    if len(requested) == 1 and requested[0].lower() in ("all", "both"):
        selected = trial_infos
    else:
        selected = {}
        for trial_name in requested:
            if trial_name not in trial_infos:
                raise FileNotFoundError(
                    f"No staged kinematics found for {case_name} {subject} "
                    f"trial {trial_name}."
                )
            selected[trial_name] = trial_infos[trial_name]

    if not selected:
        raise FileNotFoundError(
            f"No supported walking/squats kinematics found for "
            f"{case_name} {subject}."
        )
    return dict(sorted(selected.items()))


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


def staged_session_dir(args: argparse.Namespace, subject: str,
                       case_name: str) -> Path:
    return args.processing_data / staged_session_name(subject, case_name)


def source_case_session(args: argparse.Namespace, case_name: str,
                        session_name: str) -> Path:
    return args.kinematics_root / case_name / "Data" / session_name


def write_staged_metadata(source_metadata: Path, target_metadata: Path,
                          subject: str, case_name: str,
                          overwrite: bool) -> None:
    if target_metadata.exists() and not overwrite:
        return
    if not source_metadata.exists():
        raise FileNotFoundError(f"Missing metadata: {source_metadata}")

    replacements = {
        "openSimModel": CASES[case_name],
        "subjectID": staged_session_name(subject, case_name),
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


def stage_case(args: argparse.Namespace, subject: str, case_name: str,
               trial_infos: dict[str, dict[str, object]]) -> None:
    stage_dir = staged_session_dir(args, subject, case_name)
    session0 = source_case_session(args, case_name, f"{subject}_Session0")
    source_model_dir = (
        session0 / "OpenSimData" / POSE_FOLDER / CAMERA_SETUP / "Model"
    )
    source_metadata = session0 / "sessionMetadata.yaml"
    model_name = CASES[case_name]
    scaled_model = source_model_dir / f"{model_name}_scaled.osim"
    if not scaled_model.exists():
        raise FileNotFoundError(
            f"Missing scaled model for {case_name} {subject}: {scaled_model}. "
            "Run labValidationPilotKinematics.py first."
        )

    print(f"Staging {case_name} {subject} into {stage_dir}")
    write_staged_metadata(
        source_metadata,
        stage_dir / "sessionMetadata.yaml",
        subject,
        case_name,
        args.overwrite_stage,
    )
    copy_tree(
        source_model_dir,
        stage_dir / "OpenSimData" / "Model",
        args.overwrite_stage,
    )

    for trial_name, trial_info in trial_infos.items():
        source_session = source_case_session(
            args, case_name, str(trial_info["source_session"])
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

        lab_subject = args.lab_data / subject
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


def run_case_trial(args: argparse.Namespace, subject: str, case_name: str,
                   trial_name: str, trial_info: dict[str, object]) -> None:
    from UtilsDynamicSimulations.OpenSimAD.utilsOpenSimAD import (
        processInputsOpenSimAD,
    )

    stage_dir = staged_session_dir(args, subject, case_name)
    if not stage_dir.exists():
        raise FileNotFoundError(
            f"Missing staged session {stage_dir}; run --mode stage first."
        )

    motion_type = str(trial_info["motion_type"])
    if motion_type == "walking":
        time_window = walking_time_window(stage_dir, trial_name)
        repetition = None
    else:
        time_window = []
        repetition = trial_info["repetition"]

    print(
        f"Running OpenSimAD {subject} {case_name} {trial_name}: "
        f"motion_type={motion_type}, "
        f"time_window={time_window}, repetition={repetition}"
    )
    session_id = staged_session_name(subject, case_name)
    settings = processInputsOpenSimAD(
        str(PROCESSING_DIR),
        str(args.processing_data),
        session_id,
        trial_name,
        motion_type,
        time_window,
        repetition,
        0,
        "all",
        overwrite=args.overwrite_opensimad_inputs,
    )
    if args.mode == "functions":
        print(
            f"Generated OpenSimAD inputs/external function for "
            f"{subject} {case_name} {trial_name}."
        )
        return

    from UtilsDynamicSimulations.OpenSimAD.mainOpenSimAD import run_tracking

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
        session_id,
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
    subjects = selected_subjects(args, cases)
    trial_plan = {
        (subject, case_name): discover_trial_infos(args, case_name, subject)
        for subject in subjects
        for case_name in cases
    }
    for case_name in cases:
        validate_processing_model(case_name)

    if args.mode in ("stage", "all"):
        for subject in subjects:
            for case_name in cases:
                stage_case(args, subject, case_name,
                           trial_plan[(subject, case_name)])

    if args.mode in ("functions", "run", "all"):
        for subject in subjects:
            for case_name in cases:
                for trial_name, trial_info in trial_plan[
                    (subject, case_name)
                ].items():
                    run_case_trial(args, subject, case_name, trial_name,
                                   trial_info)


if __name__ == "__main__":
    main_cli()
