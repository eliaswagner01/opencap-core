"""
Stage and run OpenSimAD for the LabValidation pilot comparison.

Default pilot:
    all staged LabValidation subjects, generic and modified cases, walking
    and squats1 repetition 0.

The script can stage files only, run OpenSimAD only, or do both:

    python labValidationPilotOpenSimAD.py --mode stage
    python labValidationPilotOpenSimAD.py --mode run --case generic --trial walking
    python labValidationPilotOpenSimAD.py --mode all
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Tuple, Union

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
PREFERRED_WALKING_TRIAL = "walking1"
TRIAL_KINDS = {
    "walking": {
        "motion_type": "walking",
        "source_session_index": 1,
        "repetition": None,
    },
    "squats1": {
        "motion_type": "squats",
        "source_session_index": 0,
        "repetition": 0,
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
        choices=["walking", "walking1", "squats1", "both"],
        default="both",
        help=(
            "Which trial to process. 'walking' uses walking1 when available "
            "and falls back to the first available non-treadmill walking trial."
        ),
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=["all"],
        help=(
            "Subjects to process, for example: --subjects subject2 subject3. "
            "Use 'all' to process every subject folder in --lab-data."
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


def selected_trials(trial_arg: str) -> list[str]:
    if trial_arg == "both":
        return ["walking", "squats1"]
    if trial_arg == "walking1":
        return ["walking"]
    return [trial_arg]


def subject_sort_key(subject: str) -> Tuple[str, Union[int, str]]:
    prefix = "".join(ch for ch in subject if not ch.isdigit())
    suffix = "".join(ch for ch in subject if ch.isdigit())
    return (prefix, int(suffix) if suffix else subject)


def selected_subjects(lab_data: Path, subject_args: list[str]) -> list[str]:
    if len(subject_args) == 1 and subject_args[0].lower() == "all":
        subjects = [
            path.name for path in lab_data.iterdir()
            if path.is_dir() and path.name.startswith("subject")
        ]
    else:
        subjects = subject_args
    subjects = sorted(dict.fromkeys(subjects), key=subject_sort_key)
    if not subjects:
        raise FileNotFoundError(f"No subject folders found in {lab_data}")
    for subject in subjects:
        if not (lab_data / subject).exists():
            raise FileNotFoundError(f"Missing subject folder: {lab_data / subject}")
    return subjects


def staged_session_name(subject: str, case_name: str) -> str:
    return f"lab_{subject}_{case_name}"


def source_session_name(subject: str, session_index: int) -> str:
    return f"{subject}_Session{session_index}"


def is_walking_trial_name(trial_name: str) -> bool:
    return trial_name.startswith("walking") and "TS" not in trial_name


def strip_kinematics_suffix(path: Path) -> str:
    name = path.stem
    return name[:-5] if name.endswith("_LSTM") else name


def walking_candidates_from_kinematics(kinematics_dir: Path) -> list[str]:
    if not kinematics_dir.exists():
        return []
    candidates = {
        strip_kinematics_suffix(path)
        for path in kinematics_dir.glob("walking*.mot")
        if is_walking_trial_name(strip_kinematics_suffix(path))
    }
    if PREFERRED_WALKING_TRIAL in candidates:
        return [PREFERRED_WALKING_TRIAL] + sorted(
            candidates - {PREFERRED_WALKING_TRIAL}, key=subject_sort_key)
    return sorted(candidates, key=subject_sort_key)


def lab_reference_files(lab_subject: Path, trial_name: str) -> list[Path]:
    return [
        lab_subject / "ForceData" / f"{trial_name}_forces.mot",
        lab_subject / "EMGData" / f"{trial_name}_EMG.sto",
        lab_subject / "OpenSimData" / "Mocap" / "IK" / f"{trial_name}.mot",
        lab_subject / "OpenSimData" / "Mocap" / "ID" / f"{trial_name}.sto",
    ]


def choose_walking_trial(args: argparse.Namespace, case_name: str,
                         subject: str) -> str:
    session1 = source_case_session(args, case_name, subject, 1)
    kinematics_dir = (
        session1 / "OpenSimData" / POSE_FOLDER / CAMERA_SETUP / "Kinematics"
    )
    lab_subject = args.lab_data / subject
    for trial_name in walking_candidates_from_kinematics(kinematics_dir):
        if all(path.exists() for path in lab_reference_files(lab_subject, trial_name)):
            if trial_name != PREFERRED_WALKING_TRIAL:
                print(
                    f"WARNING: {subject} has no staged {PREFERRED_WALKING_TRIAL}; "
                    f"using {trial_name} instead."
                )
            return trial_name
    raise FileNotFoundError(
        f"No walking trial with kinematics and LabValidation references found "
        f"for {case_name} {subject} in {kinematics_dir}"
    )


def trial_spec(args: argparse.Namespace, case_name: str, subject: str,
               trial_kind: str) -> dict[str, object]:
    if trial_kind == "walking":
        trial_name = choose_walking_trial(args, case_name, subject)
    else:
        trial_name = trial_kind
    kind = "walking" if is_walking_trial_name(trial_name) else trial_kind
    info = TRIAL_KINDS[kind]
    return {
        "trial_name": trial_name,
        "motion_type": info["motion_type"],
        "source_session": source_session_name(
            subject, int(info["source_session_index"])),
        "repetition": info["repetition"],
    }


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
                        subject: str, session_index: int) -> Path:
    return (
        args.kinematics_root / case_name / "Data" /
        source_session_name(subject, session_index)
    )


def write_staged_metadata(source_metadata: Path, target_metadata: Path,
                          case_name: str, subject: str,
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


def stage_case(args: argparse.Namespace, case_name: str, subject: str,
               trial_kinds: list[str]) -> None:
    stage_dir = staged_session_dir(args, subject, case_name)
    session0 = source_case_session(args, case_name, subject, 0)
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
        case_name,
        subject,
        args.overwrite_stage,
    )
    copy_tree(
        source_model_dir,
        stage_dir / "OpenSimData" / "Model",
        args.overwrite_stage,
    )

    for trial_kind in trial_kinds:
        spec = trial_spec(args, case_name, subject, trial_kind)
        trial_name = str(spec["trial_name"])
        source_session = source_case_session(
            args,
            case_name,
            subject,
            1 if spec["motion_type"] == "walking" else 0,
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


def run_case_trial(args: argparse.Namespace, case_name: str, subject: str,
                   trial_kind: str) -> None:
    from UtilsDynamicSimulations.OpenSimAD.mainOpenSimAD import run_tracking
    from UtilsDynamicSimulations.OpenSimAD.utilsOpenSimAD import (
        processInputsOpenSimAD,
    )

    stage_dir = staged_session_dir(args, subject, case_name)
    if not stage_dir.exists():
        raise FileNotFoundError(
            f"Missing staged session {stage_dir}; run --mode stage first."
        )

    spec = trial_spec(args, case_name, subject, trial_kind)
    trial_name = str(spec["trial_name"])
    motion_type = str(spec["motion_type"])
    if motion_type == "walking":
        time_window = walking_time_window(stage_dir, trial_name)
        repetition = None
    else:
        time_window = []
        repetition = spec["repetition"]

    print(
        f"Running OpenSimAD {case_name} {subject} {trial_name}: "
        f"motion_type={motion_type}, "
        f"time_window={time_window}, repetition={repetition}"
    )
    settings = processInputsOpenSimAD(
        str(PROCESSING_DIR),
        str(args.processing_data),
        staged_session_name(subject, case_name),
        trial_name,
        motion_type,
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
        staged_session_name(subject, case_name),
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
    subjects = selected_subjects(args.lab_data, args.subjects)
    for case_name in cases:
        validate_processing_model(case_name)
    print(f"Subjects: {', '.join(subjects)}")

    if args.mode in ("stage", "all"):
        for case_name in cases:
            for subject in subjects:
                try:
                    stage_case(args, case_name, subject, trials)
                except FileNotFoundError as exc:
                    if args.subjects == ["all"]:
                        print(f"WARNING: Skipping stage for {case_name} {subject}: {exc}")
                        continue
                    raise

    if args.mode in ("run", "all"):
        for case_name in cases:
            for subject in subjects:
                for trial_name in trials:
                    try:
                        run_case_trial(args, case_name, subject, trial_name)
                    except FileNotFoundError as exc:
                        if args.subjects == ["all"]:
                            print(
                                "WARNING: Skipping OpenSimAD for "
                                f"{case_name} {subject} {trial_name}: {exc}"
                            )
                            continue
                        raise


if __name__ == "__main__":
    main_cli()
