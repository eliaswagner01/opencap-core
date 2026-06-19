"""
Reprocess a small LabValidation pilot dataset with generic and modified models.

This script is intentionally narrower than labValidationVideosToKinematics.py.
It prepares and processes the LabValidation trials needed for the model
comparison:

    <subject> Session0: extrinsics, static1, squats1
    <subject> Session1: extrinsics, walking1

Outputs are written to:

    opencap-core/Data/LabValidationPilot/<case>/Data/<session>

where <case> is "generic" or "modified".
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Optional, Tuple, Union

REPO_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = REPO_DIR.parent
if str(REPO_DIR) not in sys.path:
    sys.path.append(str(REPO_DIR))


CASES = {
    "generic": "LaiUhlrich2022",
    "modified": "LaiUhlrich2022_adjusted",
}
SESSION0_TRIALS = ["extrinsics", "static1", "squats1"]
PREFERRED_WALKING_TRIAL = "walking1"
CAMERA_SETUP = "2-cameras"
CAMERAS_TO_USE = ["Cam1", "Cam3"]
POSE_DETECTOR = "OpenPose"
RESOLUTION_POSE_DETECTION = "default"
AUGMENTER_MODEL = "v0.2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reprocess LabValidation pilot trials for model comparison."
    )
    parser.add_argument(
        "--source-data",
        type=Path,
        default=WORKSPACE_DIR / "LabValidation_withVideos",
        help="Path to LabValidation_withVideos.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_DIR / "Data" / "LabValidationPilot",
        help="Root where generic/modified pilot outputs are written.",
    )
    parser.add_argument(
        "--case",
        choices=["generic", "modified", "both"],
        default="both",
        help="Which model case to process.",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=["all"],
        help=(
            "Subjects to process, for example: --subjects subject2 subject3. "
            "Use 'all' to process every subject folder in --source-data."
        ),
    )
    parser.add_argument(
        "--modified-model-source",
        type=Path,
        default=None,
        help=(
            "Optional path to a .osim file to install as "
            "LaiUhlrich2022_subjectSpecificFemur.osim in opencap-core."
        ),
    )
    parser.add_argument(
        "--overwrite-restructure",
        action="store_true",
        help="Re-copy videos/metadata even if the pilot session folder exists.",
    )
    parser.add_argument(
        "--skip-processing",
        action="store_true",
        help="Only prepare folder structure and metadata; do not call OpenCap main().",
    )
    parser.add_argument(
        "--strict-scaling-sanity",
        action="store_true",
        help=(
            "Fail on OpenCap's attached-geometry scale-factor sanity check. "
            "By default the pilot continues if the scaled model was written, "
            "because custom femur meshes may use different mesh units."
        ),
    )
    return parser.parse_args()


def subject_sort_key(subject: str) -> Tuple[str, Union[int, str]]:
    prefix = "".join(ch for ch in subject if not ch.isdigit())
    suffix = "".join(ch for ch in subject if ch.isdigit())
    return (prefix, int(suffix) if suffix else subject)


def selected_subjects(source_data: Path, subject_args: list[str]) -> list[str]:
    if len(subject_args) == 1 and subject_args[0].lower() == "all":
        subjects = [
            path.name for path in source_data.iterdir()
            if path.is_dir() and path.name.startswith("subject")
        ]
    else:
        subjects = subject_args
    subjects = sorted(dict.fromkeys(subjects), key=subject_sort_key)
    if not subjects:
        raise FileNotFoundError(f"No subject folders found in {source_data}")
    for subject in subjects:
        subject_dir = source_data / subject
        if not subject_dir.exists():
            raise FileNotFoundError(f"Missing subject folder: {subject_dir}")
    return subjects


def session_trials_for_subject(subject: str) -> dict[str, list[str]]:
    return {
        f"{subject}_Session0": SESSION0_TRIALS,
        f"{subject}_Session1": ["extrinsics", PREFERRED_WALKING_TRIAL],
    }


def available_trials_for_session(source_data: Path, subject: str,
                                 session_index: int) -> set[str]:
    session_dir = source_data / subject / "VideoData" / f"Session{session_index}"
    if not session_dir.exists():
        return set()
    cam_dirs = [path for path in sorted(session_dir.glob("Cam*")) if path.is_dir()]
    if not cam_dirs:
        return set()
    available = None
    for cam_dir in cam_dirs:
        trials = {path.name for path in cam_dir.iterdir() if path.is_dir()}
        available = trials if available is None else available.intersection(trials)
    return available or set()


def choose_walking_trial(available_trials: set[str]) -> Optional[str]:
    if PREFERRED_WALKING_TRIAL in available_trials:
        return PREFERRED_WALKING_TRIAL
    walking_trials = [
        trial for trial in available_trials
        if trial.startswith("walking") and "TS" not in trial
    ]
    return sorted(walking_trials, key=subject_sort_key)[0] if walking_trials else None


def planned_session_trials(source_data: Path, subject: str) -> dict[str, list[str]]:
    available_session0 = available_trials_for_session(source_data, subject, 0)
    available_session1 = available_trials_for_session(source_data, subject, 1)
    if not available_session0 or not available_session1:
        print(f"WARNING: Skipping {subject}; local video sessions are incomplete.")
        return {}

    missing_session0 = [
        trial for trial in SESSION0_TRIALS if trial not in available_session0
    ]
    if missing_session0:
        print(
            f"WARNING: Skipping {subject}; Session0 is missing "
            f"{', '.join(missing_session0)}."
        )
        return {}

    walking_trial = choose_walking_trial(available_session1)
    if walking_trial is None:
        print(f"WARNING: Skipping {subject}; Session1 has no walking trial.")
        return {}
    if walking_trial != PREFERRED_WALKING_TRIAL:
        print(
            f"WARNING: {subject} has no {PREFERRED_WALKING_TRIAL}; "
            f"using {walking_trial} instead."
        )

    session1_trials = ["extrinsics", walking_trial]
    missing_session1 = [
        trial for trial in session1_trials if trial not in available_session1
    ]
    if missing_session1:
        print(
            f"WARNING: Skipping {subject}; Session1 is missing "
            f"{', '.join(missing_session1)}."
        )
        return {}

    return {
        f"{subject}_Session0": SESSION0_TRIALS,
        f"{subject}_Session1": session1_trials,
    }


def install_modified_model(source: Optional[Path]) -> None:
    if source is None:
        return
    if not source.exists():
        raise FileNotFoundError(f"Modified model source not found: {source}")
    target = REPO_DIR / "OpenSimPipeline" / "Models" / (
        CASES["modified"] + ".osim"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    print(f"Installed modified model: {target}")


def validate_model(case_name: str) -> None:
    model_name = CASES[case_name]
    model_path = REPO_DIR / "OpenSimPipeline" / "Models" / f"{model_name}.osim"
    if not model_path.exists():
        raise FileNotFoundError(
            "Missing model required for case "
            f"{case_name!r}: {model_path}. "
            "Place/copy the model there or pass --modified-model-source."
        )


def subject_dir_from_session(session_name: str) -> str:
    return session_name.split("_Session", 1)[0]


def source_session_name(session_name: str) -> str:
    return "Session" + session_name.rsplit("_Session", 1)[1]


def write_case_metadata(source_subject_dir: Path, target_session_dir: Path,
                        model_name: str) -> None:
    source_metadata = source_subject_dir / "sessionMetadata.yaml"
    target_session_dir.mkdir(parents=True, exist_ok=True)
    target_metadata = target_session_dir / "sessionMetadata.yaml"
    lines = source_metadata.read_text().splitlines()
    found_model = False
    for idx, line in enumerate(lines):
        if not line.startswith(" ") and line.split(":", 1)[0] == "openSimModel":
            lines[idx] = f"openSimModel: {model_name}"
            found_model = True
    if not found_model:
        lines.append(f"openSimModel: {model_name}")
    target_metadata.write_text("\n".join(lines) + "\n")


def copy_session_videos(source_data: Path, output_case_root: Path,
                        session_name: str, trials: list[str],
                        model_name: str, overwrite: bool) -> None:
    subject = subject_dir_from_session(session_name)
    source_subject_dir = source_data / subject
    source_video_dir = source_subject_dir / "VideoData" / source_session_name(session_name)
    target_session_dir = output_case_root / "Data" / session_name

    # Always rewrite metadata so partial/old pilot runs cannot keep a stale
    # openSimModel value.
    write_case_metadata(source_subject_dir, target_session_dir, model_name)

    for cam_dir in sorted(source_video_dir.glob("Cam*")):
        if not cam_dir.is_dir():
            continue
        target_cam_dir = target_session_dir / "Videos" / cam_dir.name
        target_input_dir = target_cam_dir / "InputMedia"
        target_cam_dir.mkdir(parents=True, exist_ok=True)

        parameters = cam_dir / "cameraIntrinsicsExtrinsics.pickle"
        if parameters.exists():
            shutil.copy2(parameters, target_cam_dir / parameters.name)

        for trial in trials:
            source_trial_dir = cam_dir / trial
            source_video = source_trial_dir / f"{trial}.avi"
            if not source_video.exists():
                raise FileNotFoundError(f"Missing video: {source_video}")
            target_trial_dir = target_input_dir / trial
            target_trial_dir.mkdir(parents=True, exist_ok=True)
            target_video = target_trial_dir / source_video.name
            if overwrite or not target_video.exists():
                shutil.copy2(source_video, target_video)


def intrinsics_folder_for_session(session_name: str) -> str:
    if "subject2" in session_name or "subject3" in session_name:
        return "Deployed_720_240fps"
    return "Deployed_720_60fps"


def process_trial(output_case_root: Path, session_name: str, trial: str) -> None:
    from main import main

    is_extrinsics = "extrinsics" in trial.lower()
    is_static = "static" in trial.lower()
    # Current OpenCap requires all cameras for calibration and neutral scaling.
    # Dynamic trials still use the selected 2-camera setup.
    cameras_to_use = ["all"] if is_extrinsics or is_static else CAMERAS_TO_USE

    main(
        session_name,
        trial,
        trial,
        cameras_to_use,
        intrinsics_folder_for_session(session_name),
        False,
        is_extrinsics,
        None,
        None,
        CAMERA_SETUP,
        4,
        POSE_DETECTOR,
        resolutionPoseDetection=RESOLUTION_POSE_DETECTION,
        scaleModel=is_static,
        bbox_thr=0.8,
        augmenter_model=AUGMENTER_MODEL,
        dataDir=str(output_case_root),
    )


def scaled_model_path(output_case_root: Path, session_name: str,
                      model_name: str) -> Path:
    return (
        output_case_root / "Data" / session_name / "OpenSimData" /
        f"{POSE_DETECTOR}_{RESOLUTION_POSE_DETECTION}" / CAMERA_SETUP /
        "Model" / f"{model_name}_scaled.osim"
    )


def is_scaling_sanity_exception(exc: Exception) -> bool:
    message = " ".join(str(arg) for arg in exc.args)
    return (
        "segment sizes are not anthropometrically realistic" in message or
        "Musculoskeletal model scaling failed" in message
    )


def copy_session0_model_to_session1(output_case_root: Path,
                                    subject: str) -> None:
    session0 = output_case_root / "Data" / f"{subject}_Session0"
    session1 = output_case_root / "Data" / f"{subject}_Session1"
    model0 = (
        session0 / "OpenSimData" /
        f"{POSE_DETECTOR}_{RESOLUTION_POSE_DETECTION}" /
        CAMERA_SETUP / "Model"
    )
    model1 = (
        session1 / "OpenSimData" /
        f"{POSE_DETECTOR}_{RESOLUTION_POSE_DETECTION}" /
        CAMERA_SETUP / "Model"
    )
    if not model0.exists():
        raise FileNotFoundError(f"Session0 scaled model folder not found: {model0}")
    model1.mkdir(parents=True, exist_ok=True)
    shutil.copytree(model0, model1, dirs_exist_ok=True)


def run_case(case_name: str, subjects: list[str],
             args: argparse.Namespace) -> None:
    validate_model(case_name)
    model_name = CASES[case_name]
    output_case_root = args.output_root / case_name
    print(f"\n=== Preparing {case_name}: {model_name} ===")
    subject_plans = {
        subject: planned_session_trials(args.source_data, subject)
        for subject in subjects
    }
    subject_plans = {
        subject: plan for subject, plan in subject_plans.items() if plan
    }
    if not subject_plans:
        raise FileNotFoundError("No subjects have the required local videos.")

    for subject, session_plan in subject_plans.items():
        print(f"\n--- Preparing {case_name}: {subject} ---")
        for session_name, trials in session_plan.items():
            copy_session_videos(
                args.source_data,
                output_case_root,
                session_name,
                trials,
                model_name,
                args.overwrite_restructure,
            )

    if args.skip_processing:
        print("Skipping OpenCap processing (--skip-processing).")
        return

    for subject, session_plan in subject_plans.items():
        print(f"\n--- Processing {case_name}: {subject} ---")
        for session_name, trials in session_plan.items():
            if session_name.endswith("Session1"):
                copy_session0_model_to_session1(output_case_root, subject)
            for trial in trials:
                print(f"Processing {case_name}: {session_name} / {trial}")
                try:
                    process_trial(output_case_root, session_name, trial)
                except Exception as exc:
                    is_static = "static" in trial.lower()
                    expected_model = scaled_model_path(
                        output_case_root, session_name, model_name
                    )
                    if (
                        is_static and
                        not args.strict_scaling_sanity and
                        is_scaling_sanity_exception(exc) and
                        expected_model.exists()
                    ):
                        print(
                            "WARNING: OpenCap scaling sanity check failed after "
                            f"writing {expected_model}. Continuing because custom "
                            "femur mesh scale factors can trigger this check. "
                            "Inspect the scaled model before interpreting results."
                        )
                        continue
                    raise


def main_cli() -> None:
    args = parse_args()
    install_modified_model(args.modified_model_source)
    subjects = selected_subjects(args.source_data, args.subjects)
    cases = ["generic", "modified"] if args.case == "both" else [args.case]
    for case_name in cases:
        validate_model(case_name)
    print(f"Subjects: {', '.join(subjects)}")
    for case_name in cases:
        run_case(case_name, subjects, args)


if __name__ == "__main__":
    main_cli()
