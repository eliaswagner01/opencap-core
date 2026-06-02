"""
Reprocess a small LabValidation pilot dataset with generic and modified models.

This script is intentionally narrower than labValidationVideosToKinematics.py.
It prepares and processes only the pilot trials needed for the model
comparison:

    subject10 Session0: extrinsics, static1, squats1
    subject10 Session1: extrinsics, walking1

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

REPO_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = REPO_DIR.parent
if str(REPO_DIR) not in sys.path:
    sys.path.append(str(REPO_DIR))


CASES = {
    "generic": "LaiUhlrich2022",
    "modified": "LaiUhlrich2022_adjusted",
}
SUBJECT = "subject10"
SESSION_TRIALS = {
    f"{SUBJECT}_Session0": ["extrinsics", "static1", "squats1"],
    f"{SUBJECT}_Session1": ["extrinsics", "walking1"],
}
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


def install_modified_model(source: Path | None) -> None:
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
    import yaml
    from utils import importMetadata

    source_metadata = source_subject_dir / "sessionMetadata.yaml"
    target_session_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_metadata, target_session_dir / "sessionMetadata.yaml")
    metadata = importMetadata(str(target_session_dir / "sessionMetadata.yaml"))
    metadata["openSimModel"] = model_name
    with open(target_session_dir / "sessionMetadata.yaml", "w") as f:
        yaml.dump(metadata, f)


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


def copy_session0_model_to_session1(output_case_root: Path) -> None:
    session0 = output_case_root / "Data" / f"{SUBJECT}_Session0"
    session1 = output_case_root / "Data" / f"{SUBJECT}_Session1"
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


def run_case(case_name: str, args: argparse.Namespace) -> None:
    validate_model(case_name)
    model_name = CASES[case_name]
    output_case_root = args.output_root / case_name
    print(f"\n=== Preparing {case_name}: {model_name} ===")

    for session_name, trials in SESSION_TRIALS.items():
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

    for session_name, trials in SESSION_TRIALS.items():
        if session_name.endswith("Session1"):
            copy_session0_model_to_session1(output_case_root)
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
    cases = ["generic", "modified"] if args.case == "both" else [args.case]
    for case_name in cases:
        validate_model(case_name)
    for case_name in cases:
        run_case(case_name, args)


if __name__ == "__main__":
    main_cli()
