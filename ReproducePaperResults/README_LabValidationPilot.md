# LabValidation Pilot Model Comparison

This pilot workflow compares the generic OpenCap model with a modified femur
model on LabValidation `subject10`, `walking1`, and `squats1`.

Current OpenCap requires all cameras for extrinsics and neutral/static scaling.
The pilot therefore uses all cameras for `extrinsics` and `static1`, then uses
the selected `Cam1`/`Cam3` two-camera setup for `walking1` and `squats1`.

The modified model is expected at both:

```text
opencap-core/OpenSimPipeline/Models/LaiUhlrich2022_subjectSpecificFemur.osim
opencap-processing/OpenSimPipeline/Models/LaiUhlrich2022_subjectSpecificFemur.osim
```

You can install the core copy while running the kinematics script:

```powershell
python opencap-core/ReproducePaperResults/labValidationPilotKinematics.py `
  --modified-model-source path\to\your_model.osim
```

Copy the same file manually to `opencap-processing/OpenSimPipeline/Models/`
before running OpenSimAD.

## 1. Reprocess Kinematics

```powershell
python opencap-core/ReproducePaperResults/labValidationPilotKinematics.py --case both
```

Outputs are written under:

```text
opencap-core/Data/LabValidationPilot/generic/Data
opencap-core/Data/LabValidationPilot/modified/Data
```

If your custom femur mesh uses different mesh units, OpenCap's scale-factor
sanity check may report non-anthropometric segment scaling even after writing
the scaled model. The pilot script continues through that specific static-scaling
exception by default if `<OpenSimModel>_scaled.osim` exists. Use
`--strict-scaling-sanity` to make that check fatal.

## 2. Stage OpenSimAD Inputs

```powershell
python opencap-core/ReproducePaperResults/labValidationPilotOpenSimAD.py --mode stage
```

This creates:

```text
opencap-processing/Data/lab_subject10_generic
opencap-processing/Data/lab_subject10_modified
```

with kinematics, model, force, EMG, mocap IK, and mocap ID files in the folder
layout expected by OpenSimAD.

## 3. Run OpenSimAD

Run one trial first:

```powershell
python opencap-core/ReproducePaperResults/labValidationPilotOpenSimAD.py `
  --mode run --case generic --trial walking1
```

Then run the rest:

```powershell
python opencap-core/ReproducePaperResults/labValidationPilotOpenSimAD.py --mode run
```

Use `--analyze-only` if the optimization has already been solved and you only
want to regenerate analysis files.

## 4. Compare Results

```powershell
python opencap-core/ReproducePaperResults/compareLabValidationPilot.py
```

Outputs:

```text
opencap-processing/Data/LabValidationPilotComparison/comparison_summary.csv
opencap-processing/Data/LabValidationPilotComparison/pilot_report.md
opencap-processing/Data/LabValidationPilotComparison/plots
```

This is a numerical/model-sensitivity check when the same modified model is
applied to LabValidation subjects; it is not subject-anatomy validation.
