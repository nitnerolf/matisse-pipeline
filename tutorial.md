# MATISSE CLI Quick Tutorial (Simple Workflow: WORK IN PROGRESS)

This tutorial explains how to reduce MATISSE data with the `matisse` command-line interface.

## 1. Prerequisites

- Python environment with this package installed (`matisse` command available).
- ESO pipeline runtime installed (`esorex`).
- MATISSE recipes (`mat_*`) visible to `esorex`.

Check your environment first:

```bash
matisse doctor
```

If recipes are not found, you can force a directory:

```bash
matisse doctor --recipe-dir /path/to/esopipes-plugins
```

## 2. Recommended Reduction Flow

Use this order:

1. `matisse reduce` (raw FITS -> reduced products in `Iter*` folders)
2. `matisse format` (collect + rename OIFITS files)
3. `matisse calibrate` (apply transfer function using calibrators)
4. Optional: `matisse bcd ...` (BCD corrections)
5. Optional: `matisse show` (visual inspection)

---

## 3. Step-by-Step Commands

### Step A: Reduce raw files

```bash
matisse reduce --data-dir /data/raw_night --result-dir /data/reduced --nbcore 4 --max-iter 2
```

What it does:

- Reads raw MATISSE FITS files.
- Builds reduction blocks by template start and detector.
- Runs ESO recipes (`esorex`) block-by-block (parallel if `--nbcore > 1`).
- Iterates (with `--max-iter`) to reuse calibration products from previous iterations.

Important options:

- `--calib-dir`: calibration archive directory (default: same as raw).
- `--resol LOW|MED|HIGH`: process only one resolution.
- `--skipL` / `--skipN`: skip one detector band.
- `--tplid`, `--tplstart`: reduce only selected templates.
- `--spectral-binning`: custom spectral binning.
- `--overwrite`: replace previous reduction outputs.
- `--check-blocks`: dry check of reducible blocks (no processing).
- `--check-cal`: check calibration attachment status.
- `--block-cal N`: print calibration files attached to block `N`.

Typical output:

- `/data/reduced/Iter1/...`
- `/data/reduced/Iter2/...`
- recipe logs and reduced FITS inside `*.rb/` block folders.

### Step B: Format reduced outputs into OIFITS folder

```bash
matisse format /data/reduced/Iter2
```

What it does:

- Scans FITS outputs recursively.
- Copies science/calibrator OIFITS products.
- Renames files with metadata (TPL start, target, configuration, band, resolution, BCD mode, chopping).

Output folder name:

- `<input_name>_OIFITS` created in your current working directory.
- Example: if input is `/data/reduced/Iter2`, output is `./Iter2_OIFITS`.

### Step C: Calibrate OIFITS

```bash
matisse calibrate --data-dir ./Iter2_OIFITS --timespan 0.04 --bands LM --bands N
```

What it does:

- Associates target files with calibrator files close in time.
- Builds SOF files.
- Runs `esorex mat_cal_oifits`.
- Writes calibrated outputs.

Important options:

- `--result-dir`: calibration output directory (default: `<data-dir>_CALIBRATED`).
- `--timespan`: max target-calibrator time separation in days.
- `--bands`: choose `LM`, `N`, or both.
- `--cumul-block / --no-cumul-block`: control `cumulBlock` recipe parameter.

---

## 4. Optional BCD Workflow

### Compute BCD magic numbers

```bash
matisse bcd compute /data/night1_OIFITS /data/night2_OIFITS \
  --bcd-mode ALL --band LM --resol LOW --plot
```

Main options:

- `--bcd-mode IN_IN|OUT_IN|IN_OUT|ALL`
- `--wavelength-range 3.3 3.8`
- `--poly-order`, `--tau0-min`
- `--output-dir`, `--prefix`
- `--results-dir` (plot existing results without recomputing)

### Apply BCD corrections

```bash
matisse bcd apply /data/Iter2_OIFITS /data/mn2025_results --merge
```

### Other BCD tools

- Remove BCD effects: `matisse bcd remove /data/Iter2_OIFITS --band LM`
- Compare BCD modes: `matisse bcd compare /data/Iter2_OIFITS`
- Merge BCD modes only: `matisse bcd merge /data/Iter2_OIFITS`

---

## 5. Visual Inspection

Display one OIFITS file:

```bash
matisse show /data/Iter2_OIFITS/my_file.fits
```

Save figure:

```bash
matisse show /data/Iter2_OIFITS/my_file.fits --save summary.png
```

Interactive multi-BCD view (same TPL START):

```bash
matisse show /data/Iter2_OIFITS/my_file.fits --interactive
```

---

## 6. Full Minimal Example

```bash
matisse doctor
matisse reduce --data-dir /data/raw --result-dir /data/reduced --nbcore 4 --max-iter 2
matisse format /data/reduced/Iter2
matisse calibrate --data-dir ./Iter2_OIFITS --bands LM --bands N
```

This is the shortest practical path to go from raw MATISSE FITS to calibrated OIFITS products.
