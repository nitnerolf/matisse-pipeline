# Changelog

All notable changes to this project will be documented in this file.

## v0.3.0 (2026-02-24)

### Feat

- add merge command to BCD CLI and enhance find_sci_filename to include CAL files
- add BCD CP mappings and update rename_calibrated_outputs function to support suffixes
- enhance category determination in _read_category method for OIFitsReader
- add remove command to handle BCD ordering in SCI OIFITS files
- improve path handling in generate_sof_files and run_esorex_calibration functions
- enhance error handling in apply_bcd_corrections for missing files
- add compare command for BCD corrections and update visualization utilities
- add plot of BCD correction
- add quality check of the BCD correction
- Add bcd correction application and save
- add BCD command group for magic numbers computation and correction

### Fix

- update help text for file directories and apply function description in BCD CLI
- correct type annotation for BCD_CP_SIGN_MAP to list[int]
- update apply_bcd_corrections to return None instead of dict
- remove unused '--no-chopping' option from BCD CLI test
- update CLI commands from 'magic' to 'bcd compute' in test cases

### Refactor

- update import statements for Sequence from collections.abc

## v0.2.0 (2026-02-04)

### Feat

- add ESO pipeline installation guidance in doctor command

## v0.1.1 (2026-02-03)

- Use of commitizen to test the automatized version updating

## v0.1.0 (2026-01-01)

- Initial release of MATISSE pipeline for pypi
- Core calibration pipeline interface
- CLI interface with `matisse` command
        - Automated calibration (`matisse calibrate`)
        - Automated data reduction (`matisse reduce`)
- Interactive viewer with Plotly visualization
- BCD (Beam-commuting device) correction module
- OIFITS data reader and processor
- Configuration system for pipeline parameters
- Comprehensive test suite

### Features

- Support for MATISSE interferometric data
- Python 3.10+ support
- CLI tools: calibrate, reduce, magic, show, format, etc.
