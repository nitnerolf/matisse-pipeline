from pathlib import Path

import pandas as pd


def load_bcd_corrections(corrections_dir, bcd_mode):
    """Load BCD polynomial correction coefficients from CSV file."""
    csv_file = Path(corrections_dir) / f"bcd_{bcd_mode}_poly_coeffs.csv"
    if not csv_file.exists():
        raise FileNotFoundError(f"Correction file not found: {csv_file}")
    return pd.read_csv(csv_file)
