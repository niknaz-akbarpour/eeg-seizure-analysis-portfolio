#!/usr/bin/env bash
set -euo pipefail

mkdir -p data/edf
base="https://physionet-open.s3.amazonaws.com/chbmit/1.0.0"

curl -L --fail -o data/edf/chb01_03.edf "$base/chb01/chb01_03.edf"
curl -L --fail -o data/edf/chb01_04.edf "$base/chb01/chb01_04.edf"
curl -L --fail -o data/edf/chb02_16.edf "$base/chb02/chb02_16.edf"
curl -L --fail -o data/edf/chb02_16+.edf "$base/chb02/chb02_16%2B.edf"
curl -L --fail -o data/edf/chb03_01.edf "$base/chb03/chb03_01.edf"
curl -L --fail -o data/edf/chb03_02.edf "$base/chb03/chb03_02.edf"

echo "Six EDF files downloaded to data/edf/."

