#!/bin/bash
# P.A.R.T.N.E.R CLI Wrapper
# Handles the Gorgonia/Go 1.25 compatibility issue

export ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.25

# Get the directory where this script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Run the partner binary
exec "${DIR}/bin/partner" "$@"
