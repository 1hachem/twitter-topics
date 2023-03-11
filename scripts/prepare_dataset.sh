#!/bin/bash
"$input_csv"="$1"
openai tools fine_tunes.prepare_data -f "$input_csv"