#!/bin/bash
# export OPENAI_API_KEY=sk-...

# check usage 
if [ $# -ne 2 ]; then
    echo "Usage: $0 <input_json> <output_jsonl>"
    exit 1
fi

"$input_json"="$1"
"$output_jsonl"="$2"

openai tools fine_tunes.prepare_data -f "$input_json" -o "$output_jsonl"
openai api fine_tunes.create -t -m ada --n_epochs 4