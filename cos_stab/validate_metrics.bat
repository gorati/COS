@ECHO OFF
python validate_metrics.py metrics.jsonl --schema extended --strict
REM python validate_metrics.py metrics.jsonl --schema minimal --strict
REM python validate_metrics.py metrics.jsonl --schema extended --strict --require-no-gap-events

