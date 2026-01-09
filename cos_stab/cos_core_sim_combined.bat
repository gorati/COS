@ECHO OFF
python cos_core_sim_combined.py --steps 1000 --seed 2025 --lambda-geom 1.0 --output metrics.jsonl --run-meta run_meta.json
rem python cos_core_sim_combined.py --steps 1000 --stop-on-gap-event
