start_time="$(date -u +%s)"
python3 ../evaluate.py --config_path ../default_config.yaml
end_time="$(date -u +%s)"
elapsed="$(($end_time-$start_time))"
echo "Total of $elapsed seconds elapsed for process"
