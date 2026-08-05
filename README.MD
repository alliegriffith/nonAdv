Use uv to set up your environment:
> uv venv .venv --python 3.11 

 subsequently activate with:
> source .venv/bin/activate


To run CEM:
1. Choose your settings in run/cem_settings.py and in run/run_cem.py
2. Run "run_cem.py" with 
> uv run python -m run.run_cem