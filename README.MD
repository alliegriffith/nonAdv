Use uv to set up your environment:
> uv venv .venv --python 3.11 

 subsequently activate with:
> source .venv/bin/activate

Within your new uv environment:
- To Use SimVBG:
1. clone repo as a seperate folder
> git clone https://github.com/Jemoka/SimVBG.git

2. add external SimVBG repository as an editable dependency
> uv add --editable /path/to/SimVBG

- To use GPT5 as judge you will need an open ai api key:
> export OPENAI_API_KEY="YOUR-KEY-HERE"

- to use bbq prompts to init simulated user-chatbot conversations:
1. clone BBQ repo in a seperate folder:
> git clone https://github.com/nyu-mll/BBQ.git 

2. update the path to the data in your cloned repo in core/data.py (line 21)
DEFAULT_BBQ_DATA_DIR = Path("..START_OF_YOUR_PATH_HERE../BBQ/data")


To run CEM:
1. Choose your settings in run/cem_settings.py and in run/run_cem.py
2. Run "run_cem.py" with 
> uv run python -m run.run_cem