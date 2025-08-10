# dtdp-non-projectivity
## Usage
- Clone the repository and cd into it
- make a venv: `python -m venv venv` and activate it `source venv/bin/activate`
- install the required libraries `pip install -r requirements.txt`
- run `parser_comparison.py` with some args, e.g., `python3 parser_comparison.py --file "english" --lang "en" --model "all"` to find all test sets in `UD_ROOT_PATH` and evaluate all english models. To run a specific conllu file, pass it with the file argument.