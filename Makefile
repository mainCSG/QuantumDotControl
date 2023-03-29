target: pre-req1 pre-req2 pre-req3
	recipes
	...

setup: requirements.txt
	pip install -r requirements.txt
	conda env create -f environment.yml

clean: 
	find . -name '__pycache__' -type d | xargs rm -fr
	find . -name '.ipynb_checkpoints' -type d | xargs rm -fr
	find . -name '*.pyc' -delete

help:
	@echo "  clean                       remove *.pyc files, __pycache__ and .ipynb_checkpoints directory"
	@echo "  setup                       install dependencies and prepare environment"