target: pre-req1 pre-req2 pre-req3
	recipes
	...

setup: requirements.txt environment.yml
	pip install -r requirements.txt
	conda env create -f environment.yml
	python -m pip install pyyaml==5.1
	python -m pip install torch torchvision torchaudio
	python -m pip install opencv-python
	python -m pip install scikit-image
	python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

clean: 
	find . -name '__pycache__' -type d | xargs rm -fr
	find . -name '.ipynb_checkpoints' -type d | xargs rm -fr
	find . -name '*.pyc' -delete

help:
	@echo "  clean                       remove *.pyc files, __pycache__ and .ipynb_checkpoints directory"
	@echo "  setup                       install dependencies and prepare environment"

csd_data_dir := ./autotuning/data/csd
config_file := ./autotuning/coarse_tuning/src/config.py

check_config_file:
	if [ -f "$(csd_data_dir)/config_copy.py" ]; then \
		if diff "$(config_file)" "$(csd_data_dir)/config_copy.py" > /dev/null; then \
			echo "Config files are identical"; \
		else \
			rm -rf "$(csd_data_dir)/processed"; \
			rm "$(csd_data_dir)/config_copy.py"; \
			cp "$(config_file)" "$(csd_data_dir)/config_copy.py"; \
			echo "Files are different, deleting the processed directory."; \
		fi; \
	else \
		echo "File doesn't exist"; \
		cp "$(config_file)" "$(csd_data_dir)/config_copy.py"; \
	fi

download_qflow_lite_data: clean
	mkdir -p $(csd_data_dir)
	mkdir -p $(csd_data_dir)/raw
	wget -P $(csd_data_dir) -nc https://data.nist.gov/od/ds/66492819760D3FF6E05324570681BA721894/data_qflow_lite.zip
	unzip -n $(csd_data_dir)/data_qflow_lite.zip -d $(csd_data_dir) > /dev/null
	mv -v $(csd_data_dir)/data_qflow_lite/* $(csd_data_dir)/raw > /dev/null

download_qflow_v2_data: clean
	mkdir -p $(csd_data_dir)
	mkdir -p $(csd_data_dir)/raw
	wget -P $(csd_data_dir) -nc https://data.nist.gov/od/ds/66492819760D3FF6E05324570681BA721894/data_qflow_v2.zip
	unzip -n $(csd_data_dir)/data_qflow_v2.zip -d $(csd_data_dir) > /dev/null
	# mv -v $(csd_data_dir)/data_qflow_v2/simulated/sim_normal/* $(csd_data_dir)/raw > /dev/null
	# mv -v $(csd_data_dir)/data_qflow_v2/simulated/sim_uniform/* $(csd_data_dir)/raw > /dev/null
	mv -v $(csd_data_dir)/data_qflow_v2/simulated/noiseless_data.hdf5 $(csd_data_dir)/raw > /dev/null
	mv -v $(csd_data_dir)/data_qflow_v2/experimental/exp_large/* $(csd_data_dir)/raw > /dev/null
	mv -v $(csd_data_dir)/data_qflow_v2/experimental/exp_small/dataset_*/*.npy $(csd_data_dir)/raw > /dev/null

convert_hdf5_to_npy: download_qflow_lite_data download_qflow_v2_data check_config_file 
	python ./autotuning/coarse_tuning/src/converter.py $(csd_data_dir) 
	rm $(csd_data_dir)/raw/noiseless_data.hdf5

process_training_data: convert_hdf5_to_npy
	python ./autotuning/coarse_tuning/src/process.py $(csd_data_dir) 
	rm -rf $(csd_data_dir)/data_qflow_v2
	rm -rf $(csd_data_dir)/data_qflow_lite
	rm -rf $(csd_data_dir)/__MACOSX

annotate_training_data: process_training_data
	python ./autotuning/coarse_tuning/src/annotate_data.py $(csd_data_dir) > /dev/null

training_data: process_training_data annotate_training_data

coarse_tuning: training_data 
	python ./autotuning/coarse_tuning/src/train_model.py $(csd_data_dir)
