This is the Pytorch code for our proposed SubPixel-BackProjection Network.
Training code will be released soon.

Dependencies:

	python 3.x
	pytorch 1.1.0
	cuda10
	torch
	torchvision
	scikit-image
	pillow
	pyyaml
	visdom
	tqdm
	opencv

	# Create virtual environment
	conda create -n sr_env

	# Install torch
	conda install pytorch torchvision cudatoolkit=10.0 -c pytorch

	# Install skimage
	conda install -c conda-forge scikit-image

	# Install visdom
	conda install -c conda-forge visdom

	# Install pyyaml
	conda install -c conda-forge pyyaml

	# Install tqdm
	conda install -c conda-forge tqdm

	conda install -c conda-forge oprncv

Datasets:

	This project contains 2/4 benchmark datasets Set5 and Set14 due to file size limitation.

	All the benchmark datasets can be downloaded from: http://vllab.ucmerced.edu/wlai24/LapSRN/results/SR_testing_datasets.zip

	To test BSDS100 and Urban100, check the directory options/test/ for SPBP_S.yaml, SPBP_M.yaml, SPBP_L.yaml and SPBP_L+.yaml 
	and add the following snippet under the 'datasets:'

	  test_3:  # the 2st test dataset
	    name: BSDS100
	    data_location: data/datasets/BSDS100/
	    shuffle: false
	    n_workers: 1  # per GPU
	    batch_size: 1
	    repeat: 1
	  test_4:  # the 2st test dataset
	    name: Urban100
	    data_location: data/datasets/Urban100/
	    shuffle: false
	    n_workers: 1  # per GPU
	    batch_size: 1
	    repeat: 1

Testing:

	To run the test, either python test.py -config options/test/CONFIG.yaml can be used or simply run the test_run.sh file.
