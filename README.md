# pairwise-argument-ranking

***-----------------This is the pairwise argument ranking model developed by Kevin Shi.-----------------***

To run the program, run the following in terminal:

	* python main.py [-do_train] [-do_eval]
	* -do_train enables training
	* -do_eval enables evaluation

***IMPORTANT***

Most of the attributes affecting training are in the config.ini file. These are the important variables hard-coded into the main program that SHOULD BE MOVED OVER TO THE CONFIG FILE by me at some point:

	- train_test_split(0.1): Split hard coded to be 90% train and 10% validation
	- logging_strategy: Set to steps
	- model_path: Directory to save and load model to and from is hard coded here.
	- os.environ['CUDA_VISIBLE_DEVICES']: Currently set to 4 for my machine, but should be variable in config.
	- rank and world_size: Both are set to 4 to match the number of visible GPUs. Should be made dependent on length of visible GPU list.
	
*proto_main.py is my development file; main.py is the last stable version.*
