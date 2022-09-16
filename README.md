# pairwise-argument-ranking

***-----------------This is the pairwise argument ranking model developed by Kevin Shi.-----------------***

To run the program, run the following in terminal:

	* python main.py [-do_train] [-do_eval]
	* -do_train enables training
	* -do_eval enables evaluation

***IMPORTANT***

Most of the attributes affecting training are in the config.ini file. These are the important variables hard-coded into the main program that SHOULD BE MOVED OVER TO THE CONFIG FILE:

	- train_test_split(0.1): Split hard coded to be 90% train and 10% validation
	- logging_strategy: Set to steps
	- model_path: Directory to save and load model to and from is hard coded here.
	
*proto_main.py is my development file; main.py is the last stable version.*
