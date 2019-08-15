# EdgeRNN：Efficient Speech Recognition Network for Edge Computing

This is an efficient speech recognition network. The paper has verified the efficiency of the network on the two tasks of speech emotion recognition and speech keyword recognition.

Speech emotion recognition uses the IEMOCAP dataset, which can be obtained from the following web page:https://sail.usc.edu/iemocap/release_form.php.

Speech keyword recognition uses the Google's Speech Commands dataset, which can be obtained from the following web page:http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz.

Decompress these two datasets and put them in the data directory. That is, the data folder contains the following files:

      ./data: 
             --IEMOCAP_full_release  folder
             --speech_commands_v0.01 folder
             --COMMANDS.py
             --COMMANDS_data.h5
             --IEMOCAP.py
             --IEMOCAP_data.h5
             

IEMOCAP_data.h5 is generated using preprocess_IEMOCAP.py.

COMMANDS_data.h5 is generated using preprocess_COMMANDS.py. It is worth noting that the COMMANDS dataset is very large. If you need to use data augmentation in your project, you must ensure that your computer's memory is larger than 48G. The memory of the computer of this project is 64G. You can increase your SWAP space by the following methods:http://smilejay.com/2012/09/new-or-add-swap/.
