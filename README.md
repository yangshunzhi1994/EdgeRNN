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

COMMANDS_data.h5 is generated using preprocess_COMMANDS.py.
