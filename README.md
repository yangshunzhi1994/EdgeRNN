# EdgeRNN: A Compact Speech Recognition Network with Spatio-temporal Features for Edge Computing

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

The thop used in the project is a statistical method of parameters and calculations, which can be installed by the following method:https://github.com/Lyken17/pytorch-OpCounter. It is worth noting that the thop cannot count the calculation amount and parameter amount of RNN/LSTM/GRU. You have to calculate this part manually. To calculate the complexity of the RNN, you can view the following tutorial: 

      1.http://www.wildml.com/2015/10/recurrent-neural-network-tutorial-part-4-implementing-a-grulstm-rnn-with-python-and-theano/
      2.https://github.com/NVIDIA-developer-blog/code-samples/issues/7
      3.https://datascience.stackexchange.com/questions/10615/number-of-parameters-in-an-lstm-model
      4.https://towardsdatascience.com/counting-no-of-parameters-in-deep-learning-models-by-hand-8f1716241889

The data augmentation uses the audiomentsations library, which can be installed by:https://github.com/iver56/audiomentations

Corresponding papers for this project：

       Yang S, Gong Z, Ye K, et al. EdgeRNN: A Compact Speech Recognition Network with Spatio-temporal Features 
       for Edge Computing[J]. IEEE Access, 2020.
