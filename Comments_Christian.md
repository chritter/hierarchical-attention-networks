# Documentation - Hirarchical Attention Networks

## Comments

* Words per documents are by default limited to 1500 words
* embedding initialized through placeholder, lookup
* chosen TF GRU implementation is not GPU-optimized (Please use tf.contrib.cudnn_rnn.CudnnGRU for better performance on GPU)
* 