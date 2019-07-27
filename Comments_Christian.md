# Documentation - Hirarchical Attention Networks

## Comments

* Words per documents are by default limited to 1500 words
* embedding initialized through placeholder, lookup
* chosen TF GRU implementation is not GPU-optimized (Please use tf.contrib.cudnn_rnn.CudnnGRU for better performance on GPU)
* Interesting implementation of tqdm inside the batch iterator

## Datasets

* IMDB has lowest number of documents and highest average number of words of 326
* Distribution of classes for IMDB set
    * 10: 51k
    * 9:31k
    * 8: 46k
    * 7: 41k
    * 6: 29k
    * 5: 21k
    * 4: 14k
    * 3: 12k
    * 2: 10k
    * 1: 18k



## Free parameteres

### Default settings
* Yelp15: Total trainable parameters: 216505, 74800 params from enbeddings alone!
* IMDB: Total trainable parameters: 18.087.010 !! 17.944.800 from embeddings
* embedding training
 
====================================================================================================

embedding/embedding_matrix:0                                                          74800 params
----------------------------------------------------------------------------------------------------
word-encoder/init_state_fw:0                                                             50 params
----------------------------------------------------------------------------------------------------
word-encoder/init_state_bw:0                                                             50 params
----------------------------------------------------------------------------------------------------
word-encoder/fw/cell_fw/gates/kernel:0                                                25000 params
----------------------------------------------------------------------------------------------------
word-encoder/fw/cell_fw/gates/bias:0                                                    100 params
----------------------------------------------------------------------------------------------------
word-encoder/fw/cell_fw/candidate/kernel:0                                            12500 params
----------------------------------------------------------------------------------------------------
word-encoder/fw/cell_fw/candidate/bias:0                                                 50 params
----------------------------------------------------------------------------------------------------
word-encoder/bw/cell_bw/gates/kernel:0                                                25000 params
----------------------------------------------------------------------------------------------------
word-encoder/bw/cell_bw/gates/bias:0                                                    100 params
----------------------------------------------------------------------------------------------------
word-encoder/bw/cell_bw/candidate/kernel:0                                            12500 params
----------------------------------------------------------------------------------------------------
word-encoder/bw/cell_bw/candidate/bias:0                                                 50 params
----------------------------------------------------------------------------------------------------
word-encoder/attention/att_W:0                                                          100 params
----------------------------------------------------------------------------------------------------
word-encoder/attention/projection/kernel:0                                            10000 params
----------------------------------------------------------------------------------------------------
word-encoder/attention/projection/bias:0                                                100 params
----------------------------------------------------------------------------------------------------
sent-encoder/init_state_fw:0                                                             50 params
----------------------------------------------------------------------------------------------------
sent-encoder/init_state_bw:0                                                             50 params
----------------------------------------------------------------------------------------------------
sent-encoder/fw/cell_fw/gates/kernel:0                                                15000 params
----------------------------------------------------------------------------------------------------
sent-encoder/fw/cell_fw/gates/bias:0                                                    100 params
----------------------------------------------------------------------------------------------------
sent-encoder/fw/cell_fw/candidate/kernel:0                                             7500 params
----------------------------------------------------------------------------------------------------
sent-encoder/fw/cell_fw/candidate/bias:0                                                 50 params
----------------------------------------------------------------------------------------------------
sent-encoder/bw/cell_bw/gates/kernel:0                                                15000 params
----------------------------------------------------------------------------------------------------
sent-encoder/bw/cell_bw/gates/bias:0                                                    100 params
----------------------------------------------------------------------------------------------------
sent-encoder/bw/cell_bw/candidate/kernel:0                                             7500 params
----------------------------------------------------------------------------------------------------
sent-encoder/bw/cell_bw/candidate/bias:0                                                 50 params
----------------------------------------------------------------------------------------------------
sent-encoder/attention/att_W:0                                                          100 params
----------------------------------------------------------------------------------------------------
sent-encoder/attention/projection/kernel:0                                            10000 params
----------------------------------------------------------------------------------------------------
sent-encoder/attention/projection/bias:0                                                100 params
----------------------------------------------------------------------------------------------------
classifier/logits/kernel:0                                                              500 params
----------------------------------------------------------------------------------------------------
classifier/logits/bias:0                                                                  5 params
----------------------------------------------------------------------------------------------------
====================================================================================================
Total trainable parameters: 216505
====================================================================================================
