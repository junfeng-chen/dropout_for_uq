# dropout_for_uq
a toy demonstration of uncertainty quantification in neural networks by dropout method.

The implementation is based on Keras sequential API. To activate dropout in the predictions, modify the source code of Keras as follows:

In class Dropout(in layers/core.py), function def call(self, inputs, training=None), change "return K.in_train_phase(dropped_inputs, inputs, training=training)" to "return K.in_train_phase(dropped_inputs, dropped_inputs, training=training)"
