import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer

class MultiHeadAttention:
    def __init__(self, num_heads, d_model, scope_name="multi_head_attention", dropout=0.1, temperature=1.0, use_layer_norm=False, use_residual=False):
        self.num_heads = num_heads
        self.d_model = d_model
        self.scope_name = scope_name
        
        assert d_model % num_heads == 0, f"d_model {d_model} must be divisible by num_heads {num_heads}"
        self.depth = d_model // num_heads
        
        self.dropout_rate = dropout
        self.use_layer_norm = use_layer_norm
        
        self.temperature = temperature
        self.use_residual = use_residual
        
        # Create variables under a unique scope
        with tf.variable_scope(self.scope_name, reuse=tf.AUTO_REUSE):
            if use_layer_norm:
                self.layer_norm = tf.contrib.layers.layer_norm
            self.input_dense = tf.layers.Dense(d_model, use_bias=True)
            # Linear layers for Q, K, V projections
            self.wq = tf.get_variable('wq', [d_model, d_model], 
                                    initializer=tf.contrib.layers.xavier_initializer())
            self.wk = tf.get_variable('wk', [d_model, d_model],
                                    initializer=tf.contrib.layers.xavier_initializer())
            self.wv = tf.get_variable('wv', [d_model, d_model],
                                    initializer=tf.contrib.layers.xavier_initializer())
            
            self.dense = tf.get_variable('dense', [d_model, d_model],
                                        initializer=tf.contrib.layers.xavier_initializer())

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def __call__(self, q, k, v, mask=None, training=True):
        with tf.variable_scope(self.scope_name, reuse=tf.AUTO_REUSE):
            batch_size = tf.shape(q)[0]
            
            # Project inputs to d_model dimension if needed
            q = self.input_dense(q) 
            k = self.input_dense(k)
            v = self.input_dense(v)
            
            # Store input for residual connection
            residual = q
            
            # Reshape inputs to match expected dimensions
            #q = tf.reshape(q, [batch_size, -1, self.d_model])
            #k = tf.reshape(k, [batch_size, -1, self.d_model])
            #v = tf.reshape(v, [batch_size, -1, self.d_model])
            
            # Linear projections
            q = tf.tensordot(q, self.wq, axes=[[2], [0]])  # (batch_size, seq_len_q, d_model)
            k = tf.tensordot(k, self.wk, axes=[[2], [0]])  # (batch_size, seq_len_k, d_model)
            v = tf.tensordot(v, self.wv, axes=[[2], [0]])  # (batch_size, seq_len_v, d_model)
            
            # Split into heads
            q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
            k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
            v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
            
            # Scaled dot-product attention
            scaled_attention = tf.matmul(q, k, transpose_b=True)  # (batch_size, num_heads, seq_len_q, seq_len_k)
            dk = tf.cast(tf.shape(k)[-1], tf.float32)
            scaled_attention = scaled_attention / tf.sqrt(dk)
            # Scale attention scores with temperature
            scaled_attention = scaled_attention / self.temperature
            
            if mask is not None:
                scaled_attention += (mask * -1e9)
            
            attention_weights = tf.nn.softmax(scaled_attention, dim=-1)
            # Apply dropout using tf.cond
            def apply_dropout():
                return tf.nn.dropout(attention_weights, keep_prob=1-self.dropout_rate)
                
            def no_dropout():
                return attention_weights
            
            # Convert training flag to tensor
            training_tensor = tf.cast(training, tf.bool)
                
            attention_weights = tf.cond(
                training_tensor,
                true_fn=apply_dropout,
                false_fn=no_dropout
            )
            #attention_weights = tf.nn.dropout(
            #    attention_weights, 
            #    keep_prob=1-self.dropout_rate if training else 1.0
            #)
            
            # Ensure attention weights have consistent shape [batch_size, max_n_msgs]
            attention_weights_reduced = tf.reduce_mean(attention_weights, axis=[1])  # Average over heads
            
            # Compute attention output
            output = tf.matmul(attention_weights, v)  # (batch_size, num_heads, seq_len_q, depth)
            
            # Combine heads
            output = tf.transpose(output, [0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
            output = tf.reshape(output, (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
            
            output = tf.tensordot(output, self.dense, axes=[[2], [0]])  # (batch_size, seq_len_q, d_model)
            
            # Apply layer normalization
            if self.use_layer_norm:
                output = self.layer_norm(output)
            
            # Add residual connection if shapes match
            if self.use_residual and residual.shape[-1] == output.shape[-1]:
                output += residual
            
            # Return reduced attention weights for visualization
            attention_weights_reduced = tf.reduce_mean(attention_weights, axis=[1])
                
            return output, attention_weights_reduced