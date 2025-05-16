#!/usr/local/bin/python
import os
import tensorflow as tf
import metrics as metrics
import stat_logger as stat_logger
from DataPipe import DataPipe
from ConfigLoader import logger
import pdb
import re
from src.sep_main.explain_module.util import summarize_trial


class Executor:

    def __init__(self, model, silence_step=200, skip_step=20):
        self.model = model
        self.silence_step = silence_step
        self.skip_step = skip_step
        self.pipe = DataPipe()

        self.saver = tf.train.Saver()
        self.tf_config = tf.ConfigProto(allow_soft_placement=True)
        self.tf_config.gpu_options.allow_growth = True
        self.explanation_history = []  # Store explanation history
        self.reflection_metrics = {
            'correct_predictions': 0,
            'total_predictions': 0,
            'improvement_rate': 0.0
        }
        
    def _log_explanations_and_reflections(self, explanations, reflections, predictions, actual):
        """Logs generated explanations and reflections"""
        logger.info("\n=== Prediction Analysis ===")
        
        for i, (expl, refl, pred, act) in enumerate(zip(explanations, reflections, predictions, actual)):
            logger.info(f"\nSample {i+1}:")
            logger.info(f"Prediction: {'Up' if pred[0] > 0.5 else 'Down'} ({pred[0]:.3f})")
            logger.info(f"Actual: {'Up' if act[0] > 0.5 else 'Down'}")
            logger.info(f"Explanation: {expl}")
            
            if refl:
                logger.info("Reflection:")
                logger.info(f"- Error Analysis: {refl['error_analysis']}")
                logger.info(f"- Proposed Improvement: {refl['proposed_improvements']}")
            
            # Update metrics
            is_correct = (pred[0] > 0.5) == (act[0] > 0.5)
            self.reflection_metrics['correct_predictions'] += int(is_correct)
            self.reflection_metrics['total_predictions'] += 1

    def _update_training_with_reflections(self, sess, feed_dict, reflections):
        """Updates model training based on reflections"""
        if reflections:
            # Add reflection-based loss to training
            reflection_feed = {
                self.model.reflection_inputs: reflections,
                **feed_dict
            }
            
            # Run optimization with reflection loss
            sess.run([self.model.optimize, self.model.reflection_loss], 
                    feed_dict=reflection_feed)


    def unit_test_train(self):
        with tf.Session() as sess:
            word_table_init = self.pipe.init_word_table()
            feed_table_init = {self.model.word_table_init: word_table_init}
            sess.run(tf.global_variables_initializer(), feed_dict=feed_table_init)
            logger.info('Word table init: done!')

            logger.info('Model: {0}, start a new session!'.format(self.model.model_name))

            n_iter = self.model.global_step.eval()

            # forward
            train_batch_loss_list = list()
            train_epoch_size = 0.0
            train_epoch_n_acc = 0.0
            train_batch_gen = self.pipe.batch_gen(phase='train')
            train_batch_dict = next(train_batch_gen)

            while n_iter < 100:
                feed_dict = {self.model.is_training_phase: True,
                             self.model.batch_size: train_batch_dict['batch_size'],
                             self.model.stock_ph: train_batch_dict['stock_batch'],
                             self.model.T_ph: train_batch_dict['T_batch'],
                             self.model.n_words_ph: train_batch_dict['n_words_batch'],
                             self.model.n_msgs_ph: train_batch_dict['n_msgs_batch'],
                             self.model.y_ph: train_batch_dict['y_batch'],
                             self.model.price_ph: train_batch_dict['price_batch'],
                             self.model.mv_percent_ph: train_batch_dict['mv_percent_batch'],
                             self.model.word_ph: train_batch_dict['word_batch'],
                             self.model.ss_index_ph: train_batch_dict['ss_index_batch'],
                             }

                ops = [self.model.y_T, self.model.y_T_,  self.model.loss, self.model.optimize]
                pdb.set_trace()
                train_batch_y, train_batch_y_,  train_batch_loss, _ = sess.run(ops, feed_dict)
                
                # training batch stat
                train_epoch_size += float(train_batch_dict['batch_size'])
                train_batch_loss_list.append(train_batch_loss)
                train_batch_n_acc = sess.run(metrics.n_accurate(y=train_batch_y, y_=train_batch_y_))
                train_epoch_n_acc += float(train_batch_n_acc)

                stat_logger.print_batch_stat(n_iter, train_batch_loss, train_batch_n_acc,
                                             train_batch_dict['batch_size'])
                n_iter += 1
    
    def _accumulate_results(self, predictions, actuals, loss, explanations):
        """
        Accumulates results during generation phase
        """
        # Store predictions and actuals
        self.y_list.append(predictions)
        self.y_list_.append(actuals)
        self.gen_loss_list.append(loss)

        # Calculate batch accuracy
        batch_size = len(predictions)
        n_correct = sum(1 for p, a in zip(predictions, actuals) 
                       if (p[0] > 0.5) == (a[0] > 0.5))
        
        self.gen_n_acc += n_correct
        self.gen_size += batch_size

        # Store explanations
        if explanations is not None:
            self.explanation_history.extend(explanations)
    
    def _compute_final_metrics(self):
        """
        Computes final metrics after generation phase
        """
        results = metrics.eval_res(
            n_acc=self.gen_n_acc,
            total=self.gen_size,
            loss_list=self.gen_loss_list,
            y_list=self.y_list,
            y_list_=self.y_list_,
            use_mcc=True
        )

        # Add explanation metrics if available
        if self.explanation_history:
            results['explanation_metrics'] = {
                'num_explanations': len(self.explanation_history),
                'avg_explanation_length': sum(len(e) for e in self.explanation_history) / len(self.explanation_history),
                'explanation_samples': self.explanation_history[:3]  # Store first 3 examples
            }

        return results

    def generation(self, sess, phase):
        generation_gen = self.pipe.batch_gen_by_stocks(phase)

        #gen_loss_list = list()
        #gen_size, gen_n_acc = 0.0, 0.0
        #y_list, y_list_ = list(), list()

        
        for gen_batch_dict in generation_gen:

            feed_dict = {self.model.is_training_phase: False,
                         self.model.batch_size: gen_batch_dict['batch_size'],
                         self.model.stock_ph: gen_batch_dict['stock_batch'],
                         self.model.T_ph: gen_batch_dict['T_batch'],
                         self.model.n_words_ph: gen_batch_dict['n_words_batch'],
                         self.model.n_msgs_ph: gen_batch_dict['n_msgs_batch'],
                         self.model.y_ph: gen_batch_dict['y_batch'],
                         self.model.price_ph: gen_batch_dict['price_batch'],
                         self.model.mv_percent_ph: gen_batch_dict['mv_percent_batch'],
                         self.model.word_ph: gen_batch_dict['word_batch'],
                         self.model.ss_index_ph: gen_batch_dict['ss_index_batch'],
                         self.model.dropout_mel_in: 0.0,
                         self.model.dropout_mel: 0.0,
                         self.model.dropout_ce: 0.0,
                         self.model.dropout_vmd_in: 0.0,
                         self.model.dropout_vmd: 0.0,
                         }

            # Run model with explanations
            results = sess.run([
                self.model.y_T,
                self.model.y_T_,
                self.model.loss,
                self.model.explanation,
                self.model.vos_attention
            ], feed_dict=feed_dict)
            
            predictions, actuals, loss, explanations, vos_attention = results

            # Log generated explanations
            self._log_explanations_and_reflections(
                explanations=explanations,
                reflections=None,  # No reflections in generation phase
                predictions=predictions,
                actual=actuals
            )

            # Store results
            self._accumulate_results(
                predictions, 
                actuals, 
                loss, 
                explanations
            )

        return self._compute_final_metrics()
    
    def train_and_dev(self):
        with tf.Session(config=self.tf_config) as sess:
            # prep: writer and init
            sanitized_path = re.sub(r'[<>:"/\\|?*;]', '_', self.model.tf_graph_path)
            #os.makedirs(os.path.dirname(sanitized_path), exist_ok=True)
            writer = tf.summary.FileWriter(sanitized_path, sess.graph)

            # init all vars with tables
            feed_table_init = {self.model.word_table_init: self.pipe.init_word_table()}
            sess.run(tf.global_variables_initializer(), feed_dict=feed_table_init)
            logger.info('Word table init: done!')

            # prep: checkpoint
            checkpoint = tf.train.get_checkpoint_state(os.path.dirname(self.model.tf_checkpoint_file_path))
            if checkpoint and checkpoint.model_checkpoint_path:
                # restore partial saved vars
                reader = tf.train.NewCheckpointReader(checkpoint.model_checkpoint_path)
                restore_dict = dict()
                for v in tf.all_variables():
                    tensor_name = v.name.split(':')[0]
                    if reader.has_tensor(tensor_name):
                        print('has tensor: {0}'.format(tensor_name))
                        restore_dict[tensor_name] = v

                checkpoint_saver = tf.train.Saver(restore_dict)
                checkpoint_saver.restore(sess, checkpoint.model_checkpoint_path)
                logger.info('Model: {0}, session restored!'.format(self.model.model_name))
            else:
                logger.info('Model: {0}, start a new session!'.format(self.model.model_name))

            for epoch in range(self.model.n_epochs):
                logger.info('Epoch: {0}/{1} start'.format(epoch+1, self.model.n_epochs))

                # training phase
                train_batch_loss_list = list()
                epoch_size, epoch_n_acc = 0.0, 0.0

                train_batch_gen = self.pipe.batch_gen(phase='train')  # a new gen for a new epoch

                for train_batch_dict in train_batch_gen:

                    # logger.info('train: batch_size: {0}'.format(train_batch_dict['batch_size']))

                    feed_dict = {self.model.is_training_phase: True,
                                 self.model.batch_size: train_batch_dict['batch_size'],
                                 self.model.stock_ph: train_batch_dict['stock_batch'],
                                 self.model.T_ph: train_batch_dict['T_batch'],
                                 self.model.n_words_ph: train_batch_dict['n_words_batch'],
                                 self.model.n_msgs_ph: train_batch_dict['n_msgs_batch'],
                                 self.model.y_ph: train_batch_dict['y_batch'],
                                 self.model.price_ph: train_batch_dict['price_batch'],
                                 self.model.mv_percent_ph: train_batch_dict['mv_percent_batch'],
                                 self.model.word_ph: train_batch_dict['word_batch'],
                                 self.model.ss_index_ph: train_batch_dict['ss_index_batch'],
                                 }

                    #ops = [self.model.y_T, self.model.y_T_,  self.model.loss, self.model.optimize,self.model.global_step]
                    #train_batch_y, train_batch_y_,   train_batch_loss, _, n_iter = sess.run(ops, feed_dict)
                    #predictions = sess.run(self.model.y_T, feed_dict=feed_dict)
                    # Run model with VoS explanation components
                    ops = [
                        self.model.y_T, 
                        self.model.y_T_,
                        self.model.loss,
                        self.model.explanation,
                        self.model.reflection,
                        self.model.vos_attention,
                        self.model.global_step
                    ]
                    
                    results = sess.run(ops, feed_dict)
                    predictions, actuals, loss, explanations, reflections, vos_attention, n_iter = results

                    # Generate explanations for VoS
                    # Log explanations and reflections
                    self._log_explanations_and_reflections(
                        explanations=explanations,
                        reflections=reflections,
                        predictions=predictions,
                        actual=actuals
                    )

                    # Update training with reflections
                    self._update_training_with_reflections(
                        sess=sess,
                        feed_dict=feed_dict,
                        reflections=reflections
                    )
                    
                    # Save VoS attention patterns
                    if n_iter % self.skip_step == 0:
                        self._save_vos_analysis(
                            vos_attention=vos_attention,
                            explanations=explanations,
                            iter_num=n_iter
                        )

                # print training epoch stat
                self._print_epoch_stats(epoch)
                
                

        writer.close()

    def _save_vos_analysis(self, vos_attention, explanations, iter_num):
        """Saves VoS attention patterns and explanations"""
        analysis_path = os.path.join(
            self.model.tf_graph_path, 
            f'vos_analysis_{iter_num}.txt'
        )
        
        with open(analysis_path, 'w') as f:
            f.write("=== VoS Analysis ===\n")
            
            # Save attention weights
            f.write("\nAttention Patterns:\n")
            f.write(f"Temporal weights: {vos_attention['temporal_weights']}\n")
            f.write(f"Feature importance: {vos_attention['feature_importance']}\n")
            
            # Save explanations
            f.write("\nGenerated Explanations:\n")
            for i, expl in enumerate(explanations):
                f.write(f"\nPrediction {i+1}:\n{expl}\n")
                
    def _print_epoch_stats(self, epoch):
        """Prints epoch statistics including explanation metrics"""
        # Calculate standard metrics
        epoch_loss, epoch_acc = metrics.basic_train_stat(
            self.train_batch_loss_list, 
            self.epoch_n_acc, 
            self.epoch_size
        )
        
        # Calculate explanation metrics
        if self.reflection_metrics['total_predictions'] > 0:
            explanation_accuracy = (
                self.reflection_metrics['correct_predictions'] / 
                self.reflection_metrics['total_predictions']
            )
        else:
            explanation_accuracy = 0.0

        # Log statistics
        logger.info(f"\n=== Epoch {epoch+1} Statistics ===")
        logger.info(f"Loss: {epoch_loss:.4f}")
        logger.info(f"Accuracy: {epoch_acc:.4f}")
        logger.info(f"Explanation Accuracy: {explanation_accuracy:.4f}")
        logger.info(f"Improvements Made: {self.reflection_metrics['improvement_rate']:.4f}")
    
    def restore_and_test(self):
        with tf.Session(config=self.tf_config) as sess:
            checkpoint = tf.train.get_checkpoint_state(os.path.dirname(self.model.tf_checkpoint_file_path))
            if checkpoint and checkpoint.model_checkpoint_path:
                logger.info('Model: {0}, session restored!'.format(self.model.model_name))
                self.saver.restore(sess, checkpoint.model_checkpoint_path)
            else:
                logger.info('Model: {0}: NOT found!'.format(self.model.model_name))
                raise IOError

            res= self.generation(sess, phase='test')
            stat_logger.print_eval_res(res,use_mcc =True)


            
            