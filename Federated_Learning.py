# ==============================
# The main function that to train model.
# ==============================


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import math
import time

from Helper_Functions import Vname_to_FeedPname, Vname_to_Pname, check_validaity_of_FLAGS, create_save_dir, \
    global_step_creator, load_from_directory_or_initialize, bring_Accountant_up_to_date, save_progress, \
    WeightsAccountant, print_loss_and_accuracy, print_new_comm_round, PrivAgent, Flag
import randomized_response


def run_federated_learning(loss, train_op, eval_correct, data, data_placeholder,
                                                   label_placeholder, privacy_agent=None, b=10, e=4,
                                                   record_privacy=False, m=0, sigma=0, eps=8, save_dir=None,
                                                   log_dir=None, max_comm_rounds=3000, rr=True, la=True, gm=False,
                                                   saver_func=create_save_dir, save_params=False):

    """
    :param loss:                TENSORFLOW node that computes the current loss.
    :param train_op:            TENSORFLOW Training_op.
    :param eval_correct:        TENSORFLOW node that evaluates the number of correct predictions.
    :param data:                A class instance with attributes:
    :param data_placeholder:    The placeholder from the tensorflow graph that is used to feed the model with data.
    :param label_placeholder:   The placeholder from the tensorflow graph that is used to feed the model with labels.
    :param privacy_agent:       A class instance that has callabels .get_m(r) .get_Sigma(r) .get_bound(), where r is the
                                communication round.
    :param b:                   Batchsize.
    :param e:                   Epochs to run on each client.
    :param record_privacy:      Whether to record the privacy or not.
    :param m:                   If specified, a privacyAgent is not used, instead the parameter is kept constant.
    :param sigma:               If specified, a privacyAgent is not used, instead the parameter is kept constant.
    :param eps:                 The epsilon for epsilon-delta privacy.
    :param save_dir:            Directory to store the process.
    :param log_dir:             Directory to store the graph.
    :param max_comm_rounds:     The maximum number of allowed communication rounds.
    :param rr:                  Whether to use a randomised response mechanism or not.
    :param la:                  Whether to use a local adaptive DP mechanism or not.
    :param gm:                  Whether to use a CDP Gaussian Mechanism or not.
    :param saver_func:          A function that specifies where and how to save progress: Note that the usual tensorflow
                                tracking will not work.
    :param save_params:         save all weights_throughout training.
    """

    # If no privacy agent was specified, the default privacy agent is used.
    if not privacy_agent:
        privacy_agent = PrivAgent(len(data.client_set), 'default_agent')

    FLAGS = Flag(len(data.client_set), b, e, record_privacy, m, sigma, eps, save_dir, log_dir, max_comm_rounds, rr,
                                                                    la, gm, privacy_agent)

    FLAGS = check_validaity_of_FLAGS(FLAGS)

    save_dir = saver_func(FLAGS)

    increase_global_step, set_global_step = global_step_creator()

    model_placeholder = dict(zip([Vname_to_FeedPname(var) for var in tf.trainable_variables()],
                                 [tf.placeholder(name=Vname_to_Pname(var),
                                                 shape=var.shape,
                                                 dtype=tf.float32)
                                  for var in tf.trainable_variables()]))

    assignments = [tf.assign(var, model_placeholder[Vname_to_FeedPname(var)]) for var in tf.trainable_variables()]

    model, loss_accountant, accuracy_accountant, delta_accountant, acc, real_round, FLAGS, computed_deltas = \
                                                                 load_from_directory_or_initialize(save_dir, FLAGS)

    m = int(FLAGS.m)

    sigma = float(FLAGS.sigma)
    # - m : amount of clients participating in a round
    # - sigma : variable for the Gaussian Mechanism.
    # Both will only be used if no Privacy_Agent is deployed.

    init = tf.global_variables_initializer()

    sess = tf.Session()

    sess.run(init)

    # If there was no loadable model, we initialize a model:
    if not model:
        model = dict(zip([Vname_to_FeedPname(var) for var in tf.trainable_variables()],
                         [sess.run(var) for var in tf.trainable_variables()]))

        model['global_step_placeholder:0'] = 0

        real_round = 0

        weights_accountant = []

    if real_round > 0:
        
        bring_Accountant_up_to_date(acc, sess, real_round, privacy_agent, FLAGS)

    # This is where the actual communication rounds start:
    data_set_asarray = np.asarray(data.sorted_x_train)
    label_set_asarray = np.asarray(data.sorted_y_train)


    #rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
    for r in xrange(FLAGS.max_comm_rounds):
        start = time.clock()
        if not (FLAGS.loaded and r == 0):
            sess.run(assignments, feed_dict=model)
            feed_dict = {str(data_placeholder.name): np.asarray(data.x_vali),
                         str(label_placeholder.name): np.asarray(data.y_vali)}
            # compute the loss on the validation set.
            global_loss = sess.run(loss, feed_dict=feed_dict)
            count = sess.run(eval_correct, feed_dict=feed_dict)
            accuracy = float(count) / float(len(data.y_vali))
            loss_accountant.append(global_loss)
            accuracy_accountant.append(accuracy)
            print_loss_and_accuracy(global_loss, accuracy)
        if record_privacy:
            if delta_accountant[-1] > privacy_agent.get_bound() or math.isnan(delta_accountant[-1]):
                print('************** The last step exhausted the privacy budget **************')
                print("Loss:")
                print(loss_accountant,"\n")
                print("Accuracy:")
                print(accuracy_accountant,"\n")
                print("delta:")
                print(delta_accountant, "\n")
                if not math.isnan(delta_accountant[-1]):
                    save_progress(save_dir, model, delta_accountant + [float('nan')], loss_accountant + [float('nan')],
                                                            accuracy_accountant + [float('nan')], privacy_agent, FLAGS)

                    return loss_accountant, accuracy_accountant, delta_accountant, model

            else:
                save_progress(save_dir, model, delta_accountant, loss_accountant, accuracy_accountant, privacy_agent,
                                                                                                                 FLAGS)

        else:
            if real_round >= 1:
                print('************** The model is convergent **************')
                print("Loss:")
                print(loss_accountant, "\n")
                print("Accuracy:")
                print(accuracy_accountant, "\n")
                save_progress(save_dir, model, delta_accountant + [float('nan')], loss_accountant + [float('nan')],
                                                   accuracy_accountant + [float('nan')], privacy_agent, FLAGS)
                return loss_accountant, accuracy_accountant, delta_accountant, model
            else:
                save_progress(save_dir, model, delta_accountant, loss_accountant, accuracy_accountant,
                                                                                             privacy_agent, FLAGS)
        ############################################################################################################
        # Start of a new communication round
        real_round = real_round + 1
        print_new_comm_round(real_round)
        if FLAGS.priv_agent:
            m = int(privacy_agent.get_m(int(real_round)))
            sigma = privacy_agent.get_Sigma(int(real_round))

        perm = np.random.permutation(FLAGS.n)
        s = perm[0:m].tolist()

        _m = m
        em = m
        
        if rr:
        # Randomly choose a total of m (out of n) client-indices that participate in this round.
        # Random response.
            em, _m, s = randomized_response.RR(FLAGS.n,m,s,eps)

        participating_clients = [data.client_set[k] for k in s]

        print('Clients participating: ' + str(_m))

        ds = []

        for c in range(_m):
            sess.run(assignments + [set_global_step], feed_dict=model)
            data_ind = np.split(np.asarray(participating_clients[c]), FLAGS.b, 0)
            ds.append(len(data_ind))
            for e in xrange(int(FLAGS.e)):
                # e = Epoch
                for step in xrange(len(data_ind)):
                    real_step = sess.run(increase_global_step)
                    batch_ind = data_ind[step]
                    feed_dict = {str(data_placeholder.name): data_set_asarray[[int(j) for j in batch_ind]],
                                 str(label_placeholder.name): label_set_asarray[[int(j) for j in batch_ind]]}
                    _ = sess.run([train_op], feed_dict=feed_dict)



            if c == 0:
                weights_accountant = WeightsAccountant(sess, model, sigma, real_round)
            else:
                weights_accountant.allocate(sess)
        # End of a communication round
        ############################################################################################################
        end = time.clock()
        print('............Communication round %s completed. Its running time: %s s............'
                                                                                %(str(real_round),(end-start)))

        model, delta = weights_accountant.Update_via_RR_GaussianMechanism(sess, rr, em, la, ds, gm, acc, FLAGS, computed_deltas)


        delta_accountant.append(delta)
        model['global_step_placeholder:0'] = real_step
        print('Epsilon-Delta:' + str([FLAGS.eps, delta]))
        if save_params:
            weights_accountant.save_params(save_dir)

    #rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
