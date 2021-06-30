import time

import tensorflow as tf

from garage.algos import RLAlgorithm
import garage.misc.logger as logger
from garage.tf.plotter import Plotter
from garage.tf.samplers import BatchSampler
from garage.tf.samplers import OnPolicyVectorizedSampler
from garage.misc.overrides import overrides

class RandomSearch(RLAlgorithm):

    def __init__(self,
                 env,
                 policy,
                 n_itr=500,
                 start_itr=0,
                 batch_size=5000,
                 max_path_length=500,
                 top_paths=None,
                 plot=False,
                 force_batch_sampler=False,
                 sampler_cls=None,
                 sampler_args=None,
                 **kwargs):
        """
        :param env: Environment
        :param policy: Policy
        :type policy: Policy
        :param n_itr: Number of iterations.
        :param start_itr: Starting iteration.
        :param batch_size: Number of samples per iteration.
        :param max_path_length: Maximum length of a single rollout.
        :param plot: Plot evaluation run after each iteration.
        :return:
        """
        self.env = env
        self.policy = policy
        self.n_itr = n_itr
        self.start_itr = start_itr
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.top_paths = top_paths
        self.plot = plot

        if sampler_cls is None:
            if self.policy.vectorized and not force_batch_sampler:
                sampler_cls = OnPolicyVectorizedSampler
            else:
                sampler_cls = BatchSampler
        if sampler_args is None:
            sampler_args = dict()
        self.sampler = sampler_cls(self, **sampler_args)
        self.init_opt()

    def start_worker(self, sess):
        self.sampler.start_worker()
        if self.plot:
            self.plotter = Plotter(self.env, self.policy, sess)
            self.plotter.start()

    def shutdown_worker(self):
        self.sampler.shutdown_worker()
        if self.plot:
            self.plotter.close()

    def obtain_samples(self, itr):
        return self.sampler.obtain_samples(itr)

    def train(self, sess=None):
        created_session = True if (sess is None) else False
        if sess is None:
            sess = tf.Session()
            sess.__enter__()

        sess.run(tf.global_variables_initializer())
        self.start_worker(sess)
        start_time = time.time()
        last_average_return = None
        for itr in range(self.start_itr, self.n_itr):
            itr_start_time = time.time()
            with logger.prefix('itr #%d | ' % itr):
                logger.log("Obtaining samples...")
                paths = self.obtain_samples(itr)

                undiscounted_returns = [sum(path["rewards"]) for path in paths]
                if not (self.top_paths is None):
                    action_seqs = [path["actions"] for path in paths]
                    [self.top_paths.enqueue(action_seq,R,make_copy=True) for (action_seq,R) in zip(action_seqs,undiscounted_returns)]

                logger.log("Saving snapshot...")
                params = self.get_itr_snapshot(itr, paths)
                if self.top_paths is not None:
                    top_paths = dict()
                    for (topi, path) in enumerate(self.top_paths):
                        top_paths[path[1]] = path[0]
                    params['top_paths'] = top_paths

                logger.save_itr_params(itr, params)
                logger.log("Saved")
                logger.record_tabular('Time', time.time() - start_time)
                logger.record_tabular('ItrTime', time.time() - itr_start_time)
                logger.record_tabular('Itr',itr)
                logger.record_tabular('StepNum',int((itr+1)*self.batch_size))
                if self.top_paths is not None:
                    for (topi, path) in enumerate(self.top_paths):
                        logger.record_tabular('reward '+str(topi), path[1])
                logger.dump_tabular(with_prefix=False)     
                if self.plot:
                    self.plotter.update_plot(self.policy, self.max_path_length)
                    if self.pause_for_plot:
                        input("Plotting evaluation run: Press Enter to "
                              "continue...")

        self.shutdown_worker()
        if created_session:
            sess.close()
        return last_average_return

    def init_opt(self):
        """
        Initialize the optimization procedure. If using tensorflow, this may
        include declaring all the variables and compiling functions
        """
        pass

    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            env=self.env,
        )
