import tensorflow as tf
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from typing import Union, Dict
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops


def get_optimizer(optimizer_name: str,
                  params: Dict,
                  learning_rate: Union[float, LearningRateSchedule] = 0.0001,
                  ) -> tf.keras.optimizers.Optimizer:
    """
    This function returns an optimizer.
    :param optimizer_name: String specifying the optimizer to use
    :param learning_rate: Float specifying the learning rate
    :param params: Dictionary containing the parameters for the optimizer
    """
    if optimizer_name == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate,
                                            momentum=params['momentum'], nesterov=True,
                                            decay=params['weight_decay'])
    elif optimizer_name == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, decay=params['weight_decay'])
    elif optimizer_name == 'adamax':
        optimizer = tf.keras.optimizers.Adamax(learning_rate=learning_rate, decay=params['weight_decay'])
    elif optimizer_name == 'nadam':
        optimizer = tf.keras.optimizers.Nadam(learning_rate=learning_rate)
    else:
        raise ValueError('Optimizer name not recognized.')
    return optimizer


class StepDecay(LearningRateSchedule):
    def __init__(self, steps_per_epoch: int, init_alpha: float = 0.01, factor: float = 0.25, drop_every: int = 10):
        super().__init__()

        self.init_lr = init_alpha
        self.factor = factor
        self.steps_per_epoch = steps_per_epoch
        self.drop_every = drop_every

    def __call__(self, step: int):
        step_int = math_ops.cast(step, tf.int32)

        init_lr = ops.convert_to_tensor_v2_with_dispatch(
            self.init_lr, name="init_lr"
        )

        steps_per_epoch = ops.convert_to_tensor_v2_with_dispatch(
            self.steps_per_epoch, name="steps_per_epoch"
        )

        factor = ops.convert_to_tensor_v2_with_dispatch(
            self.factor, name="factor"
        )

        drop_every = ops.convert_to_tensor_v2_with_dispatch(
            self.drop_every, name="drop_every"
        )
        # compute the total number of epochs that have passed
        epoch = math_ops.floordiv(step_int, steps_per_epoch)

        epoch_float = math_ops.cast(epoch, tf.float32)
        drop_every_float = math_ops.cast(drop_every, tf.float32)
        # compute the learning rate for the current epoch
        exp = math_ops.floor((math_ops.divide((1 + epoch_float), drop_every_float)))
        fact_exp = math_ops.pow(factor, exp)
        lr = math_ops.mul(init_lr, fact_exp)
        # return the learning rate
        return lr

    def get_config(self):
        return {
            "init_lr": self.init_lr,
            "factor": self.factor,
            "drop_every": self.drop_every,
            "steps_per_epoch": self.steps_per_epoch,
        }


def get_lr(
        lr_decay_scheduler_name, num_iterations: int, config: Dict
) -> Union[float, LearningRateSchedule]:
    """
    This function returns a learning rate.
    :param lr_decay_scheduler_name: String specifying the learning rate decay scheduler to use
    :param num_iterations: Integer specifying the number of iterations
    :param config: Dictionary containing the configuration
    """
    if lr_decay_scheduler_name == 'constant':
        lr = config['base_lr']
    elif lr_decay_scheduler_name == 'step':
        lr = StepDecay(
            steps_per_epoch=num_iterations,
            init_alpha=config['base_lr'],
            factor=config['decay_rate'],
            drop_every=config['drop_every'],
        )
    elif lr_decay_scheduler_name == 'exponential':
        lr = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=config['base_lr'],
            decay_steps=num_iterations,
            decay_rate=config['exp_decay_rate'],
            staircase=True,
        )
    elif lr_decay_scheduler_name == 'cosine':
        lr = tf.keras.experimental.CosineDecay(
            initial_learning_rate=config['base_lr'],
            decay_steps=num_iterations,
            alpha=config['alpha'],
        )
    else:
        raise ValueError('Learning rate decay scheduler name not recognized.')
    return lr
