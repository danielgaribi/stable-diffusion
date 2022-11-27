from clearml.automation import UniformParameterRange, UniformIntegerParameterRange, DiscreteParameterRange
from clearml.automation import HyperParameterOptimizer
# from clearml.automation.optuna import OptimizerOptuna

from clearml import Task

task = Task.init(
    project_name='Stable-Diffusion-ILVR',
    task_name='Automatic Hyper-Parameter Optimization',
    task_type=Task.TaskTypes.optimizer,
    reuse_last_task_id=False
)

optimizer = HyperParameterOptimizer(
    # specifying the task to be optimized, task must be in system already so it can be cloned
    base_task_id="afeae4f63f1045bcbf75ac3918104cc7",
    # setting the hyper-parameters to optimize
    hyper_parameters=[
        DiscreteParameterRange('Args/prompt', ["A photo of a man with mustache"]),
        DiscreteParameterRange('Args/init_img', ["ref/face/00006.png"]),
        DiscreteParameterRange('Args/down_n', [2, 4, 8, 16, 32]),
        UniformIntegerParameterRange('Args/range_t', min_value=100, max_value=800, step_size=100),
        DiscreteParameterRange('Args/sd_edit', [True, False]),
        DiscreteParameterRange('Args/latent_ilvr', [True, False]),
        UniformParameterRange('Args/strength', min_value=0.3, max_value=1.0, step_size=0.1),
        UniformParameterRange('Args/ilvr_strength', min_value=0.5, max_value=1.0, step_size=0.1),
        DiscreteParameterRange('Args/ddim_steps', [50, 100, 200])
    ],
    # setting the objective metric we want to maximize/minimize
    objective_metric_title='clip_loss',
    objective_metric_series='clip_loss',
    objective_metric_sign='min',

    # # setting optimizer
    # optimizer_class=OptimizerOptuna,
    #
    # # configuring optimization parameters
    # execution_queue='default',
    # max_number_of_concurrent_tasks=2,
    # optimization_time_limit=60.,
    # compute_time_limit=120,
    # total_max_jobs=20,
    # min_iteration_per_job=15000,
    # max_iteration_per_job=150000,
)

optimizer.start_locally()
# wait until optimization completed or timed-out
optimizer.wait()
# make sure we stop all jobs
optimizer.stop()