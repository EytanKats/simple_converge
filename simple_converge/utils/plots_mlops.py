

def metric_vs_discarded_samples_plot(
        score,
        conf,
        discarded,
        task,
        plot_name,
        metric_name
):

    if task is None:
        return

    mlops_logger = task.get_logger()
    for x, y in zip(discarded, score):
        mlops_logger.report_scalar(f'{plot_name}', f'{metric_name}', y, iteration=x)
    for x, y in zip(discarded, conf):
        mlops_logger.report_scalar(f'{plot_name}', 'confidence_thr', y, iteration=x)
