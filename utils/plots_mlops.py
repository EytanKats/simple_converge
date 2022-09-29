

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

    for x, y in zip(discarded, score):
        task.log_scalar_to_mlops_server(f'{plot_name}', f'{metric_name}', y, iteration=x)
    for x, y in zip(discarded, conf):
        task.log_scalar_to_mlops_server(f'{plot_name}', 'confidence_thr', y, iteration=x)
