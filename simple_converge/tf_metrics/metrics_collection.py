from simple_converge.tf_metrics.ZeroMetric import ZeroMetric
from simple_converge.tf_metrics.ImageMetric import ImageMetric
from simple_converge.tf_metrics.DifferenceMetric import DifferenceMetric
from simple_converge.tf_metrics.SegmentationMetric import SegmentationMetric
from simple_converge.tf_metrics.CrossEntropyMetric import CrossEntropyMetric
from simple_converge.tf_metrics.AccuracyMetric import AccuracyMetric

metrics_collection = {

    "zero_metric": ZeroMetric,
    "image_metric": ImageMetric,
    "difference_metric": DifferenceMetric,
    "segmentation_metric": SegmentationMetric,
    "cross_entropy_metric": CrossEntropyMetric,
    "accuracy_metric": AccuracyMetric

}
