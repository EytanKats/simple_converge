from tf_metrics.ZeroMetric import ZeroMetric
from tf_metrics.ImageMetric import ImageMetric
from tf_metrics.DifferenceMetric import DifferenceMetric
from tf_metrics.SegmentationMetric import SegmentationMetric
from tf_metrics.CrossEntropyMetric import CrossEntropyMetric
from tf_metrics.AccuracyMetric import AccuracyMetric

metrics_collection = {

    "zero_metric": ZeroMetric,
    "image_metric": ImageMetric,
    "difference_metric": DifferenceMetric,
    "segmentation_metric": SegmentationMetric,
    "cross_entropy_metric": CrossEntropyMetric,
    "accuracy_metric": AccuracyMetric

}
