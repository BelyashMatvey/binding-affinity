from torchmetrics import MeanAbsoluteError, MetricCollection, PearsonCorrCoef, SpearmanCorrCoef


def create_regression_metrics(prefix: str) -> MetricCollection:
    return MetricCollection(
        {
            f"{prefix}/PearsonCorrCoef": PearsonCorrCoef(),
            f"{prefix}/SpearmanCorrCoef": SpearmanCorrCoef(),
            f"{prefix}/MeanAbsoluteError": MeanAbsoluteError(),
        }
    )
