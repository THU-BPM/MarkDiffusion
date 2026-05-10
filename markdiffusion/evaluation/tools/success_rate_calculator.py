from typing import List, Dict, Union
from markdiffusion.exceptions.exceptions import TypeMismatchException, ConfigurationError
from sklearn.metrics import roc_auc_score, roc_curve

class DetectionResult:
    
    def __init__(self,
                 gold_label: bool,
                 detection_result: float,
                 ):
        
        self.gold_label = gold_label
        self.detection_result = detection_result
        
    
class BaseSuccessRateCalculator:
    
    def __init__(self,
                 labels: List[str] = ['TPR', 'TNR', 'FPR', 'FNR', 'F1', 'P', 'R', 'F1', 'ACC', 'AUC'],
                 ):
        self.labels = labels
        
    def _check_instance(self,
                        data: List[Union[bool, float]],
                        expected_type: type):
        for item in data:
            if not isinstance(item, expected_type):
                raise TypeMismatchException(expected_type, type(item))
    
    def _filter_metrics(self,
                        metrics: Dict[str, float]) -> Dict[str, float]:
        return {label: metrics[label] for label in self.labels if label in metrics}
    
    def calculate(self,
                  watermarked_results: List[DetectionResult],
                  non_watermarked_results: List[DetectionResult]) -> Dict[str, float]:
        pass
        
class FundamentalSuccessRateCalculator(BaseSuccessRateCalculator):
    """
        Calculator for fundamental success rates of watermark detection.

        This class specifically handles the calculation of success rates for scenarios involving
        watermark detection after fixed thresholding. It provides metrics based on comparisons
        between expected watermarked results and actual detection outputs.

        Use this class when you need to evaluate the effectiveness of watermark detection algorithms
        under fixed thresholding conditions.
    """

    def __init__(self, labels: List[str] = ['TPR', 'TNR', 'FPR', 'FNR', 'P', 'R', 'F1', 'ACC']) -> None:
        """
            Initialize the fundamental success rate calculator.

            Parameters:
                labels (List[str]): The list of metric labels to include in the output.
        """
        super().__init__(labels)
    
    def _compute_metrics(self, inputs: List[DetectionResult]) -> Dict[str, float]:
        """Compute metrics based on the provided inputs."""
        TP = sum(1 for d in inputs if d.detection_result and d.gold_label)
        TN = sum(1 for d in inputs if not d.detection_result and not d.gold_label)
        FP = sum(1 for d in inputs if d.detection_result and not d.gold_label)
        FN = sum(1 for d in inputs if not d.detection_result and d.gold_label)

        TPR = TP / (TP + FN) if TP + FN else 0.0
        FPR = FP / (FP + TN) if FP + TN else 0.0
        TNR = TN / (TN + FP) if TN + FP else 0.0
        FNR = FN / (FN + TP) if FN + TP else 0.0
        P = TP / (TP + FP) if TP + FP else 0.0
        R = TP / (TP + FN) if TP + FN else 0.0
        F1 = 2 * (P * R) / (P + R) if P + R else 0.0
        ACC = (TP + TN) / (len(inputs)) if inputs else 0.0
        
        # Calculate AUC
        y_true = [x.gold_label for x in inputs]
        y_score = [x.detection_result for x in inputs]
        auc = roc_auc_score(y_true=y_true, y_score=y_score)
        
        # Calculate FPR and TPR for ROC curve
        fpr, tpr, _ = roc_curve(y_true=y_true, y_score=y_score)

        return {
            'TPR': TPR, 'TNR': TNR, 'FPR': FPR, 'FNR': FNR,
            'P': P, 'R': R, 'F1': F1, 'ACC': ACC,
            'AUC': auc,
            'FPR_ROC': fpr, 'TPR_ROC': tpr
        }

    def calculate(self, watermarked_result: List[Union[bool, DetectionResult]], non_watermarked_result: List[Union[bool, DetectionResult]]) -> Dict[str, float]:
        """calculate success rates of watermark detection based on provided results."""
        
        # Convert input to DetectionResult objects if needed
        if watermarked_result and isinstance(watermarked_result[0], bool):
            self._check_instance(watermarked_result, bool)
            inputs = [DetectionResult(True, x) for x in watermarked_result]
        else:
            # Assume they are DetectionResult objects
            inputs = list(watermarked_result)
            
        if non_watermarked_result and isinstance(non_watermarked_result[0], bool):
            self._check_instance(non_watermarked_result, bool)
            inputs.extend([DetectionResult(False, x) for x in non_watermarked_result])
        else:
            # Assume they are DetectionResult objects
            inputs.extend(list(non_watermarked_result))

        metrics = self._compute_metrics(inputs)
        return self._filter_metrics(metrics)    
    

class DynamicThresholdSuccessRateCalculator(BaseSuccessRateCalculator):
    
    def __init__(self,
                 labels: List[str] = ['TPR', 'TNR', 'FPR', 'FNR', 'F1', 'P', 'R', 'F1', 'ACC', 'AUC'],
                 rule: str = 'best',
                 target_fpr: float = None,
                 reverse: bool = False,
                 ):
        super().__init__(labels)
        self.rule = rule
        self.target_fpr = target_fpr
        self.reverse = reverse
        
        if self.rule not in ['best', 'target_fpr']:
            raise ConfigurationError(f"Invalid rule: {self.rule}")
        
        if self.target_fpr is not None and not (0 <= self.target_fpr <= 1):
            raise ConfigurationError(f"Invalid target_fpr: {self.target_fpr}")
        
    def _compute_metrics(self,
                         inputs: List[DetectionResult],
                         threshold: float) -> Dict[str, float]:
        if not self.reverse:
            TP = sum(1 for x in inputs if x.gold_label and x.detection_result >= threshold)
            FP = sum(1 for x in inputs if x.detection_result >= threshold and not x.gold_label)
            TN = sum(1 for x in inputs if x.detection_result < threshold and not x.gold_label)
            FN = sum(1 for x in inputs if x.detection_result < threshold and x.gold_label)
        else:
            TP = sum(1 for x in inputs if x.gold_label and x.detection_result <= threshold)
            FP = sum(1 for x in inputs if x.detection_result <= threshold and not x.gold_label)
            TN = sum(1 for x in inputs if x.detection_result > threshold and not x.gold_label)
            FN = sum(1 for x in inputs if x.detection_result > threshold and x.gold_label)
            
        # Calculate AUC
        y_true = [1 if x.gold_label else 0 for x in inputs]
        # print(inputs)
        if not self.reverse:
            y_score = [x.detection_result for x in inputs]
        else:
            y_score = [-x.detection_result for x in inputs]
        auc = roc_auc_score(y_true=y_true, y_score=y_score)
        
        # Get ROC curve
        fpr, tpr, _ = roc_curve(y_true=y_true, y_score=y_score)
            
        metrics = {
            'TPR': TP / (TP + FN),
            'TNR': TN / (TN + FP),
            'FPR': FP / (TN + FP),
            'FNR': FN / (TP + FN),
            'P': TP / (TP + FP),
            'R': TP / (TP + FN),
            'F1': 2 * TP / (2 * TP + FP + FN),
            'ACC': (TP + TN) / (TP + TN + FP + FN),
            'AUC': auc,
            'FPR_ROC': fpr,
            'TPR_ROC': tpr,
        }
        return metrics
        
        
    def _find_best_threshold(self, inputs: List[DetectionResult]) -> float:
        best_threshold = 0
        best_metrics = None
        for i in range(len(inputs) - 1):
            threshold = (inputs[i].detection_result + inputs[i + 1].detection_result) / 2
            metrics = self._compute_metrics(inputs, threshold)
            if best_metrics is None or metrics['F1'] > best_metrics['F1']:
                best_threshold = threshold
                best_metrics = metrics
        return best_threshold
    
    def _find_threshold_by_fpr(self, inputs: List[DetectionResult]) -> float:
        
        threshold = 0
        for i in range(len(inputs) - 1):
            threshold = (inputs[i].detection_result + inputs[i + 1].detection_result) / 2
            metrics = self._compute_metrics(inputs, threshold)
            if metrics['FPR'] <= self.target_fpr:
                break
        return threshold
    
    def _find_threshold(self, inputs: List[DetectionResult]) -> float:
        
        sorted_inputs = sorted(inputs, key=lambda x: x.detection_result, reverse=self.reverse)
        
        if self.rule == 'best':
            return self._find_best_threshold(sorted_inputs)
        elif self.rule == 'target_fpr':
            return self._find_threshold_by_fpr(sorted_inputs)
        
    def calculate(self,
                  watermarked_results: List[float],
                  non_watermarked_results: List[float]) -> Dict[str, float]:
        # Check if inputs are boolean values (which suggests PRC or similar fixed-threshold algorithms)
        if (watermarked_results and isinstance(watermarked_results[0], bool)) or \
           (non_watermarked_results and isinstance(non_watermarked_results[0], bool)):
            raise ValueError(
                "DynamicThresholdSuccessRateCalculator received boolean values. "
                "For algorithms like PRC that use fixed thresholds, please use "
                "FundamentalSuccessRateCalculator instead."
            )
            
        self._check_instance(watermarked_results, float)
        self._check_instance(non_watermarked_results, float)
        
        inputs = [DetectionResult(True, d) for d in watermarked_results] + [DetectionResult(False, d) for d in non_watermarked_results]
        
        threshold = self._find_threshold(inputs)
        metrics = self._compute_metrics(inputs, threshold)
        return self._filter_metrics(metrics)
    
    