import jiwer
from deepeval.metrics import BaseMetric

_normalize = jiwer.Compose([jiwer.ToLowerCase(),jiwer.RemovePunctuation(),
                            jiwer.RemoveMultipleSpaces(),jiwer.Strip()])

class WERMetric(BaseMetric):
    """Score =1 - WER of actual_output (hypothesis) against expected_output (reference)."""
    
    def __init__(self,max_wer: float,
                 name: str= "WER"):
            self.threshold = 1 - max_wer
            self.name = name

    def measure(self, test_case, *args, **kwargs) -> float:
            wer =jiwer.wer(_normalize(test_case.expected_output), _normalize(test_case.actual_output))
            self.score = max(0.0, 1- wer)
            self.reason = f"WER={wer:.1%}"
            self.success = self.score >= self.threshold
            return self.score

    async def a_measure(self, test_case, *args, **kwargs):
           return self.measure(test_case)

    def is_successful(self) -> bool:
           return self.success

    @property
    def __name__(self):
           return self.name
    
