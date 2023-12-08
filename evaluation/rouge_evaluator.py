from rouge_metric import PyRouge

class RougeEvaluator:

    def __init__(self) -> None:
        self.rouge = PyRouge(rouge_n=(1, 2, 4), rouge_l=True, rouge_w=False, rouge_s=False, rouge_su=False)

    def batch_score(self, gen_summaries, reference_summaries):
        score = self.rouge.evaluate(gen_summaries, [[x] for x in reference_summaries])
        return score
    
    def score(self, gen_summary, reference_summary):
        score = self.rouge.evaluate([gen_summary], [[reference_summary]])
        return score