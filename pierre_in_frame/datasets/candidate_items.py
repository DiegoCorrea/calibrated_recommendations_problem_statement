from settings.save_and_load import SaveAndLoad


class CandidateItems:
    def __init__(self,
            experiment_name: str, split_methodology: str, recommender: str, dataset: str, trial: int, fold: int
                 ):
        self.candidate_items = SaveAndLoad.load_candidate_items(
            experiment_name=experiment_name, split_methodology=split_methodology,
            algorithm=recommender, dataset=dataset, trial=trial, fold=fold
        )

    def get_candidate_items(self):
        return self.candidate_items
