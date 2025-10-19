from typing import List, Union, Tuple
import re
import pandas as pd
from copy import copy
from torcheeg.datasets.module.base_dataset import BaseDataset
import os
from torcheeg.model_selection import KFoldPerSubject


class SessionPerSubject(KFoldPerSubject):
    def __init__(self,
                 n_splits: int = 5,
                 shuffle: bool = False,
                 random_state: Union[float, None] = None,
                 split_path: str = '../EEG-Conformer/split/Session_trial_per_subject'):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.split_path = split_path

        # self.k_fold = model_selection.KFold(n_splits=n_splits,
        #                                     shuffle=shuffle,
        #                                     random_state=random_state)

    def split_info_constructor(self, info: pd.DataFrame) -> None:
        subjects = list(set(info['subject_id']))
        # HACK: split by date and session
        for subject in subjects:
            subject_info = info[info['subject_id'] == subject]
            dates = list(set(subject_info['date']))
            for i, date in enumerate(dates):
                subject_date_info = subject_info[subject_info['date'] == date]
                train_index = list(range(len(subject_date_info)))
                train_info = subject_date_info.iloc[train_index]
                train_info.to_csv(os.path.join(
                    self.split_path, f'subject_{subject}_session_{i+1}.csv'),
                    index=False)

    @property
    def subjects(self) -> List:
        indice_files = list(os.listdir(self.split_path))

        def indice_file_to_subject(indice_file):
            # HACK: change the regular expression
            return re.findall(r'subject_(.*)_session_(\d*).csv',
                              indice_file)[0][0]

        subjects = list(set(map(indice_file_to_subject, indice_files)))
        subjects.sort()
        return subjects

    def split(
            self,
            dataset: BaseDataset,
            subject: Union[int,
                           None] = None) -> Tuple[BaseDataset, BaseDataset]:
        if not os.path.exists(self.split_path):
            os.makedirs(self.split_path)
            self.split_info_constructor(dataset.info)

        subjects = self.subjects
        fold_ids = self.fold_ids

        if subject is not None:
            assert subject in subjects, \
                f'The subject should be in the subject list {subjects}.'

        for local_subject in subjects:
            if (subject is not None) and (local_subject != subject):
                continue

            for fold_id in fold_ids:
                train_info = pd.read_csv(
                    os.path.join(
                        self.split_path,
                        # HACK: change the regular expression
                        f'subject_{local_subject}_session_{fold_id}.csv'))
                # test_info = pd.read_csv(
                #     os.path.join(
                #         self.split_path,
                #         f'test_subject_{local_subject}_fold_{fold_id}.csv'))

                train_dataset = copy(dataset)
                train_dataset.info = train_info

                # test_dataset = copy(dataset)
                # test_dataset.info = test_info

                yield train_dataset
