import os
import pandas as pd
from sklearn.model_selection import train_test_split

from phylo_gnn import DATA_PATH

if DATA_PATH is None:
    DATA_PATH = ''


class AbstractDataFactory:
    def __init__(
            self,
            taxa_path_bacteria="microbiote_data/lung/lung_16S_taxa_table_all_samples.csv",
            count_path_bacteria="microbiote_data/lung/lung_16St_asv_table_all_samples.csv",
            taxa_path_fungi="microbiote_data/lung/lung_ITS_taxa_table_all_samples.csv",
            count_path_fungi="microbiote_data/lung/lung_ITS_asv_table_all_samples.csv",
            **kwargs
    ):
        self.taxa_path_bacteria = os.path.join(DATA_PATH, taxa_path_bacteria)
        self.count_path_bacteria = os.path.join(DATA_PATH, count_path_bacteria)
        self.taxa_path_fungi = os.path.join(DATA_PATH, taxa_path_fungi)
        self.count_path_fungi = os.path.join(DATA_PATH, count_path_fungi)
        self.delta_path = os.path.join(DATA_PATH, "delta_16S_final.csv")

        # data tables
        self.taxa_data_bacteria = None
        self.taxa_data_fungi = None
        self.dataframe = None
        self.train = None
        self.test = None

        self.id2taxa = None
        self.taxa2id = None

        self.read_tables()


    def read_tables(self, use_fungi=False):
        # read tables
        taxa_data_bacteria = pd.read_csv(
            self.taxa_path_bacteria, sep=";", index_col=0).reset_index(names="ASV").fillna('')
        count_data_bacteria = pd.read_csv(
            self.count_path_bacteria, sep=";", index_col=0).reset_index(names="id_sample")
        taxa_data_fungi = pd.read_csv(
            self.taxa_path_fungi, sep=";", index_col=0).reset_index(names="ASV").fillna('')
        count_data_fungi = pd.read_csv(
            self.count_path_fungi, sep=";", index_col=0).reset_index(names="id_sample")
        delta_data = pd.read_csv(self.delta_path, sep=",", index_col=0)

        # rename columns to easier handling
        count_data_bacteria.rename(
            columns={col: col + '_bacteria' for col in count_data_bacteria.columns if 'ASV' in col}, inplace=True)
        count_data_fungi.rename(
            columns={col: col + '_fungi' for col in count_data_fungi.columns if 'ASV' in col}, inplace=True)

        # rename taxa ASV table
        taxa_data_bacteria.ASV += '_bacteria'
        taxa_data_fungi.ASV += '_fungi'

        delta_data = delta_data.drop(columns=[col for col in delta_data.columns if 'b_' == col[:2]])
        merged_data = delta_data[delta_data.Time == "M0"].merge(count_data_bacteria)
        if use_fungi:
            merged_data = merged_data.merge(count_data_fungi)

        merged_data["ppFEV1_cat"] = merged_data.ppFEV1_cat.fillna('Unknown')
        # merged_data = pd.get_dummies(data=merged_data, columns=["group"], drop_first=True)

        merged_data = pd.get_dummies(data=merged_data, columns=["Sex", "Col_Pyo", "Col_Afum"], drop_first=True)
        merged_data = pd.get_dummies(data=merged_data, columns=["age_group", "zscore_label", "ppFEV1_cat"])
        merged_data['group_Improvement'] = 1 - merged_data['group_Improvement']
        merged_data.rename(columns={'group_Improvement': 'Deterioration'}, inplace=True)

        # # train, test = train_test_split(merged_data,
        # #                                train_size=0.7,
        # #                                shuffle=True,
        # #                                stratify=merged_data.group_Improvement,
        # #                                random_state=42
        # #                                )
        # train["split"] = 'train'
        # test["split"] = 'test'
        # complete = pd.concat([train, test], axis=0)
        # complete[['id_patient', 'split', 'group_Improvement']].to_csv('split.csv')

        self.taxa_data_bacteria = taxa_data_bacteria
        self.taxa_data_fungi = taxa_data_fungi
        self.dataframe = merged_data
        self.train = merged_data[merged_data.split == 'train']
        self.test = merged_data[merged_data.split == 'test']

    def get_split(self, split):
        if split == 'test':
            return self.test
        elif split == 'train':
            return self.train

    def get_ids(self, ids):
        return self.dataframe.iloc[ids, :]

    def print_cols(self):
        print(self.dataframe.columns)
