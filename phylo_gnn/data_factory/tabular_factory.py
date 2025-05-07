import os
import pandas as pd

from phylo_gnn.data_factory.abstract_factory import AbstractDataFactory, DATA_PATH


class TabularDataFactory(AbstractDataFactory):
    def __init__(self, matrix_path="lung_matrix_16S_final.csv", split_path="split.csv", **kwargs):
        self.matrix_path = matrix_path
        self.spit_path = split_path
        self.metadata_columns = None
        self.composition_columns = None
        super().__init__(**kwargs)

    def read_tables(self):
        matrix_data = pd.read_csv(os.path.join(DATA_PATH, self.matrix_path), sep=';')
        matrix_data = matrix_data[matrix_data.Time == 'M0']
        split = pd.read_csv(os.path.join(DATA_PATH, self.spit_path), index_col=0)

        merged = split.merge(matrix_data)

        # patch: agregar datos sin categorías
        datos = pd.read_csv('../../data/dataset_CH/clean_data/lung_clean/delta_lung_16S.csv')
        datos = datos[datos["Time"] == 'M0'][['Age', 'ppFEV1', 'bmi_zscore']]
        merged = datos.merge(merged)


        merged.to_csv('delta_16S_final.csv')

        merged["ppFEV1_cat"] = merged.ppFEV1_cat.fillna('Unknown')
        merged['group_Improvement'] = 1 - merged['group_Improvement']
        merged.rename(columns={'group_Improvement': 'Deterioration'}, inplace=True)

        merged = pd.get_dummies(
            data=merged,
            columns=["Sex", "Col_Pyo", "Col_Afum"],
            drop_first=True
        )
        # merged = pd.get_dummies(
        #     data=merged,
        #     columns=["zscore_label", "age_group", "ppFEV1_cat"],
        #     drop_first=False
        # )
        merged[['id_patient', 'id_sample'] + [col for col in merged.columns if col not in ['id_patient', 'id_sample']]].to_csv('patient_metadata_delta_16S.csv')

        self.metadata_columns = [
            'Sex_M',
            'zscore_label_Thin',
            'zscore_label_Normal',
            'age_group_Over 8',
            'age_group_Under 5',
            'age_group_5 to 8',
            'Chao1_16S',
            'Shannon_16S',
            'Simpson_16S',
            'Col_Pyo_Yes',
            'Col_Afum_Yes',
            'ppFEV1_cat_Unknown',
            'ppFEV1_cat_≥100',
            'ppFEV1_cat_<100'
        ]
        self.metadata_columns = ['Sex_M', 'Age', 'age_group', 'bmi_zscore', 'zscore_label', 'ppFEV1', 'ppFEV1_cat',
                                 'Chao1_16S', 'Shannon_16S', 'Simpson_16S',
                                 'Col_Pyo_Yes', 'Col_Afum_Yes']
        # self.metadata_columns = ['Chao1_16S', 'Shannon_16S', 'Simpson_16S', 'Sex_M', 'zscore_label_Thin']

        self.composition_columns = [col for col in merged.columns if 'b_' == col[:2]]
        columns = ['Deterioration'] + self.metadata_columns + self.composition_columns
        columns = [col for col in merged.columns if col not in columns] + columns
        merged = merged[columns]
        merged[merged.select_dtypes(include=bool).columns] = merged.select_dtypes(include=bool).astype(int)
        print(merged.Deterioration.mean())

        self.train = merged[merged.split == 'train']
        self.test = merged[merged.split == 'test']

