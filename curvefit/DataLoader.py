import pandas as pd
import load_tools as lt



class DataLoader:
    def __init__(self):
        self.wl = None  # Масив довжин хвиль або колонок
        self.dataset = None  # Основний датасет
        self.targets = None  # Цільові колонки
        self.absorbance = None


    def load_data(self, dataset=None, wl=None):
        """
        Завантаження даних із датасету та, за необхідності, довжин хвиль (wl).
        """
        if dataset is None:
            raise ValueError("Dataset must be provided.")
        dataset = lt._load_data(dataset)
        dataset = dataset.T if dataset.shape[1]==1 else dataset
        self.targets = self._extract_targets(dataset)
        absorbance = dataset.drop(columns=self.targets.columns)
        
        self.wl = self._process_wl(wl,absorbance)
        self.dataset, self.absorbance = self._process_dataset_and_absorbance(dataset,absorbance)
        return self.wl, self.absorbance, self.targets, self.dataset
    

    def _process_wl(self, wl,absorbance):

        if wl is not None:
            return lt._load_data(wl)
        wl = self._columns_to_wl(absorbance)
        if wl:
            return wl
        return pd.DataFrame(list(range(absorbance.shape[1])))

            
        
    def _process_dataset_and_absorbance(self,dataset,absorbance):
        wl_columns = list(self.wl.T.to_numpy()[0])
        for_dataset_columns = wl_columns+list(self.targets.columns)

        if absorbance.shape[1] != len(wl_columns):
            raise ValueError(
                f"Absorbance and wavelength lengths do not match. "
                f"Absorbance length: {absorbance.shape[1]}, Wavelength length: {len(wl_columns)}"
            )
        if dataset.shape[1] != len(for_dataset_columns):
            raise ValueError(
                f"Dataset columns ({dataset.shape[1]}) do not match the combined columns ({len(for_dataset_columns)})."
            )
        absorbance.columns = wl_columns
        dataset.columns = for_dataset_columns
        return dataset,absorbance


    def _extract_targets(self, dataset):
        target_columns = [col for col in dataset.columns if pd.isna(pd.to_numeric(col, errors="coerce"))]
        if target_columns == []:
            raise TypeError(f"Dataset doesn't have any target columns: {dataset.columns}")
        targets = dataset.loc[:,target_columns]

        return targets
    

    def _columns_to_wl(self, absorbance):
        numeric_columns = []
        for col in absorbance.columns:
            # Спроба перетворити назву колонки на число
            value = pd.to_numeric(col, errors="coerce")
            if pd.notnull(value) and value >= 300:  # Перевірка на число і поріг
                numeric_columns.append(value)
            else:
                return None
        return numeric_columns
    
