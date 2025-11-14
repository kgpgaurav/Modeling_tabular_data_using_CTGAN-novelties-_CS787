# ctgan/data_transformer.py - COMPLETE MODIFIED VERSION

import numpy as np
import pandas as pd
from ctgan.normalizers import VGMNormalizer, KDENormalizer, DPMNormalizer


class DataTransformer:
    """Data Transformer with pluggable normalizers.
    
    Transforms raw tabular data into a format suitable for CTGAN training.
    Handles both continuous and discrete columns.
    """
    
    def __init__(self, max_clusters=10, weight_threshold=0.005,
                 normalizer_type='vgm', normalizer_kwargs=None):
        """Initialize Data Transformer.
        
        Args:
            max_clusters (int): Maximum number of clusters for mixture models
            weight_threshold (float): Minimum weight for a component to be valid
            normalizer_type (str): Type of normalizer - 'vgm', 'kde', or 'dpm'
            normalizer_kwargs (dict): Additional arguments for the normalizer
        """
        self.max_clusters = max_clusters
        self.weight_threshold = weight_threshold
        self.normalizer_type = normalizer_type
        self.normalizer_kwargs = normalizer_kwargs or {}
        
        self._column_transform_info_list = []
        self._column_raw_dtypes = []
        self.output_info_list = []
        self.output_dimensions = 0
        self.dataframe = True
        self.normalizer = None
        # Initialize the normalizer
        self._init_normalizer()
        
        # Storage for column information
        # self._column_transform_info_list = []
        # self._column_transform_info_list.append({
        #     'column_name': column_name,
        #     'column_type': 'discrete' or 'continuous',
        #     'transform_info': column_info
        # })

        # self._column_raw_dtypes = []
        # self.output_info_list = []
        # self.output_dimensions = 0
        # self.dataframe = True
    
    def _init_normalizer(self):
        """Initialize the appropriate normalizer based on type."""
        if self.normalizer_type == 'vgm':
            self.normalizer = VGMNormalizer(
                max_clusters=self.max_clusters,
                weight_threshold=self.weight_threshold,
                **self.normalizer_kwargs
            )
            print(f"Initialized VGM Normalizer with max_clusters={self.max_clusters}")
            
        elif self.normalizer_type == 'kde':
            self.normalizer = KDENormalizer(**self.normalizer_kwargs)
            print(f"Initialized KDE Normalizer with kwargs: {self.normalizer_kwargs}")
            
        elif self.normalizer_type == 'dpm':
            self.normalizer = DPMNormalizer(
                max_components=self.max_clusters,
                weight_threshold=self.weight_threshold,
                **self.normalizer_kwargs
            )
            print(f"Initialized DPM Normalizer with max_components={self.max_clusters}")
            
        else:
            raise ValueError(
                f"Unknown normalizer type: {self.normalizer_type}. "
                f"Choose from: 'vgm', 'kde', 'dpm'"
            )
    
    def _fit_continuous(self, data):
        """Fit normalizer to continuous column.
        
        Args:
            data (np.ndarray): 1D array of continuous values
            
        Returns:
            dict: Fit information from normalizer
        """
        # Handle NaN values
        data_clean = data[~np.isnan(data)]
        
        if len(data_clean) == 0:
            raise ValueError("Column contains only NaN values")
        
        # Fit the normalizer
        fit_info = self.normalizer.fit(data_clean)
        
        # Add metadata
        fit_info['normalizer_type'] = self.normalizer_type
        
        print(f"  Fitted {self.normalizer_type.upper()}: "
              f"Found {fit_info['num_components']} modes")
        
        return fit_info
    
    def _transform_continuous(self, data, column_info):
        """Transform continuous column using fitted normalizer.
        
        Args:
            data (np.ndarray): 1D array of values to transform
            column_info (dict): Fit information from _fit_continuous
            
        Returns:
            np.ndarray: Transformed features (n_samples, 1 + num_components)
        """
        # Handle NaN values - replace with mean of column
        nan_mask = np.isnan(data)
        if nan_mask.any():
            mean_val = column_info['means'][0]  # Use first mode mean
            data = data.copy()
            data[nan_mask] = mean_val
        
        # Transform using normalizer
        normalized_values, mode_indicators = self.normalizer.transform(
            data, column_info
        )
        
        # Clip normalized values to reasonable range
        normalized_values = np.clip(normalized_values, -10, 10)
        
        # Combine normalized values with mode indicators
        # Shape: (n_samples, 1 + num_components)
        features = np.column_stack([
            normalized_values.reshape(-1, 1),
            mode_indicators
        ])
        
        return features
    
    def _inverse_transform_continuous(self, data, column_info, sigmas=None):
        """Inverse transform continuous column.
        
        Args:
            data (np.ndarray): Transformed features (n_samples, 1 + num_components)
            column_info (dict): Fit information
            sigmas (np.ndarray, optional): Additional noise for diversity
            
        Returns:
            np.ndarray: Original scale values
        """
        # Split into normalized values and mode indicators
        normalized_values = data[:, 0]
        mode_indicators = data[:, 1:]
        
        # Add noise if provided (for sample diversity)
        if sigmas is not None:
            normalized_values = normalized_values + sigmas
        
        # Inverse transform
        original_values = self.normalizer.inverse_transform(
            normalized_values, mode_indicators, column_info
        )
        
        return original_values
    
    def _fit_discrete(self, data):
        """Fit discrete column.
        
        Args:
            data (pd.Series or np.ndarray): Discrete column values
            
        Returns:
            dict: Information about discrete column
        """
        # Get unique categories
        categories = list(set(data))
        categories.sort()  # For consistency
        
        return {
            'type': 'discrete',
            'categories': categories,
            'num_categories': len(categories)
        }
    
    def _transform_discrete(self, data, column_info):
        """Transform discrete column to one-hot encoding.
        
        Args:
            data (pd.Series or np.ndarray): Discrete values
            column_info (dict): Info from _fit_discrete
            
        Returns:
            np.ndarray: One-hot encoded values
        """
        categories = column_info['categories']
        num_categories = len(categories)
        
        # Create one-hot encoding
        one_hot = np.zeros((len(data), num_categories))
        
        for i, val in enumerate(data):
            if val in categories:
                idx = categories.index(val)
                one_hot[i, idx] = 1.0
            else:
                # Handle unseen category - assign to most frequent
                one_hot[i, 0] = 1.0
        
        return one_hot
    
    def _inverse_transform_discrete(self, data, column_info):
        """Inverse transform discrete column.
        
        Args:
            data (np.ndarray): One-hot encoded values
            column_info (dict): Info from _fit_discrete
            
        Returns:
            list: Original category values
        """
        categories = column_info['categories']
        
        # Get most probable category for each row
        indices = np.argmax(data, axis=1)
        
        return [categories[idx] for idx in indices]
    
    def fit(self, raw_data, discrete_columns=()):
        """Fit the DataTransformer.

        Args:
            raw_data (pd.DataFrame or np.ndarray): Raw tabular data
            discrete_columns (tuple): Names of discrete columns
        """
        print(f"\nFitting DataTransformer with {self.normalizer_type.upper()} normalizer")
        print("=" * 60)

        if not isinstance(raw_data, pd.DataFrame):
            self.dataframe = False
            raw_data = pd.DataFrame(raw_data)

        self._column_raw_dtypes = raw_data.infer_objects().dtypes
        self._column_transform_info_list = []
        self.output_info_list = []
        self.output_dimensions = 0

        # Process each column
        for column_name in raw_data.columns:
            print(f"\nProcessing column: {column_name}")

            column_data = raw_data[column_name].values

            if column_name in discrete_columns:
                # Discrete column
                print(f"  Type: Discrete")
                column_info = self._fit_discrete(column_data)
                # store transform info as dict for internal use
                self._column_transform_info_list.append({
                    'column_name': column_name,
                    'column_type': 'discrete',
                    'transform_info': column_info
                })

                # For discrete column we append a list with a single span (dim, activation)
                # so downstream code sees one column_info per column (list of spans)
                span = (column_info['num_categories'], 'softmax')
                self.output_info_list.append([span])
                self.output_dimensions += column_info['num_categories']

            else:
                # Continuous column
                print(f"  Type: Continuous")
                column_info = self._fit_continuous(column_data)
                self._column_transform_info_list.append({
                    'column_name': column_name,
                    'column_type': 'continuous',
                    'transform_info': column_info
                })

                # For continuous column we must append a list of spans:
                # - one 'tanh' span of dim 1 (normalized value)
                # - one 'softmax' span of dim = num_components (modes)
                spans = [
                    (1, 'tanh'),
                    (int(column_info['num_components']), 'softmax')
                ]
                self.output_info_list.append(spans)
                self.output_dimensions += 1 + int(column_info['num_components'])

        print(f"\n{'='*60}")
        print(f"Fitting complete. Total output dimensions: {self.output_dimensions}")
        print(f"{'='*60}\n")
    
    def transform(self, raw_data):
        """Transform raw data.
        
        Args:
            raw_data (pd.DataFrame): Raw data to transform
            
        Returns:
            np.ndarray: Transformed data
        """
        if not isinstance(raw_data, pd.DataFrame):
            raw_data = pd.DataFrame(raw_data)
        
        # Transform each column
        column_data_list = []
        for column_info in self._column_transform_info_list:
            column_name = column_info['column_name']
            column_data = raw_data[column_name].values
            
            if column_info['column_type'] == 'continuous':
                transformed = self._transform_continuous(
                    column_data,
                    column_info['transform_info']
                )
            else:  # discrete
                transformed = self._transform_discrete(
                    column_data,
                    column_info['transform_info']
                )
            
            column_data_list.append(transformed)
        
        # Concatenate all columns
        return np.concatenate(column_data_list, axis=1)
    
    def inverse_transform(self, data, sigmas=None):
        """Inverse transform data back to original format.
        
        Args:
            data (np.ndarray): Transformed data
            sigmas (np.ndarray, optional): Noise for continuous columns
            
        Returns:
            pd.DataFrame: Original format data
        """
        st = 0
        recovered_column_data_list = []
        column_names = []
        
        for column_info in self._column_transform_info_list:
            column_name = column_info['column_name']
            column_names.append(column_name)
            
            if column_info['column_type'] == 'continuous':
                # Get dimensions for this column
                num_components = column_info['transform_info']['num_components']
                dim = 1 + num_components
                
                # Extract data for this column
                column_data = data[:, st:st + dim]
                
                # Extract sigma if provided
                if sigmas is not None:
                    sigma = sigmas[st]
                else:
                    sigma = None
                
                # Inverse transform
                recovered_data = self._inverse_transform_continuous(
                    column_data,
                    column_info['transform_info'],
                    sigma
                )
                
                st += dim
                
            else:  # discrete
                # Get dimensions for this column
                num_categories = column_info['transform_info']['num_categories']
                
                # Extract data for this column
                column_data = data[:, st:st + num_categories]
                
                # Inverse transform
                recovered_data = self._inverse_transform_discrete(
                    column_data,
                    column_info['transform_info']
                )
                
                st += num_categories
            
            recovered_column_data_list.append(recovered_data)
        
        # Combine into DataFrame
        recovered_data = pd.DataFrame(
            dict(zip(column_names, recovered_column_data_list))
        )
        
        # Restore original dtypes
        for column_name in recovered_data.columns:
            dtype = self._column_raw_dtypes[column_name]
            
            if dtype != object:
                recovered_data[column_name] = recovered_data[column_name].astype(dtype)
        
        if not self.dataframe:
            return recovered_data.values
        
        return recovered_data
    
# """DataTransformer module."""

# from collections import namedtuple

# import numpy as np
# import pandas as pd
# from joblib import Parallel, delayed
# from rdt.transformers import ClusterBasedNormalizer, OneHotEncoder

# SpanInfo = namedtuple('SpanInfo', ['dim', 'activation_fn'])
# ColumnTransformInfo = namedtuple(
#     'ColumnTransformInfo',
#     ['column_name', 'column_type', 'transform', 'output_info', 'output_dimensions'],
# )


# class DataTransformer(object):
#     """Data Transformer.

#     Model continuous columns with a BayesianGMM and normalize them to a scalar between [-1, 1]
#     and a vector. Discrete columns are encoded using a OneHotEncoder.
#     """

#     def _init_(self, max_clusters=10, weight_threshold=0.005):
#         """Create a data transformer.

#         Args:
#             max_clusters (int):
#                 Maximum number of Gaussian distributions in Bayesian GMM.
#             weight_threshold (float):
#                 Weight threshold for a Gaussian distribution to be kept.
#         """
#         self._max_clusters = max_clusters
#         self._weight_threshold = weight_threshold

#     def _fit_continuous(self, data):
#         """Train Bayesian GMM for continuous columns.

#         Args:
#             data (pd.DataFrame):
#                 A dataframe containing a column.

#         Returns:
#             namedtuple:
#                 A `ColumnTransformInfo` object.
#         """
#         column_name = data.columns[0]
#         gm = ClusterBasedNormalizer(
#             missing_value_generation='from_column',
#             max_clusters=min(len(data), self._max_clusters),
#             weight_threshold=self._weight_threshold,
#         )
#         gm.fit(data, column_name)
#         num_components = sum(gm.valid_component_indicator)

#         return ColumnTransformInfo(
#             column_name=column_name,
#             column_type='continuous',
#             transform=gm,
#             output_info=[SpanInfo(1, 'tanh'), SpanInfo(num_components, 'softmax')],
#             output_dimensions=1 + num_components,
#         )

#     def _fit_discrete(self, data):
#         """Fit one hot encoder for discrete column.

#         Args:
#             data (pd.DataFrame):
#                 A dataframe containing a column.

#         Returns:
#             namedtuple:
#                 A `ColumnTransformInfo` object.
#         """
#         column_name = data.columns[0]
#         ohe = OneHotEncoder()
#         ohe.fit(data, column_name)
#         num_categories = len(ohe.dummies)

#         return ColumnTransformInfo(
#             column_name=column_name,
#             column_type='discrete',
#             transform=ohe,
#             output_info=[SpanInfo(num_categories, 'softmax')],
#             output_dimensions=num_categories,
#         )

#     def fit(self, raw_data, discrete_columns=()):
#         """Fit the `DataTransformer`.

#         Fits a `ClusterBasedNormalizer` for continuous columns and a
#         `OneHotEncoder` for discrete columns.

#         This step also counts the #columns in matrix data and span information.
#         """
#         self.output_info_list = []
#         self.output_dimensions = 0
#         self.dataframe = True

#         if not isinstance(raw_data, pd.DataFrame):
#             self.dataframe = False
#             # work around for RDT issue #328 Fitting with numerical column names fails
#             discrete_columns = [str(column) for column in discrete_columns]
#             column_names = [str(num) for num in range(raw_data.shape[1])]
#             raw_data = pd.DataFrame(raw_data, columns=column_names)

#         self._column_raw_dtypes = raw_data.infer_objects().dtypes
#         self._column_transform_info_list = []
#         for column_name in raw_data.columns:
#             if column_name in discrete_columns:
#                 column_transform_info = self._fit_discrete(raw_data[[column_name]])
#             else:
#                 column_transform_info = self._fit_continuous(raw_data[[column_name]])

#             self.output_info_list.append(column_transform_info.output_info)
#             self.output_dimensions += column_transform_info.output_dimensions
#             self._column_transform_info_list.append(column_transform_info)

#     def _transform_continuous(self, column_transform_info, data):
#         column_name = data.columns[0]
#         flattened_column = data[column_name].to_numpy().flatten()
#         data = data.assign({column_name: flattened_column})
#         gm = column_transform_info.transform
#         transformed = gm.transform(data)

#         #  Converts the transformed data to the appropriate output format.
#         #  The first column (ending in '.normalized') stays the same,
#         #  but the lable encoded column (ending in '.component') is one hot encoded.
#         output = np.zeros((len(transformed), column_transform_info.output_dimensions))
#         output[:, 0] = transformed[f'{column_name}.normalized'].to_numpy()
#         index = transformed[f'{column_name}.component'].to_numpy().astype(int)
#         output[np.arange(index.size), index + 1] = 1.0

#         return output

#     def _transform_discrete(self, column_transform_info, data):
#         ohe = column_transform_info.transform
#         return ohe.transform(data).to_numpy()

#     def _synchronous_transform(self, raw_data, column_transform_info_list):
#         """Take a Pandas DataFrame and transform columns synchronous.

#         Outputs a list with Numpy arrays.
#         """
#         column_data_list = []
#         for column_transform_info in column_transform_info_list:
#             column_name = column_transform_info.column_name
#             data = raw_data[[column_name]]
#             if column_transform_info.column_type == 'continuous':
#                 column_data_list.append(self._transform_continuous(column_transform_info, data))
#             else:
#                 column_data_list.append(self._transform_discrete(column_transform_info, data))

#         return column_data_list

#     def _parallel_transform(self, raw_data, column_transform_info_list):
#         """Take a Pandas DataFrame and transform columns in parallel.

#         Outputs a list with Numpy arrays.
#         """
#         processes = []
#         for column_transform_info in column_transform_info_list:
#             column_name = column_transform_info.column_name
#             data = raw_data[[column_name]]
#             process = None
#             if column_transform_info.column_type == 'continuous':
#                 process = delayed(self._transform_continuous)(column_transform_info, data)
#             else:
#                 process = delayed(self._transform_discrete)(column_transform_info, data)
#             processes.append(process)

#         return Parallel(n_jobs=-1)(processes)

#     def transform(self, raw_data):
#         """Take raw data and output a matrix data."""
#         if not isinstance(raw_data, pd.DataFrame):
#             column_names = [str(num) for num in range(raw_data.shape[1])]
#             raw_data = pd.DataFrame(raw_data, columns=column_names)

#         # Only use parallelization with larger data sizes.
#         # Otherwise, the transformation will be slower.
#         if raw_data.shape[0] < 500:
#             column_data_list = self._synchronous_transform(
#                 raw_data, self._column_transform_info_list
#             )
#         else:
#             column_data_list = self._parallel_transform(raw_data, self._column_transform_info_list)

#         return np.concatenate(column_data_list, axis=1).astype(float)

#     def _inverse_transform_continuous(self, column_transform_info, column_data, sigmas, st):
#         gm = column_transform_info.transform
#         data = pd.DataFrame(column_data[:, :2], columns=list(gm.get_output_sdtypes())).astype(float)
#         data[data.columns[1]] = np.argmax(column_data[:, 1:], axis=1)
#         if sigmas is not None:
#             selected_normalized_value = np.random.normal(data.iloc[:, 0], sigmas[st])
#             data.iloc[:, 0] = selected_normalized_value

#         return gm.reverse_transform(data)

#     def _inverse_transform_discrete(self, column_transform_info, column_data):
#         ohe = column_transform_info.transform
#         data = pd.DataFrame(column_data, columns=list(ohe.get_output_sdtypes()))
#         return ohe.reverse_transform(data)[column_transform_info.column_name]

#     def inverse_transform(self, data, sigmas=None):
#         """Take matrix data and output raw data.

#         Output uses the same type as input to the transform function.
#         Either np array or pd dataframe.
#         """
#         st = 0
#         recovered_column_data_list = []
#         column_names = []
#         for column_transform_info in self._column_transform_info_list:
#             dim = column_transform_info.output_dimensions
#             column_data = data[:, st : st + dim]
#             if column_transform_info.column_type == 'continuous':
#                 recovered_column_data = self._inverse_transform_continuous(
#                     column_transform_info, column_data, sigmas, st
#                 )
#             else:
#                 recovered_column_data = self._inverse_transform_discrete(
#                     column_transform_info, column_data
#                 )

#             recovered_column_data_list.append(recovered_column_data)
#             column_names.append(column_transform_info.column_name)
#             st += dim

#         recovered_data = np.column_stack(recovered_column_data_list)
#         recovered_data = pd.DataFrame(recovered_data, columns=column_names).astype(
#             self._column_raw_dtypes
#         )
#         if not self.dataframe:
#             recovered_data = recovered_data.to_numpy()

#         return recovered_data

#     def convert_column_name_value_to_id(self, column_name, value):
#         """Get the ids of the given column_name."""
#         discrete_counter = 0
#         column_id = 0
#         for column_transform_info in self._column_transform_info_list:
#             if column_transform_info.column_name == column_name:
#                 break
#             if column_transform_info.column_type == 'discrete':
#                 discrete_counter += 1

#             column_id += 1

#         else:
#             raise ValueError(f"The column_name {column_name} doesn't exist in the data.")

#         ohe = column_transform_info.transform
#         data = pd.DataFrame([value], columns=[column_transform_info.column_name])
#         one_hot = ohe.transform(data).to_numpy()[0]
#         if sum(one_hot) == 0:
#             raise ValueError(f"The value {value} doesn't exist in the column {column_name}.")

#         return {
#             'discrete_column_id': discrete_counter,
#             'column_id': column_id,
#             'value_id': np.argmax(one_hot),
#         }