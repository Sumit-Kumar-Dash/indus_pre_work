import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn_pandas import CategoricalImputer

class Preprocessor:
    """
        This class shall  be used to clean and transform the data before training.

        """

    def __init__(self, file_object, logger_object):
        self.file_object = file_object
        self.logger_object = logger_object

    def separate_label_feature(self, data, label_column_name):
        """
                        Method Name: separate_label_feature
                        Description: This method separates the features and a Label Coulmns.
                        Output: Returns two separate Dataframes, one containing features and the other containing Labels .
                        On Failure: Raise Exception

                """
        self.logger_object.log(self.file_object, 'Entered the separate_label_feature method of the Preprocessor class')
        try:
            self.X=data.drop(labels=label_column_name,axis=1) # drop the columns specified and separate the feature columns
            self.Y=data[label_column_name] # Filter the Label columns
            self.logger_object.log(self.file_object,
                                   'Label Separation Successful. Exited the separate_label_feature method of the Preprocessor class')
            return self.X,self.Y
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in separate_label_feature method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object, 'Label Separation Unsuccessful. Exited the separate_label_feature method of the Preprocessor class')
            raise Exception()

    def dropUnnecessaryColumnsAndDuplicates(self,data,columnNameList):
        """
                        Method Name: is_null_present
                        Description: This method drops the unwanted columns and drop duplicate rows as discussed in EDA section.
                                """
        data = data.drop(columnNameList,axis=1)
        data = data.drop_duplicates(keep='first')
        rows_list = data.loc[data['age'<0]]
        data = data.drop(rows_list,axis=0)
        return data



    def is_null_present(self,data):
        """
                                Method Name: is_null_present
                                Description: This method checks whether there are null values present in the pandas Dataframe or not.
                                Output: Returns True if null values are present in the DataFrame, False if they are not present and
                                        returns the list of columns for which null values are present.
                                On Failure: Raise Exception
                        """
        self.logger_object.log(self.file_object, 'Entered the is_null_present method of the Preprocessor class')
        self.null_present = False
        self.cols_with_missing_values=[]
        self.cols = data.columns
        try:
            self.null_counts=data.isna().sum() # check for the count of null values per column
            for i in range(len(self.null_counts)):
                if self.null_counts[i]>0:
                    self.null_present=True
                    self.cols_with_missing_values.append(self.cols[i])
            if(self.null_present): # write the logs to see which columns have null values
                self.dataframe_with_null = pd.DataFrame()
                self.dataframe_with_null['columns'] = data.columns
                self.dataframe_with_null['missing values count'] = np.asarray(data.isna().sum())
                self.dataframe_with_null.to_csv('preprocessing_data/null_values.csv') # storing the null column information to file
            self.logger_object.log(self.file_object,'Finding missing values is a success.Data written to the null values file. Exited the is_null_present method of the Preprocessor class')
            return self.null_present, self.cols_with_missing_values
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in is_null_present method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,'Finding missing values failed. Exited the is_null_present method of the Preprocessor class')
            raise Exception()

    def encodeCategoricalValues(self,data):
     """
                                        Method Name: encodeCategoricalValues
                                        Description: This method encodes all the categorical values in the training set.
                                        Output: A Dataframe which has all the categorical values encoded.
                                        On Failure: Raise Exception
                     """
     df_cat = data[data.columns[np.where(data.dtypes=='object')]]
     data = data.replace({'scholarshipAvailed':{'हाँ':'availed','नही':'Not availed','नहीं':'Not availed'},
            'hasAdhar':{'हाँ':'Yes','नही':'No'},
            'literacy':{'शिक्षित':'Literate','अशिक्षित':'Illiterate'},
            'eduType':{'औपचारिक':'Formal','अनौपचारिक':'Informal','#NAME?':'Informal'},
            'statusFormalEdu':{'शिक्षा पूर्ण':'Finished Education','जारी है':'Continuing','--- चुनिए ---':'Continuing'},
            'isHeadOfFamily':{'हाँ':'Yes','नही':'No','नहीं':'No'},
            'hasPDS':{'हाँ':'Yes','नही':'No','नहीं':'No'},
            'hasSECC':{'नही':'No','नंबर नहीं पता':'Unknown Number','नहीं':'No','हाँ':'Yes'},
            'hasAyushmanBharat':{'नही':'No','नंबर नहीं पता':'Unknown','नहीं':'No','हाँ':'Yes'},
            'hasLand':{'नहीं':'No','हाँ':'Yes'},
            'hasHealthCert':{'नहीं':'No','हाँ':'Yes','नहीं/नंबर':'Unknown'}})

     data["scholarshipAvailed"] = data["scholarshipAvailed"].map({'availed': 1, 'Not availed': 2})
     for column in df_cat.drop(['scholarshipAvailed'],axis=1).columns:
            data = pd.get_dummies(data, columns=[column],drop_first=True)

     return data


    def encodeCategoricalValuesPrediction(self,data):
        """
                                               Method Name: encodeCategoricalValuesPrediction
                                               Description: This method encodes all the categorical values in the prediction set.
                                               Output: A Dataframe which has all the categorical values encoded.
                                               On Failure: Raise Exception

                            """

     df_cat = data[data.columns[np.where(data.dtypes=='object')]]
     data = data.replace({'scholarshipAvailed':{'हाँ':'availed','नही':'Not availed','नहीं':'Not availed'},
            'hasAdhar':{'हाँ':'Yes','नही':'No'},
            'literacy':{'शिक्षित':'Literate','अशिक्षित':'Illiterate'},
            'eduType':{'औपचारिक':'Formal','अनौपचारिक':'Informal','#NAME?':'Informal'},
            'statusFormalEdu':{'शिक्षा पूर्ण':'Finished Education','जारी है':'Continuing','--- चुनिए ---':'Continuing'},
            'isHeadOfFamily':{'हाँ':'Yes','नही':'No','नहीं':'No'},
            'hasPDS':{'हाँ':'Yes','नही':'No','नहीं':'No'},
            'hasSECC':{'नही':'No','नंबर नहीं पता':'Unknown Number','नहीं':'No','हाँ':'Yes'},
            'hasAyushmanBharat':{'नही':'No','नंबर नहीं पता':'Unknown','नहीं':'No','हाँ':'Yes'},
            'hasLand':{'नहीं':'No','हाँ':'Yes'},
            'hasHealthCert':{'नहीं':'No','हाँ':'Yes','नहीं/नंबर':'Unknown'}})

     data["scholarshipAvailed"] = data["scholarshipAvailed"].map({'availed': 1, 'Not availed': 2})
     for column in df_cat.drop(['scholarshipAvailed'],axis=1).columns:
            data = pd.get_dummies(data, columns=[column],drop_first=True)

        return data

    def impute_missing_values(self, data, cols_with_missing_values):
        """
                                        Method Name: impute_missing_values
                                        Description: This method replaces all the missing values in the Dataframe using KNN Imputer.
                                        Output: A Dataframe which has all the missing values imputed.
                                        On Failure: Raise Exception
                     """
        self.logger_object.log(self.file_object, 'Entered the impute_missing_values method of the Preprocessor class')
        self.data1= data[data.columns[np.where(data.dtypes=='object')]]
        self.cols_with_missing_values=cols_with_missing_values
        try:
            self.imputer = CategoricalImputer()
            for col in self.cols_with_missing_values:
                self.data1[col] = self.imputer.fit_transform(self.data1[col])
            self.logger_object.log(self.file_object, 'Imputing missing values Successful. Exited the impute_missing_values method of the Preprocessor class')
            return self.data1
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in impute_missing_values method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,'Imputing missing values failed. Exited the impute_missing_values method of the Preprocessor class')
            raise Exception()

    def get_columns_with_zero_std_deviation(self,data):
        """
                                                Method Name: get_columns_with_zero_std_deviation
                                                Description: This method finds out the columns which have a standard deviation of zero.
                                                Output: List of the columns with standard deviation of zero
                                                On Failure: Raise Exception

                             """
        self.logger_object.log(self.file_object, 'Entered the get_columns_with_zero_std_deviation method of the Preprocessor class')
        self.columns=data.columns
        self.data_n = data.describe()
        self.col_to_drop=[]
        try:
            for x in self.columns:
                if (self.data_n[x]['std'] == 0): # check if standard deviation is zero
                    self.col_to_drop.append(x)  # prepare the list of columns with standard deviation zero
            self.logger_object.log(self.file_object, 'Column search for Standard Deviation of Zero Successful. Exited the get_columns_with_zero_std_deviation method of the Preprocessor class')
            return self.col_to_drop

        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in get_columns_with_zero_std_deviation method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object, 'Column search for Standard Deviation of Zero Failed. Exited the get_columns_with_zero_std_deviation method of the Preprocessor class')
            raise Exception()