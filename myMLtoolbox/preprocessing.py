from sklearn.pipeline import make_pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector


def get_preproc_basic():

    #transformer numerical features
    num_transformer = make_pipeline(KNNImputer(), MinMaxScaler())

    #encode categorical features
    cat_transformer = OneHotEncoder(handle_unknown='ignore', sparse=False)

    num_col = make_column_selector(dtype_include=['float64', 'int32'])

    preproc_basic = make_column_transformer(
        (num_transformer, num_col),
        (cat_transformer, make_column_selector(dtype_include='object'))
        remainder='passthrough')

    return preproc_basic
