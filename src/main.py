"""
Main execute function for stock price prediction.
"""

from .preproc import read_data, company_price, filter_keyword, merge_data, classify_curpus, text_tokenize, get_termset
from .predict import get_X_y, prediction_result

if __name__ == "__main__":

    corpus = read_data()
    data = filter_keyword(corpus, "鴻海")
    stock_price = company_price('2317 鴻海')

    data_merged = merge_data(data, stock_price, 3)
    data_classified = classify_curpus(data_merged, 0.07)
    data_cutted = text_tokenize(data_classified)

    termset = get_termset(data_cutted)
    vectorizer, X, y = get_X_y(data_classified, termset)
    
    prediction_result(X, y)

    # corpus = read_data()
    # print(corpus.shape)
    # print(corpus.head())
    
    # data = filter_keyword(corpus, "鴻海")
    # print(data.shape)
    # print(data.head())

    # stock_price = company_price('2317 鴻海')
    # print(stock_price)

    # data_merged = merge_data(data, stock_price, 3)
    # print(data_merged.head())
    
    # data_classified = classify_curpus(data_merged, 0.07)

    # data_cutted = text_tokenize(data_classified)

    # print(data_cutted.head())

    # termset = get_termset(data_cutted)
    # print(termset.shape)
    # print(termset.head())

    # # predict
    # vectorizer, X, y = get_X_y(data_classified, termset)
    
    # prediction_result(X, y)




