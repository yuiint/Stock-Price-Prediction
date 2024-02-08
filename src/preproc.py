import os
import logging
import monpa
import itertools
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import timedelta
from monpa import utils
from sklearn.feature_extraction.text import TfidfVectorizer


logging.basicConfig(format="%(asctime)s-%(name)s-%(levelname)s-%(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.info('started')


def read_data() -> pd.DataFrame:
    '''
    Read the columns we need in raw data.
    '''
    col = ['id', 'p_type', 'post_time', 'title','content']
    data = {}
    for fn in Path(f"{os.getcwd()}/data").rglob(r"bda*.csv"):
        key = str(fn).split('_')[-2] + '_' + str(fn).split('_')[-1][:-4]
        data_tmp = pd.read_csv(fn)
        data_tmp = data_tmp[col]
        data[key] = data_tmp
        logger.info(f"{key} is read")

    data_all = pd.DataFrame()
    for key in data.keys():
        data_all = pd.concat([data_all, data[key]])
    data_all = data_all[data_all['content'].notnull()]
    data_all = data_all.sort_values(by='post_time').reset_index(drop=True)
    
    return data_all


def company_price(id: int) -> pd.DataFrame:
    '''
    Retrieve the stock price associated with the ID(證券代碼) from "stock_data_2019-2023.xlsx"
    '''
    stock_price = pd.DataFrame()
    for yr in tqdm(range(2019, 2024)):
        data_tmp = pd.read_excel(f"{os.getcwd()}/data/stock_data_2019-2023.xlsx", 
                            sheet_name=f"上市{yr}", usecols=[0,1,5], names=['com_id', 'date', 'close'])
        data_tmp = data_tmp[data_tmp['com_id'] == id]
        stock_price = pd.concat([stock_price, data_tmp])

    stock_price = stock_price.sort_values(by='date').reset_index(drop=True)
    stock_price['date'] = stock_price['date'].str.replace("/", "-")
    stock_price['date'] = stock_price['date'].apply(lambda x:pd.to_datetime(x,format='%Y-%m-%d'))
    stock_price = stock_price[['date', 'close']]
    
    return stock_price


def filter_keyword(data: pd.DataFrame, keyword: str) -> pd.DataFrame:
    '''
    Filter keyword from data.
    '''
    data_filtered = data[(data['content'].str.contains(keyword)) | (data['title'].str.contains(keyword))].copy()
    data_filtered.loc[:, 'date'] = data_filtered['post_time'].str.split(' ').apply(lambda  x:x[0])
    data_filtered.loc[:, 'time'] = data_filtered['post_time'].str.split(' ').apply(lambda  x:x[1])
    data_filtered = data_filtered.drop(columns={'post_time'})
    data_filtered['date'] = data_filtered['date'].apply(lambda x:pd.to_datetime(x,format='%Y-%m-%d'))
    return data_filtered


def merge_data(data: pd.DataFrame, stock_price: pd.DataFrame, lag_day: int) -> pd.DataFrame:
  '''
  Calculate the fluctuation based on the number of delay days,
  merge the corpus with the stock price information.
  '''
  
  stock_price_lag = stock_price.copy(deep=True)
  stock_price_lag.columns = [f'date+{lag_day}', f'close+{lag_day}']

  
  data[f'date+{lag_day}'] = data['date'] + timedelta(days=lag_day)
  data = data.merge(stock_price, on='date', how='inner')
  data = data.merge(stock_price_lag, on=f'date+{lag_day}', how='inner')
  data = data[['id', 'p_type', 'title', 'content', 'time', 'date', 'close', f'date+{lag_day}', f'close+{lag_day}']].drop_duplicates()

  data_merged = data[['id', 'title', 'content','date', 'close', f'date+{lag_day}', f'close+{lag_day}']].copy()
  data_merged.loc[:, 'diff'] = (data_merged[f'close+{lag_day}'] - data_merged['close']) / data_merged['close']

  print(f"Left corpus number according to the threshold with lag day = {lag_day}")
  for rate in [0.01, 0.02, 0.03, 0.04, 0.05]:
      total = data_merged.shape[0]
      go_up = data_merged[data_merged['diff'] >= rate].shape[0]
      go_down = data_merged[data_merged['diff'] <= (rate)*-1].shape[0]
      unchanged = total - go_up - go_down
      print(f">> Threshold is {rate}:  corpus go up: {go_up}({'{:.2%}'.format(go_up/total)}) ; corpus go down: {go_down}({'{:.2%}'.format(go_down/total)}) ; corpus that price unchage: {unchanged}({'{:.2%}'.format(unchanged/total)})")
  print('='*50)

  return data_merged


# def label_with_threshold(data: pd.DataFrame, threshold: float):
#     '''
#     Label curpus with "up" and "down" according to the diff rate of close price.
#     '''
#     up = data[data['diff'] >= threshold][['id', 'title', 'content']]
#     up['type'] = 'up'
#     down = data[data['diff'] <= (threshold)*-1][['id', 'title', 'content']]
#     down['type'] = 'down'
#     return up, down


def classify_curpus(data: pd.DataFrame, threshold: float) -> pd.DataFrame:
    '''
    Get classified data by setting a threshold,
    label curpus with "up" and "down" according to the diff rate of close price.    
    '''
    # data_up, data_down = label_with_threshold(data, threshold)

    data_up = data[data['diff'] >= threshold][['id', 'title', 'content']]
    data_up['type'] = 'up'
    data_down = data[data['diff'] <= (threshold)*-1][['id', 'title', 'content']]
    data_down['type'] = 'down'

    data_up_down = pd.concat([data_up, data_down]).reset_index(drop=True)
    print(f"Number of curpus labeled up: {data_up.shape[0]}")
    print(f"Number of curpus labeled down: {data_down.shape[0]}")
   

    return data_up_down


def text_preproc(data: pd.DataFrame) -> pd.DataFrame:
    # data['content'] = data['content'].str.replace(r"[^\u4e00-\u9fa5]+", '')
    # data['title'] = data['title'].str.replace(r"[^\u4e00-\u9fa5]+", '')
    data['title_content'] = data['title'] + data['content']
    data['title_content'] = data['title_content'].str.replace(r"[^\u4e00-\u9fa5]+", '')

    return data


def cut_text(data: pd.DataFrame) -> list:
    '''
    Cut text into terms by monpa.
    '''
    cut_text = []    
    sentence_list = utils.short_sentence(data['title_content']) # 斷句
    
    for item in sentence_list:
        result_cut = monpa.cut(item) # 斷詞
        if len(result_cut) != 0:
          result_cut = pd.Series(result_cut).str.strip(' ')
          result_cut = result_cut[result_cut.str.len() > 1].to_list()
          cut_text += result_cut
    return cut_text


def text_tokenize(data: pd.DataFrame) -> pd.DataFrame:
    '''
    Preprocess curpus and cut text into terms.
    '''
    data = text_preproc(data)
    tqdm.pandas()
    data['cut_title_content'] = data.progress_apply(lambda x: cut_text(x), axis=1)
    data['cut_title_content_term'] = data['cut_title_content'].apply(lambda x:' '.join(x))
    return data


def extract_term(data: pd.DataFrame) -> list:
    '''
    Get a termset by remove duplicate terms.
    '''
    terms = list(itertools.chain(*data['cut_title_content'].to_list()))
    terms = pd.Series(terms).apply(lambda x: x.replace(' ', ''))
    terms = terms.drop_duplicates().to_list()

    return terms


def term_tfidf(data: pd.DataFrame, terms: list) -> pd.DataFrame:
    '''
    Calculate corpus' TF-IDF vector by terms.
    '''
    documents = data['cut_title_content_term'].to_list()
    vectorizer = TfidfVectorizer(vocabulary=terms, use_idf=True)

    tfidf_matrix = vectorizer.fit_transform(documents)

    term_list = []
    tfidf = []
    for _, term in enumerate(terms):
        term_list.append(term)
        term_index = vectorizer.vocabulary_.get(term)
        if term_index is not None:
            tfidf_value = tfidf_matrix[:, term_index].mean()
            tfidf.append(tfidf_value)
        else:
            tfidf.append('not_found')

    tfidf_df = pd.DataFrame(
        {
        'term':term_list,
        'tfidf':tfidf
        }
        )
    tfidf_df = tfidf_df[tfidf_df['term'].str.match(r".*\d.*") == False] #  去除含有數字的詞彙
    tfidf_df = tfidf_df[tfidf_df['tfidf'] != 0].sort_values(by='tfidf', ascending=False) #  去除tfidf為0的詞彙
    return tfidf_df


def get_termset(data: pd.DataFrame) -> pd.DataFrame:
    '''
    Select the top 200 terms with the highest TF-IDF values from both bullish and bearish articles to form the termset.
    '''
    up_terms = extract_term(data[data['type'] == 'up'])
    down_terms = extract_term(data[data['type'] == 'down'])

    same = (set(up_terms) & set(down_terms))
    up_terms = list(set(up_terms) - same)
    down_terms = list(set(down_terms) - same)

    up_terms_df = term_tfidf(data[data['type']=='up'], up_terms)
    up_terms_df = up_terms_df.head(200)
    down_terms_df = term_tfidf(data[data['type']=='down'], down_terms)
    down_terms_df = down_terms_df.head(200)

    termset = pd.concat([up_terms_df, down_terms_df]).sort_values(by='tfidf', ascending=False).reset_index(drop=True)
    termset = termset['term']

    return termset