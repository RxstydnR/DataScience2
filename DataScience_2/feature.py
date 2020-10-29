import csv
import datetime
import random
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import timeit
# import swifter


def make_date(x):
    ''' 時間indexを作成する
    '''
    day = x.day
    if len(str(day))<=1:
        day = "0{}".format(day)
    hour = x.hour
    if len(str(hour))<=1:
        hour = "0{}".format(hour)
    minute = x.minute
    if len(str(minute))<=1:
        minute = "0{}".format(minute)
    
    date = "{}-{}-{} {}:{}:{}".format(int(x.year),int(x.month),int(day),hour,minute,"00")
    
    return date


def get_date(d):
    d["date"] = d.apply(make_date, axis=1)
    d["date"] = pd.to_datetime(d["date"], utc=True)
    d.index = pd.to_datetime(d["date"])
    d.drop("date", axis=1, inplace=True)
    return d


def get_city_df():
    ''' 県庁所在地の位置情報データフレームを取得 & 作成
    '''
    df_city_lalo = pd.read_excel('lalo_data.xls')

    df_city_lalo.drop(
        index=df_city_lalo.index[0:4],
        columns=df_city_lalo.columns[[0,3,4,7]], inplace=True)

    df_city_lalo.columns = ["Pref","City","latitude","longitude"]

    df_city_lalo.reset_index(drop=True, inplace=True)
    df_city_lalo.drop(df_city_lalo.index[47:], inplace=True)

    return df_city_lalo


def get_near_city_station(df_station_loc,city_lat,city_lon):
    ''' 各県庁所在地に最も近い地点のsation番号を取得
    '''
    # 緯度の距離（２乗誤差）
    diff_lat = abs(df_station_loc["latitude"]-city_lat)**2
    # 経度の距離（２乗誤差）
    diff_lon = abs(df_station_loc["longitude"]-city_lon)**2
    # 県庁所在地に最も近いstationの行を選択
    near_idx = (diff_lat+diff_lon).idxmin()
    # station番号を取得
    near_station = df_station_loc.iloc[near_idx]["station"]
    
    return near_station


def get_city_station(df):

    # 県庁所在地の位置情報データフレームを取得 & 作成
    df_city_lalo = get_city_df()

    # Station情報取得（ユニーク）
    df_station_loc = df[~df.duplicated(subset="station")][["station","latitude","longitude"]]
    df_station_loc.reset_index(inplace=True, drop=True)

    # 各県庁所在地に最も近い地点のsation番号を取得
    df_city_lalo["station"]=0
    for i, data in df_city_lalo.iterrows():
        city_lat = data["latitude"]
        city_lon = data["longitude"]
        station_num = get_near_city_station(df_station_loc,city_lat,city_lon)
        df_city_lalo["station"].iloc[i] = station_num

    return [df_city_lalo, df_station_loc]


def get_near_station(d,df_station_loc):
    ''' あるポイントの周囲５個,10個のstation番号を取得
    '''
    city_lat=d["latitude"].iloc[0]
    city_lon=d["longitude"].iloc[0]
    
    # 緯度の距離（２乗誤差）
    diff_lat = abs(df_station_loc["latitude"]-city_lat)**2
    # 経度の距離（２乗誤差）
    diff_lon = abs(df_station_loc["longitude"]-city_lon)**2
    # 県庁所在地に最も近いstationの行を選択
    near_idxs = (diff_lat+diff_lon).sort_values().index[1:11] # 10個選択
    # station番号を取得
    near_stations = df_station_loc.iloc[near_idxs]["station"]

    return near_stations


def make_feature_around_city(df,dff,df_station_loc):
    '''
        周辺気温 特徴量
        df  : 全データ
        dff : 県庁所在地データ
    '''
    near_stations = get_near_station(df,df_station_loc)
    # 周辺 5 個のstation
    near_stations_5 = near_stations[:5]
    # 周辺 10 個のstation
    near_stations_10 = near_stations.copy()

    # 周辺のstation情報を抽出
    df_round_5 = df[df["station"].isin(near_stations_5)]
    df_round_10 = df[df["station"].isin(near_stations_10)]

    # 時間インデックス取得
    df_round_5 = get_date(df_round_5)
    df_round_10 = get_date(df_round_10)

    # 周辺５つのstation天気取得
    df_round_5_set = df_round_5.set_index([df_round_5.index.month,df_round_5.index.day,df_round_5.index.hour,df_round_5.index.minute])
    df_round_5_set.index.names = ['month','day','hour','minute']

    # 周辺10つのstation天気取得
    df_round_10_set = df_round_10.set_index([df_round_10.index.month,df_round_10.index.day,df_round_10.index.hour,df_round_10.index.minute])
    df_round_10_set.index.names = ['month','day','hour','minute']

    dff["5_around_temp"] = df_round_5_set["temp"].mean(level=['month','day','hour','minute']).values
    dff["10_around_temp"] = df_round_10_set["temp"].mean(level=['month','day','hour','minute']).values
    dff["5_around_temp"].fillna(dff["5_around_temp"].mean(), inplace=True)
    dff["10_around_temp"].fillna(dff["10_around_temp"].mean(), inplace=True)

    # リーク発生する
    # dff["5_around_temp_sub"] = dff["5_around_temp"] - dff.shift(freq='7D')["5_around_temp"]
    # dff["10_around_temp_sub"] = dff["10_around_temp"] - dff.shift(freq='7D')["10_around_temp"]
    # dff["5_around_temp_sub"].fillna(dff["5_around_temp_sub"].mean(), inplace=True)
    # dff["10_around_temp_sub"].fillna(dff["10_around_temp_sub"].mean(), inplace=True)

    # dff["5_around_temp_pct"] = dff["5_around_temp"].pct_change(freq='7D')
    # dff["10_around_temp_pct"] = dff["10_around_temp"].pct_change(freq='7D')

    # dff["5_around_temp_pct"].fillna(dff["10_around_temp_pct"].mean(), inplace=True)
    # dff["10_around_temp_pct"].fillna(dff["10_around_temp_pct"].mean(), inplace=True)

    return dff
    


def make_feature_city(df):
    ''' 気温 特徴量
    '''
    # 先週の気温
    df["7days_before"] = df.shift(freq='7D')["temp"]
    df["7days_before"].fillna(df["7days_before"].mean(), inplace=True) # 平均値で補完

    df["2days_before"] = df.shift(freq='2D')["temp"]
    df["2days_before"].fillna(df["2days_before"].mean(), inplace=True) # 平均値で補完

    df["3days_before"] = df.shift(freq='3D')["temp"]
    df["3days_before"].fillna(df["3days_before"].mean(), inplace=True) # 平均値で補完

    df["4days_before"] = df.shift(freq='4D')["temp"]
    df["4days_before"].fillna(df["4days_before"].mean(), inplace=True) # 平均値で補完

    df["5days_before"] = df.shift(freq='5D')["temp"]
    df["5days_before"].fillna(df["5days_before"].mean(), inplace=True) # 平均値で補完

    # 5周辺の気温
    df["5_around_7days_before"] = df.shift(freq='7D')["temp"]
    df["5_around_7days_before"].fillna(df["5_around_7days_before"].mean(), inplace=True) # 平均値で補完

    df["5_around_2days_before"] = df.shift(freq='2D')["temp"]
    df["5_around_2days_before"].fillna(df["5_around_2days_before"].mean(), inplace=True) # 平均値で補完

    df["5_around_3days_before"] = df.shift(freq='3D')["temp"]
    df["5_around_3days_before"].fillna(df["5_around_3days_before"].mean(), inplace=True) # 平均値で補完

    df["5_around_4days_before"] = df.shift(freq='4D')["temp"]
    df["5_around_4days_before"].fillna(df["5_around_4days_before"].mean(), inplace=True) # 平均値で補完

    df["5_around_5days_before"] = df.shift(freq='5D')["temp"]
    df["5_around_5days_before"].fillna(df["5_around_5days_before"].mean(), inplace=True) # 平均値で補完

    # 10周辺の気温
    df["10_around_7days_before"] = df.shift(freq='7D')["temp"]
    df["10_around_7days_before"].fillna(df["10_around_7days_before"].mean(), inplace=True) # 平均値で補完

    df["10_around_2days_before"] = df.shift(freq='2D')["temp"]
    df["10_around_2days_before"].fillna(df["10_around_2days_before"].mean(), inplace=True) # 平均値で補完

    df["10_around_3days_before"] = df.shift(freq='3D')["temp"]
    df["10_around_3days_before"].fillna(df["10_around_3days_before"].mean(), inplace=True) # 平均値で補完

    df["10_around_4days_before"] = df.shift(freq='4D')["temp"]
    df["10_around_4days_before"].fillna(df["10_around_4days_before"].mean(), inplace=True) # 平均値で補完

    df["10_around_5days_before"] = df.shift(freq='5D')["temp"]
    df["10_around_5days_before"].fillna(df["10_around_5days_before"].mean(), inplace=True) # 平均値で補完

    # 5周辺との比率
    df["around_7days_sub"] = df["7days_before"] - df["5_around_7days_before"]
    df["around_7days_sub"].fillna(df["around_7days_sub"].mean(), inplace=True) # 平均値で補完
    df["around_7days_pct"] = df["7days_before"] / (df["5_around_7days_before"]+1)
    df["around_7days_pct"].fillna(df["around_7days_pct"].mean(), inplace=True) # 平均値で補完

    df["around_2days_sub"] = df["2days_before"] - df["5_around_2days_before"]
    df["around_2days_sub"].fillna(df["around_2days_sub"].mean(), inplace=True) # 平均値で補完
    df["around_2days_pct"] = df["2days_before"] / (df["5_around_2days_before"]+1)
    df["around_2days_pct"].fillna(df["around_2days_pct"].mean(), inplace=True) # 平均値で補完

    df["around_3days_sub"] = df["3days_before"] - df["5_around_3days_before"]
    df["around_3days_sub"].fillna(df["around_3days_sub"].mean(), inplace=True) # 平均値で補完
    df["around_3days_pct"] = df["3days_before"] / (df["5_around_3days_before"]+1)
    df["around_3days_pct"].fillna(df["around_3days_pct"].mean(), inplace=True) # 平均値で補完

    df["around_4days_sub"] = df["4days_before"] - df["5_around_4days_before"]
    df["around_4days_sub"].fillna(df["around_4days_sub"].mean(), inplace=True) # 平均値で補完
    df["around_4days_pct"] = df["4days_before"] / (df["5_around_4days_before"]+1)
    df["around_4days_pct"].fillna(df["around_4days_pct"].mean(), inplace=True) # 平均値で補完

    df["around_5days_sub"] = df["5days_before"] - df["5_around_5days_before"]
    df["around_5days_sub"].fillna(df["around_5days_sub"].mean(), inplace=True) # 平均値で補完
    df["around_5days_pct"] = df["5days_before"] / (df["5_around_5days_before"]+1)
    df["around_5days_pct"].fillna(df["around_5days_pct"].mean(), inplace=True) # 平均値で補完

    return df




def make_feature_all(df):
    ''' 全データに対する特徴量の作成
    '''

    ''' 高度情報 '''
    # 100m規模で測定
    df["altitude_effect_1"] = df["altitude"]//100

    # 0.6度下がることを考慮
    df["altitude_effect_2"] = df["altitude"]//100 * 0.6

    return df



# if __name__ == "__main__":

    # df_nov = pd.read_csv('../all_nov.csv', sep=',', header=None)
    # df_dec = pd.read_csv('../all_dec.csv', sep=',', header=None)

    # df = pd.read_csv('data.csv', sep=',')
    
    # # testのyを事前に抜いて起きてリークを防ぐ！！
    # df["temp_spare"] = df["temp"]
    # df["temp"].iloc[len(df_nov):] = np.nan
    
    # df_city_lalo, df_station_loc = get_city_station(df)

    # df = make_feature_all(df)

    # df_new = pd.DataFrame(columns=df.columns)
    # for st_num in tqdm(sorted(list(set(df.station)))):
    #     dff = df[df["station"]==st_num]#.reset_index(drop=True)
    #     dff = make_feature_aruod_city(df,dff,df_station_loc)
    #     dff = make_feature_city(dff)
    #     df_new = pd.concat([df_new,dff])

    # df_new.to_csv("data_2.csv")