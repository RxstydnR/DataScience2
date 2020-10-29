# データサイエンス演習 2



### 取り組んだ課題

全国に位置する観測地点で観測された11月の気象データを使用して、県庁所在地の12月の気温を予測する。



### 提出データの内容

- **Datascience2.key** <br>
  進捗報告と最終プレゼン資料
- **feature.py** <br>
  データの整形と特徴量の作成を行う（特徴量の作成は実験では使用していない）
- **model.py** <br>
  作成した予測回帰モデルを記述
- **lalo_data.xls** <br>
  県庁所在地の場所を示すファイル
- **TimeSeriese.ipynb** <br>
  Main Notebook
  
 

### コードの説明

- **feature.py** <br>
  このファイルでは、データの整形と特徴量の作成を行う。ここでは、記述されているメソッドについて説明する。最終的な実験に使用していないものも説明する。
  - make_date, get_date <br>
    pandasで容易に時間を扱えるように、datetime形式にindexを変更する
  - get_city_df, get_city_station <br>
    lalo_data.xlsに格納されている県庁所在地の位置情報データフレームを作成する。
  - get_near_city_station, get_city_station <br>
    各県庁所在地に最も近い地点のsation番号を取得する
  - get_near_station <br>
    あるステーションに近い周囲5地点,10地点のステーション番号を取得する
  - make_feature_arond_city <br>
    あるステーションに近い周囲5地点,10地点のステーションの気象データを取得する
  - make_feature_city <br>
    様々な特徴量を作成する
    - 1~7日前の気温
    - 1~7日前の気温との差分
    - 1~7日前の気温との比率
    - 周辺5地点の1~7日前の気温
    - 周辺5地点の1~7日前との差分
    - 周辺5地点の1~7日前との比率
    - 周辺10地点の1~7日前の気温
    - 周辺10地点の1~7日前との差分
    - 周辺10地点の1~7日前との比率
  - make_feature_all <br>
    高度に関する特徴量を作成する。高度が100m上がるにつれて温度が0.6度下がることを考慮した特徴量。
    - 1oom単位でビン化  → `altitude//100`
    - 0.6度の低下  → `altitude//100*0.6`

- **model.py** <br>
  作成したモデルを簡単に紹介する
  - 1DCNN
  - LSTM
  - RNN
  - CNN+RNN
  - CNN+LSTM
  - Ensemble
