import pandas as pd
import numpy as np

class Data(list):
    def __init__(self):
        self.features = ['top', 'crown', 'material', 't', 'd/2h', 'd', 'fai', 'knuckle', 'xita', 'smax', 'buckling']
        self.dfs = []
        self.good_dfs = []
        self.default = {
            'top': 0,
            'crown': 0,
            'material': 205,
            't': 5,
            'd/2h': 2,
            'd': 5000,
            'fai': 20,
            'knuckle': 0,
            'xita': 10,
            'smax': 0
        }

    def read(self, excel, features, x, y):
        self.append([pd.read_excel(excel, sheet_name = None), features, x, y])

    def process(self, ii):
        all_df = self[ii][0]
        x = self[ii][2]
        y = self[ii][3]
        
        #重命名main
        all_df['main'].columns = self[ii][1]
        main = all_df['main']
        
        #初始化dic
        dic = {}
        for i in main.columns[:-1]:
            dic[i] = []
        dic[x] = []
        dic[y] = []
        dic['buckling'] = []
        
        #录入数据
        for i in main.index:
            sheet = main['Sheet'][i]
            df0 = all_df[sheet]
            
            #重命名df0
            knuckles = df0.columns.tolist()
            for n in range(len(knuckles) - 1):
                knuckles[n + 1] = str(knuckles[n + 1])
            df0.columns = knuckles
            
            #正式录入
            for index in df0.index:
                
                for knuckle in knuckles[1:]:
                    t = df0[knuckles[0]][index]
                    t = float(str(t).replace('mm', '').strip())
                    
                    buckling = df0[knuckle][index]
                    
                    dic[x].append(t)
                    dic[y].append(float(knuckle))
                    dic['buckling'].append(buckling)
                    
                    for name in main.columns[:-1]:
                        dic[name].append(main[name][i])
        
        #生成DataFrame
        df = pd.DataFrame(dic)
        
        #材料变换为屈服强度
        trans = {'S30408': 205, 'Q345R': 345, 'SA738B': 415}
        df['material'] = df['material'].map(trans)
        
        #清洗数据
        df = df.dropna(subset=['buckling', 'material'])

        #赋值
        self.dfs.append(df)

    def translate(self, df):
        if df.empty:
            return
            
        lenth = len(df.index)
        
        less_features = df.columns.tolist()
        all_features = self.features
        difference = [i for i in all_features if i not in less_features and i != 'buckling']
        
        for i in difference:
            column = [self.default[i]] * lenth
            df[i] = column

        divi = ['top', 'crown', 't', 'knuckle', 'smax']
        df[divi] = df[divi].div(df['d'].replace(0, np.nan), axis = 0)
        df = df.drop(columns = 'd')
        values = df.columns.tolist()
        values = [i if i not in divi else i + '/d' for i in values]
        df.columns = values
        self.values = values

        self.good_dfs.append(df)

    def process_all(self):
        self.dfs = []
        self.good_dfs = []
        
        lenth = len(self)
        for i in range(lenth):
            self.process(i)
            self.translate(self.dfs[i])

    def merge(self):
        self.process_all()
        
        if not self.good_dfs:
            return pd.DataFrame()
            
        df = pd.concat(self.good_dfs, ignore_index=True)

        return df.dropna()