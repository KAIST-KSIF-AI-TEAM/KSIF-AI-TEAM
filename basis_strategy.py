###################################### Futures basis strategy_daily rebalancing
#*****************************************************************************#
###############################################################################
# Import

import os
# 자료가 있는 주소로 정할 것!
os.chdir('C:/Users/CSJSK/Desktop/KSIF 대안투자(AI운용)/Future - Basis 전략/basis_strategy')

import pandas as pd
from pandas import DataFrame
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import math

# Load Excelfile
path="futures_20170827.xlsx" # xlsx version (not csv version)
data=pd.ExcelFile(path)

# Creat sheet name(1)(For sorting and ranking)
SheetName = ["LeanHog", "LiveCattle", "NaturalGasEmini", "CrudeOilEmini", "NaturalGas",
              "SoybeanOil", "Corn", "Oats", "RoughRice",
              "Wheat", "SoybeanMeal", "Soybeans", "Platinum", 
              "Sugar", "Cocoa", "OrangeJuice", "Cotton"]


###############################################################################
# Create the dataframe for point change
PointChange = [40000, 40000, 2500, 500, 10000, 60000, 5000, 5000, 2000, 5000, 100, 5000, 50, 112000, 10, 15000, 50000]
DfPointChange = {}

for i in SheetName:
     DfPointChange[i] = PointChange[SheetName[:]==i]
     
# First, parsing data and make dataframe for each sheet name

DfSheet={}
for i in SheetName:
    DfSheet[i]=pd.DataFrame()
    DfSheet[i]=data.parse(i)
    DfSheet[i]=pd.DataFrame.from_dict(DfSheet[i].iloc[1:,:])
    
    CountCol=len(DfSheet[i].columns)
    VacantColumnNum = list(range(3, CountCol, 4))
    DfSheet[i]=DfSheet[i].drop(DfSheet[i].columns[VacantColumnNum], axis=1).reset_index(drop=True)
    
# df_sheet['Sugar'].iloc[0,0]

# df_sheet를 통해서 각 상품들에 대한 선물가격 데이터 저장

## ChaCha's Coding ############################################################

DfFuturesDate = {}
DfFuturesRollover = {}
DfRollover = {}
a=0

for name in SheetName:
     
     print(name)
     Df = DfSheet[name]                                                        
     DfRollover[name] = []
     CountCol=len(Df.columns)
     n = int(CountCol/3)
     Df.columns = np.zeros(CountCol)
     
     for i in range(n):
          if i==0:
               DfDate = DfSheet[name].iloc[:,0].dropna()
          else:
               DfDate = pd.concat([DfDate, DfSheet[name].iloc[:,3*i].dropna()], join = 'outer', ignore_index=True)
               DfDate = DfDate.drop_duplicates().reset_index(drop=True)     
          DfRollover[name].append(DfDate.iloc[-1])
     
     DfFuturesDate[name] = DfDate     
     
     if a==0:
          DfFuturesAllDate = pd.DataFrame(DfDate)

     else:
          DfFuturesAllDate = DfFuturesAllDate.merge(pd.DataFrame(DfDate))
     
     a+=1
###############################################################################
# 수요일 추출(rebalancing date)

DfRebalDate = DfFuturesAllDate[DfFuturesAllDate[0].dt.dayofweek==2].reset_index(drop=True)


"""
DfSheet['LeanHog'].iloc[:,0]==DfRolloverRebalDate['LeanHog'][0]
DfRolloverRebalDate['LeanHog']
DfRebalDate[DfRebalDate<DfRollover['LeanHog'][0]].dropna().iloc[-1,0]
"""
###############################################################################




###############################################################################
# 롤오버 직전 Rebalancing 날짜 계산 완료

DfRolloverRebalDate = {}

for name in SheetName:
     DfRolloverRebalDate[name] = []
     for i in range(len(DfRollover[name])):
          DfRolloverRebalDate[name].append(DfRebalDate[DfRebalDate<DfRollover[name][i]].dropna().iloc[-1,0]) 
     DfRolloverRebalDate[name] = pd.DataFrame(DfRolloverRebalDate[name])
     DfRolloverRebalDate[name] = DfRolloverRebalDate[name].drop_duplicates()
###############################################################################

"""
#Checking 용도

DfRebalDate[DfRebalDate.iloc[:,0].isin(DfRollover['LeanHog'])]
DfRollover['LeanHog']


Df_Period_Rebal = DfFuturesAllDate[DfFuturesAllDate <= DfRolloverRebalDate['LeanHog'].iloc[0,0]].dropna().iloc[:,0].tolist()
df_first  = DfSheet['LeanHog'].iloc[:,:3][DfSheet['LeanHog'].iloc[:,0].isin(Df_Period_Rebal)]
df_first.columns = ['Date', 'First Price', 'First Volume']
df_second = DfSheet['LeanHog'].iloc[:,3:6][DfSheet['LeanHog'].iloc[:,3].isin(Df_Period_Rebal)]
df_second.columns = ['Date', 'Second Price', 'Second Volume']

df_first_second = pd.merge(df_first.iloc[:-1,:], df_second.iloc[:-1,:], on='Date')

pd.DataFrame(data = [df_first.iloc[-1,:].values], columns = df_first.iloc[-1,:].index)

# Rollover Rebalancing 기간
pd.merge(pd.DataFrame(data = [df_first.iloc[-1,:].values], columns = df_first.iloc[-1,:].index), pd.DataFrame(data = [df_second.iloc[-1,:].values], columns = df_second.iloc[-1,:].index), on='Date')

pd.concat([pd.merge(df_first, df_second, on='Date'),pd.merge(df_first, df_second, on='Date')])


Df_Period_Rebal = DfFuturesAllDate[DfFuturesAllDate < DfRolloverRebalDate['LeanHog'].iloc[1,0]][DfFuturesAllDate >= DfRolloverRebalDate['LeanHog'].iloc[0,0]].dropna().iloc[:,0].tolist()
df_first  = DfSheet['LeanHog'].iloc[:,3:6][DfSheet['LeanHog'].iloc[:,3].isin(Df_Period_Rebal)]
df_first.columns = ['Date', 'First Price', 'First Volume']
df_second = DfSheet['LeanHog'].iloc[:,6:9][DfSheet['LeanHog'].iloc[:,6].isin(Df_Period_Rebal)]
df_second.columns = ['Date', 'Second Price', 'Second Volume']
pd.concat([pd.merge(df_first, df_second, on='Date'),pd.merge(df_first, df_second, on='Date')])

DfSheet['LeanHog'].iloc[: , 3*i    :3*(i+1)][DfSheet['LeanHog'].iloc[:,3*i    ].isin(PeriodAfterRolloverRebal)]
"""


# Date, 근월가격, 근월거래량, 원월가격, 원월거래량
# Date, RolloverRebal시, 근월가격, 근월거래량, 원월가격, 원월거래량
# Dataframe 생성

DfFuturesPriceVolume = {}
DfFuturesPriceVolumeRolloverRebal = {}

for name in SheetName:
     
     print(name)
     
     for i in range(len(DfRolloverRebalDate[name])):
          if i == 0:
               
               # 일부러 리밸런싱 기간도 포함함
               # Rollover할 때, 근월물, 원월물 데이터를 기록한다음 다음 기간에서 근원월물, 원원월물 데이터를 입력하기 위함
               
               PeriodBeforeRolloverRebal = DfFuturesAllDate[DfFuturesAllDate <= DfRolloverRebalDate[name].iloc[i,0]].dropna().iloc[:,0].tolist()
               
               DfFirst  = DfSheet[name].iloc[: , 3*i : 3*(i+1)    ][DfSheet[name].iloc[:,3*i].isin(PeriodBeforeRolloverRebal)]
               DfFirst.columns  = ['Date', 'First Price', 'First Volume']
               
               DfSecond = DfSheet[name].iloc[: , 3*(i+1) : 3*(i+2)][DfSheet[name].iloc[:,3*(i+1)].isin(PeriodBeforeRolloverRebal)]
               DfSecond.columns = ['Date', 'Second Price', 'Second Volume']
               
               # 총 날짜에 대한 근월물, 차월물의 가격과 거래량 데이터
               DfFuturesPriceVolume[name] =              pd.merge(DfFirst.iloc[:-1,:], DfSecond.iloc[:-1,:], on='Date')
               # 롤오버 리밸런싱 날짜에 대한 근월물, 차월물의 가격과 거래량 데이터
               DfFuturesPriceVolumeRolloverRebal[name] = pd.DataFrame()
               DfFuturesPriceVolumeRolloverRebal[name] = pd.merge(pd.DataFrame(data = [DfFirst.iloc[-1,:].values], columns = DfFirst.iloc[-1,:].index), pd.DataFrame(data = [DfSecond.iloc[-1,:].values], columns = DfSecond.iloc[-1,:].index), on='Date')
               
          else:
               
               PeriodForRolloverRebal = DfFuturesAllDate[DfFuturesAllDate <= DfRolloverRebalDate[name].iloc[i,0]][DfFuturesAllDate >= DfRolloverRebalDate[name].iloc[i-1,0]].dropna().iloc[:,0].tolist()
               
               DfFirst  = DfSheet[name].iloc[: , 3*i:3*(i+1)][DfSheet[name].iloc[:,3*i].isin(PeriodForRolloverRebal)]
               DfFirst.columns  = ['Date', 'First Price', 'First Volume']
               
               DfSecond = DfSheet[name].iloc[: , 3*(i+1):3*(i+2)][DfSheet[name].iloc[:,3*(i+1)].isin(PeriodForRolloverRebal)]
               DfSecond.columns = ['Date', 'Second Price', 'Second Volume']
               
               # 총 날짜에 대한 근월물, 차월물의 가격과 거래량 데이터
               DfFuturesPriceVolume[name] = pd.concat( [DfFuturesPriceVolume[name], pd.merge(DfFirst.iloc[:-1,:], DfSecond.iloc[:-1,:], on='Date')] )
               # 롤오버 리밸런싱 날짜에 대한 근월물, 차월물의 가격과 거래량 데이터
               DfFuturesPriceVolumeRolloverRebal[name] = pd.concat( [DfFuturesPriceVolumeRolloverRebal[name], pd.merge(pd.DataFrame(data = [DfFirst.iloc[-1,:].values], columns = DfFirst.iloc[-1,:].index), pd.DataFrame(data = [DfSecond.iloc[-1,:].values], columns = DfSecond.iloc[-1,:].index), on='Date')] )
               
     i=len(DfRolloverRebalDate[name])
     
     PeriodAfterRolloverRebal = DfFuturesAllDate[DfFuturesAllDate >= DfRolloverRebalDate[name].iloc[i-1,0]].dropna().iloc[:,0].tolist()
               
     DfFirst  = DfSheet[name].iloc[: , 3*(i-1) : 3*i][DfSheet[name].iloc[:,3*(i-1)].isin(PeriodAfterRolloverRebal)]
     DfFirst.columns  = ['Date', 'First Price', 'First Volume']
     
     DfSecond = DfSheet[name].iloc[: , 3*i : 3*(i+1)][DfSheet[name].iloc[:,3*i    ].isin(PeriodAfterRolloverRebal)]
     DfSecond.columns = ['Date', 'Second Price', 'Second Volume']

     # 총 날짜에 대한 근월물, 차월물의 가격과 거래량 데이터
     DfFuturesPriceVolume[name] = pd.concat( [DfFuturesPriceVolume[name], pd.merge(DfFirst.iloc[:-1,:], DfSecond.iloc[:-1,:], on='Date')] )








from xlwt.Workbook import *
from pandas import ExcelWriter
import xlsxwriter

listforSheet =[]
wb= Workbook()
i=0
for name in SheetName:
     ws = wb.add_sheet(name)
     DfFuturesPriceVolume[name].to_excel(ws, name)
     i+=1
writer.save('DfFuturesPriceVolume.xlsx')
     ws1 = wb.add_sheet('original')
ws2 = wb.add_sheet('result')
original.to_excel(writer,'original')
data.to_excel(writer,'result')
writer.save('final.xls')

'DfFuturesPriceVolume'+'LeanHog'
for name in SheetName:
     DfFuturesPriceVolume[name].reset_index()





myDF = pd.DataFrame(DfFuturesPriceVolume)
writer = ExcelWriter('DfFuturesPriceVolume')
myDF.to_excel(writer)
writer.save()







# roll over 일자와 rebalancing 날짜에 맞춰 roll over 없는 근,차월물 가격 테이블 만들 것!

# roll over 직전 rebalancing date 찾기
df_futures_date['Sugar'][1]-df_futures_date['Sugar'][0]>Timedelta(')
df_futures_roll_rebaldate = {}
df_futures1 = {}
df_futures2 = {}
for name in sheet_name:
     for i in range(len(df_rollover[name]))
          if i == 0:
               df_futures1[name] = df_sheet
          
# 근월물 : 1st month contract
# 차월물 : 2nd month contract

for name in sheet_name: # input sheet name(1) first and input sheet name(2) next

    df = df_sheet[name]
    countcol=len(df.columns)
    vacantColumnNum = list(range(2,countcol,3))
    
    df2 = df.drop(df.columns[vacantColumnNum], axis=1) 
    countcol2=len(df2.columns)
    n=int( (countcol2)/2)
    
    # Make df, dff, dfuni
    df={} # df: df2.iloc[:, 0 : 2].dropna() including daily dates and daily prices(PX_last) of monthly futures 
    dff={} # dff : df2.iloc[:, 0 : 1].dropna() including daily dates only of monthly futures 
    for i in range(n):
        df[i]=pd.DataFrame()
        dff[i]=pd.DataFrame()
        df[i]=df2.iloc[:, 2*i : 2*i+2].dropna()
        dff[i]=df2.iloc[:, 2*i : 2*i+1].dropna()
    
    dfuni={} # dfuni : df2.iloc[:, 0].dropna() including daily dates only in 'k'th cumulative monthly futures as increasing k
    dfhelp_list=[] # help_list for making dfuni
    k=0
    while k < n:
        dfuni[k]=pd.DataFrame()
        dfhelp_list.append(df2.iloc[:,2*k].dropna()) # make list first and convert dataframe
        dfuni[k]=pd.DataFrame(dfhelp_list).T
        k +=1
    
    # Make cumulative sum_columns and pure_columns using df, dff, dfuni
    cumsumcol={} # Last cumsumcol is including total dates(every transaction dates)
    purecol={} # pure col is including pure dates in each monthly futures( B-(AUB)^C ) like Complement set
    s0 = df[0].iloc[:,0] # cumulative first monthly futures only
    n0 = df[0].iloc[:,0] # pure first monthly futures only (s0 = n0)
    cumsumcol[0] = pd.concat([s0, df[1][~dff[1].iloc[:,0].isin(dfuni[0].iloc[:,0])].iloc[:,0] ], ignore_index=True, axis=0) # cumulative columns including second monthly futures
    purecol[0] = df[1][~dff[1].iloc[:,0].isin(dfuni[0].iloc[:,0])].iloc[:,0] # pure second monthly futures only
    for i in range(n-2): # iterate from third columns to the end columns using column genereation
        cumsumcol[i+1] = pd.DataFrame()
        purecol[i+1] = pd.DataFrame()
        cumsumcol[i+1] = pd.concat( [ cumsumcol[i] , df[i+2][~dff[i+2].iloc[:,0].isin(cumsumcol[i])].iloc[:,0] ], ignore_index=True, axis=0)
        purecol[i+1] = df[i+2][~dff[i+2].iloc[:,0].isin(cumsumcol[i])].iloc[:,0] # 순수 월물들
    
    # count length of purecol for next step in advnace    
    count_purecol=[]
    purecol_date=[]
    for i in range(n-2):
        a=int(len(purecol[i])-1)
        b=purecol[i].iloc[-1]
        count_purecol.append(a)
        purecol_date.append(b)
        
        
###############################################################################        
#*****************************************************************************#    
###############################################################################    


    # daily rebalacning version only
    # basis = 근월물 - 차월물
    # make basis_b(B), basis_c(C) when basis is positive(b) and negative(c) each
    # and save basis & dates
    
    # first futures and next futures only
    count_before=[]
    count_after=[]
    basis_init_date=[]
    basis_b=[]
    basis_c=[]
    for j in range(len(n0)-1):
        basis_binit = df2[ df2 [ df2.columns[0] ] == n0[j+1] ].iloc[0, 1]
        basis_ainit = df2[ df2 [ df2.columns[2] ] == n0[j+1] ].iloc[0, 3]
        basis_init = basis_binit - basis_ainit
        basis_init_date.append(n0[j+1])
        basis_b.append( round(basis_init,2) ) 
        basis_c.append( -round(basis_init,2) )
        
    
    # second and third // third and forth // ... continually
    # make  basis_rebal_pos(B), basis_rebal_neg(C) when basis is positive(b) and negative(c) each
    # and save basis & dates
    basis_rebal_pos=[]
    basis_rebal_neg=[]
    rebal_date=[]
    k=0
    l=0
    # structure : two for loop iteration
    for l in range(len(count_purecol)): # l : length of purecol
        for m in range(k, len(cumsumcol[len(cumsumcol)-1])): # m : length of cumsumcol 
            # check the dates statement is true, if 'if' statement pass through, 
            # then choose first two columns and calculate 
            # check the dates statement is true if 'if' statement do not pass through(go to the else statement), 
            # then k=m(for set arange correctly), l +=1(choose next two columns)
            if cumsumcol[len(cumsumcol)-1][ m + len(df2.iloc[:,0].dropna()) ] <= purecol[l].iloc[-1]:
                daily_before = df2[ df2 [ df2.columns[2*l+2] ] == cumsumcol[len(cumsumcol)-1][ m + len(df2.iloc[:,0].dropna()) ] ].iloc[0,2*l+3]
                daily_after = df2[ df2 [ df2.columns[2*l+4] ] == cumsumcol[len(cumsumcol)-1][ m + len(df2.iloc[:,0].dropna()) ] ].iloc[0,2*l+5]
                basis_next=daily_before - daily_after # calculate basis
                rebal_date.append( cumsumcol[len(cumsumcol)-1][ m + len(df2.iloc[:,0].dropna()) ] ) # save rebal dates
                basis_rebal_pos.append( round((basis_next),2) ) # save positive basis
                basis_rebal_neg.append( -round((basis_next),2) ) # save negative basis (basis_pos= -basis_neg)
                
            else: # False:
               k=m
               l+=1
               break
        
    # For appending init_basis(using first two columns), reverse the order 
    # because when you append the value in the list, it is appended back to front
    basis_b.reverse()
    basis_c.reverse()
    basis_init_date.reverse()
    # if the last value of cumsum(basis) is positive(backwardation), go through this if statement
    if np.cumsum(basis_rebal_pos)[-1] >= 0:
        plot_df=pd.DataFrame()
        for i in range(len(basis_b)):
            if np.cumsum(basis_b)[-1] >= 0: # for append the values(basis & dates each)
                basis_rebal_pos.insert(0, basis_b[i])
                rebal_date.insert(0, basis_init_date[i])
        # make dataframe for plotting
        plot_df=pd.concat( [pd.DataFrame(rebal_date), pd.DataFrame(basis_rebal_pos)], ignore_index=True, axis=1 )
        plot_df=pd.concat( [plot_df, pd.DataFrame(np.cumsum(basis_rebal_pos))], ignore_index=True, axis=1)
        plt.plot(plot_df.dropna().iloc[:,0], plot_df.dropna().iloc[:,2])
        
        df_name_b.append(name)
        df_basis_b.append(np.cumsum(basis_rebal_pos)[-1])
        df_basis_total_b.append(basis_rebal_pos)
    
        print(name, "is", "Backwardation")
        print(np.cumsum(basis_rebal_pos)[-1])
        
    # if the last value of cumsum(basis) is negative(contango), go through this else if statement    
    else: 
        plot_df=pd.DataFrame()
        for i in range(len(basis_c)):
            if np.cumsum(basis_b)[-1] <= 0: # for append the values(basis & dates each)
                basis_c.reverse()
                basis_rebal_neg.insert(0, basis_c[i])
        # make dataframe for plotting
        plot_df=pd.concat( [pd.DataFrame(rebal_date), pd.DataFrame(basis_rebal_neg)], ignore_index=True, axis=1 )
        plot_df=pd.concat( [plot_df, pd.DataFrame(np.cumsum(basis_rebal_neg))], ignore_index=True, axis=1)
        plt.plot(plot_df.dropna().iloc[:,0], plot_df.dropna().iloc[:,2])
        
        df_name_c.append(name)
        df_basis_c.append(np.cumsum(basis_rebal_neg)[-1])
        df_basis_total_c.append(basis_rebal_neg)
        print(name, "is", "Contango")
        print(np.cumsum(basis_rebal_neg)[-1])


# For ranking of the futues(optional part when you use sheet_name(2))        
df_portfolio_b=pd.DataFrame()
df_portfolio_c=pd.DataFrame()

# save the futures names & basis
df_name_b=pd.Series(df_name_b)
df_basis_b=pd.Series(df_basis_b)
df_name_c=pd.Series(df_name_c)
df_basis_c=pd.Series(df_basis_c)

# concat portfolio_backwardation, portfolio_contagno(incuding futures names & basis)
df_portfolio_b=pd.concat([ pd.DataFrame(df_name_b), pd.DataFrame(df_basis_b) ], ignore_index=True, axis=1)
df_portfolio_c=pd.concat([ pd.DataFrame(df_name_c), pd.DataFrame(df_basis_c) ], ignore_index=True, axis=1)

# reset index for making new columns names
df_portfolio_b.reset_index()
df_portfolio_c.reset_index()
df_portfolio_b.columns=['b_name', 'b_basis']
df_portfolio_c.columns=['c_name', 'c_basis']

# sort the portfolio basis values in ascending(low -> high)
df_portfolio_b.sort_values(by='b_basis').index=range(len(df_portfolio_b))
df_portfolio_c.sort_values(by='c_basis').index=range(len(df_portfolio_c))

# pick number of futures name as you want 1 or 2 or 3, ...
pick_number=2
print('Backwardation 2nd, 1st', 'is\n', df_portfolio_b.sort_values(by='b_basis').iloc[:,0][(len(df_portfolio_b)-pick_number):len(df_portfolio_b)])
print(df_portfolio_b.sort_values(by='b_basis').iloc[:,1][(len(df_portfolio_b)-pick_number):len(df_portfolio_b)])
print('Contagno 2nd, 1st', 'is\n', df_portfolio_c.sort_values(by='c_basis').iloc[:,0][(len(df_portfolio_c)-pick_number):len(df_portfolio_c)])
print(df_portfolio_c.sort_values(by='c_basis').iloc[:,1][(len(df_portfolio_c)-pick_number):len(df_portfolio_c)])


###################################################################### 2nd part 
#****************************************************************************** 
###############################################################################
# Iterate 1st part one more after changing sheet_name(2)

# For making portfolio after sorting the basis values
# "Sugar"(b), "Soybean"(b), "Wheat"(c), "Aluminum"(c)
# initial margin / maintenance margin / trading unit in sequence
# Leverage = ( balanced price(정산가) * trading unit ) / initial margin
# balanced price is unit of daily price
ini_mar = {'soybean_meal' : [1980,1800,100], 'soybean' : [462,420,1000], 'soybean_oil' : [935, 850, 60000], 'corn' : [187,170,1000], 'wheat':[308, 280, 1000],'rough_rice' : [1073,975,2000],'live_cattle':[2255,2050,40000], 'lean_hog' :[1320,1200,40000], 'oats' : [1045,950,5000], 'copper':[511,511,1], 'aluminum' : [134,134,1], 'zinc':[895,895,5], 'platinum' : [2090,1900,50], 'lead':[198,198,1],'aluminum_alloy':[2380,2380,20],'crude_oil':[1375,1250,500],'natural_gas':[481,437,2500],'singapore':[1540,1400,2500],'brent_oil':[3300,3300,1000],'eua':[750,750,1000],'cocoa':[1595,1450,10],'sugar':[1232,1120,112000]}
ini_mar['sugar']
ini_mar['soybean']
ini_mar['wheat']
ini_mar['aluminum']
leverage_sugar=[]
leverage_soybean=[]
leverage_wheat=[]
leverage_aluminum=[]
lev_sugar=round(ini_mar['sugar'][2] / ini_mar['sugar'][0],2) # 90.91
lev_soybean=round(ini_mar['soybean'][2] / ini_mar['soybean'][0],2) # 2.16
lev_wheat=round(ini_mar['wheat'][2] / ini_mar['wheat'][0],2) # 3.25
lev_aluminum=round(ini_mar['aluminum'][2] / ini_mar['aluminum'][0],2) # 0.01

# make total portfolio and sub_basis_total(for multiplying basis * leverage)
total_portfolio=pd.DataFrame()
sub_basis_total=pd.DataFrame()

# sub_basis_total=pd.concat( [pd.DataFrame(np.multiply(df_basis_total_b[0], lev_sugar)), pd.DataFrame(np.multiply(df_basis_total_c[0], lev_wheat)) ], ignore_index=True, axis=1 )
sub_basis_total=pd.concat( [pd.DataFrame(np.multiply(df_basis_total_b[0], lev_sugar)), pd.DataFrame(np.multiply(df_basis_total_b[1], lev_soybean)), pd.DataFrame(np.multiply(df_basis_total_c[0], lev_wheat)),  pd.DataFrame(np.multiply(df_basis_total_c[1], lev_aluminum)) ], ignore_index=True, axis=1 )
sub_basis_total.T
total_portfolio=pd.concat( [ pd.DataFrame(rebal_date), sub_basis_total ], ignore_index=True, axis=1)
total_portfolio.reset_index()

# total_portfolio.columns=['Date', 'b_basis', 'c_basis']
total_portfolio.columns=['Date', 'b_basis1', 'b_basis2', 'c_basis1', 'c_basis2']

# total_portfolio['sum']=total_portfolio.b_basis+total_portfolio.c_basis
total_portfolio['sum']=total_portfolio.b_basis1+total_portfolio.b_basis2+total_portfolio.c_basis1+total_portfolio.c_basis2

# total_portfolio['cumsum']=np.cumsum(total_portfolio.iloc[:,3])
total_portfolio['cumsum']=np.cumsum(total_portfolio.iloc[:,5])
total_portfolio.dropna()

# plt.plot(total_portfolio.dropna().iloc[:,0], total_portfolio.dropna().iloc[:,4])
plt.plot(total_portfolio.dropna().iloc[:,0], total_portfolio.dropna().iloc[:,6])  


###################################################################### 3rd part 
#****************************************************************************** 
###############################################################################  
# calculate MDD using cumsum basis*lev
def Max_Drawdown(Input_data):
    Drawdown=[]
    for i in range(len(Input_data)):
        Drawdown.append((Input_data[i]-max(Input_data[0:i+1]))/max(Input_data[0:i+1]))

    return min(Drawdown)

Max_Drawdown(total_portfolio.dropna().iloc[:,4]) # -0.12862 (using each 1)
Max_Drawdown(total_portfolio.dropna().iloc[:,6]) # -0.11877 (using each 2)



