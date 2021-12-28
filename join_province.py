
import pandas as pd


filename=r"G:\file\学校\可视化\大作业\COVID-19\COVID-19-Data-master\China\Province_level_summary\China_Province_summary_covid19_confirmed.csv"

df_province=pd.read_csv(filename,encoding="gbk")


df_province=df_province[["hasc","Province"]]
print("df_province")
print(df_province)
# filename_all_fac=r"D:\proj\visualization\bigwork\all_factors_china.xlsx"
filename_all_fac=r"D:\proj\visualization\bigwork\all_factors_china_no_idx.xlsx"


# df_all_fac=pd.read_csv(filename_all_fac,encoding="gbk")
# df_all_fac=pd.read_excel(filename_all_fac,encoding="gbk")
df_all_fac=pd.read_excel(filename_all_fac)
print("df_all_fac")
print(df_all_fac)
# df_all_fac.join(df_province,on="hasc")
# join( on df

pd_merge=pd.merge(df_all_fac,df_province,left_on="HASC_1",right_on="hasc")
print("pd_merge")
print(pd_merge)
# pd_merge.drop("Unnamed: 0",replace=True)
# pd_merge=pd_merge.drop("Unnamed: 0")
# pd_merge=pd_merge.drop(["Unnamed: 0"],axis=0)

# print("pd_merge")
# print(pd_merge)

pd_merge.to_excel("fac_china_with_province.xlsx",index=False)