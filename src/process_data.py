import pandas

def CDR(df):	
	df["CDR"] = df["CDR"].fillna(0.0)
	df["CDR"] = df["CDR"].apply(lambda x: 2*x)
	df["CDR"] = df["CDR"].apply(lambda x: 3.0 if x >= 4 else x)
	return df

df = pandas.read_csv("../data/data_summary.csv", sep=',')
CDR(df).to_csv("../data/processed_data_summary.csv")