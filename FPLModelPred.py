#basics
import numpy as np
import pandas as pd

#models
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, Lasso, ElasticNetCV, RidgeCV, LassoCV

#for timing model runtimes
import time

#plotting
import seaborn as sns
import matplotlib.pyplot as plt

#Stats/preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OrdinalEncoder
from scipy import stats
from sklearn.model_selection import GridSearchCV, RepeatedKFold



#Load parsed files
eventsData = pd.read_csv("C:/Users/bradl/OneDrive/Desktop/Professional/FPL/ParsedFiles/events.csv")
playerData = pd.read_csv("C:/Users/bradl/OneDrive/Desktop/Professional/FPL/ParsedFiles/elements.csv") 
fixtureData = pd.read_csv("C:/Users/bradl/OneDrive/Desktop/Professional/FPL/ParsedFiles/fixtures.csv") 
teamData = pd.read_csv('C:/Users/bradl/OneDrive/Desktop/Professional/FPL/ParsedFiles/teamData.csv')
gameweekData = pd.read_csv('C:/Users/bradl/OneDrive/Desktop/Professional/FPL/ParsedFiles/gameWeekData.csv')


#Current team by ID: (Jose Sa, trent AA, Coufal, Cancelo, Bowen, James Ward-Prowse, Gabriel Martinelli, Mo Salah, James Maddison, Gabriel Jesus, Ollie Watkins, Matty Cash, Tanganga, Edouard)
currTeam = [478, 285, 463, 306, 465, 407, 19, 283, 261, 28, 40, 43, 439, 166]



def displayNullStats(display, dataframe):
    if display == 0:
        pass
    else:

    	#length of dataframe columns
        columnLength = dataframe.shape[0]

        #loop through columns
        for col in dataframe.columns:

            countNulls = dataframe[col].isnull().sum()

            percentNulls = round((countNulls / columnLength) * 100, 1)

            print("{} has {} nulls and {}% of nulls.".format(str(col), countNulls, percentNulls))
            

#This is solely used for the final predicted gameweek. will aggregate these, and then use the distinct value for that week on the rest
def aggregatePrevGameweeks(startweek: int, endweek: int, dataframe, features):

	#dataframe to hold aggregate gameweek data
	aggRes = pd.DataFrame()

	#output
	print("Aggregating data for gameweeks {} to {}".format(startweek, endweek))

	#get gameweek data between desired parameters, store in aggRes
	dataframe = dataframe[(dataframe['gameweek'] >= startweek) & (dataframe['gameweek'] <= endweek)]
	aggRes = dataframe.groupby('playerID')[features].mean()

	return aggRes


#Test a list of regression models, prints r2 score for analysis
def testRegressor(xTrain, yTrain, xTest, yTest, playerIDList, modelList, fName):
        
    #loop through models
	for mdl in modelList:

		#train/cross validate LinearRegression, others have pre-built cross validation model
		if mdl == LinearRegression():
			print("Training {} on {}".format(str(mdl), fName))
			scores = cross_val_score(mdl, xTrain, yTrain, scoring='r2', cv=5)
			print(scores)

		#train/cross validate other models
		else:

			print("Training {} on {}".format(str(mdl), fName))

			#Train and time model
			model = mdl
			startTime = time.time()
			model.fit(xTrain, yTrain)
			print(model.score(xTest, yTest))
			stopTime = time.time()
			timeToRun = round(stopTime - startTime, 3)
			print("Trained in {}s".format(timeToRun),"\n")

			#print coefficients if desired
			coefs = dict(zip(xTest.columns, model.coef_))
			print(coefs)



#Function to tune hyperparameters for chosen model        
def tuneHyperparameters(xTrainList, yTrainList, nameList, mdl):

	#hyperparameter grid
	grid = dict()
	grid['alpha'] = ([0.1, 0.25, 0.5, 0.75, 1, 2, 3, 4, 5, 7.5, 10])
	
	#get index to access all datasets/lists
	for i in range(len(xTrainList)):
		print("Training ", nameList[i])

		#chosen model
		model = mdl
		startTime = time.time()

		#set up cross validation
		cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

		#do grid search since there is only 1 hyperparameter being tuned
		search = GridSearchCV(model, grid, scoring='r2', cv=cv, n_jobs=-1)

		#store results
		results = search.fit(xTrainList[i], yTrainList[i])

		# summarize
		print('Best R2: %.3f' % results.best_score_)
		print('Best Param: %s' % results.best_params_)
	    
	    #end timer
		stopTime = time.time()
		timeToRun = round(stopTime - startTime, 3)
	        
		print("Trained in {}s".format(timeToRun),"\n")


#Make predictions, output to CSV files 
def makePredictions(xTrainList, yTrainList, xTestList, yTestList, playerIDLists, mdl, fNameList):
        
	#create index to loop through all datasets
	for i in range(len(xTrainList)):

		print("Training {} on {}".format(str(mdl), fNameList[i]))

		#chosen model
		model = mdl

		#time and train model
		startTime = time.time()
		model.fit(xTrainList[i], yTrainList[i])

		#create predictions
		yPred = np.around(model.predict(xTestList[i]),2)
	    
		#end timer
		stopTime = time.time()
		timeToRun = round(stopTime - startTime, 3)
		print("Trained in {}s".format(timeToRun))

		#store predictions in dataframe
		xTestList[i]['predPoints'] = yPred

		#add player IDs back in
		xTestList[i]['playerID'] = playerIDLists[i]

		#Merge in relevant data from player datasets
		xTestList[i] = pd.merge(xTestList[i], playerData[['id', 'first_name','second_name','now_cost']], left_on = 'playerID', right_on='id', how='left').drop('id',axis=1)

		#Create full player name
		xTestList[i]['playerName'] = xTestList[i]['first_name'] + " " + xTestList[i]['second_name']

		#dataframe to hold predictions, to be exported
		dataToExport = pd.DataFrame()
		dataToExport['playerName'] = xTestList[i]['playerName']
		dataToExport['predPoints'] = xTestList[i]['predPoints']
		dataToExport['now_cost'] = xTestList[i]['now_cost']
		dataToExport['PlayerID'] = xTestList[i]['playerID']
		dataToExport['TrueY'] = yTestList[i]
		dataToExport['pointsToCost'] = dataToExport['predPoints'] / dataToExport['now_cost']
		dataToExport['InTeam'] = np.where(np.isin(dataToExport['PlayerID'], currTeam), 1, 0)
		dataToExport = dataToExport.sort_values('predPoints', ascending = False)
		filePath = 'C:/Users/bradl/OneDrive/Desktop/Professional/FPL/Final/pred' + fNameList[i] + '.csv'
		dataToExport.to_csv(filePath,index=False)
		yield dataToExport


#takes in all dataframes, find largest differentials in the same cost, and sort by that
#look only to swap out mids or attackers as they are more consistnety
def suggestTransfer(transferDF, currTeam):


	transferListDF = pd.DataFrame(transferDF[~transferDF['PlayerID'].isin(currTeam)], columns = transferDF.columns)

	currTeamDF = pd.DataFrame(transferDF[transferDF['PlayerID'].isin(currTeam)], columns = transferDF.columns)

	suggestedTransfer = pd.DataFrame(columns = ['currPlayer','suggPlayer','currPoints','suggPoints','currCost','suggCost'])

	for idx1 in range(len(currTeamDF)):
		for idx2 in range(len(transferListDF)):
			#if current plater has a higher cost than the possible player....and has a lower predicted points
			if (currTeamDF.iloc[idx1,2] > transferListDF.iloc[idx2,2]) and (currTeamDF.iloc[idx1,1] < transferListDF.iloc[idx2,1]):
				suggestedTransfer.loc[len(suggestedTransfer.index)] = [currTeamDF.iloc[idx1,0], transferListDF.iloc[idx2,0], currTeamDF.iloc[idx1,1],transferListDF.iloc[idx2,1], currTeamDF.iloc[idx1,2], transferListDF.iloc[idx2,2]]

	suggestedTransfer['pointDiff'] = suggestedTransfer['currPoints'] - suggestedTransfer['suggPoints']
	suggestedTransfer['costDiff'] = suggestedTransfer['currCost'] - suggestedTransfer['suggCost']
	suggestedTransfer = suggestedTransfer.sort_values(by = 'pointDiff', ascending = True)
	suggestedTransfer.to_csv('C:/Users/bradl/OneDrive/Desktop/Professional/FPL/Final/transfers.csv',index=False, encoding='utf-8-sig')



#DRIVER CODE#####################################################################################################################################

#Global maps, variables

#Current gameweek variable (one to be predicted)
currGameweek = 9



positionMap = {'GK'  : 1,
			   'DEF' : 2,
			   'MID' : 3,
			   'FWD' : 4,
}


#TEAM DATA PREPROCESSING###########################################

#Clean up columns
teamDataColsToDrop = ['code','draw','loss','played','points','position','team_division','unavailable','win','pulse_id']
teamData = teamData.drop(teamDataColsToDrop,axis=1)

#Create map for team names (id : team name)
teamMap = dict(teamData[['id','name']].values)



#FIXTURE PREPROCESSING################################################

#Clean up columns
fixtureColsToDrop = ['code','finished_provisional','kickoff_time','provisional_start_time','started','stats','pulse_id']
fixtureData = fixtureData.drop(fixtureColsToDrop,axis=1)

#bring in home and away team names
fixtureData['homeTeamName'] = fixtureData['team_h'].map(teamMap)
fixtureData['awayTeamName'] = fixtureData['team_a'].map(teamMap)

#join in stregnth at home/away stats
fixtureData = pd.merge(fixtureData, teamData[['id','strength_overall_home']], how='left', left_on = 'team_h', right_on = 'id').drop('id_y',axis=1)
fixtureData = pd.merge(fixtureData, teamData[['id','strength_overall_away']], how='left', left_on = 'team_a', right_on = 'id').drop('id_x',axis=1)


#Positive means favoring the home team, negative means favoring away
fixtureData['difficultyDifferential'] = fixtureData['strength_overall_home'] - fixtureData['strength_overall_away']


#PLAYER PREPROCESSING#################################################

#columns we'll want to drop
playerDataColsToDrop = [ 'code','cost_change_event','cost_change_event_fall','cost_change_start','cost_change_start_fall','news','news_added','photo','special',
						'squad_number','team_code','transfers_in', 'transfers_in_event','transfers_out', 'transfers_out_event','value_form','value_season','web_name',
						'corners_and_indirect_freekicks_order','corners_and_indirect_freekicks_text', 'direct_freekicks_order','direct_freekicks_text', 'penalties_order', 
						'penalties_text']

#drop columns
playerData = playerData.drop(playerDataColsToDrop, axis = 1)

#join in team name to player data
playerData['teamName'] = playerData['team'].map(teamMap)



#GAMEWEEK PREPROCESSING##############################################################################################################




#last 4 weeks of data for training:

#split trainig and testing sets
trainData = gameweekData[gameweekData['gameweek'] != currGameweek]
testData = gameweekData[gameweekData == (currGameweek-1)]


#Add chance_of_playing_next_round?? form instead of average it??
testData = playerData[['id','team','status','element_type']]

#needed to merge fixture data
testData['event'] = currGameweek


#join in player data for train data
trainData = pd.merge(trainData, playerData[['id','element_type','form','team','status']], left_on = 'playerID', right_on = 'id', how ='left').drop('id',axis=1)


#Filter for active players (not injured, suspended, on loan, etc.)
trainData = trainData[trainData['status'] == 'a']
testData = testData[testData['status'] == 'a']

#drop status
trainData = trainData.drop('status',axis=1)
testData = testData.drop('status',axis=1)


#join for home team...check if team matches with team_h otherwise we know they'rw away
#1st step - join for the home team in a gameweek
trainData = pd.merge(trainData, fixtureData[['event','team_h']], left_on = ['gameweek','team'], right_on = ['event','team_h'], how='left').drop('event',axis=1)
testData = pd.merge(testData, fixtureData[['event','team_h']], left_on = ['event','team'], right_on = ['event','team_h'], how='left')

#Then join back to get away team
trainData = pd.merge(trainData, fixtureData[['event','team_a']], left_on = ['gameweek','team'], right_on = ['event','team_a'], how='left').drop('event',axis=1)
testData = pd.merge(testData, fixtureData[['event','team_a']], left_on = ['event','team'], right_on = ['event','team_a'], how='left')



#JOIN FOR OPPONENT TEAM
#opponent is home team, drop the away column as it is just the joining column. merge the team_h_x
trainData = pd.merge(trainData, fixtureData[['event','team_h','team_a']], left_on = ['gameweek','team'], right_on = ['event','team_a'], how='left', suffixes = ("","_y"))

#fill null home team values with merged home team, drop columns
trainData['team_h'] = trainData['team_h'].fillna(trainData['team_h_y'])
trainData = trainData.drop(['event', 'team_h_y', 'team_a_y'],axis=1)



#same process for testing data
testData = pd.merge(testData, fixtureData[['event','team_h','team_a']], left_on = ['event','team'], right_on = ['event','team_a'], how='left', suffixes = ("","_y"))

#fill null home team values with merged home team, drop columns
testData['team_h'] = testData['team_h'].fillna(testData['team_h_y'])
testData = testData.drop(['team_h_y', 'team_a_y'],axis=1)


#opponent is away team, drop the away column as it is just the joining column. merge the team_h_x or whatever
trainData = pd.merge(trainData, fixtureData[['event','team_h','team_a']], left_on = ['gameweek','team'], right_on = ['event','team_h'], how='left', suffixes = ("","_y"))
trainData['team_a'] = trainData['team_a'].fillna(trainData['team_a_y'])
trainData = trainData.drop(['event', 'team_h_y', 'team_a_y'],axis=1)

#same process for test data
testData = pd.merge(testData, fixtureData[['event','team_h','team_a']], left_on = ['event','team'], right_on = ['event','team_h'], how='left', suffixes = ("","_y"))
testData['team_a'] = testData['team_a'].fillna(testData['team_a_y'])
testData = testData.drop(['event', 'team_h_y', 'team_a_y'],axis=1)

#Create isHome flag to tell if player is at home that week
trainData['isHome'] = np.where(trainData['team'] == trainData['team_h'], 1, 0)
testData['isHome'] = np.where(testData['team'] == testData['team_h'], 1, 0)

#drop team column
trainData.drop('team',axis=1,inplace=True)
testData.drop('team',axis=1,inplace=True)


#Bring in home strength 
trainData = pd.merge(trainData, teamData[['id', 'strength_overall_home']], left_on = 'team_h', right_on='id', how='left').drop('id',axis=1)
testData = pd.merge(testData, teamData[['id', 'strength_overall_home']], left_on = 'team_h', right_on='id', how='left', suffixes = ['','_x']).drop('id_x',axis=1)

#bring in away strength
trainData = pd.merge(trainData, teamData[['id', 'strength_overall_away']], left_on = 'team_a', right_on='id', how='left').drop('id',axis=1)
testData = pd.merge(testData, teamData[['id', 'strength_overall_away']], left_on = 'team_a', right_on='id', how='left', suffixes = ['','_x']).drop('id_x',axis=1)

#get differential. positive means home is favored, negative means away is
trainData['strDiff'] = trainData['strength_overall_home'] - trainData['strength_overall_away']
testData['strDiff'] = testData['strength_overall_home'] - testData['strength_overall_away']


#drop gameweek 8 as most games havent been played
trainData = trainData[trainData['gameweek'] != 8]



#Check nulls in both dataframes
displayNullStats(0, trainData)
displayNullStats(0, testData)


#Split into ATK, MID, DEF, GK groups
trainDataGK = trainData[trainData['element_type'] == 1]
testDataGK = testData[testData['element_type'] == 1]

trainDataDEF = trainData[trainData['element_type'] == 2]
testDataDEF = testData[testData['element_type'] == 2]

trainDataMID = trainData[trainData['element_type'] == 3]
testDataMID = testData[testData['element_type'] == 3]

trainDataATK = trainData[trainData['element_type'] == 4]
testDataATK = testData[testData['element_type'] == 4]

#Get relevant stats by position
statsToAvgATK = ['minutes','goals_scored','assists','own_goals','penalties_saved','penalties_missed','yellow_cards','red_cards','bonus','influence','creativity','threat',
				'ict_index','form','total_points']

statsToAvgDEF = ['minutes','goals_scored','assists','clean_sheets','goals_conceded','own_goals','penalties_saved','penalties_missed','yellow_cards','red_cards','bonus','influence',
				'creativity','threat','ict_index','form','total_points']

statsToAvgGK = ['minutes','goals_scored','assists','clean_sheets','goals_conceded','own_goals','penalties_saved','penalties_missed','yellow_cards','red_cards','saves','bonus',
				'influence','creativity','threat','ict_index','form','total_points']

#columns that we'll use upcoming game's data for
nonAggregatedValues = ['strDiff','strength_overall_home','strength_overall_away','isHome']

#aggregate previous weeks between parameter 1 and 2
prevWeeksAvgATK = aggregatePrevGameweeks(4,7, trainDataATK, statsToAvgATK)
prevWeeksAvgMID = aggregatePrevGameweeks(4,7, trainDataMID, statsToAvgATK)
prevWeeksAvgDEF = aggregatePrevGameweeks(4,7, trainDataDEF, statsToAvgDEF)
prevWeeksAvgGK = aggregatePrevGameweeks(4,7, trainDataGK, statsToAvgGK)

#train model on x weeks of data
trainDataATK = trainDataATK[trainDataATK['gameweek'].isin([1,2,3])]
trainDataMID = trainDataMID[trainDataMID['gameweek'].isin([1,2,3])]
trainDataDEF = trainDataDEF[trainDataDEF['gameweek'].isin([1,2,3])]
trainDataGK = trainDataGK[trainDataGK['gameweek'].isin([1,2,3])]


#Clean up columns for ATK
testDataATK = testDataATK.drop(['element_type', 'team_h', 'team_a'],axis=1)
trainDataATK = trainDataATK.drop(['gameweek','clean_sheets', 'goals_conceded', 'saves','bps', 'element_type', 'team_h', 'team_a'],axis=1)

#Clean up columns for MID
testDataMID = testDataMID.drop(['element_type', 'team_h', 'team_a'],axis=1)
trainDataMID = trainDataMID.drop(['gameweek','clean_sheets', 'goals_conceded', 'saves','bps', 'element_type', 'team_h', 'team_a'],axis=1)

#Clean up columns for DEF
testDataDEF = testDataDEF.drop(['element_type', 'team_h', 'team_a'],axis=1)
trainDataDEF = trainDataDEF.drop(['gameweek', 'saves','bps', 'element_type', 'team_h', 'team_a'],axis=1)

#Clean up columns for GK
testDataGK = testDataGK.drop(['element_type', 'team_h', 'team_a'],axis=1)
trainDataGK = trainDataGK.drop(['gameweek','bps', 'element_type', 'team_h', 'team_a'],axis=1)



#Merge testing data to aggregated columns
testDataATK = pd.merge(testDataATK, prevWeeksAvgATK, left_on = 'id', right_on = 'playerID', how='left')
testDataMID = pd.merge(testDataMID, prevWeeksAvgMID, left_on = 'id', right_on = 'playerID', how='left')
testDataDEF = pd.merge(testDataDEF, prevWeeksAvgDEF, left_on = 'id', right_on = 'playerID', how='left')
testDataGK = pd.merge(testDataGK, prevWeeksAvgGK, left_on = 'id', right_on = 'playerID', how='left')

#testDataMID.to_csv('TESTMID.csv')

testDataMID = testDataMID.dropna()
testDataDEF = testDataMID.dropna()
testDataGK = testDataMID.dropna()

#Store player ID to insert again after predictions
testPlayerIdATK = testDataATK['id']
testPlayerIdMID = testDataMID['id']
testPlayerIdDEF = testDataDEF['id']
testPlayerIdGK = testDataGK['id']

#drop player IDs for all dataframes
trainDataATK = trainDataATK.drop('playerID',axis=1)
testDataATK = testDataATK.drop('id',axis=1)

trainDataMID = trainDataMID.drop('playerID',axis=1)
testDataMID = testDataMID.drop('id',axis=1)

trainDataDEF = trainDataDEF.drop('playerID',axis=1)
testDataDEF = testDataDEF.drop('id',axis=1)

trainDataGK = trainDataGK.drop('playerID',axis=1)
testDataGK = testDataGK.drop('id',axis=1)




#target variable
yTrainATK = trainDataATK['total_points']
yTestATK = testDataATK['total_points']

yTrainMID = trainDataMID['total_points']
yTestMID = testDataMID['total_points']

yTrainDEF = trainDataDEF['total_points']
yTestDEF = testDataDEF['total_points']

yTrainGK = trainDataGK['total_points']
yTestGK = testDataGK['total_points']


trainDataATK = trainDataATK.drop(['total_points','in_dreamteam'],axis=1)
testDataATK = testDataATK.drop(['total_points'],axis=1)

trainDataMID = trainDataMID.drop(['total_points','in_dreamteam'],axis=1)
testDataMID = testDataMID.drop(['total_points'],axis=1)

trainDataDEF = trainDataDEF.drop(['total_points','in_dreamteam'],axis=1)
testDataDEF = testDataDEF.drop(['total_points'],axis=1)

trainDataGK = trainDataGK.drop(['total_points','in_dreamteam'],axis=1)
testDataGK = testDataGK.drop(['total_points'],axis=1)


#align column order
trainDataATK = trainDataATK[testDataATK.columns]
trainDataMID = trainDataMID[testDataMID.columns]
trainDataDEF = trainDataDEF[testDataDEF.columns]
trainDataGK = trainDataGK[testDataGK.columns]

#for col in trainDataATK:
#	sns.histplot(data=trainDataATK, x=col, kde=True, palette='dark').set(title='Histogram of {}'.format(str(col)))
#	plt.show()



##MODEL PREPROCESSING (Scaling)
minMaxScaler = MinMaxScaler(feature_range=(0,1))

dfset = [trainDataATK, testDataATK, trainDataMID, testDataMID, trainDataDEF, testDataDEF, trainDataGK, testDataGK]

#Scale variables
for df in dfset:
	for col in df.columns:
		df[col] = minMaxScaler.fit_transform(df[col].values.reshape(-1,1))



#Check nulls in both dataframes
displayNullStats(0, trainDataATK)
displayNullStats(0, testDataATK)
displayNullStats(0, trainDataMID)
displayNullStats(0, testDataMID)
displayNullStats(0, trainDataDEF)
displayNullStats(0, testDataDEF)
displayNullStats(0, trainDataGK)
displayNullStats(0, testDataGK)






#MODELING##################################################################

#model to test
modelList = [LinearRegression(), 
			 ElasticNetCV(cv=5), 
			 RidgeCV(cv=5, alphas = [.1, 1.0, 5.0]),
			 LassoCV(cv=5, alphas = [.1, 1.0, 5.0])]



#test each model across the different datasets, print r2 score
#testRegressor(trainDataATK, yTrainATK, testDataATK, yTestATK, testPlayerIdATK, modelList, 'ATK')
#testRegressor(trainDataMID, yTrainMID, testDataMID, yTestMID, testPlayerIdMID, modelList, 'MID')
#testRegressor(trainDataDEF, yTrainDEF, testDataDEF, yTestDEF, testPlayerIdDEF, modelList, 'DEF')
#testRegressor(trainDataGK, yTrainGK, testDataGK, yTestGK, testPlayerIdGK, modelList, 'GK')

#create list of datasets to loop through
xTrL = [trainDataATK, trainDataMID, trainDataDEF, trainDataGK]
xTeL = [testDataATK, testDataMID, testDataDEF, testDataGK]
yTrL = [yTrainATK, yTrainMID, yTrainDEF, yTrainGK]
yTeL = [yTestATK, yTestMID, yTestDEF, yTestGK]

#names to print which dataset is being accessed
nameList = ['ATK', 'MID', 'DEF', 'GK']

#player ID lists to add into predictions
playerIDs = [testPlayerIdATK, testPlayerIdMID, testPlayerIdDEF, testPlayerIdGK]

#find best alpha value
#tuneHyperparameters(xTrL, yTrL, nameList, Ridge())

#Make Predictions for next week, output to CSV
atkPred, midPred, defPred, gkPred = makePredictions(xTrL, yTrL, xTeL, yTeL, playerIDs, Ridge(alpha = 0.1), nameList)


atkMidJoined = pd.concat([atkPred, midPred])

suggestTransfer(atkMidJoined, currTeam)

