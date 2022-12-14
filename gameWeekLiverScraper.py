import requests
import json
import numpy as np
import pandas as pd
import datetime
import time

#NOTES



#Loading, parsing for fixtures API page


#grab all gameweek json files between startWeek and endWeek. First gameweek starts at 1. 
def scrapeGameweekFiles(startWeek: int, endWeek: int, toRun: int):

	if toRun == 0:
		pass

	else:
		#file path
		filePath = "C:/Users/bradl/OneDrive/Desktop/Professional/FPL/IntFiles/GameWeekFiles/"

		#loop through designated gameweeks
		for i in range(startWeek, endWeek+1):

			#variable for file name, set full path
			fileName = "gameWeek" + str(i) + "Data.json"
			fullPath = filePath + fileName

			#URL to be passed in requests.get
			gameWeekURL = "https://fantasy.premierleague.com/api/event/" + str(i) + "/live/"

			#send response, store json data
			response = requests.get(gameWeekURL)
			data = response.json()
			with open(fullPath, "w", encoding="utf-8") as f:
			    json.dump(data, f, ensure_ascii = False, indent = 4)
			f.close()

			print("Gameweek ", i , " data has been stored.")

			#wait until next request is sent to be mindful of amount of requests at once
			time.sleep(10)


def parseGameweekFiles(startWeek: int, endWeek: int, toRun: int):

	if toRun == 0:
		pass

	else:

		#will hold final dataframe to be sent into parsedFiles
		parsedData = pd.DataFrame()
		
		intFilePath = "C:/Users/bradl/OneDrive/Desktop/Professional/FPL/IntFiles/GameWeekFiles/"

		#loop through gameweeks
		for i in range(startWeek, endWeek+1):


			#Name for json file in intFile directory, set path to that directory as well
			fullIntPath = intFilePath + "gameWeek" + str(i) + "Data.json"


			f = open(fullIntPath, encoding="utf-8")
			data = json.load(f)

			#get throgh 1st main element of json file (elements)
			for block in data['elements']:
				
				#dict to hold all data per gamewekk for a player
				playerData = {}

				#set gameweek column
				playerData['gameweek'] = i

				#set player ID
				playerData['playerID'] = block['id']

				#loop through stats
				for statName,statVal in block['stats'].items():
					
					#Store data in dict as statistic : value (ex. minutes:90)
					playerData[statName] = statVal
			
				parsedData = parsedData.append(playerData, ignore_index=True)
					
			f.close()

		#send parsed data to folder as CSV
		parsedData.to_csv('C:/Users/bradl/OneDrive/Desktop/Professional/FPL/ParsedFiles/gameWeekData.csv',index=False)








#DRIVER############################

#scrape needed weeks, only ones that have not already been done before
scrapeGameweekFiles(6,8,0)

#parse all weeks 
parseGameweekFiles(1,8,1)
