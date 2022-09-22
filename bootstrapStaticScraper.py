import requests
import json
import pandas as pd




#Connect to page, get response, convert to JSON, and write to a .json file in folder
def scrapeBootStrapStatic(toRun):
	if toRun == 0:
		pass
	else:
		link = "https://fantasy.premierleague.com/api/bootstrap-static/"
		response = requests.get(link)
		data = response.json()
		with open("C:/Users/bradl/OneDrive/Desktop/Professional/FPL/IntFiles/response.json", "w", encoding="utf-8") as f:
		    json.dump(data, f, ensure_ascii = False, indent = 4)
		f.close()    




#EVENTS DATA################################

def parseEventData(toRun):
	if toRun == 0:
		pass
	else:
		print("Parsing event data.")
		eventDataDF = pd.DataFrame()

		#loop through each gameweek
		for event in data['events']:

			#dict to hold data
			eventData = {}

			#loop through gameweek info
			for eventItem, eventVal in event.items():

				#pass chip plays, don't need these
				if eventItem == 'chip_plays':
					pass

				#want this data - loop through its elements
				elif eventItem == 'top_element_info' and eventData['id'] <= 4:

					#loop through next level of top_element_info
					for topElemInfo, topElemVal in eventVal.items():

							#dict key name
							keyName = "top_element_info_" + topElemInfo

							#store
							eventData[keyName] = topElemVal

				#otherwise, drop right into dict
				elif eventItem not in ['chip_plays','top_element_info']:
					eventData[eventItem] = eventVal

			#append dict data to dataframe
			eventDataDF = eventDataDF.append(eventData, ignore_index=True)

		#push to CSV
		eventDataDF.to_csv('C:/Users/bradl/OneDrive/Desktop/Professional/FPL/ParsedFiles/events.csv',index=False)




#TEAM DATA
def parseTeamData(toRun):
	if toRun == 0:
		pass
	else:
		print("Parsing team data")
		teamDataDF = pd.DataFrame()

		for team in data['teams']:

			#dict to hold team data
			teamData = {}


			#print(gameWeek,"\n")
			for teamItem,teamVal in team.items():

				#store data in dict
				teamData[teamItem] = teamVal

			teamDataDF = teamDataDF.append(teamData, ignore_index=True)


		teamDataDF.to_csv('C:/Users/bradl/OneDrive/Desktop/Professional/FPL/ParsedFiles/teamData.csv',index=False)




#ELEMENTS DATA#####################
#player data

def parseElementsData(toRun):
	if toRun == 0:
		pass
	else:
		print("Parsing elements data")
		dfElements = pd.json_normalize(data['elements'])
		dfElements.to_csv('C:/Users/bradl/OneDrive/Desktop/Professional/FPL/ParsedFiles/elements.csv',encoding='utf-8-sig',index=False)







#DRIVER CODE###########

#scapes entire page, stores as JSON. 
scrapeBootStrapStatic(1)


#Reading and parsing JSON from scrapeBootStrapStatic(), read into data variable
f = open('C:/Users/bradl/OneDrive/Desktop/Professional/FPL/IntFiles/response.json', encoding="utf-8")
data = json.load(f)


#parses out event data from data 
parseEventData(1)

#parses out team data from data 
parseTeamData(1)

#parses out player data
parseElementsData(1)

#close file
f.close()