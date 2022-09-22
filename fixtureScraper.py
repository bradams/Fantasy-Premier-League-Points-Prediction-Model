import requests
import json
import pandas as pd


#Loading, parsing for fixtures API page


#Scraping function, input 0 if we just need to parse data
def fixtureScraper(toRun):
	#Connect to page, get response, convert to JSON, and write to a .json file in folder
	if toRun == 0:
		pass
	else:
		print("Scraping fixture data")

		link = "https://fantasy.premierleague.com/api/fixtures/"
		response = requests.get(link)
		data = response.json()
		with open("C:/Users/bradl/OneDrive/Desktop/Professional/FPL/IntFiles/fixtures.json", "w", encoding="utf-8") as f:
		    json.dump(data, f, ensure_ascii = False, indent = 4)
		f.close()


#Parsing function
def fixtureParser(toRun):
	if toRun == 0:
		pass
	else:
		print("Parsing fixture data")

		#Reading and parsing JSON
		f = open('C:/Users/bradl/OneDrive/Desktop/Professional/FPL/IntFiles/fixtures.json', encoding="utf-8")
		data = json.load(f)

		#normalize JSON file
		dfFixtures = pd.json_normalize(data)

		#Send fixture data to CSV
		dfFixtures.to_csv('C:/Users/bradl/OneDrive/Desktop/Professional/FPL/ParsedFiles/fixtures.csv',index=False)

		f.close()



#DRIVER CODE##############

fixtureScraper(1)
fixtureParser(1)