This project predicts the amount of fantasy points that a player will earn in a given gameweek.

It start by scraping the data from the Fantasy Premier League (FPL) API and storing the data as JSON files in a specified directory.
The data is parsed and stored in separate files as Pandas DataFrames, as these are significantly easier to work with. The data stored is as following:

  1) playerData.csv
      -This file contains all data relating to players, including:
  3)
  4)
  5)
  6)

The final piece of this project is the script for preprocessing and building the model, which outputs each player and their predicted fantasy points for the next week. The first part of this process is preprocessing, which joins together relvant data from all the different files. Once the data is joined together the columns are scaled. 

The model is trained on all data except for the last 5 weeks, which are used to make the predictions. Most of the features are passed into an aggregation function which takes in a list of features to aggregate and takes the mean over the last 5 games (so average minutes played, goals scored, etc. in the last 5 games). There are also some other features, such as whether the player is at home, strength of the opposition team that week, etc. that are not aggregated but included in the model. Finally, a regression model is fit to predict fantasy points on those features

