import numpy as np 
import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment
import os
import re
import time
from pathlib import Path

YEARS_TO_SCRAPE = range(2024, 2025)

SECONDS_PER_REQUEST = 2


def scrape_all_draft_data(years_to_scrape: range) -> pd.DataFrame:
    all_player_records = []
    #helps with our requests
    preURL = 'https://www.sports-reference.com/cfb/'
    for year in YEARS_TO_SCRAPE:
        time.sleep(3)
        try:
            url = preURL+ "years" + "/" + str(year) + '-receiving.html'
            time.sleep(3)
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
           
            #This gets us the data for all receivers in a year into a BeautifulSoup Object

            table = soup.find('table',{'id': 'receiving_standard'})

            #Gives the column labels so we know how to seperate our information
            #table_thead  = table.find('thead')

            #We are trying to get the data for each row seperately
            table_body= table.find('tbody')
            rows = table_body.find_all('tr')

            for row in rows[:-1]:
                #This first line may not be needed I doubt it
                cells = row.find_all('td')
                firstColumn = row.find('td')

                #Use the table to find the player receiving yards in a certain year
                player_receiving_yards = int(row.find('td', {'data-stat': 'rec_yds'}).text.strip())
                print(player_receiving_yards)
                
                
                a_tag = firstColumn.find('a')
                # We are getting all the players's unique links and names
                if a_tag: 
                    player_name = a_tag.text

                    print(player_name) 

                    # Get draft pick, college stats
                    player_college_href = 'https://www.sports-reference.com/' + a_tag["href"]

                    print(player_college_href)

                    #Get player rookie year href?
                    # player_rookie_href = lastName_href_list(player_name)
                    try:
                        player_record = get_college_player(player_college_href, player_name, player_receiving_yards)
                        if player_record['position'] == 'WR':
                            all_player_records.append(player_record)
                    except Exception as e:
                        continue
                else:
                    continue


                #should decide what other paramateres we want to add to the data frame when we create the rows since
                #right now we are only storing player_href and maybe player_name 
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                print(f"Rate limit exceeded when fetching year {year}. Waiting for 60 seconds.")
                time.sleep(60)
                continue # Try the next year after waiting
            else:
                print(f"HTTP error occurred for year {year}: {e}")
                continue # Move to the next year on other HTTP errors
        except requests.exceptions.RequestException as e:
            print(f"Could not fetch data for year {year}: {e}")
            continue

    return pd.DataFrame(all_player_records)

def get_college_player(college_player_href: str, player_name: str, player_receiving_yards: int) -> dict:
    try:
        time.sleep(SECONDS_PER_REQUEST)
        #Dive into the college player's profiles
        playerURL = college_player_href
        response = requests.get(playerURL)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        #Finds the table with recieving data and finds the player position
        try:
            table_receiving = soup.find('table', {'id': 'receiving_standard'})
            table_receiving_body = table_receiving.find('tr')
            cells = table_receiving.find_all('td')
            player_position = str(cells[3].text).strip()
            print(player_position)
            return {
                'player_name': player_name,
                'position': player_position,
                'college_receiving_yards': player_receiving_yards
                # 'draft_pick':
            }
        except Exception as e:
            return {
                'position': "NOT RECEIVER"
            }
        
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            print(f"  --> Rate limit exceeded for {player_name}. Waiting 60s and skipping player.")
            time.sleep(60)
        else:
            print(f"  --> HTTP error for {player_name}: {e}")
        return None # Return None to indicate failure
    except Exception as e:
        print(f"  --> An error occurred processing {player_name}: {e}")
        return None


            
            
#Get list of hrefs with same last name as the player
def lastName_href_list(playerName: str) -> list:

    #Split player names to get the first Initial of their last name
    thisPlayerName = playerName
    splitPlayerName = thisPlayerName.split("")
    last_Name = splitPlayerName[-2]
    last_Name_first_Initial = last_Name[0]

    #Use the list to find the player specific url
    # list_url = https://www.pro-football-reference.com/players + last_Name_first_Initial

# def create_Nfl_Receiver_List(years: range) -> dict:
#     preURL = 'https://www.pro-football-reference.com/years/'
#     for year in years:
#         proURL = preURL + str(year) + '/draft.htm'
#         response = requests.get(proURL)
#         response.raise_for_status()
#         soup = BeautifulSoup(response.content, 'html.parser')

#         #This rushing standard is def not the actual id name
#         table = soup.find('table', {'id': ' rushing-standard' })


def main():
    # 1. Run the scraper to get the DataFrame
    player_data = scrape_all_draft_data(YEARS_TO_SCRAPE)

    # 2. Check if the DataFrame is not empty before saving
    if not player_data.empty:
        print("\nScraping complete! A sample of the data:")
        print(player_data.head())

        # --- NEW CODE TO DEFINE THE CORRECT PATH ---
        
        # Get the path to the directory of the current script (e.g., .../FF_MODEL/src)
        current_dir = Path(__file__).parent
        
        # Go up one level to the project root (e.g., .../FF_MODEL)
        project_root = current_dir.parent
        
        # Build the path to your raw data folder and define the filename
        # This creates a path like .../FF_MODEL/data/raw/college_wr_stats.csv
        output_path = project_root / 'data' / 'raw' / 'college_wr_stats.csv'
        
        # --- END OF NEW CODE ---

        # 3. Save the DataFrame to the newly defined path
        player_data.to_csv(output_path, index=False)
        
        print(f"\nSuccessfully saved data to: {output_path}")
    else:
        print("\nScraping finished, but no data was collected.")

if __name__ == "__main__":
    main()




