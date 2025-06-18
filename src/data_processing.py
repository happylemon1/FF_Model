import numpy as np 
import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment
import os
import re
import time
from pathlib import Path

YEARS_TO_SCRAPE = range(2015, 2025)

CURRENT_ROOKIES_YEAR = 2025
CURRENT_ROOKIES_STATS = 2024

def scrape_all_player_data(YEARS_TO_SCRAPE: range):
    all_QB_records = []
    all_RB_records = []
    all_WR_records = []
    all_TE_records = []
    #Scrape the draft html for every year
    for year in YEARS_TO_SCRAPE:
        print(year)
        draftURL = 'https://www.pro-football-reference.com/years/' + str(year) + '/draft.htm'
        time.sleep(3)
        response = requests.get(draftURL)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')


        table = soup.find('table', {'id': 'drafts'})
        table_body = table.find('tbody')
        rows = table_body.find_all('tr')
        
        for row in rows:
            time.sleep(3)
            playerName= str(row.find('td', {'data-stat':'player'}).text).strip()
            age = int(row.find('td', {'data-stat': 'age'}).text.strip())
            print(age)
            draft_pick = int(row.find('td', {'data-stat': 'draft_pick'}).text.strip())
            print(draft_pick)
            position = str(row.find('td', {'data-stat': 'pos'}).text).strip()
            print(position)

            #URL for player's college stat
            collegeURLRow = row.find('td', {'data-stat':'college_link'})
            a_tag = collegeURLRow.find('a')
            player_college_URL = a_tag['href']
            #This is to get our a href thing 
            

            #playerURL Attmepts for their rookie stats:
            urlRow = row.find('td', {'data-stat': 'player'})
            a_tag = urlRow.find('a')
            player_nfl_href = 'https://www.pro-football-reference.com/' + a_tag['href']
            rookie_stats_dict = get_rookie_stats(year, position, player_nfl_href)
            
            
            #Get Player college stats through helper method
            college_stats_dict = generate_college_stats(position, player_college_URL)
            player_dict = {
                'name': playerName, 
                'age': age, 
                'position': position, 
                'draft_pick': draft_pick, 
                'college_stats_dict': college_stats_dict, 
                'rookie_stats_dict': rookie_stats_dict
            }
            print(player_dict)

            if player_dict['position'] == 'QB': 
                all_QB_records.append(player_dict)
            elif player_dict['position'] == 'RB': 
                all_RB_records.append(player_dict)
            elif player_dict['position'] == 'WR': 
                all_WR_records.append(player_dict)
                # I don't think we should do Tight end but we can
            elif player_dict['position'] == 'TE': 
                all_TE_records.append(player_dict)
            else: 
                continue
    return all_QB_records, all_RB_records, all_WR_records, all_TE_records

def get_rookie_stats(year: int, position: str, draftURL: str) -> dict:
    response = requests.get(draftURL)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, 'html.parser')
    playerDraftYear = year
    if position == 'QB':
        passing_table = soup.find('table', {'id': 'passing'})
        passing_Rookie_Row = passing_table.find('tr', {'id':'passing.' + str(year)})
        
        passing_cmp = float(passing_Rookie_Row.find('td', {'data-stat':'pass_comp_pct'}).text.strip())
        passing_yds = int(passing_Rookie_Row.find('td', {'data-stat':'games'}).text.strip())
        passing_tds = int(passing_Rookie_Row.find('td', {'data-stat':'pass_td'}).text.strip())
        passing_int = int(passing_Rookie_Row.find('td', {'data-stat':'pass_int'}).text.strip())   

        rushing_table = soup.find('table',{'id': 'rushing_and_receiving'}) 
        rushing_Rookie_Row =  rushing_table.find('tr', {'id':'rushing_and_receiving.' + str(year)})
        rushingYDS = int(rushing_Rookie_Row.find('td', {'data-stat':'rush-yds'})).text.strip()
        rushingTDS = int(rushing_Rookie_Row.find('td', {'data-stat':'rush-yds'})).text.strip()
    """
    if position == 'QB':
        passing_completion = float((int(row.find('td', {'data-stat': 'pass_cmp'}).text.strip()))/(int(row.find('td', {'data-stat': 'pass_att'}).text.strip())))
        passing_yds = int(row.find('td', {'data-stat': 'pass_yds'}).text.strip())
        passing_tds = int(row.find('td', {'data-stat': 'pass_td'}).text.strip())
        passing_int = int(row.find('td', {'data-stat': 'pass_int'}).text.strip())
        rushing_yds = int(row.find('td', {'data-stat': 'rush_yds'}).text.strip())
        rushing_tds = int(row.find('td', {'data-stat': 'rush_td'}).text.strip())

        return {
            'passCMP': passing_completion,
            'passingYDS': passing_yds,
            'passingTDS': passing_tds,
            'passingINT': passing_int,
            'rushingYDS': rushing_yds,
            'rushingTDS': rushing_tds
        }
    if position == 'RB':
        rushing_yds = int(row.find('td', {'data-stat': 'rush_yds'}).text.strip())
        rushing_tds = int(row.find('td', {'data-stat': 'rush_td'}).text.strip())
        receptions = int(row.find('td', {'data-stat': 'rec'}).text.strip())
        rec_tds = int(row.find('td', {'data-stat': 'rec_td'}).text.strip())
        rec_yds =  int(row.find('td', {'data-stat': 'rec_yds'}).text.strip())
        rec_tds = int(row.find('td', {'data-stat': 'rec_td'}).text.strip())

        return {
            'rushingYDS': rushing_yds,
            'rushingTDS': rushing_tds,
            'recs': receptions,
            'recYDS': rec_yds,
            'recTDS': rec_tds
        }
    if position == 'WR' or position == 'TE':
        receptions = int(row.find('td', {'data-stat': 'rec'}).text.strip())
        rec_yds =  int(row.find('td', {'data-stat': 'rec_yds'}).text.strip())
        rec_tds = int(row.find('td', {'data-stat': 'rec_td'}).text.strip())

        return {
            'recs':receptions,
            'recYDS': rec_yds,
            'recTDS': rec_tds
        }
    """

def generate_college_stats(position: str, player_college_URL: str) -> dict:
    time.sleep(3)
    response = requests.get(player_college_URL)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, 'html.parser')
    college_dict = {}
    if position == 'QB':
        passing_table = soup.find('table',{'id': 'passing_standard'})
        passing_table_body= passing_table.find('tbody')
        passing_rows = passing_table_body.find_all('tr')

        rushing_table = soup.find('table', {'id': 'rushing_standard'})
        rushing_table_body = rushing_table.find('tbody')


        for row in passing_rows: 
            time.sleep(3)
            key = str(row.find('th').text).strip()
            key = key.replace('*', '')
            print(key)
            cells = row.find_all('td')
            rushing_row = rushing_table_body.find('tr', {'id': 'rushing_standard.' + key})
            rushing_cells = rushing_row.find_all('td')
            college_dict[key] = {
                'games': int(cells[4].text.strip()),
                'cmp': int(cells[5].text.strip()),
                'att': int(cells[6].text.strip()),
                'cmp%': float(cells[7].text.strip()),
                'yds': int(cells[8].text.strip()),
                'TDs': int(cells[9].text.strip()),
                'TD%': float(cells[10].text.strip()),
                'Int': int(cells[11].text.strip()),
                'Int%': float(cells[12].text.strip()),
                'Y/A': float(cells[13].text.strip()),
                'Rating': float(cells[17].text.strip()),
                'Rush Attempts': int(rushing_cells[5].text.strip()),
                'Rush Yards': int(rushing_cells[6].text.strip()),
                'Rush TDs': float(rushing_cells[8].text.strip())
            }
    elif position == 'RB':
        rushing_table = soup.find('table', {'id': "rushing_standard"})
        if rushing_table == None:
            rushing_table = soup.find('table',{'id': 'receiving_standard'})
        rushing_body = rushing_table.find('tbody')
        rushing_rows = rushing_body.find_all('tr')
                
        for row in rushing_rows:
            key = str(row.find('th').text).strip()
            cells = row.find_all('td')
            college_dict[key] = {
                'games': int(cells[4].text.strip()),
                'rushATT': int(cells[5].text.strip()),
                'rushYDS': int(cells[6].text.strip()),
                'rushTDS': int(cells[8].text.strip()),
                'recs': int(cells[10].text.strip()),
                'recYDS': int(cells[11].text.strip()),
                'recTDS': int(cells[13].text.strip()),
            }
    elif position == 'WR' or position == 'TE': 
        receiving_table = soup.find('table', {'id': 'receiving_standard'})
        receiving_table_body = receiving_table.find('tbody')
        receiving_rows = receiving_table_body.find_all('tr')

        for row in receiving_rows: 
            key = str(row.find('th').text).strip()
            cells = row.find_all('td')
            college_dict[key] = {
                'games': int(cells[4].text.strip()),
                'recs': int(cells[5].text.strip()),
                'recYDS': int(cells[6].text.strip()), 
                'recTDS': int(cells[8].text.strip()), 
                'Y/R': float(cells[7].text.strip()), 
                'rushAttempts': int(cells[10].text.strip()),
                'rushYards': int(cells[11].text.strip()), 
                'rushTDS': int(cells[12].text.strip())
            }
    else: 
        print("Don't care about defense")
    return college_dict

    

def main():
    qb_list, rb_list, wr_list, te_list = scrape_all_player_data(YEARS_TO_SCRAPE)


if __name__ == "__main__":
    main()