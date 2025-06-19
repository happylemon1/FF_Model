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
        print(f"--- Scraping Year: {year} ---") 
        try:
            draftURL = 'https://www.pro-football-reference.com/years/' + str(year) + '/draft.htm'
            time.sleep(3)
            response = requests.get(draftURL)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')


            table = soup.find('table', {'id': 'drafts'})
            if not table:
                print(f"Could not find draft table for {year}. Skipping.")
                continue

            table_body = table.find('tbody')
            rows = table_body.find_all('tr')
            
            for row in rows:
                try:
                    time.sleep(3)
                    player_cell = row.find('td', {'data-stat': 'player'})
                    if player_cell is None:
                        continue  # skip bad rows like round separators

                    playerName = player_cell.text.strip()

                    age = int(row.find('td', {'data-stat': 'age'}).text.strip())
                    
                    print(age)
                    draft_pick = int(row.find('td', {'data-stat': 'draft_pick'}).text.strip())
                    print(draft_pick)
                    position = str(row.find('td', {'data-stat': 'pos'}).text).strip()
                    print(position)
                    school = str(row.find('td', {'data-stat':'college_id'}).text).strip()
                    if position not in ['QB', 'RB', 'WR', 'TE']:
                        continue
                        
                    print(f"Processing {year} {position}: {playerName}")

                    #URL for player's college stat
                    collegeURLRow = row.find('td', {'data-stat':'college_link'})
                    a_tag = collegeURLRow.find('a')

                    if not a_tag:
                        try:
                            SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
                            parts = playerName.lower().split()

                        # Remove suffix if it's at the end
                            if parts[-1].strip(".") in SUFFIXES:
                                parts = parts[:-1]

                            first = parts[0]
                            last = parts[-1]
            
                            player_college_URL = f"https://www.sports-reference.com/cfb/players/{first}-{last}-1.html"


                        except Exception as e:
                            print(f"SKIPPING: No college stats link for {playerName}.")
                            continue
                    player_college_URL = a_tag['href']
                    #This is to get our a href thing 
                    

                    #playerURL Attmepts for their rookie stats:
                    urlRow = row.find('td', {'data-stat': 'player'})
                    nfl_a_tag = urlRow.find('a')                   
                    player_nfl_href = 'https://www.pro-football-reference.com/' + nfl_a_tag['href']
                    print(player_nfl_href)

                    college_stats_dict = generate_college_stats(position, player_college_URL)
                    print(college_stats_dict)
                    rookie_stats_dict = get_rookie_stats(year, position, player_nfl_href)
                    print(rookie_stats_dict)
                    #Get Player college stats through helper method

                    if rookie_stats_dict is None or college_stats_dict is None:
                        continue

                    for season_year, season_stats in college_stats_dict.items():
                        flat_row = {
                            'name': playerName,
                            'age': age,
                            'position': position,
                            'school': school,
                            'draft_pick': draft_pick,
                            'season': season_year,
                            **season_stats,
                            **rookie_stats_dict
                        }
                        print(flat_row)
                        if position == 'QB':
                            all_QB_records.append(flat_row)
                        elif position == 'RB':
                            all_RB_records.append(flat_row)
                        elif position == 'WR':
                            all_WR_records.append(flat_row)
                        elif position == 'TE':
                            all_TE_records.append(flat_row)


                except Exception as e: 
                    print("Tough luck")
                    continue
        except: 
            print("Oof")
            continue

    return all_QB_records, all_RB_records, all_WR_records, all_TE_records

def get_rookie_stats(year: int, position: str, draftURL: str) -> dict:
    try:
        time.sleep(3)
        response = requests.get(draftURL)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        playerDraftYear = year
        if position == 'QB':
            passing_table = soup.find('table', {'id': 'passing'})
            passing_tbody = passing_table.find('tbody')
            passing_Rookie_Row = passing_tbody.find('tr')

            games = int(passing_Rookie_Row.find('td', {'data-stat':'games'}).text.strip())
            passing_cmp = float(passing_Rookie_Row.find('td', {'data-stat':'pass_cmp_pct'}).text.strip())
            passing_yds = int(passing_Rookie_Row.find('td', {'data-stat':'pass_yds'}).text.strip())
            passing_tds = int(passing_Rookie_Row.find('td', {'data-stat':'pass_td'}).text.strip())
            passing_int = int(passing_Rookie_Row.find('td', {'data-stat':'pass_int'}).text.strip())   

            rushing_table = soup.find('table',{'id': 'rushing_and_receiving'}) 
            rushing_tbody = rushing_table.find('tbody')
            rushing_Rookie_Row =  rushing_tbody.find('tr')
            rushingYDS = int(rushing_Rookie_Row.find('td', {'data-stat':'rush_yds'}).text.strip())
            rushingTDS = int(rushing_Rookie_Row.find('td', {'data-stat':'rush_td'}).text.strip())
            receptions = int(rushing_Rookie_Row.find('td', {'data-stat': 'rec'}).text.strip())
            rec_tds = int(rushing_Rookie_Row.find('td', {'data-stat': 'rec_td'}).text.strip())
            rec_yds =  int(rushing_Rookie_Row.find('td', {'data-stat': 'rec_yds'}).text.strip())

            return {
                'games': games,
                'passCMP': passing_cmp,
                'passingYDS': passing_yds,
                'passingTDS': passing_tds,
                'passingINT': passing_int,
                'rushingYDS': rushingYDS,
                'rushingTDS': rushingTDS
            } 
        
        elif position == 'RB':
            rushing_table = soup.find('table',{'id': 'rushing_and_receiving'}) 
            rushing_tbody = rushing_table.find('tbody')
            rushing_Rookie_Row = rushing_tbody.find('tr')
            games = int(rushing_Rookie_Row.find('td', {'data-stat':'games'}).text.strip())
            rec_tds = int(rushing_Rookie_Row.find('td', {'data-stat': 'rec_td'}).text.strip())
            rec_yds =  int(rushing_Rookie_Row.find('td', {'data-stat': 'rec_yds'}).text.strip())
            receptions = int(rushing_Rookie_Row.find('td', {'data-stat': 'rec'}).text.strip())
            rushingYDS = int(rushing_Rookie_Row.find('td', {'data-stat':'rush_yds'}).text.strip())
            rushingTDS = int(rushing_Rookie_Row.find('td', {'data-stat':'rush_td'}).text.strip())

            return {
                'rushingYDS': rushingYDS,
                'rushingTDS': rushingTDS,
                'games':games,
                'recs': receptions,
                'recYDS': rec_yds,
                'recTDS': rec_tds
            }

        elif position == 'WR' or position == 'TE':
            receiving_table = soup.find('table', {'id': 'receiving_and_rushing'})
            receiving_tbody = receiving_table.find('tbody')
            receiving_rookie_row = receiving_tbody.find('tr')
            games = int(receiving_rookie_row.find('td', {'data-stat':'games'}).text.strip())
            receptions = int(receiving_rookie_row.find('td', {'data-stat': 'rec'}).text.strip())
            rec_yds =  int(receiving_rookie_row.find('td', {'data-stat': 'rec_yds'}).text.strip())
            rec_tds = int(receiving_rookie_row.find('td', {'data-stat': 'rec_td'}).text.strip())

            return {
                'games':games,
                'recs':receptions,
                'recYDS': rec_yds,
                'recTDS': rec_tds
            }
        else:
            print("Defense is retarded")
            return {}
    except Exception as e:
        return {}

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
"""
def generate_college_stats(position: str, player_college_URL: str) -> dict:
    time.sleep(3)
    print(player_college_URL)
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
            rushing_row = rushing_table_body.find('tr', {'id': 'rushing_standard.' + key})
            college_dict[key] = {
                'games': int(row.find('td', {'data-stat': 'games'}).text.strip()),
                'pass_cmp': int(row.find('td', {'data-stat': 'pass_cmp'}).text.strip()),
                'pass_att': int(row.find('td', {'data-stat': 'pass_att'}).text.strip()),
                'pass_cmp_pct': float(row.find('td', {'data-stat': 'pass_cmp_pct'}).text.strip()),
                'pass_yds': int(row.find('td', {'data-stat': 'pass_yds'}).text.strip()),
                'pass_td': int(row.find('td', {'data-stat': 'pass_td'}).text.strip()),
                'pass_td_pct': float(row.find('td', {'data-stat': 'pass_td_pct'}).text.strip()),
                'pass_int': int(row.find('td', {'data-stat': 'pass_int'}).text.strip()),
                'pass_int_pct': float(row.find('td', {'data-stat': 'pass_int_pct'}).text.strip()),
                'pass_yds_per_att': float(row.find('td', {'data-stat': 'pass_yds_per_att'}).text.strip()),
                'pass_rating': float(row.find('td', {'data-stat': 'pass_rating'}).text.strip()),
                'Rush Attempts': int(rushing_row.find('td', {'data-stat': 'rush_att'}).text.strip()),
                'Rush Yards': int(rushing_row.find('td', {'data-stat': 'rush_yds'}).text.strip()),
                'Rush TDs': float(rushing_row.find('td', {'data-stat': 'rush_td'}).text.strip()),
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
                'games': int(row.find('td', {'data-stat': 'games'}).text.strip()),
                'rush_att': int(row.find('td', {'data-stat': 'rush_att'}).text.strip()),
                'rush_yds': int(row.find('td', {'data-stat': 'rush_yds'}).text.strip()),
                'rush_td': int(row.find('td', {'data-stat': 'rush_td'}).text.strip()),
                'rec': int(row.find('td', {'data-stat': 'rec'}).text.strip()),
                'rec_yds': int(row.find('td', {'data-stat': 'rec_yds'}).text.strip()),
                'rec_td': int(row.find('td', {'data-stat': 'rec_td'}).text.strip()),
            }
    elif position == 'WR' or position == 'TE': 
        receiving_table = soup.find('table', {'id': 'receiving_standard'})
        receiving_table_body = receiving_table.find('tbody')
        receiving_rows = receiving_table_body.find_all('tr')

        for row in receiving_rows: 
            key = str(row.find('th').text).strip()
            cells = row.find_all('td')
            college_dict[key] = {
                'games': int(row.find('td', {'data-stat': 'games'}).text.strip()),
                'rec': int(row.find('td', {'data-stat': 'rec'}).text.strip()),
                'rec_yds': int(row.find('td', {'data-stat': 'rec_yds'}).text.strip()), 
                'rec_td': int(row.find('td', {'data-stat': 'rec_td'}).text.strip()), 
                'rush_att': int(row.find('td', {'data-stat': 'rush_att'}).text.strip()),
                'rush_yds': int(row.find('td', {'data-stat': 'rush_yds'}).text.strip()), 
                'rush_td': int(row.find('td', {'data-stat': 'rush_td'}).text.strip())
            }
    else: 
        print("Don't care about defense")
    return college_dict
"""

def generate_college_stats(position: str, player_college_URL: str) -> dict:
    try: 
        time.sleep(3)
        response = requests.get(player_college_URL)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        # with open("debug_page_source.html", "w", encoding="utf-8") as f:
            # f.write(str(soup))
        # print(soup.prettify()) # Or print a portion if too large
        college_dict = {}
        if position == 'QB':
            passing_table = soup.find('table',{'id': 'passing_standard'})
            passing_table_body= passing_table.find('tbody')
            passing_rows = passing_table_body.find_all('tr')

            rushing_table = soup.find('table', {'id': 'rushing_standard'})
            rushing_table_body = rushing_table.find('tbody')


            for row in passing_rows: 
                if row.get('class') is None:
                    time.sleep(3)
                    key = str(row.find('th').text).strip()
                    key = key.replace('*', '')
                    print(key)
                    rushing_row = rushing_table_body.find('tr', {'id': 'rushing_standard.' + key})
                    college_dict[key] = {
                        'games': int(row.find('td', {'data-stat': 'games'}).text.strip()),
                        'pass_cmp': int(row.find('td', {'data-stat': 'pass_cmp'}).text.strip()),
                        'pass_att': int(row.find('td', {'data-stat': 'pass_att'}).text.strip()),
                        'cmp%': float(row.find('td', {'data-stat': 'pass_att'}).text.strip()),
                        'yds': int(row.find('td', {'data-stat': 'pass_yds'}).text.strip()),
                        'TDs': int(row.find('td', {'data-stat': 'pass_td'}).text.strip()),
                        'TD%': float(row.find('td', {'data-stat': 'pass_td_pct'}).text.strip()),
                        'Int': int(row.find('td', {'data-stat': 'pass_int'}).text.strip()),
                        'Int%': float(row.find('td', {'data-stat': 'pass_int_pct'}).text.strip()),
                        'Rating': float(row.find('td', {'data-stat': 'pass_rating'}).text.strip()),
                        'Rush Attempts': int(rushing_row.find('td', {'data-stat': 'rush_att'}).text.strip()),
                        'Rush Yards': int(rushing_row.find('td', {'data-stat': 'rush_yds'}).text.strip()),
                        'Rush TDs': float(rushing_row.find('td', {'data-stat': 'rush_td'}).text.strip())
                    }
                else:
                    continue
        elif position == 'RB':
            rushing_table = soup.find('table', {'id': "rushing_standard"})
            if rushing_table == None:
                rushing_table = soup.find('table',{'id': 'receiving_standard'})
            rushing_body = rushing_table.find('tbody')
            rushing_rows = rushing_body.find_all('tr')
                    
            for row in rushing_rows:
                if row.get('class') is None:
                    key = str(row.find('th').text).strip()
                    # cells = row.find_all('td')
                    college_dict[key] = {
                        'games': int(row.find('td', {'data-stat': 'games'}).text.strip()),
                        'rushATT': int(row.find('td', {'data-stat': 'rush_att'}).text.strip()),
                        'rushYDS': int(row.find('td', {'data-stat': 'rush_yds'}).text.strip()),
                        'rushTDS': int(row.find('td', {'data-stat': 'rush_td'}).text.strip()),
                        'recs': int(row.find('td', {'data-stat': 'rec'}).text.strip()),
                        'recYDS': int(row.find('td', {'data-stat': 'rec_yds'}).text.strip()),
                        'recTDS': int(row.find('td', {'data-stat': 'rec_td'}).text.strip()),
                    }
                else: 
                    continue
        elif position == 'WR' or position == 'TE':
            receiving_table = soup.find('table', {'id': 'receiving_standard'})
            receiving_table_body = receiving_table.find('tbody')
            receiving_rows = receiving_table_body.find_all('tr')

            for row in receiving_rows:
                if row.get('class') is None:
                    key = str(row.find('th').text).strip()
                    cells = row.find_all('td')
                    college_dict[key] = {
                        'games': int(row.find('td', {'data-stat': 'games'}).text.strip()),
                        'recs': int(row.find('td', {'data-stat': 'rec'}).text.strip()),
                        'recYDS': int(row.find('td', {'data-stat': 'rec_yds'}).text.strip()), 
                        'recTDS': int(row.find('td', {'data-stat': 'rec_td'}).text.strip()), 
                        'rushAttempts': int(row.find('td', {'data-stat': 'rush_att'}).text.strip()),
                        'rushYards': int(row.find('td', {'data-stat': 'rush_yds'}).text.strip()), 
                        'rushTDS': int(row.find('td', {'data-stat': 'rush_td'}).text.strip())
                    }
                else:
                    continue
        else: 
            print("Don't care about defense")
        return college_dict
    except Exception as e: 
        return {}


    

def main():
        
    qb_list, rb_list, wr_list, te_list = scrape_all_player_data(YEARS_TO_SCRAPE)

    qb_df = pd.DataFrame(qb_list)
    rb_df = pd.DataFrame(rb_list)
    wr_df = pd.DataFrame(wr_list)
    te_df = pd.DataFrame(te_list)

    # 2. Write each DataFrame to a separate CSV file
    # You can specify the path and filename.
    # index=False prevents Pandas from writing the DataFrame index as a column in the CSV.

    qb_df.to_csv('qb_data.csv', index=False)
    rb_df.to_csv('rb_data.csv', index=False)
    wr_df.to_csv('wr_data.csv', index=False)
    te_df.to_csv('te_data.csv', index=False)

    print("Player data successfully saved to CSV files:")
    print("- qb_data.csv")
    print("- rb_data.csv")
    print("- wr_data.csv")
    print("- te_data.csv")


if __name__ == "__main__":
    main()