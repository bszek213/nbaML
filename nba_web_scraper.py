#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
html parse code - NBA
@author: brianszekely
"""
import requests
from bs4 import BeautifulSoup
from pandas import DataFrame
import cfbd
from numpy import nan
from time import sleep
from os.path import join, exists
from os import getcwd

def html_to_df_web_scrape_NBA(URL,URL1,team,year):
    while True:
        try:
            page = requests.get(URL)
            soup = BeautifulSoup(page.content, "html.parser")
            page1 = requests.get(URL1)
            soup1 = BeautifulSoup(page1.content, "html.parser")
            break
        except:
            print('HTTPSConnectionPool(host="www.sports-reference.com", port=443): Max retries exceeded. Retry in 10 seconds')
            sleep(10)
    table = soup.find(id="all_tgl_basic")
    table1 = soup.find(id="all_tgl_basic")
    tbody = table.find('tbody')
    tbody1 = table1.find('tbody')
    tr_body = tbody.find_all('tr')
    tr_body1 = tbody1.find_all('tr')
    # game_season = []
    # date_game = []
    # game_location = []
    # opp_id= []
    # BASIC STATS
    game_result= []
    pts= []
    opp_pts= []
    fg= []
    fga= []
    fg_pct= []
    fg3= []
    fg3a= []
    fg3_pct= []
    ft= []
    fta= []
    ft_pct= []
    orb= []
    total_board= []
    ast= []
    stl= []
    blk= []
    tov= []
    pf= []
    opp_fg = []
    opp_fga= []
    opp_fg_pct= []
    opp_fg3= []
    opp_fg3a= []
    opp_fg3_pct= []
    opp_ft= []
    opp_fta= []
    opp_ft_pct= []
    opp_orb= []
    opp_trb= []
    opp_ast= []
    opp_stl= []
    opp_blk= []
    opp_tov= []
    opp_pf= []
    for trb in tr_body:
        for td in trb.find_all('td'):
            if td.get('data-stat') == "game_result":
                if td.get_text() == 'W':
                    game_result.append(1)
                else:
                    game_result.append(0)
            if td.get('data-stat') == "pts":
                pts.append(td.get_text())
            if td.get('data-stat') == "opp_pts":
                opp_pts.append(td.get_text())
            if td.get('data-stat') == "fg":
                fg.append(td.get_text())
            if td.get('data-stat') == "fga":
                fga.append(td.get_text())
            if td.get('data-stat') == "fg_pct":
                fg_pct.append(td.get_text())
            if td.get('data-stat') == "fg3":
                fg3.append(td.get_text())
            if td.get('data-stat') == "fg3a":
                fg3a.append(td.get_text())
            if td.get('data-stat') == "fg3_pct":
                fg3_pct.append(td.get_text())
            if td.get('data-stat') == "ft":
                ft.append(td.get_text())
            if td.get('data-stat') == "fta":
                fta.append(td.get_text())
            if td.get('data-stat') == "ft_pct":
                ft_pct.append(td.get_text())
            if td.get('data-stat') == "orb":
                orb.append(td.get_text())
            if td.get('data-stat') == "trb":
                total_board.append(td.get_text())
            if td.get('data-stat') == "ast":
                ast.append(td.get_text())
            if td.get('data-stat') == "stl":
                stl.append(td.get_text())
            if td.get('data-stat') == "blk":
                blk.append(td.get_text())
            if td.get('data-stat') == "tov":
                tov.append(td.get_text())
            if td.get('data-stat') == "pf":
                pf.append(td.get_text())
            if td.get('data-stat') == "opp_fg":
                opp_fg.append(td.get_text())
            if td.get('data-stat') == "opp_fga":
                opp_fga.append(td.get_text())
            if td.get('data-stat') == "opp_fg_pct":
                opp_fg_pct.append(td.get_text())
            if td.get('data-stat') == "opp_fg3":
                opp_fg3.append(td.get_text())
            if td.get('data-stat') == "opp_fg3a":
                opp_fg3a.append(td.get_text())
            if td.get('data-stat') == "opp_fg3_pct":
                opp_fg3_pct.append(td.get_text())
            if td.get('data-stat') == "opp_ft":
                opp_ft.append(td.get_text())
            if td.get('data-stat') == "opp_fta":
                opp_fta.append(td.get_text())
            if td.get('data-stat') == "opp_ft_pct":
                opp_ft_pct.append(td.get_text())
            if td.get('data-stat') == "opp_orb":
                opp_orb.append(td.get_text())
            if td.get('data-stat') == "opp_trb":
                opp_trb.append(td.get_text())
            if td.get('data-stat') == "opp_ast":
                opp_ast.append(td.get_text())
            if td.get('data-stat') == "opp_stl":
                opp_stl.append(td.get_text())
            if td.get('data-stat') == "opp_blk":
                opp_blk.append(td.get_text())
            if td.get('data-stat') == "opp_tov":
                opp_tov.append(td.get_text())
            if td.get('data-stat') == "opp_pf":
                opp_pf.append(td.get_text())      
    #ADVANCED STATS
    off_rtg = []
    def_rtg = []
    pace = []
    fta_per_fga_pct = []
    fg3a_per_fga_pct = []
    ts_pct = []
    trb_pct = []
    ast_pct = []
    stl_pct = [] 
    blk_pct = []
    efg_pct = []
    tov_pct = []
    orb_pct = []
    ft_rate = []
    opp_efg_pct= []
    opp_tov_pct = []
    drb_pct = []
    opp_ft_rate = []
    DataFrame(list(zip(game_result,pts,opp_pts,fg,fga,
    fg_pct,fg3,fg3a,fg3_pct,ft,fta,ft_pct,orb,total_board,ast,
    stl,blk,tov,pf,opp_fg,opp_fga,opp_fg_pct,opp_fg3,opp_fg3a,opp_fg3_pct,
    opp_ft,opp_fta,opp_ft_pct,opp_orb,opp_trb,opp_ast,opp_stl,opp_blk,opp_tov,
    opp_pf, off_rtg,def_rtg,pace,fta_per_fga_pct,fg3a_per_fga_pct,ts_pct,
    trb_pct,ast_pct,stl_pct,blk_pct,efg_pct,tov_pct,orb_pct,ft_rate,opp_efg_pct,
    opp_tov_pct,drb_pct,opp_ft_rate)),
            columns =['game_result','pts','opp_pts','fg','fga',
            'fg_pct','fg3','fg3a','fg3_pct','ft','fta','ft_pct','orb','total_board','ast',
            'stl','blk','tov','pf','opp_fg','opp_fga','opp_fg_pct','opp_fg3','opp_fg3a','opp_fg3_pct',
            'opp_ft','opp_fta','opp_ft_pct','opp_orb','opp_trb','opp_ast','opp_stl','opp_blk','opp_tov',
            'opp_pf','off_rtg','def_rtg','pace','fta_per_fga_pct','fg3a_per_fga_pct','ts_pct',
            'trb_pct','ast_pct','stl_pct','blk_pct','efg_pct','tov_pct','orb_pct','ft_rate','opp_efg_pct',
            'opp_tov_pct','drb_pct','opp_ft_rate'])
    for trb in tr_body1:
        for td in trb.find_all('td'):
            if td.get('data-stat') == "off_rtg":
                off_rtg.append(td.get_text())
            if td.get('data-stat') == "def_rtg":
                def_rtg.append(td.get_text())
            if td.get('data-stat') == "pace":
                pace.append(td.get_text())
            if td.get('data-stat') == "fta_per_fga_pct":
                fta_per_fga_pct.append(td.get_text())
            if td.get('data-stat') == "fg3a_per_fga_pct":
                fg3a_per_fga_pct.append(td.get_text())
            if td.get('data-stat') == "ts_pct":
                ts_pct.append(td.get_text())
            if td.get('data-stat') == "trb_pct":
                trb_pct.append(td.get_text())
            if td.get('data-stat') == "ast_pct":
                ast_pct.append(td.get_text())
            if td.get('data-stat') == "stl_pct":
                stl_pct.append(td.get_text())
            if td.get('data-stat') == "blk_pct":
                blk_pct.append(td.get_text())
            if td.get('data-stat') == "efg_pct":
                efg_pct.append(td.get_text())
            if td.get('data-stat') == "tov_pct":
                tov_pct.append(td.get_text())
            if td.get('data-stat') == "orb_pct":
                orb_pct.append(td.get_text())
            if td.get('data-stat') == "ft_rate":
                ft_rate.append(td.get_text())
            if td.get('data-stat') == "opp_efg_pct":
                opp_efg_pct.append(td.get_text())
            if td.get('data-stat') == "opp_tov_pct":
                opp_tov_pct.append(td.get_text())
            if td.get('data-stat') == "drb_pct":
                drb_pct.append(td.get_text())
            if td.get('data-stat') == "opp_ft_rate":
                opp_ft_rate.append(td.get_text())
            
            
    return 