from .Util import get_data_from_csv
import numpy as np
from os import path
import csv

def select_data(data, categories, dates):
    selected_data = []
    for item in data:
        is_selected = True
        for k, v in categories.items():
            if item[k] != v:
                is_selected = False
        if is_selected:
            selected_item = [item[date] for date in dates]
            selected_data.append(selected_item)
    return selected_data

def select_dates(dates, years, months):
    selected_dates = []
    for date in dates:
        year, month, _ = date.split('-')
        if int(year) in years and int(month) in months:
            selected_dates.append(date)
    return selected_dates

def import_weekly_sales_and_prices(dataset_path):
    sales = get_data_from_csv(
        path.join(
            dataset_path,
            'sales_train_validation_weekly.csv'
                 )
    )
    prices = get_data_from_csv(
        path.join(
            dataset_path,
            'sell_prices_weekly.csv'
                 )
    )
    return sales, prices

def import_weekly_dates(dataset_path):
    with open(path.join(dataset_path, 'weekly_dates.txt'),'r') as date_file:
        dates = date_file.read().split(',')
    return dates

def generate_daily_prices(dataset_path):
    # import data
    print('import sales, calendar and sell prices ...')
    sales = get_data_from_csv(
        path.join(
            dataset_path,
            'sales_train_validation.csv'
                 )
    )
    calendar = get_data_from_csv(
        path.join(
            dataset_path,
            'calendar.csv'
                 )
    )
    prices = get_data_from_csv(
        path.join(
            dataset_path,
            'sell_prices.csv'
                 )
    )
    
    # build dictionary for price
    print('build price dictionary ...')
    price_dict = {}
    for price in prices:
        key =\
            price['store_id'] + '+' + \
            price['item_id'] + '+' + \
            price['wm_yr_wk']
        price_dict[key] = float(price['sell_price'])
    
    # input daily prices
    print('input daily prices ...')
    day_prices = []
    basic_info = {'id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'}
    for i, sale in enumerate(sales):
        day_price = {info:sale[info] for info in basic_info}
        for j in range(6, len(sales[i])):
            key =\
                sale['store_id'] + '+' + \
                sale['item_id'] + '+' + \
                calendar[j - 6]['wm_yr_wk']
            if key in price_dict:
                day_price['d_' + str(j - 5)] = float(price_dict[key])
            else:
                day_price['d_' + str(j - 5)] = 0.0
        day_prices.append(day_price)
        
    # save file
    print('save daily prices ...')
    with open(path.join(dataset_path, 'day_prices.csv'), 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames = list(day_prices[0].keys()))
        writer.writeheader()
        for row in day_prices:
            writer.writerow(row)

def generate_weekly_sales_and_prices(dataset_path):
    # import data
    print('import sales, calendar and daily prices ...')
    sales = get_data_from_csv(
        path.join(
            dataset_path,
            'sales_train_validation.csv'
                 )
    )
    calendar = get_data_from_csv(
        path.join(
            dataset_path,
            'calendar.csv'
                 )
    )
    day_prices = get_data_from_csv(
        path.join(
            dataset_path,
            'day_prices.csv'
                 )
    )
    
    # generate weekly sales
    print('generate weekly sales ...')
    wk_sales = [[] for _ in range(len(sales))]
    wk_sale = 0
    for i in range(len(sales)):
        for j in range(1,1914):
            wk_sale += int(sales[i]['d_' + str(j)])
            if calendar[j - 1]['weekday'] == 'Sunday':
                wk_sales[i].append(wk_sale)
                wk_sale = 0
        if wk_sale:
            wk_sales[i].append(wk_sale)
            wk_sale = 0
                
    # generate weekly prices
    print('generate weekly prices ...')
    wk_prices = [[] for _ in range(len(sales))]
    avg_prices = []
    for i in range(len(sales)):
        for j in range(1,1914):
            avg_prices.append(int(sales[i]['d_' + str(j)]))
            if calendar[j - 1]['weekday'] == 'Sunday':
                wk_prices[i].append(np.mean(avg_prices))
                avg_prices = []
        if len(avg_prices):
            wk_prices[i].append(np.mean(avg_prices))
            avg_prices = []
            
    # generate weekly dates
    print('generate weekly dates ...')
    dates = []
    for i in range(1913):
        if calendar[i]['weekday'] == 'Sunday':
            dates.append(calendar[i]['date'])
    if calendar[-1]['weekday'] != 'Sunday':
        dates.append(calendar[-1]['date'])

    print('save weekly sales ...')
    basic_info = {'id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'}
    with open(path.join(dataset_path, 'sales_train_validation_weekly.csv'), 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames = list(basic_info) + dates)
        writer.writeheader()
        for sale, wk_sale in zip(sales, wk_sales):
            row = {k:sale[k] for k in basic_info}
            for i, wk_v in enumerate(wk_sale):
                row[dates[i]] = wk_v
            writer.writerow(row)
    print('save weekly prices ...')
    with open(path.join(dataset_path, 'sell_prices_weekly.csv'), 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames = list(basic_info) + dates)
        writer.writeheader()
        for sale, wk_price in zip(sales, wk_prices):
            row = {k:sale[k] for k in basic_info}
            for i, wk_p in enumerate(wk_price):
                row[dates[i]] = wk_p
            writer.writerow(row)
    print('save weekly dates ...')
    with open(path.join(dataset_path, 'weekly_dates.txt'), 'w') as txt_file:
        txt_file.write(','.join(dates))