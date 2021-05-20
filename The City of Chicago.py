#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install geocoder')
get_ipython().system('pip install folium')
get_ipython().system('pip install geopy')


# In[2]:


import pandas as pd
import numpy as np
import geocoder
import requests
import folium
import matplotlib.cm as cm
import matplotlib.colors as colors
import json
import xml
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")

from pandas.io.json import json_normalize 
from sklearn.cluster import KMeans
from geopy.geocoders import Nominatim 

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

print("All Required Libraries Imported!")


# In[3]:


df = pd.read_excel (r'/Users/prakshi/Downloads/Chicago.xlsx')
print(df.shape)
df.head()


# In[4]:


def get_latilong(neighborhood):
    lati_long_coords = None
    while(lati_long_coords is None):
        g = geocoder.arcgis('{}, Chicago, Illinois'.format(neighborhood))
        lati_long_coords = g.latlng
    return lati_long_coords
    
get_latilong('Galewood')


# In[5]:


neighborhoods = df['Neighborhood']
coords = [ get_latilong(neighborhood) for neighborhood in neighborhoods.tolist() ]


# In[6]:


df_coords = pd.DataFrame(coords, columns=['Latitude', 'Longitude'])
df['Latitude'] = df_coords['Latitude']
df['Longitude'] = df_coords['Longitude']
df.head()


# In[7]:


df[df.Neighborhood == 'Crestline']


# In[8]:


address = 'Chicago,Illinois'

geolocator = Nominatim(user_agent="myemailaddress@gmail.com")
location = geolocator.geocode(address)
latitude_x = location.latitude
longitude_y = location.longitude
print('The Geograpical Co-ordinate of Chicago, Illinois are {}, {}.'.format(latitude_x, longitude_y))


# In[9]:


chicago_map = folium.Map(location=[latitude_x, longitude_y], zoom_start=10)

for lat, lng, nei in zip(df['Latitude'], df['Longitude'], df['Neighborhood']):
    
    label = '{}'.format(nei)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(chicago_map)  
    
chicago_map


# In[10]:


address = 'Chicago,Illinois'

geolocator = Nominatim(user_agent="myemailaddress@gmail.com")
location = geolocator.geocode(address)
latitude_n1 = location.latitude
longitude_n1 = location.longitude
print('The Geograpical Co-ordinate of Neighborhood are {}, {}.'.format(latitude_x, longitude_y))


# In[11]:


CLIENT_ID = 'My Foursquare Client ID'
CLIENT_SECRET = 'My Foursquare Client Secret'
VERSION = '20180604'
LIMIT = 30


# In[12]:


radius = 700 
LIMIT = 100
url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
    CLIENT_ID, 
    CLIENT_SECRET, 
    VERSION, 
    latitude_n1, 
   longitude_n1, 
    radius, 
   LIMIT)
results = requests.get(url).json()


# In[13]:


venues=results['response']['groups'][0]['items']
nearby_venues = json_normalize(venues)
nearby_venues.columns


# In[14]:


def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']


# In[15]:


filtered_columns = ['venue.name', 'venue.categories', 'venue.location.lat', 'venue.location.lng']
nearby_venues =nearby_venues.loc[:, filtered_columns]
nearby_venues.head()


# In[16]:


nearby_venues['venue.categories'] = nearby_venues.apply(get_category_type, axis=1)
nearby_venues.columns = [col.split(".")[-1] for col in nearby_venues.columns]
nearby_venues.head()


# In[17]:


a = pd.Series(nearby_venues.categories)
a.value_counts()[:10]


# In[18]:


def getNearbyVenues(names, latitudes, longitudes, radius=700):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        venue_results = requests.get(url).json()["response"]['groups'][0]['items']
        
        venues_list.append([(
            name,
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in venue_results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


# In[19]:


chicago_venues = getNearbyVenues(names=df['Neighborhood'],latitudes=df['Latitude'],longitudes=df['Longitude'])


# In[20]:


print('There are {} Uniques Categories.'.format(len(chicago_venues['Venue Category'].unique())))
chicago_venues.groupby('Neighborhood').count().head()


# In[21]:


chicago_onehot = pd.get_dummies(chicago_venues[['Venue Category']], prefix="", prefix_sep="")
chicago_onehot['Neighborhood'] = chicago_venues['Neighborhood'] 
fixed_columns = [chicago_onehot.columns[-1]] + list(chicago_onehot.columns[:-1])
chicago_onehot = chicago_onehot[fixed_columns]
chicago_grouped = chicago_onehot.groupby('Neighborhood').mean().reset_index()
chicago_onehot.head()


# In[22]:


num_top_venues = 5
for hood in chicago_grouped['Neighborhood']:
    print("---- "+hood+" ----")
    temp = chicago_grouped[chicago_grouped['Neighborhood'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')


# In[23]:


def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# In[24]:


import numpy as np
num_top_venues = 10

indicators = ['st', 'nd', 'rd']

columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = chicago_grouped['Neighborhood']

for ind in np.arange(chicago_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(chicago_grouped.iloc[ind, :], num_top_venues)

print(neighborhoods_venues_sorted.shape)
neighborhoods_venues_sorted.head()


# In[25]:


chicago_grouped_clustering = chicago_grouped.drop('Neighborhood', 1)
kmeans = KMeans(n_clusters = 3, random_state = 0).fit(chicago_grouped_clustering)
kmeans.labels_


# In[26]:


neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)
chicago_merged = df.iloc[:246,:]
chicago_merged = chicago_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighborhood')
chicago_merged.head()


# In[27]:


kclusters = 10


# In[28]:


map_clusters = folium.Map(location=[latitude_x, longitude_y], zoom_start=11)

x = np.arange(kclusters)
colors_array = cm.rainbow(np.linspace(0, 1, kclusters))
rainbow = [colors.rgb2hex(i) for i in colors_array]
print(rainbow)

markers_colors = []
for lat, lon, nei , cluster in zip(chicago_merged['Latitude'], 
                                   chicago_merged['Longitude'], 
                                   chicago_merged['Neighborhood'], 
                                   chicago_merged['Cluster Labels']):
    label = folium.Popup(str(nei) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# In[29]:


chicago_avg_housing_price = pd.read_excel (r'/Users/prakshi/Downloads/Median Housing Prices By Community Area.xlsx')
chicago_avg_housing_price.head()


# In[30]:


chicago_avg_housing_price.set_index('Community Area',inplace=True,drop=True)


# In[31]:


chicago_avg_housing_price.plot(kind='bar',figsize=(24,18),alpha=0.75)


# In[32]:


chicago_school_ranking = pd.read_excel (r'/Users/prakshi/Downloads/Top School Ratings By Community Area.xlsx')
chicago_school_ranking.head()


# In[33]:


chicago_school_ranking.set_index('Community Area',inplace=True,drop=True)


# In[34]:


chicago_school_ranking.plot(kind='bar',figsize=(16,10),color='pink',alpha=0.75)


# In[35]:


crime_rate = pd.read_excel (r'/Users/prakshi/Downloads/Crime Rate By Community Area.xlsx')
crime_rate.head()


# In[36]:


crime_rate.set_index('Community Area',inplace=True,drop=True)


# In[37]:


crime_rate.plot(kind='bar',figsize=(16,10),color='yellow',alpha=0.75)

