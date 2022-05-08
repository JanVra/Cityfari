#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import iqr
from scipy.spatial.distance import euclidean, cdist, pdist
from sklearn.metrics.pairwise import nan_euclidean_distances
import geojson
from itertools import combinations, chain, product
import plotly.express as px
from functools import reduce
import random
import networkx as nx
from pyroutelib3 import Router
px.set_mapbox_access_token("your_mapbox_access_token")

# %%

medoids = pd.read_csv("../data/medoids2.csv")
# %%
medoids = medoids[medoids.Label == "walk"]
# %%
fig = px.scatter_mapbox(medoids, lat="Latitude", lon="Longitude")
fig
# %%
with open("../data/POIs.geojson",encoding='utf-8') as f:
    gj = geojson.load(f)
features = gj['features']
# %%
poi = pd.DataFrame(features)
poi["lon_p"] = poi.geometry.apply(lambda x: x["coordinates"][0])
poi["lat_p"] = poi.geometry.apply(lambda x: x["coordinates"][1])
poi["name"] = poi.properties.apply(lambda x: x.get("name:en"))
poi["tourism"] = poi.properties.apply(lambda x: x.get("tourism"))
p = poi[["lon_p","lat_p","name", "tourism"]].reset_index(drop=False)

# %%
distance = cdist(medoids[["Longitude", "Latitude"]], p[["lon_p", "lat_p"]])
distance
# %%
medoids["join_idx"] = np.argmin(distance, axis = 1)
medoids["distance_to_poi"] = np.min(distance, axis = 1)
joined = medoids.merge(p, how = "left", left_on="join_idx", right_index = True)
# %%
# %%
j = joined[joined.distance_to_poi < 0.005]

# %%
j = j.sort_values(by = ["Id_user", "Id_perc", "Date_Time"], axis =0).copy()
# %%
j["Date"] = pd.to_datetime(j.Date_Time).dt.date
# %%
j.groupby("Date").apply(lambda x: len(x)).value_counts()
# %%
sequences = j.groupby(["Date", "Id_user", "Id_perc"]).apply(lambda x: set(list(x["name"])))
# %%
sequences.apply(lambda x: len(x)).mean()

# %%
seq = sequences.reset_index().rename(columns = {0 : "seq"})

# %%
def flatten(t):
    return [item for sublist in t for item in sublist]
# %%
unique_places = set(flatten(seq.seq.values))
# %%
def init_graphs(places):
    p_G = {}
    for place in places:
        p_G[place] = nx.Graph(name = place)
        p_G[place].add_node(place, name = place)
    return p_G


def build_place_graphs(p_G, tr, p):
    for trace in tr:
        for ix, place in enumerate(trace):
            if ix < len(trace) - 1:
                ix = ix + 1
            else:
                break
            neighbor = trace[ix]
            dist = calc_dist(place, neighbor, p)

            p_G[place].add_node(neighbor, name = neighbor)
            p_G[place].add_edge(place, neighbor, weight = dist)
            p_G[neighbor].add_node(place, name = place)
            p_G[neighbor].add_edge(neighbor, place, weight = dist)
    return p_G

def build_tour(node, p_G, n = 5):
    tour = nx.DiGraph()
    tour.add_node(node, name = node)
    for _ in range(n):
        place = p_G[node]
        restricted = set(place.nodes) - set(tour.nodes)

        if len(restricted) == 0:
            return tour

        elif len(restricted) == 1:
            restricted = [restricted.pop()]

        next_node = random.choice(list(restricted)) # this can be done via heuristics
        tour.add_node(node, name = node)
        tour.add_edge(node, next_node)
        node = next_node
    return tour

def calc_dist(p1, p2, p):
    router = Router("cycle")
    start_lat, start_lon = p[p.name == p1][["lat_p", "lon_p"]].values[0]
    end_lat, end_lon = p[p.name == p2][["lat_p", "lon_p"]].values[0]
    start = router.findNode(start_lat, start_lon) # Find start and end nodes
    end = router.findNode(end_lat, end_lon)

    return router.distance((start_lat, start_lon), (end_lat, end_lon))

def prune_graph(p_G):
    keys = [key for key, val in p_G.items() if len(val.nodes) == 1]
    for key in keys:
        p_G.pop(key)
    return p_G
# %%
p_G = init_graphs(unique_places)
p_G = build_place_graphs(p_G, [list(s) for s in seq.seq.values], p)
# %%
p_G = prune_graph(p_G)
c_g = reduce(nx.compose, p_G.values())
nx.draw(c_g,pos=nx.spring_layout(c_g), with_labels = True)
# %%
g = build_tour("Water Source", p_G, n = 5)
nx.draw(g, pos=nx.spring_layout(g),with_labels = True)
# %%
tour_df = p[p.name.isin(list(g.nodes))]

px.scatter_mapbox(tour_df, lat="lat_p", lon="lon_p", color="name")
# %%
