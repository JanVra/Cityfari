# Cityfari

![logo](https://github.com/JanVra/Cityfari/blob/main/images/logo.png)

--- 

Citiyfari is a data driven platform that enables creators to share unique experiences and show their city in a new light. Utilizing cutting edge artificial intelligence algorithms, Cityfari provides a headstart to shape your vacation and keep your leisure time spontaneous and fun. Using Cityfari, the user is going to notice a decrease in time spent having to worry about what to do and instead make the most of their time.

What we do:
- We find popular points of interest, by leveraging GPS trajectory data by mapping these to open source third party data portals via medoid clustering.
- We provide a concept for a creator space, which promotes sustainable/slow forms of tourism.
- We offer unique and tailor made experiences to our users.

Future Outlook:
- gamification through achievement badges and sustainability scores
- community-made scavenger hunts with high scores
- extending AI capabilities to relief creator work-load


# Data Sources and Disclaimer
https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/

Preprocessed Data by Pietro Colombo:
https://medium.com/analytics-vidhya/data-mining-of-geolife-dataset-560594728538

OpenStreetMap Data via Overpass Turbo with query based on Bejing:

```
[out:json][timeout:25];
// gather results
(
  node["tourism"]({{bbox}});
  node["name:"]({{bbox}});
);
out body;
>;
out skel qt;
```
