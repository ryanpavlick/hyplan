#%%
import ee
ee.Authenticate()
from hyplan.clouds import create_cloud_data_array_with_limit, simulate_visits, plot_yearly_cloud_fraction_heatmaps_with_visits

#%%



# Parameters
polygon_file = 'data/hyspiri.geojson'
year_start = 2003
year_stop = 2006
day_start = 1  # January 1
day_stop = 75  # February 29 (non-leap year)

# Assuming `cloud_data_df` has already been created and contains the necessary data
cloud_data_df = create_cloud_data_array_with_limit(polygon_file, year_start, year_stop, day_start, day_stop)

#%%

day_start = 30
day_stop = 74

# Simulate visits for each year
visit_days_df, visit_tracker, rest_days = simulate_visits(
    cloud_data_df,
    day_start,
    day_stop,
    year_start,
    year_stop,
    cloud_fraction_threshold=0.25,  # Maximum cloudiness for a flight box to be considered
    rest_day_threshold=2,  # Add rest day after N consecutive flight days
    exclude_weekends=True, # Exclude weekends from possible flight days
    debug=True
)

#%%
plot_yearly_cloud_fraction_heatmaps_with_visits(
    cloud_data_df, visit_tracker, rest_days,
    cloud_fraction_threshold=0.25, exclude_weekends=True,
    day_start=day_start, day_stop=day_stop
)# %%
# %%
