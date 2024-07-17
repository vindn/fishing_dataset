# %% 
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import geopandas as gpd
import movingpandas as mpd
import shapely as shp
import hvplot.pandas
import time
from datetime import datetime, timedelta
import folium
from geopy.distance import geodesic
from folium.plugins import MeasureControl
from shapely.geometry import MultiPoint
from shapely.geometry import Point
from shapely.geometry import LineString


# https://globalfishingwatch.org/datasets-and-code-vessel-identity/

# %%

# Class for preprocessing GFW dataset
class GFWDatasetProcessor:
    def __init__(self):
        self.data = pd.DataFrame()
        self.gdf_data = None
        self.trajs = None

    def read_and_clean(self, file_path):
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Data cleaning steps:
        # - Remove duplicates
        df.drop_duplicates(inplace=True)
        
        # - Handle missing values (for simplicity, we'll drop rows with any NaNs)
        df.dropna(inplace=True)
        
        return df

    def process_files(self, file_paths):
        for file_path in file_paths:
            cleaned_df = self.read_and_clean(file_path)
            self.data = pd.concat([self.data, cleaned_df], ignore_index=True)

   
    # is_fishing: Label indicating fishing activity.
    # 0 = Not fishing
    # >0 = Fishing. Data values between 0 and 1 indicate the average score for the position if scored by multiple people.
    # -1 = No data
    def set_only_fishing_data( self, is_fishing=0 ):
        self.data = self.data[ self.data['is_fishing'] > is_fishing ]
    
    def convert_types( self ):
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'], unit='s')        
        self.convert_to_nautical_miles(['distance_from_shore', 'distance_from_port'])
        self.data['mmsi'] = self.data['mmsi'].astype(int)
        self.data['name'] = 'gfw_' + self.data['source']

    def rename_columns( self ):
        columns_mapping = {
            'speed': 'SOG',
            'course': 'COG'
        }
        # Renaming columns based on the provided mapping
        self.data.rename(columns=columns_mapping, inplace=True)

    def convert_to_nautical_miles(self, columns):
        # Converting the specified columns from meters to nautical miles
        for column in columns:
            if column in self.data.columns:
                self.data[column] = self.data[column] / 1852        

    def get_fishing_data( self, is_fishing=0 ):
        return self.data[ self.data['is_fishing'] > is_fishing ]

    def create_gdf(self, df):
        import warnings
        warnings.filterwarnings('ignore')

        gdf = gpd.GeoDataFrame(
            df.set_index('timestamp'), geometry=gpd.points_from_xy(df.lon, df.lat))

        gdf.set_crs('epsg:4326')
        return gdf

    def calc_distance_diff_nm( self, df, LAT_colLON, lon_coluna):
        """
        Calcula as diferenças de distância entre pares de pontos de LATitudLON longitude em um DataFrame.
        df: DataFrame contendo as colunas de LATitudLON longitude
        LAT_colLON: Nome da coluna de LATitudLON   lon_coluna: Nome da coluna de longitude
        Retorna uma lista com as diferenças de distância entre as linhas.
        """
        diferencas = []
        for i in range(len(df) - 1):
            ponto1 = (df[LAT_colLON].iloc[i], df[lon_coluna].iloc[i])
            ponto2 = (df[LAT_colLON].iloc[i + 1], df[lon_coluna].iloc[i + 1])
            distancia = geodesic(ponto1, ponto2).nautical
            diferencas.append(distancia)

        diferencas.insert(0, diferencas[0])
        return diferencas

    def calc_time_diff_h( self, df, coluna_tempo):
        """
        Calcula as diferenças de tempo em horas entre linhas consecutivas de um DataFrame.
        df: DataFrame contendo a coluna de tempo
        coluna_tempo: Nome da coluna de tempo
        Retorna uma lista com as diferenças de tempo em horas entre as linhas.
        """
        diferencas = []
        for i in range(len(df) - 1):
            tempo1 = pd.to_datetime(df[coluna_tempo].iloc[i])
            tempo2 = pd.to_datetime(df[coluna_tempo].iloc[i + 1])
            diferenca = (tempo2 - tempo1).total_seconds() / 3600  # Diferença em horas
            diferencas.append(diferenca)
        
        diferencas.insert(0, diferencas[0])
        return diferencas

    def angular_diff(self, direction1, direction2):
        """
        Calcula a menor diferença angular entre duas séries de direções.

        Parâmetros:
        direcao1 (pandas.Series ou array-like): Primeira série de direções (em graus).
        direcao2 (pandas.Series ou array-like): Segunda série de direções (em graus).

        Retorna:
        array-like: A menor diferença angular entre as direções.
        """
        # Converter de graus para radianos
        direction1_rad = np.radians(direction1)
        direction2_rad = np.radians(direction2)

        # Calcular a diferença angular em radianos
        difference = np.arctan2(np.sin(direction1_rad - direction2_rad), 
                                np.cos(direction1_rad - direction2_rad))

        # Converter de radianos para graus
        degrees_diff = np.degrees(difference)

        # Ajustar para que o resultado esteja entre -180 e 180 graus
        degrees_diff = (degrees_diff + 180) % 360 - 180

        # Ajustar o primeiro ponto pra ficar igual ao segundo ponto
        degrees_diff[0] = degrees_diff[1]

        return degrees_diff

    def calculate_compass_bearing(self, pointA, pointB):
        """
        Calcular o azimute entre dois pontos.
        :param pointA: Tuple com latitude e longitude do primeiro ponto (latA, lonA)
        :param pointB: Tuple com latitude e longitude do segundo ponto (latB, lonB)
        :return: Azimute em graus
        """
        if (type(pointA) != tuple) or (type(pointB) != tuple):
            raise TypeError("Only tuples are supported as arguments")

        lat1 = np.radians(pointA[0])
        lat2 = np.radians(pointB[0])

        diffLong = np.radians(pointB[1] - pointA[1])

        x = np.sin(diffLong) * np.cos(lat2)
        y = np.cos(lat1) * np.sin(lat2) - (np.sin(lat1) * np.cos(lat2) * np.cos(diffLong))

        initial_bearing = np.arctan2(x, y)

        # Converte de radianos para graus e ajusta para 0-360°
        initial_bearing = np.degrees(initial_bearing)
        compass_bearing = (initial_bearing + 360) % 360

        return compass_bearing

    def calculate_cog(self, df):
        """
        Calcular o COG para cada ponto de uma trajetória.
        :param df: DataFrame com colunas 'LAT' e 'LON'
        :return: DataFrame com coluna adicional 'COG'
        """
        # Verifica se as colunas 'LAT' e 'LON' existem no DataFrame
        if 'lat' not in df.columns or 'lon' not in df.columns:
            raise KeyError("DataFrame must contain 'LAT' and 'LON' columns")

        # Certifica-se de que os índices do DataFrame são contínuos
        df = df.reset_index(drop=True)

        cogs = [np.nan]  # O primeiro ponto não tem COG

        for i in range(1, len(df)):
            pointA = (df.iloc[i-1]['lat'], df.iloc[i-1]['lat'])
            pointB = (df.iloc[i]['lat'], df.iloc[i]['lon'])        
            cog = self.calculate_compass_bearing(pointA, pointB)
            cogs.append(cog)

        # df['COG'] = cogs
        cogs[0] = cogs[1]
        return cogs


    def create_trajectories(self, verbose=True, gap_minutes=120, traj_min_size=10):

        self.gdf_data = self.create_gdf( self.data )

        # reset index
        self.gdf_data = self.gdf_data.reset_index()
        self.gdf_data['timestamp'] = pd.to_datetime(self.gdf_data['timestamp'], utc=True)
        self.gdf_data['traj_id'] = self.gdf_data['mmsi']

        start_time = time.time()

        collection = mpd.TrajectoryCollection(
            self.gdf_data, "traj_id",  t='timestamp', min_length=0.001, crs='epsg:4326')

        collection = mpd.ObservationGapSplitter(
            collection).split(gap=timedelta(minutes=gap_minutes))

        trajs = []        
        for traj in collection.trajectories:
            if len(traj.df) > traj_min_size:
                traj.df['dist_diff']  = self.calc_distance_diff_nm( traj.df, 'lat', 'lon')
                traj.df['time_diff_h'] = self.calc_time_diff_h( traj.df.reset_index(), 'timestamp' )
                traj.df['time_diff'] = traj.df['time_diff_h'] * 3600
                traj.df['speed_nm'] = traj.df['dist_diff'] / traj.df['time_diff_h']
                traj.df['ang_diff_cog'] = self.angular_diff( traj.df['COG'], traj.df['COG'].shift(1))
                traj.df['cog_calculated'] = self.calculate_cog( traj.df )
                traj.df['ang_diff_cog_calculated'] = self.angular_diff( traj.df['cog_calculated'], traj.df['cog_calculated'].shift(1))

                trajs.append( traj )

        end_time = time.time()
        if verbose:
            print("Time creation trajectories: ", (end_time-start_time)/60,  " min")

        self.trajs = mpd.TrajectoryCollection( trajs )
        return self.trajs

    def create_trajectories_ex(self, gdf, verbose=True, gap_minutes=120, traj_min_size=10):

        # reset index
        gdf = gdf.reset_index()
        gdf['timestamp'] = pd.to_datetime(gdf['timestamp'], utc=True)
        gdf['traj_id'] = gdf['mmsi']

        start_time = time.time()

        collection = mpd.TrajectoryCollection(
            gdf, "traj_id",  t='timestamp', min_length=0.001, crs='epsg:4326')

        collection = mpd.ObservationGapSplitter(
            collection).split(gap=timedelta(minutes=gap_minutes))

        trajs = []        
        for traj in collection.trajectories:
            if len(traj.df) > traj_min_size:
                traj.df['dist_diff']  = self.calc_distance_diff_nm( traj.df, 'lat', 'lon')
                traj.df['time_diff_h'] = self.calc_time_diff_h( traj.df.reset_index(), 'timestamp' )
                traj.df['time_diff'] = traj.df['time_diff_h'] * 3600
                traj.df['speed_nm'] = traj.df['dist_diff'] / traj.df['time_diff_h']
                traj.df['ang_diff_cog'] = self.angular_diff( traj.df['COG'], traj.df['COG'].shift(1))
                traj.df['cog_calculated'] = self.calculate_cog( traj.df )
                traj.df['ang_diff_cog_calculated'] = self.angular_diff( traj.df['cog_calculated'], traj.df['cog_calculated'].shift(1))

                trajs.append( traj )

        end_time = time.time()
        if verbose:
            print("Time creation trajectories: ", (end_time-start_time)/60,  " min")

        trajs = mpd.TrajectoryCollection( trajs )
        return trajs


    def write_pickle_obj(self, data, file_name):
        with open( file_name, 'wb') as data_file:
            pickle.dump(data, data_file)

    def read_pickle_obj(self, file_name):
        try:
            with open(file_name, 'rb') as data_file:
                data = pickle.load(data_file)
                return data
        except Exception as e:
            print(e, "File not Found!")

    def calculate_statistics(self, column_name):
        # Calculating statistics for all numerical columns
        return self.data[column_name].describe()

    def plot_statistics(self, column_name):
        # Calculate statistics for the 'speed' column
        speed_stats = self.data[column_name].describe()

        # Plotting the statistics
        fig, ax = plt.subplots(figsize=(10, 6))
        stats_to_plot = speed_stats[['25%', '50%', '75%']]
        stats_to_plot.plot(kind='bar', ax=ax)
        
        # Adding mean, std, and min as text
        textstr = '\n'.join((
            f"Mean: {speed_stats['mean']:.2f}",
            f"Std: {speed_stats['std']:.2f}",
            f"Min: {speed_stats['min']:.2f}",
            f"Max: {speed_stats['max']:.2f}"
        ))

        # Text box properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(1.05, 0.5, textstr, transform=ax.transAxes, fontsize=12,
                verticalalignment='center', bbox=props)

        ax.set_title('Statistics for ' + column_name)
        ax.set_ylabel('Values')
        ax.set_xlabel('Statistics')
        plt.show()

# %%

########
## Class for trajectories plot
#######
from folium.plugins import MeasureControl
class TrajectoriesPlot( ):
    def __init__( self, trajs=None, traj=None ):
        pass

    def plot_one_trajectory( self, traj ):

        if traj is None:
            print("Trajectory was not set.")
            return None
        
        m = self.plot_trajectory( traj.df, '' )

        return m

    def plot_trajectories( self, trajs ):

        if trajs is None:
            print("Trajectories was not set.")
            return None
                
        m = None
        i = 0
        for traj in trajs:
            if m:
                m = self.plot_trajectory( traj.df, i, m )
            else:
                m = self.plot_trajectory( traj.df, i )
            i += 1
        return m        

    def plot_encounter( self, gdf1, gdf2, m=None ):
        if m is None:
            m = self.plot_trajectory( gdf1, "vessel 1", m=None, color='blue' )
            m = self.plot_trajectory( gdf2, "vessel 2", m, color='red' )
        else:
            m = self.plot_trajectory( gdf1, "vessel 1", m=m, color='blue' )
            m = self.plot_trajectory( gdf2, "vessel 2", m, color='red' )

        return m
    def plot_encounters( self, trajs1, trajs2 ):

        id = 0
        m = self.plot_trajectory( trajs1.trajectories[0].df, "vessel" + str(id), m=None, color='blue' )
        m = self.plot_trajectory( trajs2.trajectories[0].df, "vessel" + str(id), m, color='red' )
        for i in range(1, len(trajs1.trajectories)):
            id += 1
            m = self.plot_trajectory( trajs1.trajectories[i].df, "vessel" + str(id), m, color='blue' )
            m = self.plot_trajectory( trajs2.trajectories[i].df, "vessel" + str(id), m, color='red' )
        return m

    # plot gdf points
    def plot_gdf( self, gdf, vessel_description, m=None, color='blue' ):
        import folium

        latitude_initial = gdf.iloc[0]['lat']
        longitude_initial = gdf.iloc[0]['lon']

        if not m:
            m = folium.Map(location=[latitude_initial, longitude_initial], zoom_start=10)
            m.add_child(MeasureControl())

        for _, row in gdf.reset_index().iterrows():

            # vessel_description = vessel_type_mapping.get( int( row['VesselType'] ), "Unknown")
            # vessel_description = vessel_type_mapping.get( int( row['VesselType'] ), "Unknown")

            # Concatenar colunas para o popup
            popup_content = f"<b>Traj_id:</b> {row.traj_id}<br><b>Timestamp:</b> {row.timestamp}<br><b>VesselName:</b> {row['name']}<br><b>MMSI</b>: {row['mmsi']}<br><b>LAT:</b> {row['lat']}<br><b>LON:</b> {row['lon']}<br><b>SOG:</b> {row['SOG']}<br><b>Type:</b> {vessel_description}<br><b>COG:</b> {row['COG']}"
            # color = mmsi_to_color( row['MMSI'] )
            
            folium.CircleMarker(
                location=[row['geometry'].y, row['geometry'].x],
                popup=popup_content,
                radius=1,  # Define o tamanho do ponto
                color=color,  # Define a cor do ponto
                fill=True,
                fill_opacity=1,
            ).add_to(m)            

        return m

    def create_linestring(self, group):        
            # Ordenar por timestamp
            group = group.sort_values(by='timestamp')      
            # Se há mais de um ponto no grupo, crie uma LineString, caso contrário, retorne None
            return LineString(group.geometry.tolist()) if len(group) > 1 else None

    # plot trajectories from points
    # plot trajectories from points
    def plot_trajectory( self, gdf, vessel_description, m=None, color='blue' ):
        import folium

        lines = gdf.groupby('mmsi').apply(self.create_linestring)

        # Remove possíveis None (se algum grupo tiver apenas um ponto)
        lines = lines.dropna()

        # Crie um novo GeoDataFrame com as LineStrings
        lines_gdf = gpd.GeoDataFrame(lines, columns=['geometry'], geometry='geometry')

        lines_gdf.reset_index(inplace=True)

        # start_point = Point(lines_gdf.iloc[0].geometry.coords[0])
        # m = folium.Map(location=[start_point.y, start_point.x], zoom_start=10)

        if not m:
            m = self.plot_gdf( gdf, vessel_description, color=color )
        else:
            self.plot_gdf( gdf, vessel_description, m, color=color )

        for _, row in lines_gdf.iterrows():            
            if row['geometry'].geom_type == 'LineString':
                popup_content = f"{row['mmsi']}"
                coords = list(row['geometry'].coords)
                    
                folium.PolyLine(locations=[(lat, lon) for lon, lat in coords], 
                            popup=popup_content,
                            weight=0.5,
                            color=color
                ).add_to(m)

        return m

#####################
## MAIN
####################
# %%

file_paths = [ 
    'raw_dataset/drifting_longlines.csv', 
    'raw_dataset/fixed_gear.csv', 
    'raw_dataset/purse_seines.csv', 
    'raw_dataset/trollers.csv', 
    'raw_dataset/pole_and_line.csv', 
    'raw_dataset/trawlers.csv', 
    # 'raw_dataset/unknown.csv'
    ]
processor = GFWDatasetProcessor()
processor.process_files(file_paths)
# set only fishing drop other trajectories!
processor.set_only_fishing_data(is_fishing=0.5)
processor.convert_types( )
processor.rename_columns( )

# %%

combined_data = processor.get_fishing_data(is_fishing=0.4)
combined_data

###########
# Statistics
############

# %%

processor.plot_statistics( 'SOG' )
# %%

processor.plot_statistics( 'distance_from_shore' )
# %%

trajs = processor.create_trajectories()

# %%

###########################
## Ploting trajectories for visual conference
###########################

tp = TrajectoriesPlot( )
traj = trajs.trajectories[101]
                          
#Normal traj
print("Normal Traj")
tp.plot_one_trajectory( traj )

# %%

# DouglasPeuckerGeneralizer
print("DouglasPeuckerGeneralizer")
traj = mpd.DouglasPeuckerGeneralizer(traj).generalize(tolerance=1.0)
tp.plot_one_trajectory( traj )

# %%
print("MinDistanceGeneralizer")
traj = mpd.MinDistanceGeneralizer(traj).generalize(tolerance=1.0)
tp.plot_one_trajectory( traj )


# %%
print("MinTimeDeltaGeneralizer")
traj = mpd.MinTimeDeltaGeneralizer(traj).generalize(tolerance=timedelta(minutes=1))
tp.plot_one_trajectory( traj )

# %%
print("TopDownTimeRatioGeneralizer")
traj = mpd.TopDownTimeRatioGeneralizer(traj).generalize(tolerance=1.0)
tp.plot_one_trajectory( traj )

# %%

detector = mpd.TrajectoryStopDetector(traj)
stops = detector.get_stop_segments(min_duration=timedelta(seconds=60),
                                       max_diameter=100)
tp.plot_trajectories( stops )

# %%

#################
## Writing files
#################

processor.write_pickle_obj( processor.trajs, 'output/movingpandas_fishing_trajectories.pickle')
processor.write_pickle_obj( processor.gdf_data, 'output/gdf_fishing_trajectories.pickle')

# %%

# use as normal trajectories the dataset from marine cadastre for vessel with min distance of 10 NM from shore.
gdf_normal_trajectories = processor.read_pickle_obj( '/home/vindn/SynologyDrive/4_UFRJ/projeto_tese/codigos/projeto_datasets/loitering/datasets_output/gdf_not_encounters_3meses.pickle' )

# %%
columns_mapping = {
    'MMSI': 'mmsi',
    'LAT': 'lat',
    'LON': 'lon',
    'VesselName': 'name',
    'BaseDateTime': 'timestamp',
    'distance_to_coast': 'distance_from_shore'
}
# Renaming columns based on the provided mapping
gdf_normal_trajectories = gdf_normal_trajectories.reset_index()
gdf_normal_trajectories.rename(columns=columns_mapping, inplace=True)

# %%
# Create normal trajectories
trajs_normal = processor.create_trajectories_ex( gdf_normal_trajectories[:] )


# %%
processor.write_pickle_obj( processor.trajs, 'output/movingpandas_normal_trajectories.pickle')
processor.write_pickle_obj( processor.gdf_data, 'output/gdf_normal_trajectories.pickle')

# %%
