import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class ProcessGameState():
  def __init__(self, data_path):
    self.df = pd.read_parquet(data_path)

    # convert data types
    self.df['side'] = self.df['side'].astype('category')
    self.df['team'] = self.df['team'].astype('category')
    self.df['area_name'] = self.df['area_name'].astype('category')
    self.df['map_name'] = self.df['map_name'].astype('category')
    self.df['player'] = self.df['player'].astype('category')


  def is_point_in_area(self, vertices, x, y, z, z_lower_bound=285, z_upper_bound=421):
    j = len(vertices) - 1  # The last vertex is the 'previous' one to the first

    result = False

    for i in range(len(vertices)):
        if ((vertices[i][1] > y) != (vertices[j][1] > y)) and \
                (x < vertices[i][0] + (vertices[j][0] - vertices[i][0]) * (y - vertices[i][1]) /
                (vertices[j][1] - vertices[i][1]) and \
                  z_lower_bound <= z <= z_upper_bound):  # compute the x-coordinate of the point of intersection of the line from the point to the outside of the vertices with the current edge
            result = not result
        j = i

    return result


  def has_weapon(self, inventory, weapon_class):
    if inventory is not None:
      for weapon in inventory:
        if weapon['weapon_class'] == weapon_class:
          return True
    return False


  def clock_time_to_seconds(self, clock_time):
    time = clock_time.split(':')
    return int(time[0]) * 60 + int(time[1])


  def seconds_to_clock_time(self, seconds):
    minutes = seconds // 60
    seconds = seconds % 60
    return f"{minutes}:{seconds}"


  def answer_question_2a(self, vertices, z_lower_bound, z_upper_bound):
    # add column for whether player is in area
    self.df['in_area'] = self.df.apply(
      lambda row:
        self.is_point_in_area(vertices, row['x'], row['y'], row['z'], z_lower_bound, z_upper_bound)\
        , axis=1
    )

    # filter data to only include Team2, T side, and in area
    columns = ['round_num', 'side', 'team', 'player', 'in_area']
    data = self.df[columns]

    team2 = data['team'] == 'Team2'
    sideT = data['side'] == 'T'
    choke_point = data['in_area'] == True
    filter_condition = team2 & sideT & choke_point
    data = data[filter_condition]


    # number of rounds at least one member of Team 2 found in choke point as T
    num_rounds = data['round_num'].nunique()

    # total number of rounds Team 2 played as T
    total_rounds = self.df[team2 & sideT]['round_num'].nunique()

    # percentage of rounds at least one member of Team 2 found in choke point as T
    percentage = num_rounds / total_rounds

    result = f"""***** Answer to Question 2a *****

    Out of {total_rounds} total rounds Team2 played as T, we only found {num_rounds} round where at least one of their players within the given area.

    {num_rounds} / {total_rounds} = {percentage:.2%}

    Therefor, entering via the light blue boundary is not a common strategy used by Team2 on T side.



    """

    return result


  def answer_question_2b(self):
    # add column for wether or not a player has a SMG in their inventory
    self.df['SMG'] = self.df.apply(
      lambda row:
        self.has_weapon(row['inventory'], 'SMG'), axis=1
    )

    # add column for whether or not a player has a Rifle in their inventory
    self.df['rifle'] = self.df.apply(
      lambda row:
        self.has_weapon(row['inventory'], 'Rifle'), axis=1
    )

    columns = ['round_num', 'clock_time', 'team', 'side', 'player', 'area_name', 'SMG', 'rifle']
    data = self.df[columns]

    # filter data
    team2 = data['team'] == 'Team2'
    sideT = data['side'] == 'T'
    smg_or_rifle = data['SMG'] | data['rifle']
    bombsite_b = data['area_name'] == 'BombsiteB'
    filter_condition = team2 & sideT & smg_or_rifle & bombsite_b
    data = data[filter_condition]

    # number of players each round that entered bombsite B with a SMG or Rifle
    grouped_data = data.groupby('round_num').agg({'player': 'nunique', 'clock_time': 'max'})

    # rounds where at least two players entered bombsite B with a SMG or Rifle
    grouped_data = grouped_data[grouped_data['player'] >= 2].reset_index()

    # convert clock time to seconds
    grouped_data['seconds_remaining'] = grouped_data['clock_time'].apply(self.clock_time_to_seconds)

    # average clock time when at least two players entered bombsite B with a SMG or Rifle
    average_time = self.seconds_to_clock_time(int(grouped_data['seconds_remaining'].mean()))

    result = f"""***** Answer to Question 2b *****

    Team2 enters "BomsiteB" with atleast 2 rifles or SMGs at an average of {average_time} in-game clock timer



    """

    return result

  def answer_question_2c(self, map_path):
    columns = ['round_num', 'clock_time', 'team', 'side', 'player', 'area_name', 'x', 'y']
    data = self.df[columns]

    # filter data for Team2 CT @ Bombsite B
    team2 = data['team'] == 'Team2'
    sideCT = data['side'] == 'CT'
    bombsite_b = data['area_name'] == 'BombsiteB'
    filter_condition = team2 & bombsite_b & sideCT
    data = data[filter_condition]

    # group data to isolate timepoints with highest time remaining
    grouped = data.groupby(['round_num']).agg({'clock_time': 'max'})

    # merge grouped data with filtered data to access x and y
    merged = pd.merge(data, grouped, on=['round_num', 'clock_time'])

    # create heatmap data
    merged['count'] = 1
    heat_map_data = merged.groupby(['x', 'y']).agg({'count': 'sum'}).reset_index()

    # plot heatmap data
    map_img = mpimg.imread('de_overpass_bombsite_b.jpeg')

    # create subplots
    fig, ax1 = plt.subplots(figsize=(6, 7))

    # Display bombsite B map
    ax1.imshow(map_img, aspect='auto', extent=[heat_map_data['x'].min(), heat_map_data['x'].max(), heat_map_data['y'].min(), heat_map_data['y'].max()])

    # Create a second axes for the heatmap
    ax2 = fig.add_axes(ax1.get_position(), frameon=False)

    # Create the heatmap
    sns.kdeplot(x=heat_map_data['x'], y=heat_map_data['y'], cmap='viridis', fill=True, alpha=0.5)

    # hide heatmap axes
    ax2.xaxis.set_visible(False)
    ax2.yaxis.set_visible(False)

    # Display the plot
    plt.title('Heatmap of Team2 CT Defense Points Bombsite B')
    plt.show()


if __name__ == "__main__":
  data_path = "game_state_frame_data.parquet"
  result = ProcessGameState(data_path)

  map_path = "de_overpass_radar.jpeg"
  vertices = [(-1735, 250), (-2024, 398), (-2806, 742), (-2472, 1233), (-1565, 580)]
  z_lower_bound = 285
  z_upper_bound = 421

  goodbye = """Thank you for the opportunity to participate in this assessment.  I thoroughly enjoyed the assignment.


  """

  menu = """Please select a question to answer:

  a. Is entering via the light blue boundary a common strategy used by Team2 on T (terrorist) side?

  b. What is the average timer that Team2 on T (terrorist) side enters “BombsiteB” with least 2 rifles or SMGs?

  c. Heatmap reflecting the locations where Team2 is found in Bombsite B at the beginning of each round.

  q. To exit.

  Enter (a | b | c | q):  """

  while(True):
    response = input(menu)

    if response == 'a':
      print('\n\n', result.answer_question_2a(vertices, z_lower_bound, z_upper_bound))
    elif response == 'b':
      print('\n\n', result.answer_question_2b())
    elif response == 'c':
      result.answer_question_2c(map_path)
    elif response == 'q':
      print('\n\n', goodbye)
      quit()
    else:
      continue
