import pandas as pd
import numpy as np
from scipy.stats import gamma
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from scipy.stats import gamma, norm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


''' format of the data  : 
Mandal wise Day wise Rainfall Data from 2004 June to 2023 December			
District	Mandal	Date	Rainfall (mm)
Hyderabad	Shaikpet	06/01/2004	0
Hyderabad	Shaikpet	06/02/2004	0
Hyderabad	Shaikpet	06/03/2004	0
'''

df = pd.read_csv('IIIT Rainfall Data.csv')

# convert the date column to datetime
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
df['Day'] = df['Date'].dt.day
df['MonthYear'] = df['Year'].astype(str) + '-' + df['Month'].astype(str)

##
# df['MonthYear'] = pd.to_datetime(df['MonthYear'], format='%m-%Y')
# print(df)
# exit(0)
##sort df by MonthYear

df = df.groupby(['District', 'Mandal', 'MonthYear'])['Rainfall(mm)'].sum().reset_index()
df = df.sort_values(by='MonthYear')

'''
     District    Mandal  MonthYear  Rainfall(mm)
0   Hyderabad  Ameerpet 2004-06-01          29.5
1   Hyderabad  Ameerpet 2004-07-01         221.3
2   Hyderabad  Ameerpet 2004-08-01          77.5
3   Hyderabad  Ameerpet 2004-09-01         114.0
4   Hyderabad  Ameerpet 2004-10-01          72.5
5   Hyderabad  Ameerpet 2004-11-01           0.0
6   Hyderabad  Ameerpet 2004-12-01           0.0
7   Hyderabad  Ameerpet 2005-01-01           0.0
8   Hyderabad  Ameerpet 2005-02-01           0.0
9   Hyderabad  Ameerpet 2005-03-01           0.0

'''

def SPI(data, scale):
    accumulation_period = scale
    data = data.reset_index(drop=True)
    data['SPI'] = np.nan
    rainfall_data = data['Rainfall(mm)'].values
    rainfall_accumulated = np.convolve(rainfall_data, np.ones(scale) / scale, mode='valid')
    rainfall_accumulated[rainfall_accumulated == 0] = 1e-15
    shape, loc, scale = gamma.fit(rainfall_accumulated, floc=0)
    gamma_cdf_rainfall = gamma.cdf(rainfall_accumulated, shape, loc, scale=scale)
    spi_values = norm.ppf(gamma_cdf_rainfall)
    data.loc[accumulation_period - 1:, 'SPI'] = spi_values
    return data

df['SPI-1'] = np.nan
df['SPI-3'] = np.nan
mandals = df['Mandal'].unique()
print(mandals)
for i in mandals:
    data = df[df['Mandal'] == i].copy()
    ret = SPI(data, 1)
    df.loc[df['Mandal'] == i, 'SPI-1'] = ret['SPI'].values
    data = df[df['Mandal'] == i].copy()
    ret = SPI(data, 3)
    df.loc[df['Mandal'] == i, 'SPI-3'] = ret['SPI'].values
print(df)
# exit(0)

##iterate over each mandal and plot its spi value over the years
import matplotlib.pyplot as plt
import seaborn as sns

mandals = df['Mandal'].unique()
for i in mandals:
    data = df[df['Mandal'] == i]
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data, x='MonthYear', y='SPI-3')
    ##label x axis not contunusoly rather in steps
    plt.xticks(data['MonthYear'][::24])
    plt.title(f'SPI-3 for Mandal: {i}')
    # plt.xticks(rotation=45)
    plt.show()


def categorize_spi(spi_value):
    if spi_value >= 2:
        return 'Extremely wet'
    elif 1.5 <= spi_value < 2:
        return 'Severely wet'
    elif 1 <= spi_value < 1.5:
        return 'Moderately wet'
    elif 0 <= spi_value < 1:
        return 'Mildly wet'
    elif -1 < spi_value < 0:
        return 'Mild drought'
    elif -1.5 < spi_value <= -1:
        return 'Moderate drought'
    elif -2 < spi_value <= -1.5:
        return 'Severe drought'
    else:
        return 'Extreme drought'

extreme_drought_count = np.sum(df['SPI-3'].apply(categorize_spi) == 'Extreme drought')
moderate_drought_count = np.sum(df['SPI-3'].apply(categorize_spi) == 'Moderate drought')
mild_drought_count = np.sum(df['SPI-3'].apply(categorize_spi) == 'Mild drought')
severe_drought_count = np.sum(df['SPI-3'].apply(categorize_spi) == 'Severe drought')


'''
        District           Mandal  MonthYear  Rainfall(mm)     SPI-1     SPI-3
0      Hyderabad         Ameerpet 2004-06-01          29.5  0.776753       NaN
1      Hyderabad         Ameerpet 2004-07-01         221.3  1.333543       NaN
2      Hyderabad         Ameerpet 2004-08-01          77.5  1.005546  0.865395
3      Hyderabad         Ameerpet 2004-09-01         114.0  1.114248  0.974492
4      Hyderabad         Ameerpet 2004-10-01          72.5  0.987912  0.768827
...          ...              ...        ...           ...       ...       ...
1640  Rangareddy  Serilingampally 2023-08-01          50.3  0.945854  1.192192
1641  Rangareddy  Serilingampally 2023-09-01         338.9  1.555445  1.405369
1642  Rangareddy  Serilingampally 2023-10-01           0.0 -1.723672  1.014658
1643  Rangareddy  Serilingampally 2023-11-01          38.9  0.886376  1.000390
1644  Rangareddy  Serilingampally 2023-12-01           3.7  0.464710  0.230298

'''

'''

SPI value Category Probability (%)
2.00 or more Extremely wet 2.3
1.50 to 1.99 Severely wet 4.4
1.00 to 1.49 Moderately wet 9.2
0 to 0.99 Mildly wet 34.1
0 to −0.99 Mild drought 34.1
−1.00 to −1.49 Moderate drought 9.2
−1.50 to −1.99 Severe drought 4.4
−2 or less Extreme drought
'''

df = df.dropna(subset=['SPI-3'])

def categorize_spi_number(spi_value):
    if spi_value >= 2:
        return 0
    elif 1.5 <= spi_value < 2:
        return 1
    elif 1 <= spi_value < 1.5:
        return 2
    elif 0 <= spi_value < 1:
        return 3
    elif -1 < spi_value < 0:
        return 4
    elif -1.5 < spi_value <= -1:
        return 5
    elif -2 < spi_value <= -1.5:
        return 6
    else:
        return 7




##create one rnn model 
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Function to categorize SPI values
def categorize_spi(spi_value):
    if spi_value >= 2:
        return 'Extremely wet'
    elif 1.5 <= spi_value < 2:
        return 'Severely wet'
    elif 1 <= spi_value < 1.5:
        return 'Moderately wet'
    elif 0 <= spi_value < 1:
        return 'Mildly wet'
    elif -1 < spi_value < 0:
        return 'Mild drought'
    elif -1.5 < spi_value <= -1:
        return 'Moderate drought'
    elif -2 < spi_value <= -1.5:
        return 'Severe drought'
    else:
        return 'Extreme drought'

# Function to create an somple rmnn
def create_lstm_model(input_shape, output_dim):
    model = keras.Sequential([
        layers.LSTM(80, input_shape=input_shape),
        ##use ropout
        layers.Dropout(0.2),
        layers.Dense(100, activation='relu'),
        layers.Dense(32, activation='relu'),

        layers.Dense(output_dim, activation='softmax')
    ])
    ##use loss functions so peaks are detected
    loss = tf.keras.losses.CategoricalFocalCrossentropy(from_logits=True)
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    return model


import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Function to categorize SPI values
def categorize_spi(spi_value):
    if spi_value >= 2:
        return 'Extremely wet'
    elif 1.5 <= spi_value < 2:
        return 'Severely wet'
    elif 1 <= spi_value < 1.5:
        return 'Moderately wet'
    elif 0 <= spi_value < 1:
        return 'Mildly wet'
    elif -1 < spi_value < 0:
        return 'Mild drought'
    elif -1.5 < spi_value <= -1:
        return 'Moderate drought'
    elif -2 < spi_value <= -1.5:
        return 'Severe drought'
    else:
        return 'Extreme drought'

# Function to create an RNN model
def create_rnn_model(input_shape, output_dim):
    model = keras.Sequential([
        layers.SimpleRNN(128, input_shape=input_shape),
        ##dropout
        # layers.Dropout(0.1),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),

        layers.Dense(output_dim, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


X = df[['SPI-3','Rainfall(mm)','Mandal']].values[:-1]  # Drop the last row

##just map mandal to array index of this
'''
['Ameerpet' 'Khairatabad' 'Shaikpet' 'Balanagar' 'Kukatpally'
 'Quthbullapur' 'Serilingampally']
'''
mandal_map = {'Ameerpet':0, 'Khairatabad':1, 'Shaikpet':2, 'Balanagar':3, 'Kukatpally':4,
    'Quthbullapur':5, 'Serilingampally':6}

X[:,2] = [mandal_map[i] for i in X[:,2]]
print(X)
X = np.asarray(X).astype(np.float32)
y = df['SPI-3'][1:].apply(categorize_spi).values  # Shift SPI-3 by 1 month and categorize

# Encode category labels
le = LabelEncoder()
y = le.fit_transform(y)

# Train Test Splitting - RNN
X_train, X_test, y_train, y_test = X[:int(0.8*len(X))], X[int(0.8*len(X)):], y[:int(0.8*len(y))], y[int(0.8*len(y)):]
# Reshape input data for RNN
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

model = create_rnn_model(input_shape=(1, X_train.shape[2]), output_dim=len(le.classes_))
model.fit(X_train, y_train, epochs=100, batch_size=32)

# Make predictions for the test set
y_pred = model.predict(X_test)
y_pred_categories = le.inverse_transform(np.argmax(y_pred, axis=1))
y_test_categories = le.inverse_transform(y_test)

# Print some sample predictions
print("Sample predictions:")
for i in range(5):
    print(f"Actual: {y_test_categories[i]}, Predicted: {y_pred_categories[i]}")

# Print accuracy of the model
accuracy = accuracy_score(y_test_categories, y_pred_categories)
print(f"Accuracy: {accuracy}")


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Create confusion matrix
extreme_drought_count = np.sum(y_test_categories == 'Extreme drought')
print(f"Number of instances of 'Extreme drought' in y_test: {extreme_drought_count}")
print(y_test_categories)
print(y_pred_categories)
cm = confusion_matrix(y_test_categories, y_pred_categories)

# Plot confusion matrix
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


import matplotlib.pyplot as plt
import numpy as np

# Define a mapping dictionary for the categories
category_mapping = {
    'Extremely wet': 7,
    'Severely wet': 6,
    'Moderately wet': 5,
    'Mildly wet': 4,
    'Mild drought': 3,
    'Moderate drought': 2,
    'Severe drought': 1,
    'Extreme drought': 0
}


plot_test = [category_mapping[category]  for category in y_test_categories]
plot_pred = [category_mapping[category] for category in y_pred_categories]

##select random set of 100 values
plot_test = plot_test[:100]
plot_pred = plot_pred[:100]
plt.figure(figsize=(10, 10))
plt.plot(plot_test, label='Actual')
plt.plot(plot_pred, label='Predicted')

# # Highlight points where the predictions are an exact match
exact_match_indices = np.where(np.array(plot_test) == np.array(plot_pred))[0]
plt.scatter(exact_match_indices, np.array(plot_test)[exact_match_indices], color='pink', marker='o', label='Exact Match')

plt.xlabel('Index')
plt.ylabel('Drought Category')
plt.yticks(list(category_mapping.values()), list(category_mapping.keys()))
plt.title('Actual vs Predicted Drought Category')
plt.legend()
plt.show()