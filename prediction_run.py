import requests

# Initialization
url = 'http://127.0.0.1:9000/predict'


test_dict_sample = {'Gender': 'Male',
                    'Customer Type': 'Loyal Customer',
                    'Age': 56,
                    'Type of Travel': 'Business travel',
                    'Class': 'Business',
                    'Flight Distance': 600,
                    'Inflight wifi service': 2,
                    'Departure/Arrival time convenient': 1,
                    'Ease of Online booking': 2,
                    'Gate location': 2,
                    'Food and drink': 3,
                    'Online boarding': 4,
                    'Seat comfort': 4,
                    'Inflight entertainment': 4,
                    'On-board service': 4,
                    'Leg room service': 4,
                    'Baggage handling': 4,
                    'Checkin service': 4,
                    'Cleanliness': 4,
                    'Departure Delay in Minutes': 92,
                    'Arrival Delay in Minutes': 82.0}

# Main
response = requests.post(url, json=test_dict_sample)
result = response.json()
print(result)