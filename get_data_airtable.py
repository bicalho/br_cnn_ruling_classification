import pandas as pd
from sklearn.model_selection import train_test_split
from airtable import Airtable

airtable = Airtable('appwyZMLKcg4sYvpC', 'RC', api_key='keyMqApx7X0uKHXu6')
rc_data = []
for page in airtable.get_iter():
    for record in page:
        rc_data.append({
            'code' : record['fields']['2'],
            'uf' : record['fields']['3'],
            'type' : record['fields']['4'],
            'rapporteur' : record['fields']['5'],
            'date' : record['fields']['6'],
            'class' : record['fields']['7'],
            'court_decision' : record['fields']['10'],
            'process_type' : record['fields']['13'],
            'process_number' : record['fields']['15'],
            'court_type' : record['fields']['15'],
            'city' : record['fields']['18'],
            'merito_modo' : record['fields']['mérito_modo'],
            'merito_dispositivo' : record['fields']['merito_dispositivo'],
            'merito_abrangência' : record['fields']['merito_abrangêcia'],
            'preliminar_modo' : record['fields']['preliminar_modo'],
            'preliminar_dispositivo' : record['fields']['preliminar_dispositivo'],
            'sentimento' : record['fields']['sentimento'],
        })

df = pd.DataFrame(rc_data)
df.to_csv('data/rc.csv', encoding='utf-8')

