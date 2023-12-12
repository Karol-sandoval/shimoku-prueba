import shimoku_api_python as shimoku
from os import getenv
import numpy as np
import pandas as pd

from typing import List, Dict, Union
from aux import get_data

from dotenv import load_dotenv



load_dotenv()
#--------------- GET THE DATA DICTIONARY ----------------#
data = get_data('data/leads.csv', 'data/offers.csv')

#----------------- CLIENT INITIALIZATION ----------------#
api_key: str = getenv('API_TOKEN')
universe_id: str = getenv('UNIVERSE_ID')
workspace_id: str = getenv('WORKSPACE_ID')
#environment: str = getenv('ENVIRONMENT')

s = shimoku.Client(
access_token=api_key,
universe_id=universe_id,
#environment=environment,
async_execution=True,
verbosity='INFO',
)
s.set_workspace(workspace_id)

s.set_board('Custom Dashboard')

s.set_menu_path('Thinking process')


prediction_header = (
    "<head>"
    "<style>"  # Styles title
    ".component-title{height:auto; width:100%; "
    "border-radius:16px; padding:16px;"
    "display:flex; align-items:center;"
    "background-color:var(--chart-C1); color:var(--color-white);}"
    "</style>"
    # Start icons style
    "<style>.big-icon-banner"
    "{width:48px; height: 48px; display: flex;"
    "margin-right: 16px;"
    "justify-content: center;"
    "align-items: center;"
    "background-size: contain;"
    "background-position: center;"
    "background-repeat: no-repeat;"
    "background-image: url('https://uploads-ssl.webflow.com/619f9fe98661d321dc3beec7/63594ccf3f311a98d72faff7_suite-customer-b.svg');}"
    "</style>"
    # End icons style
    "<style>.base-white{color:var(--color-white);}</style>"
    "</head>"  # Styles subtitle
    "<div class='component-title'>"
    "<div class='big-icon-banner'></div>"
    "<div class='text-block'>"
    "<h1>1. Data Load</h1>"
    "<p class='base-white'>"
    "The first step corresponds to the loading of data, which in this case is leads and offers,"
    "containing information on potential leads and customers who have had at least one demo (offers).</p>"
    "</div>"
    "</div>"
)

s.plt.html(html=prediction_header, order=0)

leads_table = data['leads_table']
offers_table = data['offers_table']

prediction_table_header1 = (
        '<div style="width:100%; height:90px; "><h4>Table Leads</h4>'
    )

s.plt.html(html=prediction_table_header1, order=1)

s.plt.table(
        data=pd.DataFrame(leads_table).replace({np.nan: 'missing'}),
        # label_columns=leads_table.columns.values,
        # columns_options={
        #     'Lead ID': {'width': 100},
        #     'Lead Scoring': {'width': 120},
        #     'Probability': {'width': 120},
        #     'Positive Impact Factors': {'width': 590},
        #     'Negative Impact Factors': {'width': 590}
        # },
        order=2
    )


prediction_table_header2 = (
        '<div style="width:100%; height:90px; "><h4>Table Offers</h4>'
    )

s.plt.html(html=prediction_table_header2, order=3)

s.plt.table(
        data=pd.DataFrame(offers_table).replace({np.nan: 'missing'}),
        # label_columns=leads_table.columns.values,
        # columns_options={
        #     'Lead ID': {'width': 100},
        #     'Lead Scoring': {'width': 120},
        #     'Probability': {'width': 120},
        #     'Positive Impact Factors': {'width': 590},
        #     'Negative Impact Factors': {'width': 590}
        # },
        order=4
    )


prediction_header1 = (
"<head>"
"<style>"  # Styles title
".component-title{height:auto; width:100%; "
"border-radius:16px; padding:16px;"
"display:flex; align-items:center;"
"background-color:var(--chart-C1); color:var(--color-white);}"
"</style>"
# Start icons style
"<style>.big-icon-banner"
"{width:48px; height: 48px; display: flex;"
"margin-right: 16px;"
"justify-content: center;"
"align-items: center;"
"background-size: contain;"
"background-position: center;"
"background-repeat: no-repeat;"
"background-image: url('https://uploads-ssl.webflow.com/619f9fe98661d321dc3beec7/63594ccf3f311a98d72faff7_suite-customer-b.svg');}"
"</style>"
# End icons style
"<style>.base-white{color:var(--color-white);}</style>"
"</head>"  # Styles subtitle
"<div class='component-title'>"
"<div class='big-icon-banner'></div>"
"<div class='text-block'>"
"<h1>2. Data Preprocessing</h1>"
"<p class='base-white'>"
"</p>"
"</div>"
"</div>"
)
s.plt.html(html=prediction_header1, order=5)



table_explanaiton8 = (
    "<head>"
    "<style>.banner"
    "{height:100%; width:100%; border-radius:var(--border-radius-m); padding:24px;"
    "background-size: cover;"

    "color:var(--color-black);}"
    "</style>"
    "</head>"
    "<div class='banner'>"
    "<p class='base-black'>"
    "For processing, the response variable (purchase) was identified, a single value per ID was extracted"
    "in order to fit the model with unduplicated data and some small transformations were made on the data. For example:<br>" 
    "</p>"
    "</div>"
    "</a>"
)
s.plt.html(html=table_explanaiton8, order=6)

table_explanaiton7 = (
    "<head>"
    "<style>.banner"
    "{height:100%; width:100%; border-radius:var(--border-radius-m); padding:24px;"
    "background-size: cover;"
    "background-image: url('https://ajgutierrezcommx.files.wordpress.com/2022/12/bg-info-predictions.png');"
    "color:var(--color-white);}"
    "</style>"
    "</head>"
    "<div class='banner'>"
    "<p class='base-white'>"
    "<pre><code>"
    "def f(row):<br>"
    "if row['Status'] == 'Closed Won':<br>"
    "   val = 1<br>"
    "elif row['Status'] == 'Closed Lost':<br>"
    "  val = 0<br>"
    "else:<br>"
    "  val = -1<br>"
    "return val<br>"
    "offers['status_num'] = offers.apply(f, axis=1)<br>"
    "</code></pre>"
    "</p>"
    "</div>"
    "</a>"
)
s.plt.html(html=table_explanaiton7, order=7)



prediction_header2 = (
    "<head>"
    "<style>"  # Styles title
    ".component-title{height:auto; width:100%; "
    "border-radius:16px; padding:16px;"
    "display:flex; align-items:center;"
    "background-color:var(--chart-C1); color:var(--color-white);}"
    "</style>"
    # Start icons style
    "<style>.big-icon-banner"
    "{width:48px; height: 48px; display: flex;"
    "margin-right: 16px;"
    "justify-content: center;"
    "align-items: center;"
    "background-size: contain;"
    "background-position: center;"
    "background-repeat: no-repeat;"
    "background-image: url('https://uploads-ssl.webflow.com/619f9fe98661d321dc3beec7/63594ccf3f311a98d72faff7_suite-customer-b.svg');}"
    "</style>"
    # End icons style
    "<style>.base-white{color:var(--color-white);}</style>"
    "</head>"  # Styles subtitle
    "<div class='component-title'>"
    "<div class='big-icon-banner'></div>"
    "<div class='text-block'>"
    "<h1>3. Descriptive Analysis and considerations</h1>"
    "<p class='base-white'>"
    "</p>"
    "</div>"
    "</div>"
)
s.plt.html(html=prediction_header2, order=8)



table_explanaiton1 = (
    "<head>"
    "<style>.banner"
    "{height:100%; width:100%; border-radius:var(--border-radius-m); padding:24px;"
    "background-size: cover;"
    "background-image: url('https://ajgutierrezcommx.files.wordpress.com/2022/12/bg-info-predictions.png');"
    "color:var(--color-white);}"
    "</style>"
    "</head>"
    "<div class='banner'>"
    "<p class='base-white'>"
    "It was found in the data set that most of the potential leads have corporate events use case, that the most frequent"
    "cities are Chicago, San Francisco and San Diego, that the most frequent acquisition campaign in the leads is the virtual" 
    "meetups and as for the days of the week and the months of creation of the leads are mostly Monday, Tuesday, Saturday" 
    "and Sunday and of the months are June and September those with the highest frequency. For example:<br>" 
    "</p>"
    "</div>"
    "</a>"
)
s.plt.html(html=table_explanaiton1, order=9)

# consolidado = pd.DataFrame(data['lead_final'].groupby(['Use Case'])['Id'].count()).reset_index()
leads_final= data['lead_final']
s.plt.bar(
    order=10, title='Language expressiveness',
    data=leads_final, y=['Id'],
    x=['Use Case'],
)

prediction_header3 = (
    "<head>"
    "<style>"  # Styles title
    ".component-title{height:auto; width:100%; "
    "border-radius:16px; padding:16px;"
    "display:flex; align-items:center;"
    "background-color:var(--chart-C1); color:var(--color-white);}"
    "</style>"
    # Start icons style
    "<style>.big-icon-banner"
    "{width:48px; height: 48px; display: flex;"
    "margin-right: 16px;"
    "justify-content: center;"
    "align-items: center;"
    "background-size: contain;"
    "background-position: center;"
    "background-repeat: no-repeat;"
    "background-image: url('https://uploads-ssl.webflow.com/619f9fe98661d321dc3beec7/63594ccf3f311a98d72faff7_suite-customer-b.svg');}"
    "</style>"
    # End icons style
    "<style>.base-white{color:var(--color-white);}</style>"
    "</head>"  # Styles subtitle
    "<div class='component-title'>"
    "<div class='big-icon-banner'></div>"
    "<div class='text-block'>"
    "<h1>3.1 Considerations</h1>"
    "<p class='base-white'>"
    "</p>"
    "</div>"
    "</div>"
)
s.plt.html(html=prediction_header3, order=11)

table_explanaiton2 = (
    "<head>"
    "<style>.banner"
    "{height:100%; width:100%; border-radius:var(--border-radius-m); padding:24px;"
    "background-size: cover;"
    "background-image: url('https://ajgutierrezcommx.files.wordpress.com/2022/12/bg-info-predictions.png');"
    "color:var(--color-white);}"
    "</style>"
    "</head>"
    "<div class='banner'>"
    "<p class='base-white'>"
    "Within the Leads and Offers databases, missing data were found in several variables, this generally"
    "implies loss of valuable information in case of not using techniques such as data imputation to treat"
    "them, however, for variables such as the ID, imputation is not the best solution since the"
    "characteristics in each of the databases are not very specific to approximate in the most accurate"
    "way the value of the Id, therefore, a significant set of data was not considered within the analysis"
    "and the machine learning model; in short, there was a significant amount of Leads and Offers data"
    "that could not be matched and, therefore, were not considered for this test. Specifically,"
    "the following was found:<br><br>"
    ""
    "- Of the 61639 records in Leads a total of 17667 (29%) missing data IDs were found.<br>"
    "- Of the 6130 records in Offers a total of 1168 (19%) missing data in ID was found. <br><br>"
    ""
    "Considering that there is not much information on why there is so much missing data, it would be"
    "important to identify the reason and avoid problems in the future with some suggestions such as:<br><br>"
    ""
    "- Implement a more robust data collection strategy: improve data capture forms (training people if applicable)"
    "or conduct more rigorous data validation to ensure proper data collection and storage.<br>"
    "Create a data backup system: If data is lost or corrupted it is important to have a backup system to recover"
    "lost information.<br>"
    "- Continuously monitor data quality: Create periodic reports to validate data quality, perform data audits"
    "or implement key performance indicators (KPIs) to monitor data quality.<br><br>"
    ""
    "Another recommendation may be to add other types of variables to better characterize customers and thus better "
    "predict the probability of lead purchase, such as: age, gender, employment, among other factors."
    "cities are Chicago, San Francisco and San Diego, that the most frequent acquisition campaign in the leads is the virtual" 
    "meetups and as for the days of the week and the months of creation of the leads are mostly Monday, Tuesday, Saturday" 
    "and Sunday and of the months are June and September those with the highest frequency.<br><br>" 
    ""
    "</p>"
    "</div>"
    "</a>"
)
s.plt.html(html=table_explanaiton2, order=12)


prediction_header4 = (
    "<head>"
    "<style>"  # Styles title
    ".component-title{height:auto; width:100%; "
    "border-radius:16px; padding:16px;"
    "display:flex; align-items:center;"
    "background-color:var(--chart-C1); color:var(--color-white);}"
    "</style>"
    # Start icons style
    "<style>.big-icon-banner"
    "{width:48px; height: 48px; display: flex;"
    "margin-right: 16px;"
    "justify-content: center;"
    "align-items: center;"
    "background-size: contain;"
    "background-position: center;"
    "background-repeat: no-repeat;"
    "background-image: url('https://uploads-ssl.webflow.com/619f9fe98661d321dc3beec7/63594ccf3f311a98d72faff7_suite-customer-b.svg');}"
    "</style>"
    # End icons style
    "<style>.base-white{color:var(--color-white);}</style>"
    "</head>"  # Styles subtitle
    "<div class='component-title'>"
    "<div class='big-icon-banner'></div>"
    "<div class='text-block'>"
    "<h1>4. Feature Selection and Feature Engineering</h1>"
    "<p class='base-white'>"
    "</p>"
    "</div>"
    "</div>"
)
s.plt.html(html=prediction_header4, order=13)


table_explanaiton2 = (
    "<head>"
    "<style>.banner"
    "{height:100%; width:100%; border-radius:var(--border-radius-m); padding:24px;"
    "background-size: cover;"
    "background-image: url('https://ajgutierrezcommx.files.wordpress.com/2022/12/bg-info-predictions.png');"
    "color:var(--color-white);}"
    "</style>"
    "</head>"
    "<div class='banner'>"
    "<p class='base-white'>"
    "Due to the categorical nature of the variables, a logistic regression model was used to establish which variables were significant,"
    "following some feature engineering steps such as: the establishment of coding for dummy variables and the imputation of missing data"
    "either by the highest frequency category or in case of more than 10% of null values the creation of a new category called missing. For more details see the notebook.<br>"
    "</p>"
    "</div>"
    "</a>"
)
s.plt.html(html=table_explanaiton2, order=14)



prediction_header5 = (
    "<head>"
    "<style>"  # Styles title
    ".component-title{height:auto; width:100%; "
    "border-radius:16px; padding:16px;"
    "display:flex; align-items:center;"
    "background-color:var(--chart-C1); color:var(--color-white);}"
    "</style>"
    # Start icons style
    "<style>.big-icon-banner"
    "{width:48px; height: 48px; display: flex;"
    "margin-right: 16px;"
    "justify-content: center;"
    "align-items: center;"
    "background-size: contain;"
    "background-position: center;"
    "background-repeat: no-repeat;"
    "background-image: url('https://uploads-ssl.webflow.com/619f9fe98661d321dc3beec7/63594ccf3f311a98d72faff7_suite-customer-b.svg');}"
    "</style>"
    # End icons style
    "<style>.base-white{color:var(--color-white);}</style>"
    "</head>"  # Styles subtitle
    "<div class='component-title'>"
    "<div class='big-icon-banner'></div>"
    "<div class='text-block'>"
    "<h1>5.Predictions</h1>"
    "<p class='base-white'>"
    "Lead scoring prediction</p>"
    "</div>"
    "</div>"
)
s.plt.html(html=prediction_header5, order=15)


table_explanaiton4 = (
    "<head>"
    "<style>.banner"
    "{height:100%; width:100%; border-radius:var(--border-radius-m); padding:24px;"
    "background-size: cover;"
    "background-image: url('https://ajgutierrezcommx.files.wordpress.com/2022/12/bg-info-predictions.png');"
    "color:var(--color-white);}"
    "</style>"
    "</head>"
    "<div class='banner'>"
    "<p class='base-white'>"
    "Regarding the variables for the construction of the model, within the descriptive statistics they do not seem to be very decisive,"
    "however, lead status was discarded since it indicates the state of rejection and as the intention is to predict the probability of"
    "purchase it is not useful, in turn, Discarded/Nurturing Reason was not taken for the same reason. In addition, some additional"
    "variables such as month, day of the week and day of the month were created and, for a future occasion, a standardization and"
    "regrouping of the classes of categorical variables is recommended, in order to simplify the interpretation of the model, avoid"
    "estimation problems or avoid overfitting problems."
    "</p>"
    "</div>"
    "</a>"
)
s.plt.html(html=table_explanaiton4, order=16)





prediction_table_header = (
        '<div style="width:100%; height:90px; "><h4>Lead Scoring predictions and factors</h4>'
        '<p></p></div>'
    )
def get_label_columns(table_data: pd.DataFrame) -> Dict:
    low_threshold = table_data["Probability"][table_data["Lead Scoring"] == "Low"].max() + 1e-10
    mid_threshold = table_data["Probability"][table_data["Lead Scoring"] == "Medium"].max() + 1e-10
    return {
        ('Positive Impact Factors', 'outlined'): '#20C69E',
        ('Negative Impact Factors', 'outlined'): '#ED5627',
        'Lead Scoring': {
            'Low': '#F86C7D',
            'High': '#001E50',
            'Medium': '#F2BB67',
        },
        'Probability': {
            (0, low_threshold): '#F86C7D',
            (low_threshold, mid_threshold): '#F2BB67',
            (mid_threshold, np.inf): '#001E50',
        },
    }

s.plt.html(html=prediction_table_header, order=17)

binary_prediction_table = data['binary_prediction_table']

label_columns = get_label_columns(binary_prediction_table)

s.plt.table(
        data=binary_prediction_table[:200],
        label_columns=label_columns, categorical_columns=['Lead Scoring'],
        columns_options={
            'Lead ID': {'width': 100},
            'Lead Scoring': {'width': 120},
            'Probability': {'width': 120},
            'Positive Impact Factors': {'width': 590},
            'Negative Impact Factors': {'width': 590}
        },
        order=18
    )

# s.boards.force_delete_board(name='Thinking process')

s.run()