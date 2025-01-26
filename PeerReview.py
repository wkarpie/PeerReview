#!/usr/bin/env python
# coding: utf-8

# # <ins>**Monitoring and Reviewing Academic Publications by Author**<ins>  
# 

# # **Outline**
# 1. Introduction
# 2. Getting Publication Data from INSPIRE
# 3. Data Formatting
# 4. Using the Gemini API
# 5. Notification

# # **1. Introduction**

# ## Summary
# ### This script uses requests, BeautifulSoup, Google's Gemini API to retrieve academic publications from a specific author and review the newest articles.

# ### Importing Libraries and Setting up Gemini API access

# ### ensure the following are installed before running, pip install google-generativeai requests beautifulsoup4 httpx


# In[1]:


import pandas as pd
import requests
from bs4 import BeautifulSoup
import json
import sys
import httpx
import base64
import google.generativeai as genai
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os

# In[2]:

# Retrieve the API keys from the environment variable
api_key = os.getenv('GOOGLE_GENAI_API_KEY')
GOOGLE_EMAIL_APPWORD = os.getenv('GOOGLE_EMAIL_APPWORD')

# Check if the API keys were retrieved successfully
if api_key is None:
    print("Error: GOOGLE_GENAI_API_KEY environment variable is not set.")
    sys.exit()
if GOOGLE_EMAIL_APPWORD is None:
    print("Error: GOOGLE_EMAIL_APPWORD environment variable is not set.")
    sys.exit()

# In[3]:



genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")


# # 2. Getting Publication Data from INSPIRE

# ### INSPIRE offers [API](https://github.com/inspirehep/rest-api-doc) access and documentation. For this project, retrieving a single author's current list of publications was easiest to complete by using the literature identifier and the value 'Joseph.Karpie.1'. 

# In[4]:

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

response = requests.get('https://inspirehep.net/api/literature?sort=mostrecent&size=250&page=1&q=a%20Joseph.Karpie.1', headers=headers)
print(response.headers)


soup = BeautifulSoup(response.text, "html.parser")


# # 3. Data Formatting

# ### The content inside of the first 'p' tag is the result of the search for publications. Below we transform that text into a dataframe.

# In[5]:


# Parse the response as JSON (instead of using BeautifulSoup)
data = response.json()  # Automatically decodes the JSON response

# Now you can work with the 'data' dictionary and process it
hits = data.get('hits', {}).get('hits', [])

#Normalize the nested structure and convert it to a DataFrame. This will flatten the nested JSON structure into a more tabular form
df_publications = pd.json_normalize(hits)

#Display info for the DataFrame
df_publications.info()


# ### There is a lot of metadata that will be useful for this project, below are select usefull columns.

# In[6]:


df_publications[['id','metadata.titles','metadata.abstracts','updated','created','metadata.citation_count','metadata.number_of_pages']]


# ### The 'id' column has unique values used by INSPIRE to track publications, this can be used to check against a tracker file to indicate if a publication is new or not.

# In[7]:


df_previous_publications = pd.read_excel('previous_publications.xlsx')


# In[8]:


df_publications['id'] = pd.to_numeric(df_publications['id'], errors='coerce') # need to convert in working df because the excel changes 'id' column to integers
df_new_publications = df_publications[~df_publications['id'].isin(df_previous_publications['id'])]
df_new_publications


# ### If df is empty, then there are no new publications so we can end the script here.

# In[9]:


# Check if the DataFrame is empty
if len(df_new_publications) == 0:
    print("No publications to process. Exiting the script.")
    sys.exit()


# ### The metadata is structured in dictionaries, below we make a function to extract the values from those dictionaries that we care about and append them to the end of the data frame.

# In[ ]:


#Extract 'value' from each list of dictionaries, handle NaN or non-iterable values
def extract_value(entry):
    if isinstance(entry, list) and len(entry) > 0 and isinstance(entry[0], dict):
        return entry[0].get('value')  # Safely extract the 'value'
    return None  # Return None for invalid or empty entries

#Extract 'title' from each list of dictionaries, handle NaN or non-iterable values
def extract_title(entry):
    if isinstance(entry, list) and len(entry) > 0 and isinstance(entry[0], dict):
        return entry[0].get('title')  # Safely extract the 'value'
    return None  # Return None for invalid or empty entries

# Apply the extraction functions
df_new_publications['titles'] = df_new_publications['metadata.titles'].apply(extract_title)

df_new_publications['abstracts'] = df_new_publications['metadata.abstracts'].apply(extract_value)
df_new_publications['arxiv_value'] = df_new_publications['metadata.arxiv_eprints'].apply(extract_value)
df_new_publications['arxiv_link'] = "https://arxiv.org/pdf/"+ df_new_publications['arxiv_value']


# Display the DataFrame with the new columns
df_new_publications[['id','titles','abstracts','updated','created','metadata.citation_count','metadata.number_of_pages','arxiv_value','arxiv_link']]


# # 4. Using the Gemini API

# In[ ]:


# Function to get and encode PDF
def get_encoded_pdf(url):
    try:
        response = httpx.get(url)
        response.raise_for_status()  # Raise an error for bad responses
        return base64.standard_b64encode(response.content).decode("utf-8")
    except httpx.HTTPError as e:
        print(f"HTTP error occurred: {e}")
        return None

# Function to generate summary
def generate_summary(doc_data, prompt):
    try:
        response = model.generate_content([{'mime_type': 'application/pdf', 'data': doc_data}, prompt])
        return response.text
    except Exception as e:
        print(f"Error generating summary: {e}")
        return None

# Function to generate suggestions
def generate_suggestions(doc_data, prompt):
    try:
        response = model.generate_content([{'mime_type': 'application/pdf', 'data': doc_data}, prompt])
        return response.text
    except Exception as e:
        print(f"Error generating suggestions: {e}")
        return None

# Initialize an empty list to store summaries and suggestions
summaries = []
suggestions = []

# Iterate over each row in the DataFrame
for index, row in df_new_publications.iterrows():
    doc_url = row['arxiv_link']
    print(f"Processing URL: {doc_url}")

    # Get and encode the PDF
    doc_data = get_encoded_pdf(doc_url)
    if doc_data is None:
        summaries.append(None)
        suggestions.append(None)
        continue

    # Generate summary
    summary_prompt = """Summarize this as if you’re the author trying to explain it to a five year old, call them "little scientist."
    Keep it simple, fun, and as if you're giving a crash course for someone who barely remembers anything from high school physics."""
    summary = generate_summary(doc_data, summary_prompt)
    summaries.append(summary)

    # Generate suggestions for improvement
    suggestions_prompt = """Based on the article's content, draft an email to the author, your older brother Joseph. 
    Begin the email by expressing appreciation for the article, highlighting specific aspects you found insightful or engaging. 
    After the positive introduction, kindly offer constructive suggestions for improvement. Focus on areas such as structure, clarity, 
    and any missing details or explanations that could enhance understanding and accessibility of the topic. Conclude the email with a 
    thoughtful Albert Einstein quote, prefacing it like this: "Remember what Einstein said," Please format your response in HTML."""

    suggestion = generate_suggestions(doc_data, suggestions_prompt)
    suggestions.append(suggestion)

# Add summaries and suggestions to the DataFrame
df_new_publications['summary'] = summaries
df_new_publications['suggestion'] = suggestions


# In[ ]:


df_new_publications['summary'][0]


# In[ ]:


df_new_publications['suggestion'][0]


# # 5. Notification

# In[ ]:


# Email credentials
email_address = "wkarpie.dev@gmail.com"
email_password = GOOGLE_EMAIL_APPWORD  # Use gmail App Password if 2FA is enabled

# Recipient
to_email = "wkarpie.dev@gmail.com"

# Connect to the Gmail SMTP server
try:
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()  # Secure the connection
    server.login(email_address, email_password)
    print("Logged in to the server successfully!")

    # Loop through each row in the DataFrame
    for index, row in df_new_publications.iterrows():
        # Subject and body for each email
        subject = f"New Publication: {row['titles']}!"
        body = (
            f"<html>"
            f"<body>"
            f"<p>Hello there,</p>"
            f"<p><strong>Joseph published something new:</strong> {row['titles']}</p>"
            f"<p><strong>Here is a Summary (written for a five-year-old):</strong><br>{summary}</p>"
            f"<p><strong>Link:</strong> <a href='{row['arxiv_link']}'>{row['arxiv_link']}</a></p>"
            f"<p><strong>Here is an email draft to Joseph, with some suggetsions for improvement:</strong> <br>{suggestion}</p>"
            f"</body>"
            f"</html>"
        )
        
        # Create a MIMEMultipart object
        msg = MIMEMultipart()
        msg['From'] = email_address
        msg['To'] = to_email
        msg['Subject'] = subject

        # Attach the body with the msg instance
        msg.attach(MIMEText(body, 'html'))

        # Send the email
        text = msg.as_string()
        server.sendmail(email_address, to_email, text)
        print(f"Email sent successfully for publication: {row['titles']}")

except Exception as e:
    print(f"Failed to send email: {e}")
finally:
    server.quit()
    print("Server connection closed.")


# ### Append the new publications to df_previous_publications and save an updated version of the previous publications tracker excel.

# In[ ]:


df_previous_publications = pd.concat([df_previous_publications['id'], df_new_publications['id']], ignore_index=True).reset_index(drop=True)


# In[ ]:


df_previous_publications.to_excel(f"previous_publications.xlsx", index=False)

