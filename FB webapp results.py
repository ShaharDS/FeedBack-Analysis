import streamlit as st
import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import langchain
import openai
import numpy as np
import re

# Create the Streamlit app
def main():
    st.title("Feedback Analysis")
    # Display an input box for the Google Drive link
    drive_link = st.text_input("Enter Google Drive link")

    if st.button("Run Analysis"):
        # Check if a link is provided
        if drive_link:
            OpenAIKey = "Your-Api-key"

            Prompt = """
            The reviews you see are for a product that we sell.
            Give most prevalent examples in bullets whether Positive or Negative feedback
            Next to each feedback is a quote that describes the feedback
            What do you suggest we focus on improving? 
            And Overall Insights Summary

            use this prompt:

            Positive feedback:
            1 - 5 Feedbacks  " Quote " 

            Negative:
            1-5 Feedbacks " Quote " 

            What to Improve:

            "improve ...."

            Overall Insights:

            The product have received positive/negative reviews... customers found .... some customers also said
            """

            # Vector dimension: 768
            # Loading JSON's from Google-Drive
            
            with st.spinner("Processing the data please wait..."):
                
                def read_json_from_google_drive_link(drive_link):
                    file_id = drive_link.split('/')[-2]
                    download_url = f"https://drive.google.com/uc?id={file_id}"
                    response = requests.get(download_url)
                    content = response.content.decode('utf-8')

                    if response.status_code != 200:
                        raise requests.exceptions.HTTPError(response)

                    json_data = json.loads(content)

                    return json_data


                drive_link = drive_link  # Set the drive_link variable

                response = requests.get(drive_link)
                html_text = response.text

                # Extract file IDs using regular expression
                file_ids = re.findall(r"/file/d/([^/]+)", html_text)

                # Storing the json files from the drive into a list for easier access.
                json_list = []

                for file_id in file_ids:
                    json_data = read_json_from_google_drive_link(
                        f"https://drive.google.com/file/d/{file_id}/view?usp=sharing")
                    if json_data not in json_list:  # Check if the data is already in the list
                        json_list.append(json_data)

                # Transfering all JSON uploaded from Google-Drive into Dataframes
                dataframes = []  # Creating a DataFrame list
                max_text_length = 600

                def truncate_review(text):  # Function to set limit on text length
                    return text[:max_text_length]

                for json_data in json_list:
                    df = pd.DataFrame(json_data)

                    # Renaming
                    new_columns = {
                        0: 'Name', 1: 'Key Words', 2: 'Text', 3: 'Stars'
                    }
                    df.rename(columns=new_columns, inplace=True)
                    #

                    # Removing Unwanted words for the model to be easier to understand and analyze.
                    df['Text'] = df['Text'].str.replace('The media could not be loaded.\n                \n\n\n\n\xa0', '')
                    df['truncated'] = df.apply(lambda row: truncate_review(row['Text']), axis=1)  # Applying "truncate_review" on each DataFrame
                    dataframes.append(df)
                
                total_dataframes = len(dataframes)

                # Importing NLTK (Vader) Progress.  --- Sentiment Analysis
                # + Function to decide whether Positive/Negative/Neutral
                from nltk.sentiment import SentimentIntensityAnalyzer

                sia = SentimentIntensityAnalyzer()

                ## Creating a function that will take each text from 'Text' column and apply their sentiment compound score to the new column
                def analyze_sentiment(text):
                    analyzer = SentimentIntensityAnalyzer()
                    sentiment_scores = analyzer.polarity_scores(text)
                    if sentiment_scores['compound'] >= 0.05:
                        return 'Positive'
                    elif sentiment_scores['compound'] <= -0.05:
                        return 'Negative'
                    else:
                        return 'Neutral'

                # Importing Embedding libraries
                # + Creating a new Embedded Column and uploading embedded values
                from langchain.embeddings import HuggingFaceEmbeddings

                embeddings = HuggingFaceEmbeddings()
                st.info("Embedding Feedbacks")
                for i, df in tqdm(enumerate(dataframes), desc="Embedding Feedbacks", total=total_dataframes):
                    df['embeddings'] = df.apply(lambda row: embeddings.embed_query(row['truncated']), axis=1)


                # Import Pinecone using their API.
                # + Storing vectors in
                # + Importing Langchain QA Model and OpenAI use
                import pinecone
                from langchain.vectorstores import Pinecone

                PINECONE_API_KEY = 'Your-Api-key'
                PINECONE_ENV = 'us-central1-gcp'

                pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

                #################

                from langchain.chains import RetrievalQA
                from langchain.chat_models import ChatOpenAI

                chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0, openai_api_key=OpenAIKey)

                query_results = []  ## Saving all query results
                

                st.info("Uploading Vectors into Pinecone")
                # Running in a loop on all dataframes
                # + Making "Truncated" column into a list
                # + Uploading all vectors to Pipecone with a unique namespace so we can identify which is what
                # + Using NLTK Vader's Sentiment Analysis and creating a new column that shows the result
                # + Returning all queries into a list (Goods Bads and what's to improve in your product based on your feedbacks)
                for i, df in tqdm(enumerate(dataframes), desc="Uploading Vectors", total=total_dataframes):
                    texts = df['truncated'].tolist()
                    namespace = 'p' + str(i + 1)  # Generate the namespace dynamically
                    
                    vstore = Pinecone.from_texts(texts, embeddings, index_name='geenie', namespace=namespace)
                    review_chain = RetrievalQA.from_chain_type(llm=chat, chain_type="stuff", retriever=vstore.as_retriever())
                    df['Sentiment Score'] = df['Text'].apply(analyze_sentiment)

                    result = review_chain.run(Prompt)
                    query_results.append(result)


                # Plotting the sentiment distribution for each dataframe
                for i, df in tqdm(enumerate(dataframes), desc="Generating Charts", total=total_dataframes):
                    st.subheader(f"DataFrame {i+1} Information:")
                    sentiment_counts = df['Sentiment Score'].value_counts()

                    # Print the query results
                    st.subheader(f"Query Results for DataFrame {i+1}:")
                    st.write(query_results[i])

                    # Create the Pie Chart
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
                    ax1.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%')
                    ax1.set_title("Sentiment Distribution (Pie Chart)")

                    # Create the bar chart
                    ax2.bar(sentiment_counts.index, sentiment_counts.values)
                    ax2.set_title("Sentiment Distribution (Bar Chart)")
                    ax2.set_xlabel("Sentiment")
                    ax2.set_ylabel("Count")

                    # Add counts to each bar
                    for index, value in enumerate(sentiment_counts.values):
                        ax2.text(index, value, str(value), ha='center', va='bottom')

                    # Display the charts
                    st.pyplot(fig)

        else:
            st.warning("Please enter a valid Google Drive link.")


if __name__ == '__main__': main()
