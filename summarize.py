import time
import pandas as pd
import torch
from gpt4all import GPT4All

start_time = time.time()

gpt_model = GPT4All("orca-mini-3b-gguf2-q4_0.gguf")

curr_date = "1-11-2019"
df_news = pd.read_excel('data/news.xlsx', engine='openpyxl', header=None)
df = df_news[df_news[0] == curr_date].copy()

summarized_texts = []

for i in range(len(df)):
    row = df.loc[i]
    with gpt_model.chat_session():
        summarize_text = gpt_model.generate(prompt='Summarize this article: ' + row[2], temp=0.1)
    summarized_texts.append(summarize_text)

df['Summarized Text'] = summarized_texts

end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

df.to_csv("data/summarized_news.csv", header=False)

# Print the elapsed time
print(f"Elapsed time: {elapsed_time} seconds")
