# from transformers import  AutoTokenizer, AutoModelForTokenClassification
# from scipy.special import softmax
#
# tweet = ""

# import snscrape.modules.twitter as sntwitter
# import  pandas as pd
# import time
#
# query = "python"
#
# for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
#     if i > 0 and i % 10 == 0:  # Every 10 iterations, take a break
#         time.sleep(60)  # Wait for 60 seconds
#     print(vars(tweet))