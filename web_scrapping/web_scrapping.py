import pandas as pd
import praw

# Reddit API kimlik bilgileri
reddit = praw.Reddit(
    client_id="C6fFCQfHo40z6Bt7miFQ9A",  # Reddit'ten aldığın Client ID
    client_secret="s46O4keWw_mMsNTGVcvwZICUZLy5iw",  # Reddit'ten aldığın Client Secret
    user_agent="ShallotBig5912",  # Örneğin: 'my chatbot scraper'
)

# Veri toplayacağın subreddit ve anahtar kelimeler
subreddit_name = "depression"  # İlgili subreddit (örneğin: depression, mentalhealth)
keyword = "help"  # Anahtar kelime (isteğe bağlı)

# Subredditteki gönderileri çek
posts = []
subreddit = reddit.subreddit(subreddit_name)

# İlk 100 gönderiyi çek
for post in subreddit.search(keyword, limit=10000):
    posts.append({
        "id": post.id,
        "title": post.title,
        "selftext": post.selftext,
        "created_utc": post.created_utc,
        "upvotes": post.score,
        "comments_count": post.num_comments
    })

# Verileri bir DataFrame'e dönüştür
df = pd.DataFrame(posts)

# CSV dosyasına kaydet
df.to_csv("web_scrapping/reddit_data.csv", index=False)
print("Veri çekme işlemi tamamlandı!")
