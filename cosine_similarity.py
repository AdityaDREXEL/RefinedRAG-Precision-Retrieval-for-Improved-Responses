import json
import os

from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

# Initialize the Langchain OpenAI Embedding API
embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

# Sample texts
text1 = "What large language models are discussed in the class? List all the LLMs."
text2 = """also keep developing. So the column three now we are using now is very powerful multimodal AI. Now AI
can not only just do text but they can do image and audio. So here actually you can directly say audio to
it. Okay, so this is uh two. How do you like it? Yeah, this one is not that good, right. But this one seems
better, you know, but it's also the number are not that good. So this is an area they can further improve.
Finally, let's see the answer from Gemini Gemny gave us answer, right. Alpha. Go zero. Right. Okay.
From here you can see some preference, right. Some buyers. So each company's two is certainly biased
towards a product of that company, Alpha. Right. Lambda. So different tools. Okay. Yes, darling. Yeah, the
text is darling and also is video generation tools. Right. So also this is very hard I think what's the name of
the company? So last year there was a very, let me see. So maybe I can find it for you guys. This remind
me that. What's that it. Yeah, I just forgot the name. Anyone can remind me. Here was a. What is that?
Yeah, somebody says going deeper. No, that's. Yeah, yeah. Nevermind. So basically the reason. Yeah,
Sora. Yes. Yeah. Thank you. So let's see what's sora currently doing? Sora is part of OpenAI, but the
reason I want to mention this one because even for Sora, for all those video generating also how to
handle text is a big issue because here if you look at the traditional ones, like an image, right. So even
though there may be some issues right in the background, right. So we cannot tell. But if there is some
word there, some text there, we can easily tell whether it is good or bad. Right. Like this one. Right. For
other part you feel good, but for the text you can easily see the issues. Okay. This is also because they
didn't use a separate text generator for the text, I think that's the root cause of the issue. Okay, let's come
back. Okay, so with all those great improvement, right, so what I want to share with you the, the slides,"""

# Vectorize the texts
vector1 = embeddings.embed_query(text1)
vector2 = embeddings.embed_query(text2)

# Compute the cosine similarity
cosine_sim = cosine_similarity([vector1], [vector2])

# Print the cosine similarity
print("Cosine Similarity:", cosine_sim[0][0])
