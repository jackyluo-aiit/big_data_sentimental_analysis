from textblob import TextBlob

testimonial = TextBlob("ha- good job. that's right - we gotta throw that tag EVERYWHERE! I wanna get it trending before I start")
print(testimonial.polarity)
print(testimonial.words)