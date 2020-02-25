import twint

def scraping_by_time(t1, t2, x, y, k):
    c = twint.Config()
    c.Store_csv = True
    c.Output = "tweets.csv"
    c.Until = t1
    c.Since = t2
    c.Geo = x + "," + y + "," + k + "km"
    twint.run.Search(c)
    # "38.1405227,13.2870764,50km" Pisa
