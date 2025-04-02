import requests

# Single domain example
response = requests.get('https://siterank.redirect2.me/api/rank.json?domain=www.angelfire.com')
data = response.json()
print(data)

