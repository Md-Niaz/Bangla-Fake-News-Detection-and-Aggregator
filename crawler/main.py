# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.common.keys import Keys
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from selenium.webdriver.chrome.options import Options
# import pandas as pd

# options = Options()
# options.add_argument("--headless")

# # Setup WebDriver
# driver = webdriver.Chrome(options=options)  # Or replace with the WebDriver for your browser
# url = "https://www.bd-pratidin.com/"
# list_of_articles = []
# list_of_titles = []

# try:
#     # Navigate to the target URL
#     driver.get(url)

#     # Wait for the <a> tags to be present
#     links = WebDriverWait(driver, 5).until(
#         EC.presence_of_all_elements_located((By.TAG_NAME, "a"))
#     )

#     # Extract and print the href attributes
#     hrefs = [link.get_attribute("href") for link in links if link.get_attribute("href")]
#     for href in hrefs:
#         splitted_url = href.split('/')
        
#         if len(splitted_url) == 8:
#             list_of_articles.append(href)
#     count = 0
#     for aricle in list_of_articles:
#         count += 1
#         print(f"getting article from: {aricle}, {count}/{len(list_of_articles)}")
#         driver.get(aricle)
#         title = driver.find_element(By.CLASS_NAME, "n_head")
#         print(title.text)
#         list_of_titles.append(title.text.encode('utf-8').decode('utf-8'))
#         if count == 5:
#             break

#     title_data = pd.DataFrame(list_of_titles)
#     title_data.to_csv('title_data.csv', index=False)
    
# finally:
#     driver.quit()


from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import pandas as pd

# Configure Selenium WebDriver options
options = Options()
options.add_argument("--headless")
driver = webdriver.Chrome(options=options)

# Target URL and storage lists
url = "https://www.bd-pratidin.com/"
list_of_articles = []
list_of_titles = []

try:
    # Navigate to the target URL
    driver.get(url)

    # Wait for the <a> tags to be present
    links = WebDriverWait(driver, 5).until(
        EC.presence_of_all_elements_located((By.TAG_NAME, "a"))
    )

    # Extract href attributes
    hrefs = [link.get_attribute("href") for link in links if link.get_attribute("href")]
    for href in hrefs:
        splitted_url = href.split('/')
        if len(splitted_url) == 8:
            list_of_articles.append(href)

    # Visit each article and extract titles
    count = 0
    for article in list_of_articles:
        count += 1
        print(f"Fetching article from: {article}, {count}/{len(list_of_articles)}")
        driver.get(article)
        
        try:
            # Extract article title
            title = WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.CLASS_NAME, "n_head"))
            ).text
            list_of_titles.append({"url": article, "title": title})
        except Exception as e:
            print(f"Failed to fetch title for {article}: {e}")
        
        # Fetch only the first 5 articles for testing
        if count == 5:
            break

    # Save data to a CSV file
    df = pd.DataFrame(list_of_titles)
    df.to_csv('articles_and_titles.csv', index=False, encoding='utf-8')
    print("Data saved to 'articles_and_titles.csv'.")

finally:
    driver.quit()

