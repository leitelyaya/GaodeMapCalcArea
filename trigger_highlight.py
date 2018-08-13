import time
import os

from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

os.environ.setdefault("webdriver.chrome.bin", r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe")
os.environ.setdefault("webdriver.chrome.driver", r"D:\Program Files\chromedriver\chromedriver.exe")

o = open("extsource/other")
o2 = open("extsource/other2", "w")
o3 = open("extsource/other3", "w")

if __name__ == '__main__':
    browser = webdriver.Chrome(executable_path=os.environ.get("webdriver.chrome.driver"))
    for line in o:
        no = line.strip()
        browser.get("http://localhost:5000/%s"%no)

        try:
            locator = (By.CSS_SELECTOR, '.error')
            WebDriverWait(browser, 5, 0.5).until(EC.presence_of_element_located(locator))

            convas = browser.find_element_by_css_selector(".error")
            o2.write(line)
        except:
            o3.write(line)
        finally:
            # browser.quit()
            pass