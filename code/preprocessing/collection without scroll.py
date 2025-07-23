import json
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import requests

BASE_URL = "https://community.sap.com"
MAIN_URL = "/t5/enterprise-resource-planning-q-a/qa-p/erp-questions"

# Configure Selenium WebDriver
def get_driver():
    options = Options()
    options.headless = False  # Set to False to debug the browser window
    driver_path = 'C:\webdrivers\chromedriver.exe'  # Update this with the path to your ChromeDriver
    service = Service(driver_path)
    driver = webdriver.Chrome(service=service, options=options)
    return driver


def get_main_page(driver):
    driver.get(BASE_URL + MAIN_URL)
    # Wait for the page to load fully
    WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.CSS_SELECTOR, '.custom-message-tile')))
    return BeautifulSoup(driver.page_source, 'html.parser')


def load_more(driver):
    try:
        # Wait until the "Load More" button is clickable
        load_more_button = WebDriverWait(driver, 60).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, '.load-more-button-class'))  # Update with the actual selector
        )

        if load_more_button.is_displayed():
            load_more_button.click()  # Click the 'Load More' button
            print("Clicked 'Load More'")
            # Wait until new content is loaded, assuming the page updates content after clicking 'Load More'
            WebDriverWait(driver, 60).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, '.custom-message-tile'))
            )
            return True
        else:
            print("No 'Load More' button found or visible.")
            return False  # No more content to load

    except Exception as e:
        print(f"Error while clicking 'Load More': {e}")
        return False  # Handle errors like timeouts or element not found

def get_question_details(soup):
    questions = []
    containers = soup.find_all('article', class_='custom-message-tile')

    for container in containers:
        title_tag = container.find('h3').find('a')
        title = ' '.join(title_tag.text.strip().split())  # Ensure proper spacing
        detail_page_url = BASE_URL + title_tag['href']

        replies_tag = container.find('li', class_='custom-tile-replies')
        replies = replies_tag.find('b').text.strip() if replies_tag else "0"

        question_text = ' '.join(container.find('footer').find('p').text.strip().split()) if container.find('footer').find('p') else "Question text not available"

        questions.append({
            "title": title,
            "detail_page_url": detail_page_url,
            "replies": replies,
            "question_text": question_text
        })

    return questions


def get_complete_question(detail_page_url):
    # Fetch the complete question text from the detail page
    try:
        response = requests.get(detail_page_url)
        response.raise_for_status()
        detail_soup = BeautifulSoup(response.content, 'html.parser')

        # Assuming the question is inside a <div> or a specific tag
        question_container = detail_soup.find('div', class_='lia-message-body-content')  # Adjust this based on actual HTML
        question_text = ' '.join(question_container.get_text(strip=True).split()) if question_container else "No question text found"

        return question_text
    except Exception as e:
        print(f"Error fetching question details: {e}")
        return "Error fetching question details"


def get_answers(detail_page_url):
    # Fetching answers from the detail page
    try:
        response = requests.get(detail_page_url)
        response.raise_for_status()
        detail_soup = BeautifulSoup(response.content, 'html.parser')

        # Find all sections where answers are located
        answer_elements = detail_soup.find_all('div', class_='lia-message-body-content')

        answers = []
        for ans in answer_elements:
            # Exclude content inside <div class="lia-comment-main-wrapper">
            comment_wrappers = ans.find_all('div', class_='lia-comment-main-wrapper')
            for wrapper in comment_wrappers:
                wrapper.decompose()  # Remove the unwanted wrapper content

            # Extract the cleaned answer text
            answer_text = ' '.join(ans.get_text(strip=True).split())  # Ensure proper spacing
            if answer_text:
                answers.append(answer_text)

        return answers
    except Exception as e:
        print(f"Error fetching answers: {e}")
        return ["Error fetching answers"]


def main():
    driver = get_driver()
    main_soup = get_main_page(driver)

    start_time = time.time()  # Record the start time
    max_duration = 60 * 60  # 1 hour in seconds (3600 seconds)
    total_questions_collected = 0  # Track the number of collected questions
    target_questions = 1000  # Set the target to 1000 questions

    load_more_found = True
    while load_more_found and total_questions_collected < target_questions:
        elapsed_time = time.time() - start_time  # Calculate elapsed time
        if elapsed_time >= max_duration:
            print("One hour has passed. Stopping.")
            break  # Stop if 1 hour has passed

        try:
            load_more_found = load_more(driver)
            # Increase the sleep time to ensure more time for loading
            time.sleep(15)  # Increased sleep time to give more time for loading new content

            # Re-fetch the page content after clicking 'Load More'
            main_soup = BeautifulSoup(driver.page_source, 'html.parser')
            question_details = get_question_details(main_soup)

            # Add newly collected questions
            total_questions_collected += len(question_details)
            print(f"Collected {total_questions_collected} questions so far.")
        except Exception as e:
            print(f"Error during loading more questions: {e}")
            break  # Exit if there's an error during loading more content

    print(f"Total Questions Collected: {total_questions_collected}")
    # After loading more content, parse the page again
    main_soup = BeautifulSoup(driver.page_source, 'html.parser')

    # Extract all question details from the loaded page
    question_details = get_question_details(main_soup)

    # List to store data in SQuAD format
    squad_data = []

    # Extract data for each question
    for question in question_details:
        question_title = question['title']
        question_text = question['question_text']
        replies = question['replies']

        # Get complete question from the answer page
        complete_question = get_complete_question(question['detail_page_url'])

        # Get answers from the detail page if there are replies
        if int(replies) > 0:
            answers = get_answers(question['detail_page_url'])
        else:
            answers = ["No answers yet."]

        # Create a SQuAD format entry for each question
        qas = []
        for index, answer in enumerate(answers):
            if index == 0:  # Skip the first answer (index 0) to avoid redundant data
                continue
            qas.append({
                "question": question_text,
                "id": f"{question_title}-{index}",
                "answers": [{"text": answer, "answer_start": 0}],  # Placeholder for start position
            })

        # Each question is a new entry in the data with title and context
        squad_data.append({
            "title": question_title,
            "paragraphs": [{
                "context": complete_question,
                "qas": qas
            }]
        })

        print(f"Processed question: {question_title}")
        print("-" * 40)

    # Save the data in SQuAD format to a JSON file
    with open('sap_questions_answers_squad2.json', 'w', encoding='utf-8') as json_file:
        json.dump({"data": squad_data}, json_file, ensure_ascii=False, indent=4)

    driver.quit()


if __name__ == "__main__":
    main()

