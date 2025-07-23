import json
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://community.sap.com"


def get_question_from_detail_page(detail_page_url):
    try:
        # Fetch the detail page content
        response = requests.get(detail_page_url)
        response.raise_for_status()

        # Parse the HTML content of the detail page
        detail_soup = BeautifulSoup(response.content, 'html.parser')

        # Extract the question text from the body section
        question_text_section = detail_soup.find('div', class_='lia-message-body-content')

        if question_text_section:
            question_text = clean_html_content(str(question_text_section))
            return question_text
        else:
            return "Question text not found"

    except Exception as e:
        print(f"Error fetching question: {e}")
        return "Error fetching question"
def get_question_details(soup):
    questions = []
    containers = soup.find_all('div', class_='lia-message-subject-wrapper')

    for container in containers:
        # Extract title from the <h2> tag inside the container
        title_tag = container.find('h2', class_='message-subject').find('a')
        title = ' '.join(title_tag.text.strip().split())  # Ensure proper spacing
        detail_page_url = BASE_URL + title_tag['href']

        # Extract question text from the body section
        question_text = ' '.join(container.find_next('div', class_='lia-message-body-content').find('div',
                                                                                                    class_='lia-truncated-body-container').text.strip().split()) if container.find_next(
            'div', class_='lia-message-body-content') else "Question text not available"

        questions.append({
            "title": title,
            "detail_page_url": detail_page_url,
            "question_text": question_text
        })

    return questions


def clean_html_content(content):
    # This function will clean the HTML content by extracting the text and adding spaces
    # It will handle <p>, <a>, <em>, <strong>, and other tags appropriately
    soup = BeautifulSoup(content, 'html.parser')

    # Replace all <p> tags and other tags with their text content, ensuring spaces are added
    paragraphs = soup.find_all(['p', 'a', 'em', 'strong', 'br', 'span'])

    cleaned_content = ''
    for tag in paragraphs:
        if tag.name == 'p':
            cleaned_content += ' ' + tag.get_text(strip=True) + ' '
        else:
            cleaned_content += ' ' + tag.get_text(strip=True) + ' '

    # Return the cleaned content with proper spacing
    return ' '.join(cleaned_content.split())


def get_accepted_solution(detail_page_url):
    # Fetch the accepted solution and title from the detail page
    try:
        response = requests.get(detail_page_url)
        response.raise_for_status()
        detail_soup = BeautifulSoup(response.content, 'html.parser')

        # Extract the title of the detail page
        title_tag = detail_soup.find('h2', {'itemprop': 'name', 'class': 'message-subject'})
        page_title = title_tag.text.strip() if title_tag else "Title not found"

        # Locate the "Accepted Solutions" section
        accepted_solution_section = detail_soup.find('div',
                                                     class_='ComponentToggler lia-component-solutions-with-toggle')

        if accepted_solution_section:
            # Extract the answer text from the accepted solution
            solution_text = accepted_solution_section.find('div', class_='lia-message-body-content')
            if solution_text:
                cleaned_solution = clean_html_content(str(solution_text))
                return page_title, cleaned_solution
            else:
                return page_title, "No solution text found"
        else:
            return page_title, "No accepted solution found"
    except Exception as e:
        print(f"Error fetching accepted solution: {e}")
        return "Error fetching title", "Error fetching accepted solution"


def scrape_sap_forums(url,data):
    # Replace with the URL of the page you want to scrape

    # Fetch the page content
    response = requests.get(url)
    response.raise_for_status()

    # Parse the HTML content
    soup = BeautifulSoup(response.content, 'html.parser')

    # Get list of questions
    questions = get_question_details(soup)



    for question in questions:
        page_title, accepted_solution = get_accepted_solution(question['detail_page_url'])

        print(f"Title: {page_title}")
        print(f"Detail Page URL: {question['detail_page_url']}")
        question_text = get_question_from_detail_page(question['detail_page_url'])
        print(f"Question Text: {question_text}")


        # Get accepted solution from detail page

        print(f"Accepted Solution: {accepted_solution}")
        print("=" * 50)


        # Save question and answer in the specified format
        question_data = {
            "title":  page_title,
            "paragraphs": [
                {
                    "context": " ",
                    "qas": [
                        {
                            "question":question_text ,
                            "id": "1",
                            "answers": [
                                {
                                    "text": accepted_solution
                                }
                            ]
                        }
                    ]
                }
            ]
        }
        data["data"].append(question_data)
    return data

    # Save the data to a JSON file

if __name__ == "__main__":

    data = {"data": []}

    for i in range(15, 200):
        j = str(i)

        url = "https://community.sap.com/t5/forums/searchpage/tab/message?filter=location&q=which&noSynonym=false&advanced=true&location=qanda-board:erp-questions&page="+j+"&collapse_discussion=true&search_type=thread&search_page_size=50"
        data = scrape_sap_forums(url, data)
        with open('sap_forum_data_which2.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)


