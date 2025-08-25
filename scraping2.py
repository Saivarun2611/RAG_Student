import requests
from bs4 import BeautifulSoup
import json
import os

URL = "https://catalog.northeastern.edu/graduate/university-interdisciplinary-programs/science-data-ms-bos/#programrequirementstext"

def scrape_courses():
    response = requests.get(URL)
    soup = BeautifulSoup(response.text, "html.parser")
    course_links = soup.find_all("a", href=lambda x: x and "/search/?" in x)

    courses = []
    for link in course_links:
        course_number = link.get_text(strip=True)  # current codecol text
        title_td = link.find_parent("td").find_next_sibling("td")
        course_title = title_td.get_text(strip=True) if title_td else ""
        course_text = f"{course_number}. {course_title}"
        course_url = "https://catalog.northeastern.edu" + link["href"]

        # fetch the course page
        course_page = requests.get(course_url)
        course_soup = BeautifulSoup(course_page.text, "html.parser")

        description_div = course_soup.find("div", class_="courseblock")
        description = description_div.get_text(strip=True) if description_div else "No description found"
        


        courses.append({
            "text": course_text,
            "url": course_url,
            "description": description
        })

    return courses


if __name__ == "__main__":
    courses = scrape_courses()

    
    os.makedirs("data", exist_ok=True)

    # save as JSON
    with open("data/courses2.json", "w") as f:
        json.dump(courses, f, indent=2)

    # check
    print(f"Saved {len(courses)} courses to data/courses.json")
    print("Example course:", courses[0]["text"])
