import json
import re

def clean_text(text: str) -> str:
    # Replace non-breaking spaces and weird whitespace with normal spaces
    return text.replace("\u00a0", " ").strip()

# Load the raw courses.json
with open("data/courses2.json", "r") as f:
    courses = json.load(f)

processed_courses = []

for course in courses:
    raw_text = clean_text(course["text"])  # normalize spaces
    description = clean_text(course["description"])
    url = course["url"]

    # Extract course number (pattern: letters + digits)
    match_number = re.match(r"([A-Z]+\s*\d+)", raw_text)
    course_number = match_number.group(1) if match_number else ""

    # Extract credits
    match_credits = re.search(r"\((\d+)\s*Hours\)", raw_text)
    credits = int(match_credits.group(1)) if match_credits else None

    # Extract title
    title_part = re.sub(r"^[A-Z]+\s*\d+\.\s*", "", raw_text)  # remove course number
    title_part = re.sub(r"\(\d+\s*Hours\)", "", title_part)   # remove credits
    title = clean_text(title_part.strip(" ."))

    # Build document (I'll embed this)
    document = f"Course {course_number} - {title} ({credits} credits). {description}"

    processed_courses.append({
        "course_number": course_number,
        "title": title,
        "credits": credits,
        "url": url,
        "description": description,
        "document": document
    })

# Save processed_courses2.json
with open("data/processed_courses2.json", "w") as f:
    json.dump(processed_courses, f, indent=2, ensure_ascii=False)

print(" Preprocessing complete! Saved to data/processed_courses.json")
