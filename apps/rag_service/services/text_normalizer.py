import re
import json

# this both functions format_blocks and normalize_json are used to normalize the json data only

def normalize_json(data: str) -> str:
    if isinstance(data, dict):
        return {k: normalize_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [normalize_json(v) for v in data]
    elif isinstance(data, str):
        value = data.strip()
        
        if value.startswith("/url?q="):
            value = value[7:]
        
        if "&" in value and value.startswith(("http://", "https://")):
            idx = value.find("&")
            if idx != -1:
                value = value[:idx]
        
        value = value.replace("\n", " ").replace("\\", "").replace(".", "").strip()
        
        return value
    else:
        return data


def format_blocks(data, indent=0, none_placeholder="None", show_list_index=True) -> str:
    result = []
    spacing = "  " * indent
    
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, (dict, list)):
                result.append(f"{spacing}{k}:\n")
                result.append(format_blocks(v, indent + 1, none_placeholder, show_list_index))
            else:
                if v is None:
                    v = none_placeholder
                result.append(f"{spacing}{k}: {v}\n")
    
    elif isinstance(data, list):
        for i, v in enumerate(data, 1):
            if show_list_index:
                result.append(f"{spacing}- Item {i}:\n")
            result.append(format_blocks(v, indent + 1, none_placeholder, show_list_index))
    
    else:
        if data is None:
            data = none_placeholder
        result.append(f"{spacing}{data}\n")
    
    return "".join(result)


# Load JSON from file
# with open('filter.json', 'r', encoding='utf-8') as file:
#     raw_json = json.load(file)

# # Normalize
# cleaned_json = normalize_json(raw_json)

# formatted_output = format_blocks(cleaned_json, show_list_index=False)
# print(formatted_output)

# function for refining the txt file data 

def normalize_text(text: str) -> str:

    def clean_url(match):
        url = match.group(1)
        url = re.sub(r'([?&](sa|ved|usg|utm_[^=]+|opi)=[^&]*)', '', url)
        url = re.sub(r'[?&]$', '', url)
        return url

    text = re.sub(r'/url\?q=(https?://[^\s&]+)', clean_url, text)
    text = text.replace('\\n', ' ').replace('\\t', ' ').replace('\\\\', '')

    urls = re.findall(r'https?://[^\s]+', text)
    placeholder = "URLPLACEHOLDER{}"
    url_map = {}
    for i, u in enumerate(urls):
        key = placeholder.format(i)
        text = text.replace(u, key)
        url_map[key] = u

    text = re.sub(r'[!@#$%^&*()_+\[\]{}|;:\'",<>?]', '', text)
    text = text.replace("\n", " ").replace("\\", "").replace(".", "").replace("-", "").replace("=", "").strip()
    text = re.sub(r'\s+', ' ', text)

    for key, u in url_map.items():
        text = text.replace(key, u)

    return text


# Example usage with your text file
# with open("text2.txt", "r", encoding="utf-8") as f:
#     raw_text = f.read()

# cleaned_text = normalize_text(raw_text)
# print(cleaned_text)

# print("Cleaning complete! Check 'cleaned_test_data.txt'.")
