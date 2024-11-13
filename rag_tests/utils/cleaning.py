import re
import unidecode
# TODO: fix unbolding and unitalicising

# for now we are using unidecode to normalise the text

# def unbold_text(text):
#     # Mapping of bold numbers to their regular equivalents
#     bold_numbers = {
#         "𝟬": "0",
#         "𝟭": "1",
#         "𝟮": "2",
#         "𝟯": "3",
#         "𝟰": "4",
#         "𝟱": "5",
#         "𝟲": "6",
#         "𝟳": "7",
#         "𝟴": "8",
#         "𝟵": "9",
#     }

#     # Function to convert bold characters (letters and numbers)
#     def convert_bold_char(match):
#         char = match.group(0)
#         # Convert bold numbers
#         if char in bold_numbers:
#             return bold_numbers[char]
#         # Convert bold uppercase letters
#         elif "\U0001d5d4" <= char <= "\U0001d5ed":
#             return chr(ord(char) - 0x1D5D4 + ord("A"))
#         # Convert bold lowercase letters
#         elif "\U0001d5ee" <= char <= "\U0001d607":
#             return chr(ord(char) - 0x1D5EE + ord("a"))
#         else:
#             return char  # Return the character unchanged if it's not a bold number or letter

#     # Regex for bold characters (numbers, uppercase, and lowercase letters)
#     bold_pattern = re.compile(
#         r"[\U0001D5D4-\U0001D5ED\U0001D5EE-\U0001D607\U0001D7CE-\U0001D7FF]"
#     )
#     text = bold_pattern.sub(convert_bold_char, text)

#     return text


# def unitalic_text(text):
#     # Function to convert italic characters (both letters)
#     def convert_italic_char(match):
#         char = match.group(0)
#         # Unicode ranges for italic characters
#         if "\U0001d608" <= char <= "\U0001d621":  # Italic uppercase A-Z
#             return chr(ord(char) - 0x1D608 + ord("A"))
#         elif "\U0001d622" <= char <= "\U0001d63b":  # Italic lowercase a-z
#             return chr(ord(char) - 0x1D622 + ord("a"))
#         else:
#             return char  # Return the character unchanged if it's not an italic letter

#     # Regex for italic characters (uppercase and lowercase letters)
#     italic_pattern = re.compile(r"[\U0001D608-\U0001D621\U0001D622-\U0001D63B]")
#     text = italic_pattern.sub(convert_italic_char, text)

#     return text


def remove_emojis_and_symbols(text):
    # Extended pattern to include specific symbols like ↓ (U+2193) or ↳ (U+21B3)
    emoji_and_symbol_pattern = re.compile(
        "["
        "\U0001f600-\U0001f64f"  # emoticons
        "\U0001f300-\U0001f5ff"  # symbols & pictographs
        "\U0001f680-\U0001f6ff"  # transport & map symbols
        "\U0001f1e0-\U0001f1ff"  # flags (iOS)
        "\U00002193"  # downwards arrow
        "\U000021b3"  # downwards arrow with tip rightwards
        "\U00002192"  # rightwards arrow
        "]+",
        flags=re.UNICODE,
    )

    return emoji_and_symbol_pattern.sub(r" ", text)


def replace_urls_with_placeholder(text, placeholder="[URL]"):
    # Regular expression pattern for matching URLs
    url_pattern = r"https?://\S+|www\.\S+"

    return re.sub(url_pattern, placeholder, text)


def remove_non_ascii(text: str) -> str:
    text = text.encode("ascii", "ignore").decode("ascii")
    return text

def replace_newlines(text: str) -> str:
    text = text.replace("\n\n", "<DoubleLineBreak/>")
    text = text.replace("\n", "<NewLine/>")
    return text
def remove_hashtags(text: str) -> str:
    text = re.sub(r'#\s*(\w+)', '', text)  # Remove hashtag, any following whitespace, and the word
    text = text.strip()  # Remove any leading/trailing whitespace
    return text

def clean_text(text_content: str) -> str:
    text_with_modified_newlines = replace_newlines(text_content)
    text_with_removed_emojis = remove_emojis_and_symbols(text_with_modified_newlines)

    text_with_removed_hashtags = remove_hashtags(text_with_removed_emojis)
    text_with_replaced_urls = replace_urls_with_placeholder(text_with_removed_hashtags)

    normalised_text = unidecode.unidecode(text_with_replaced_urls)

    return normalised_text