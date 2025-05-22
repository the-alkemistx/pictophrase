# caption_utils.py
import re
import nltk
from nltk.corpus import stopwords
import language_tool_python

# Download once — check if already downloaded
try:
    _ = stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

# Initialize
stop_words = set(stopwords.words('english'))
tool = language_tool_python.LanguageTool('en-US')

def remove_repetitions(text, window=3):
    """
    Removes consecutive repeating phrases of a given window size.
    """
    words = text.split()
    result = []
    seen_phrases = set()

    i = 0
    while i < len(words):
        phrase = " ".join(words[i:i+window])
        if phrase in seen_phrases:
            i += window  # Skip full phrase, not just one word
        else:
            seen_phrases.add(phrase)
            result.append(words[i])
            i += 1
    return " ".join(result)

def post_process_caption(caption, remove_stopwords=False, grammar_check=True):
    """
    Post-processes the generated caption:
    - Cleans unwanted tokens
    - Removes repetition
    - Optionally removes stopwords
    - Optionally corrects grammar
    """
    # Clean <start> and <end> tokens
    caption = caption.replace("<start>", "").replace("<end>", "").strip()

    # Lowercase and remove excess spaces
    caption = re.sub(r'\s+', ' ', caption).lower()

    # Remove repetitive patterns
    caption = remove_repetitions(caption)

    # Remove stopwords (optional)
    if remove_stopwords:
        caption = ' '.join(word for word in caption.split() if word not in stop_words)

    # Grammar correction (optional and SLOW)
    if grammar_check and caption:
        try:
            caption = tool.correct(caption)
        except Exception as e:
            print(f"⚠️ Grammar tool error: {e}")

    return caption.strip()
