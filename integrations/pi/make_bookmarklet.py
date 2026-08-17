# make_bookmarklet.py — generates the bookmark URL from bookmarklet.js.
#
# Run:
#   python3 integrations/pi/make_bookmarklet.py
#
# This outputs a single "javascript:..." string that you paste as the URL of
# a browser bookmark. The script strips comments and compresses whitespace
# so the URL is short enough that browsers don't truncate it.

import re           # for stripping comments
import sys          # for printing to stdout
import urllib.parse # for percent-encoding the JavaScript

# Read the raw bookmarklet source.
src_path = __file__.replace("make_bookmarklet.py", "bookmarklet.js")
with open(src_path, "r") as f:
    js = f.read()

# Remove block comments (/* ... */).
js = re.sub(r'/\*.*?\*/', '', js, flags=re.DOTALL)

# Remove single-line comments (// ...) but only when they appear on their own
# line or after code. We keep the logic simple: strip from // to end-of-line.
# This regex avoids stripping URLs (http://) by requiring whitespace before //.
js = re.sub(r'(?m)^\s*//.*$', '', js)   # full-line comments
js = re.sub(r'\s//[^\n]*', '', js)      # trailing comments after code

# Collapse runs of whitespace (newlines, tabs, multiple spaces) into one space.
js = re.sub(r'\s+', ' ', js).strip()

# Wrap in the javascript: protocol.
bookmarklet_url = "javascript:" + urllib.parse.quote(js, safe="(){}[];,'\"=!<>&|+-*/%^~?.:")

# Write to bookmarklet_url.txt so the user can open the file and copy from it.
out_path = src_path.replace("bookmarklet.js", "bookmarklet_url.txt")
with open(out_path, "w") as f:
    f.write(bookmarklet_url)

print("Bookmarklet URL written to:", out_path)
print()
print("Next steps:")
print("  1. Open", out_path)
print("  2. Select all and copy the entire contents")
print("  3. In your browser, create a new bookmark")
print("  4. Paste the copied text as the bookmark URL (not the name)")
print("  5. Save — then navigate to pi.ai and click the bookmark to test")
