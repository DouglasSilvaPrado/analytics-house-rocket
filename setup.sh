mkdir -p~/.stremlit/

echo"\
[general]\n\
email = \"douglassilvadaprado@gmail.com\"\n\
"> ~/.streamlit//credentials.toml

echo "\
[serve]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
">~/.streamlit/config.toml