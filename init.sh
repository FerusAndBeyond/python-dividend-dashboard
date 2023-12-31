env_content="# note if \$OPENAI_API_KEY is set, it will override below value
OPENAI_API_KEY=...
FMP_API_KEY=...
OPENAI_MODEL=gpt-3.5-turbo"

if [ -f ".env" ]; then
    echo ".env already exists"
else
    (
        echo "creating .env" &&
        touch .env &&
        echo "$env_content" >> .env
    )
fi