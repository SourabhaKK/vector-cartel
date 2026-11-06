# Security policy

## Reporting a vulnerability

Please report security issues privately, not in a public issue.

Use GitHub's private reporting: open the repository's **Security** tab and choose **Report a vulnerability**. If that option is not shown, open a public issue that says only that you have a security concern, without details, and I will arrange a private channel.

This is a personal hackathon project, so there is no guaranteed response time. I aim to acknowledge reports within a week.

## Supported versions

Only the latest commit on `main` is supported.

## Scope notes

- API keys belong in a local `.env` file, which is git-ignored. If you find a real key anywhere in this repository or its history, please report it privately.
- `index_store/bm25.pkl` is a pickle file that `src/index.py` loads. Pickle can run code when loaded, so only load index files you built yourself or trust, and rebuild the index (`python -m src.index`) if in doubt.
