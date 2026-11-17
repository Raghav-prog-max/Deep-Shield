# DEEP SHIELD

DEEP SHIELD is a software-only computer vision project intended to analyze images and videos and present explainable findings for human review. It is under active development; features are documented only as they are implemented and verified.

All findings are advisory, not proof of authenticity, identity, intent, or wrongdoing. Do not use this project for face recognition, identity lookup, or sensitive-attribute inference. Only use media you have permission to analyze.

## Repository contents

- `backend/` — first FastAPI and SQLite backend prototype. See its README for setup, API routes, and current limits.
- `docs/CODEX_INTEGRATION_AUDIT.md` — source baseline and verified status.

The currently available backend supports local media intake and job records. No analysis engine, report generation, or frontend integration is claimed complete.

## Contributing

Use a feature branch and open a pull request against `main`:

```bash
git clone https://github.com/Raghav-prog-max/Deep-Shield.git
cd Deep-Shield
git switch -c feature/short-description
# make your changes
git add <files>
git commit -m "Describe your change"
git push -u origin feature/short-description
```

Do not commit secrets, private footage, personal data, or large model/data files without coordinating with the maintainers.
