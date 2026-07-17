# Shivom Gupta — Academic Portfolio

A responsive, accessible portfolio emphasizing human physiology, cognitive neuroscience, experimental design, reproducible scientific computing and research leadership. It deploys directly on GitHub Pages.

## Local preview

```bash
python -m http.server 8000
```

Open `http://localhost:8000`. No build step or package installation is required.

## Structure

- `index.html` — main portfolio
- `research-software.html` — reviewed research-code and experiment portfolio
- `code/` — public, repository-ready project bundles with technical notes
- `assets/css/portfolio.css` — design system and responsive layout
- `assets/js/portfolio.js` — mobile navigation, persistent theme control and copyright year
- `images/` — portraits, project images and CV
- `sitemap.xml`, `robots.txt` — search engine discovery

## Deployment

Push the site to the default branch of the `Shivom7/Shivom7.github.io` repository. In **Settings → Pages**, select **Deploy from a branch**, then the repository root. GitHub Pages will publish it at `https://shivom7.github.io/`.

## Content updates

Project descriptions are based on a static review of the supplied source. Before scientific use, execute the projects with their compatible environments, datasets and acquisition hardware, then retain the generated quality-control evidence. The three folders under `code/` can be moved into separate Git repositories if independent versioning is desired.
